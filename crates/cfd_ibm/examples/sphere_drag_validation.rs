//! **Resolved single-sphere drag `Cd(Re)` via the 3-D ghost-cell IBM** — the 3-D
//! analogue of `cylinder_drag_validation`, and the mandatory *first* step of the
//! resolved DEM-CFD track: a live 3-D Navier–Stokes solve past a `cfd_ibm::Sphere`
//! recovering the standard-drag coefficient, matched to an **independent**
//! correlation (Schiller & Naumann 1935; cross-checked against Clift, Grace &
//! Weber 1978), reusing the exact refinement-independent surface-marker traction
//! integrator (`MarkerForce`) the resolved seam hands to the DEM particle.
//!
//! A sphere of diameter `D` sits in a uniform stream `U` (subsonic velocity inlet,
//! pressure outlet, symmetry far-field on the four lateral faces). The Reynolds
//! number `Re = ρ U D / μ` is pinned by the constant viscosity `μ = ρ U D / Re`
//! (an incompressible-limit case run at low Mach in this compressible solver).
//! The drag coefficient is
//!
//! ```text
//!   Cd = F_x / (½ ρ U² · π R²),   F_x = marker-integrated pressure+viscous load.
//! ```
//!
//! ## Why the chosen `Re` is steady, and the anti-transient gate
//!
//! Flow past a sphere is **steady & axisymmetric** up to `Re ≈ 210` (first
//! bifurcation), so a valid result is a *plateau*: `Cd(t)` must go flat. As in the
//! cylinder case, this compressible solver marched to low Mach traps acoustic
//! energy against the reflective inlet/outlet, so a downstream **sponge** relaxes
//! the solution toward the undisturbed freestream over the last stretch of the
//! wake, letting outgoing waves exit. The `[validation]` block gates on BOTH (1)
//! `Cd` within `tol_rel` of the correlation and (2) a steady plateau (`Cd`
//! peak-to-peak over the averaging window below `steady_ptp_tol`) — a run that
//! merely passes *through* the reference while ringing is rejected.
//!
//! The tolerance is set from the **grid-convergence** behaviour of this
//! first-order staircase/ghost-cell IBM (the drag converges ≈linearly in `Δx`),
//! measured across `D/Δx` levels — it is *not* tuned to pass. Everything
//! case-specific (geometry, `Re`, mesh, run length, sponge, reference value +
//! tolerance) is declarative, read from the TOML passed as `argv[1]`:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example sphere_drag_validation -- \
//!     crates/cfd_ibm/examples/sphere_drag_re20.toml
//! ```

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, PressureOutlet, SubsonicInflow, Symmetry};
use cfd_eos::{Eos, IdealGas, Viscosity};
use cfd_ibm::{cd_clift_grace_weber, cd_schiller_naumann, GhostCellIbmPlugin, MarkerForce, SphereBody, SphereIbmPlugin, WallForce};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, RkStage, SolverConfig,
    SolverPlugin, SolverState,
};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, StructuredMesh,
    UniformMesh, UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_scheduler::Res;
use serde::Deserialize;

const R_GAS: f64 = 287.058; // matches IdealGas::air()

/// `[flow]`: freestream state + Reynolds number (sets μ).
#[derive(Deserialize, Default)]
struct FlowCfg {
    rho: f64,
    p: f64,
    u_inf: f64,
    reynolds: f64,
}

/// `[sphere]`: geometry (center + diameter); markers = surface quadrature count.
#[derive(Deserialize, Default)]
struct SphereCfg {
    cx: f64,
    cy: f64,
    cz: f64,
    diameter: f64,
    #[serde(default = "default_markers")]
    markers: usize,
}
fn default_markers() -> usize {
    2000
}

/// `[run]`: iteration budget and trailing fraction to average / test steadiness.
#[derive(Deserialize, Default)]
struct RunCfg {
    steps: usize,
    print_every: usize,
    average_frac: f64,
}

/// `[validation]`: cited correlation value + tolerances.
#[derive(Deserialize, Default)]
struct ValidationCfg {
    cd_ref: f64,
    tol_rel: f64,
    steady_ptp_tol: f64,
}

/// `[sponge]`: downstream absorbing layer relaxing to the uniform freestream.
#[derive(Deserialize, Default)]
struct SpongeCfg {
    x_start: f64,
    sigma: f64,
}

/// Uniform-freestream sponge parameters (see [`SpongeCfg`]).
#[derive(Clone, Copy)]
struct Sponge {
    x_start: f64,
    x_end: f64,
    sigma: f64,
    target: ConsVar,
}

/// `PostUpdate` (once per completed RK step): relax the conserved state toward the
/// undisturbed freestream inside the downstream sponge with a quadratic ramp
/// `σ(x)=σ·s²`, `s=(x−x_start)/(x_end−x_start)`; per-step blend `min(σ(x)·Δt,1)`.
/// A passive absorber (can only pull toward the freestream), applied only on a
/// completed step so it does not disturb the RK3 stage registers.
fn outlet_sponge_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    sponge: Res<Sponge>,
    sstate: Res<SolverState>,
) {
    if sstate.rk_stage != RkStage::First {
        return;
    }
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let ng = mesh.n_ghost();
    let [ni, nj, nk] = mesh.dims();
    let dt = sstate.dt;
    let (x0, x1, sig) = (sponge.x_start, sponge.x_end, sponge.sigma);
    let t = sponge.target;
    let inv_w = 1.0 / (x1 - x0);
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let idx = mesh.idx_raw(ir, jr, kr);
                let c = mesh.cell_centroid(idx);
                if c[0] < x0 {
                    continue;
                }
                let s = ((c[0] - x0) * inv_w).clamp(0.0, 1.0);
                let f = (sig * s * s * dt).min(1.0);
                let u = &mut state.u[idx];
                u.rho += f * (t.rho - u.rho);
                u.rho_u += f * (t.rho_u - u.rho_u);
                u.rho_v += f * (t.rho_v - u.rho_v);
                u.rho_w += f * (t.rho_w - u.rho_w);
                u.rho_e += f * (t.rho_e - u.rho_e);
            }
        }
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: sphere_drag_validation <case.toml>  (see crates/cfd_ibm/examples/sphere_drag_re20.toml)");
    let toml_src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let mesh_cfg: UniformMeshConfig = cfg.section("grid");
    let flow: FlowCfg = cfg.section("flow");
    let sc: SphereCfg = cfg.section("sphere");
    let run: RunCfg = cfg.section("run");
    let valid: ValidationCfg = cfg.section("validation");
    let sponge_cfg: SpongeCfg = cfg.section("sponge");
    let mut solver_cfg: SolverConfig = cfg.section("solver");
    solver_cfg.viscous = true; // resolved viscous drag: always a Navier–Stokes case

    let d = sc.diameter;
    let r = 0.5 * d;
    let mu = flow.rho * flow.u_inf * d / flow.reynolds;
    let dx = (mesh_cfg.bounds_hi[0] - mesh_cfg.bounds_lo[0]) / mesh_cfg.nx as f64;
    let mesh_hi_x = mesh_cfg.bounds_hi[0];
    let (nx, ny, nz) = (mesh_cfg.nx, mesh_cfg.ny, mesh_cfg.nz);

    let eos = IdealGas::air();
    let t_inf = flow.p / (flow.rho * R_GAS);
    let freestream = eos.prim_to_cons(&PrimVar::new(flow.rho, flow.u_inf, 0.0, 0.0, flow.p, t_inf));
    let init = move |_x: Vec3| freestream;

    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, SubsonicInflow { rho: flow.rho, u: flow.u_inf, v: 0.0, w: 0.0, t: t_inf })
        .with(BoundarySide::XHi, PressureOutlet { p: flow.p })
        .with_axis(1, Symmetry) // lateral far-field (symmetry tunnel)
        .with_axis(2, Symmetry);

    let sphere = SphereBody::fixed([sc.cx, sc.cy, sc.cz], r);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    app.add_resource(Viscosity::Constant(mu));
    app.add_resource(Sponge {
        x_start: sponge_cfg.x_start,
        x_end: mesh_hi_x,
        sigma: sponge_cfg.sigma,
        target: freestream,
    });
    app.add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_update_system(outlet_sponge_system, MeshScheduleSet::PostUpdate)
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(solver_cfg))
        .add_plugins(GhostCellIbmPlugin)
        .add_plugins(SphereIbmPlugin::new(vec![sphere], freestream).with_markers(sc.markers));
    app.prepare();

    let re_actual = flow.rho * flow.u_inf * d / mu;
    let m_inf = flow.u_inf / eos.sound_speed(&freestream);
    let q = 0.5 * flow.rho * flow.u_inf * flow.u_inf * std::f64::consts::PI * r * r; // ½ρU²·πR²
    let cd_of = |fx: f64| fx / q;
    let cd_sn = cd_schiller_naumann(re_actual);
    let cd_cgw = cd_clift_grace_weber(re_actual);

    println!("# Resolved single-sphere drag Cd(Re) — 3-D ghost-cell IBM");
    println!("# Re = {re_actual:.3}   mu = {mu:.6}   U = {:.3}   M = {m_inf:.3}", flow.u_inf);
    println!("# D = {d}   D/dx = {:.1}   cells = {nx}x{ny}x{nz}   markers/sphere = {}", d / dx, sc.markers);
    println!("# Cd(SN 1935) = {cd_sn:.4}   Cd(CGW 1978) = {cd_cgw:.4}   |SN−CGW|/CGW = {:.2}%", 100.0 * (cd_sn - cd_cgw).abs() / cd_cgw);
    println!("# step        Cd(marker)   Cd(staircase)");

    let avg_start = ((1.0 - run.average_frac) * run.steps as f64) as usize;
    let mut cd_samples: Vec<f64> = Vec::new();
    for step in 0..run.steps {
        app.run();
        let marker = app.get_resource_ref::<MarkerForce>().unwrap().total();
        if step >= avg_start {
            cd_samples.push(cd_of(marker[0]));
        }
        if run.print_every > 0 && (step % run.print_every == 0 || step + 1 == run.steps) {
            let wall = app.get_resource_ref::<WallForce>().unwrap().total();
            println!("{step:>8}    {:>10.4}    {:>10.4}", cd_of(marker[0]), cd_of(wall[0]));
        }
    }

    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        assert!(
            state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0 && u.rho_e.is_finite()),
            "sphere run went non-physical",
        );
    }

    let n = cd_samples.len().max(1) as f64;
    let cd = cd_samples.iter().sum::<f64>() / n;
    let cd_max = cd_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cd_min = cd_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let cd_ptp_rel = if cd.abs() > 0.0 { (cd_max - cd_min) / cd.abs() } else { f64::INFINITY };
    let rel = (cd - valid.cd_ref).abs() / valid.cd_ref;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!("# Cd (marker, averaged over last {:.0}%): {cd:.4}", 100.0 * run.average_frac);
    println!("# Cd reference (correlation): {:.4}", valid.cd_ref);
    println!("# Cd relative error: {:.2}%   (tolerance {:.1}%)", 100.0 * rel, 100.0 * valid.tol_rel);
    println!("# Cd peak-to-peak over window: {:.2}%   (steady gate {:.1}%)", 100.0 * cd_ptp_rel, 100.0 * valid.steady_ptp_tol);
    let pass_cd = rel <= valid.tol_rel;
    let pass_steady = cd_ptp_rel <= valid.steady_ptp_tol;
    if pass_cd && pass_steady {
        println!(
            "VALIDATION: PASS  (Cd = {cd:.4} vs {:.4}, {:.2}% <= {:.1}%; ptp {:.2}% <= {:.1}%)",
            valid.cd_ref, 100.0 * rel, 100.0 * valid.tol_rel, 100.0 * cd_ptp_rel, 100.0 * valid.steady_ptp_tol,
        );
    } else {
        println!(
            "VALIDATION: FAIL  (Cd_ok={pass_cd} [{:.2}% vs {:.1}%]  steady_ok={pass_steady} [ptp {:.2}% vs {:.1}%])",
            100.0 * rel, 100.0 * valid.tol_rel, 100.0 * cd_ptp_rel, 100.0 * valid.steady_ptp_tol,
        );
        std::process::exit(1);
    }
}
