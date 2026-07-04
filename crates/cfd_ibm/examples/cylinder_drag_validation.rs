//! **Laminar cylinder drag via the ghost-cell IBM** — the Schäfer & Turek (1996)
//! DFG flow-around-a-cylinder benchmark, case 2D-1 (steady, `Re = 20`).
//!
//! A circular cylinder (`D`) sits in a plane channel (`H`) with a parabolic inflow
//! `u(y) = 4 Uₘ y (H − y) / H²`, mean velocity `Ū = ⅔ Uₘ`, no-slip channel walls,
//! and a pressure outlet. The Reynolds number `Re = Ū D / ν = 20` is set by the
//! constant viscosity `μ = ρ Ū D / Re` (an incompressible-limit case run at a low
//! Mach number in this compressible solver). The immersed cylinder is resolved by
//! the sharp-interface ghost-cell IBM; the drag comes from the refinement-
//! independent surface-marker traction integration (`MarkerForce`).
//!
//! The **reference value** is the widely-quoted benchmark drag coefficient
//! `c_D = 5.5795` (Schäfer & Turek 1996; consensus ≈ 5.57–5.59, e.g. John 2004),
//! with `c_D = 2 F_x / (ρ Ū² D L_z)`. The tolerance in the `[validation]` section
//! reflects the first-order staircase/ghost-cell IBM's expected drag error at this
//! resolution — it is *not* tuned to pass.
//!
//! Everything case-specific (geometry, channel, Reynolds number, mesh, run length,
//! reference value + tolerance) is **declarative**, read from the TOML passed as
//! `argv[1]`. Run:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example cylinder_drag_validation -- \
//!     crates/cfd_ibm/examples/cylinder_drag_re20.toml
//! ```

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, NoSlipWall, PressureOutlet, SlipWall, SupersonicOutflow};
use cfd_eos::{Eos, EosResource, IdealGas, Viscosity};
use cfd_ibm::{Cylinder, CylinderIbmPlugin, GhostCellIbmPlugin, MarkerForce, WallForce};
use cfd_solver::{CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, StructuredMesh,
    UniformMesh, UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_scheduler::Res;
use serde::Deserialize;

const R_GAS: f64 = 287.058; // matches IdealGas::air()

/// `[flow]`: freestream + Reynolds number (mean velocity is the characteristic one).
#[derive(Deserialize, Default)]
struct FlowCfg {
    rho: f64,
    p: f64,
    u_mean: f64,
    reynolds: f64,
}

/// `[cylinder]`: geometry and the channel height that defines the parabolic inlet.
#[derive(Deserialize, Default)]
struct CylinderCfg {
    cx: f64,
    cy: f64,
    diameter: f64,
    channel_height: f64,
}

/// `[run]`: iteration budget and the trailing fraction to time-average `c_D` over.
#[derive(Deserialize, Default)]
struct RunCfg {
    steps: usize,
    print_every: usize,
    average_frac: f64,
}

/// `[validation]`: the cited benchmark value and the accepted relative tolerance.
#[derive(Deserialize, Default)]
struct ValidationCfg {
    cd_ref: f64,
    tol_rel: f64,
}

/// Parabolic-inlet parameters, held as a resource so the example-local inflow
/// system (below) can see the peak velocity and channel height.
#[derive(Clone, Copy)]
struct InflowProfile {
    u_max: f64,
    height: f64,
    rho: f64,
}

/// `Boundary` (registered *after* the stock boundary fill, so it overrides the
/// x-lo ghosts): impose the parabolic velocity inlet `u(y) = 4 Uₘ y (H−y)/H²`,
/// extrapolating pressure from the first interior cell (a subsonic velocity inlet,
/// but `y`-dependent — which the position-free `BoundaryCondition` trait can't
/// express, so it is a case-local system here rather than a core BC).
fn parabolic_inflow_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    prof: Res<InflowProfile>,
) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    let ng = mesh.n_ghost();
    let [_, nj, nk] = mesh.dims();
    let (um, h, rho) = (prof.u_max, prof.height, prof.rho);
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            let first = mesh.idx_raw(ng, jr, kr);
            let y = mesh.cell_centroid(first)[1];
            let uy = 4.0 * um * y * (h - y) / (h * h);
            let p_in = eos.pressure(&state.u[first]);
            let t = p_in / (rho * R_GAS);
            let cons = eos.prim_to_cons(&PrimVar::new(rho, uy, 0.0, 0.0, p_in, t));
            for gi in 0..ng {
                state.u[mesh.idx_raw(gi, jr, kr)] = cons;
            }
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect(
        "usage: cylinder_drag_validation <case.toml>  (see crates/cfd_ibm/examples/cylinder_drag_re20.toml)",
    );
    let toml_src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let mesh_cfg: UniformMeshConfig = cfg.section("grid");
    let flow: FlowCfg = cfg.section("flow");
    let cyl_cfg: CylinderCfg = cfg.section("cylinder");
    let run: RunCfg = cfg.section("run");
    let valid: ValidationCfg = cfg.section("validation");
    let mut solver_cfg: SolverConfig = cfg.section("solver");
    solver_cfg.viscous = true; // this benchmark is a viscous case, always

    // Transport: pin μ so Re = ρ Ū D / μ hits the target exactly.
    let d = cyl_cfg.diameter;
    let mu = flow.rho * flow.u_mean * d / flow.reynolds;
    let u_max = 1.5 * flow.u_mean; // parabolic peak = 3/2 · mean
    let l_z = mesh_cfg.bounds_hi[2] - mesh_cfg.bounds_lo[2];
    let (nx, ny) = (mesh_cfg.nx, mesh_cfg.ny); // captured before the mesh_cfg move

    let eos = IdealGas::air();
    let t_inf = flow.p / (flow.rho * R_GAS);
    // Uniform-mean freestream: seeds solid→fluid flips and the initial field.
    let freestream = eos.prim_to_cons(&PrimVar::new(flow.rho, flow.u_mean, 0.0, 0.0, flow.p, t_inf));
    // Initialize the channel already carrying the parabolic profile (faster to steady).
    let (um, h, rho0, p0) = (u_max, cyl_cfg.channel_height, flow.rho, flow.p);
    let init = move |x: Vec3| {
        let uy = 4.0 * um * x[1] * (h - x[1]) / (h * h);
        eos.prim_to_cons(&PrimVar::new(rho0, uy, 0.0, 0.0, p0, p0 / (rho0 * R_GAS)))
    };

    let cyl = Cylinder { center_xy: [cyl_cfg.cx, cyl_cfg.cy], radius: 0.5 * d };
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, SupersonicOutflow) // base fill; overridden by parabolic_inflow_system
        .with(BoundarySide::XHi, PressureOutlet { p: flow.p })
        .with_axis(1, NoSlipWall) // channel walls (no-slip)
        .with_axis(2, SlipWall); // 2-D span

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    // Override the default Sutherland transport with the constant μ for this Re.
    app.add_resource(Viscosity::Constant(mu));
    app.add_resource(InflowProfile { u_max, height: cyl_cfg.channel_height, rho: flow.rho });
    app.add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        // Registered after BoundaryPlugin ⇒ runs after the stock boundary fill in
        // the same `Boundary` phase, overriding the x-lo ghosts with the profile.
        .add_update_system(parabolic_inflow_system, MeshScheduleSet::Boundary)
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(solver_cfg))
        .add_plugins(GhostCellIbmPlugin)
        .add_plugins(CylinderIbmPlugin::new(cyl, freestream));
    app.prepare();

    let re_actual = flow.rho * flow.u_mean * d / mu;
    let m_max = u_max / eos.sound_speed(&freestream);
    let q = 0.5 * flow.rho * flow.u_mean * flow.u_mean * d * l_z; // ½ρŪ²·D·L_z
    let cd_of = |fx: f64| fx / q;

    println!("# Schäfer–Turek 1996 DFG-2D-1 cylinder drag (ghost-cell IBM)");
    println!("# Re = {re_actual:.3}   mu = {mu:.5}   U_mean = {:.3}   U_max = {u_max:.3}   M_max = {m_max:.3}", flow.u_mean);
    println!("# D = {d}   H = {}   L_z = {l_z}   cells = {nx}x{ny}", cyl_cfg.channel_height);
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

    // Physicality guard.
    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        assert!(
            state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0 && u.rho_e.is_finite()),
            "cylinder run went non-physical",
        );
    }

    let cd = cd_samples.iter().sum::<f64>() / cd_samples.len().max(1) as f64;
    let marker = app.get_resource_ref::<MarkerForce>().unwrap().total();
    let cl = marker[1] / q;
    let rel = (cd - valid.cd_ref).abs() / valid.cd_ref;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!("# Cd (marker, averaged over last {:.0}%): {cd:.4}", 100.0 * run.average_frac);
    println!("# Cd reference (Schäfer–Turek 1996): {:.4}", valid.cd_ref);
    println!("# Cl (marker): {cl:.4}");
    println!("# relative error: {:.2}%   (tolerance {:.1}%)", 100.0 * rel, 100.0 * valid.tol_rel);
    if rel <= valid.tol_rel {
        println!("VALIDATION: PASS  (Cd = {cd:.4} vs {:.4}, {:.2}% <= {:.1}%)", valid.cd_ref, 100.0 * rel, 100.0 * valid.tol_rel);
    } else {
        println!("VALIDATION: FAIL  (Cd = {cd:.4} vs {:.4}, {:.2}% > {:.1}%)", valid.cd_ref, 100.0 * rel, 100.0 * valid.tol_rel);
        std::process::exit(1);
    }
}
