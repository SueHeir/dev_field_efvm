//! **Resolved sphere surface-load handoff** — the IBM / cut-cell integration path
//! from `cfd_ibm::Sphere` geometry to marker-integrated net force.
//!
//! This validates the resolved IBM path's data handoff: given a
//! `cfd_ibm::Sphere` immersed in a FIELD gas state, the cut-cell embedded-boundary
//! integrator ([`CutCellMetrics`]) must turn the fluid pressure field into the
//! correct **net force on the body**. We check it against two fields with
//! *exact* closed-form answers, so
//! the validation is independent, not self-consistent:
//!
//! 1. **Uniform pressure ⇒ zero net force.** A constant pressure over a closed
//!    body integrates to zero; the residual measures the discrete consistency of
//!    the embedded-boundary normal-area vectors `n̂_eb·A_eb`.
//! 2. **Hydrostatic pressure ⇒ Archimedes buoyancy.** For `p(z) = p₀ + ρ_f g_z z`
//!    the net pressure force is the buoyancy `F = −ρ_f V g` (Archimedes) — here
//!    `F_z = −ρ_f V_sphere g_z` (upward, since `g_z < 0`). This is the resolved
//!    analytic buoyancy result for the same immersed geometry.
//!
//! A third check confirms the cut-cell metrics recover the sphere's **volume**
//! `Σ (1−α_v)V ≈ 4/3 π R³` — the geometry the void-fraction deposition also rests
//! on. FIELD owns the mesh; the sphere is immersed (`signed_distance`), never a
//! cell owner.
//!
//! This example validates the surface-load INTEGRATOR only; it runs **no flow
//! solve**, so the resolved *viscous* drag `Cd(Re)` is **not established here**.
//! Recovering it needs a live 3-D Navier–Stokes solve past the sphere (the 3-D
//! analogue of the `cylinder_drag_validation` case), held to a grid-convergence-
//! justified tolerance against an independent standard-drag correlation
//! (Schiller–Naumann 1935 / Clift–Grace–Weber 1978); it reuses this exact load
//! integrator, now validated on the analytically-solvable pressure loads. The
//! `Cd(Re)` values this example prints below are that follow-up's *target*,
//! printed for reference —
//! **not** a validation performed here.
//!
//! Declarative TOML (`argv[1]`):
//!
//! ```text
//! cargo run --release -p dev_field_efvm --bin sphere_drag_resolved -- \
//!     examples/sphere_drag_resolved/config.toml
//! ```

use cfd_eos::{Eos, IdealGas};
use cfd_ibm::{cd_schiller_naumann, CutCellMetrics, Sphere};
use cfd_state::{ConsVar, PrimVar};
use field_core::{FvMesh, StructuredMesh, UniformMesh, UniformMeshConfig, Vec3};
use grass_io::Config;
use serde::Deserialize;

const R_GAS: f64 = 287.058;

#[derive(Deserialize, Default)]
struct GasCfg {
    rho: f64,
    p: f64,
    mu: f64,
}

#[derive(Deserialize, Default)]
struct SphereCfg {
    cx: f64,
    cy: f64,
    cz: f64,
    radius: f64,
}

#[derive(Deserialize, Default)]
struct GravityCfg {
    gz: f64,
}

/// `[flow]`: a reference relative velocity + Reynolds number, used only to print
/// the target `Cd(Re)` the resolved viscous follow-up aims at.
#[derive(Deserialize, Default)]
struct FlowCfg {
    u_inf: f64,
    reynolds: f64,
}

#[derive(Deserialize, Default)]
struct ValidationCfg {
    /// Relative tolerance on the recovered buoyancy vs Archimedes.
    tol_buoyancy: f64,
    /// Max ratio |F_uniform| / |F_hydrostatic| (constant-field residual).
    tol_uniform_ratio: f64,
    /// Relative tolerance on the cut-cell solid volume vs 4/3 π R³.
    tol_volume: f64,
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: sphere_drag_resolved <case.toml>");
    let toml_src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let gas: GasCfg = cfg.section("gas");
    let sc: SphereCfg = cfg.section("sphere");
    let grav: GravityCfg = cfg.section("gravity");
    let flow: FlowCfg = cfg.section("flow");
    let valid: ValidationCfg = cfg.section("validation");
    let mesh_cfg: UniformMeshConfig = cfg.section("grid");

    // FIELD owns the mesh; the sphere is immersed via its signed-distance function.
    let mesh = UniformMesh::from_config(&mesh_cfg);
    let sphere = Sphere {
        center: [sc.cx, sc.cy, sc.cz],
        radius: sc.radius,
    };
    let eos = IdealGas::air();
    let t0 = gas.p / (gas.rho * R_GAS);

    // Interior cell geometry (centroid, size), plus two analytic pressure fields
    // built in the SAME order so they align with the cut-cell metrics indexing.
    let [ni, nj, nk] = mesh.dims();
    let mut geom: Vec<(Vec3, Vec3)> = Vec::with_capacity(ni * nj * nk);
    let mut u_uniform: Vec<ConsVar> = Vec::with_capacity(ni * nj * nk);
    let mut u_hydro: Vec<ConsVar> = Vec::with_capacity(ni * nj * nk);
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let c = mesh.idx(i, j, k);
                let cc = mesh.cell_centroid(c);
                let size = [mesh.spacing(0, i), mesh.spacing(1, j), mesh.spacing(2, k)];
                geom.push((cc, size));
                // Uniform field: p = p0 everywhere.
                u_uniform.push(eos.prim_to_cons(&PrimVar::new(gas.rho, 0.0, 0.0, 0.0, gas.p, t0)));
                // Hydrostatic field: dp/dz = ρ_f g_z ⇒ p(z) = p0 + ρ_f g_z (z − z_c),
                // referenced at the sphere centre so p stays ~p0 near the body.
                let p_hydro = gas.p + gas.rho * grav.gz * (cc[2] - sc.cz);
                let t = p_hydro / (gas.rho * R_GAS);
                u_hydro.push(eos.prim_to_cons(&PrimVar::new(gas.rho, 0.0, 0.0, 0.0, p_hydro, t)));
            }
        }
    }

    let metrics = CutCellMetrics::compute(&geom, &sphere);

    // ── Geometry: recovered solid volume vs analytic sphere volume ────────────
    let vol_num = metrics.solid_volume(&geom);
    let vol_ref = 4.0 / 3.0 * std::f64::consts::PI * sc.radius.powi(3);
    let vol_err = (vol_num - vol_ref).abs() / vol_ref;

    // ── Surface-load handoff: pressure force under each analytic field ────────
    let f_uniform = metrics.pressure_force(&u_uniform, &eos);
    let f_hydro = metrics.pressure_force(&u_hydro, &eos);

    // Archimedes reference: F = −ρ_f V g  ⇒  F_z = −ρ_f V g_z (> 0, upward).
    let buoy_ref = [0.0, 0.0, -gas.rho * vol_ref * grav.gz];
    let buoy_ref_mag = buoy_ref[2].abs();
    let f_hydro_mag = (f_hydro[0].powi(2) + f_hydro[1].powi(2) + f_hydro[2].powi(2)).sqrt();
    let buoy_err = (f_hydro[2] - buoy_ref[2]).abs() / buoy_ref_mag;

    let f_uniform_mag = (f_uniform[0].powi(2) + f_uniform[1].powi(2) + f_uniform[2].powi(2)).sqrt();
    let uniform_ratio = f_uniform_mag / f_hydro_mag.max(1e-30);

    // Reference drag coefficient the resolved viscous follow-up targets.
    // NOTE: informational only — NOT validated by this example (no flow solve).
    let cd_stokes = 24.0 / flow.reynolds.max(1e-12);
    let cd_sn = cd_schiller_naumann(flow.reynolds.max(1e-12));

    let d = 2.0 * sc.radius;
    println!("# Resolved DEM-CFD seam — sphere surface-load handoff (cut-cell IBM)");
    println!(
        "# D = {d}   D/dx = {:.1}   cells = {ni}x{nj}x{nk}",
        d / mesh.spacing(0, 0)
    );
    println!(
        "# rho_f = {}   mu = {:.3e}   g_z = {}   V_sphere = {:.4e} m^3",
        gas.rho, gas.mu, grav.gz, vol_ref
    );
    println!("#");
    println!(
        "# solid volume:   {vol_num:.6e}  (ref {vol_ref:.6e}, err {:.2}%, tol {:.1}%)",
        100.0 * vol_err,
        100.0 * valid.tol_volume
    );
    println!(
        "# uniform p force: {f_uniform:?}   |F|/|F_hydro| = {:.2e}  (tol {:.1e})",
        uniform_ratio, valid.tol_uniform_ratio
    );
    println!(
        "# hydrostatic Fz:  {:.6e}  (Archimedes {:.6e}, err {:.2}%, tol {:.1}%)",
        f_hydro[2],
        buoy_ref[2],
        100.0 * buoy_err,
        100.0 * valid.tol_buoyancy
    );
    println!("# follow-up TARGET (informational, NOT validated here — no flow solve) resolved Cd @ Re={:.3}: Cd_stokes = 24/Re = {cd_stokes:.4}   Cd_SN = {cd_sn:.4}   (U_inf = {})", flow.reynolds, flow.u_inf);

    let pass_vol = vol_err <= valid.tol_volume;
    let pass_uniform = uniform_ratio <= valid.tol_uniform_ratio;
    let pass_buoy = buoy_err <= valid.tol_buoyancy;

    println!("#");
    if pass_vol && pass_uniform && pass_buoy {
        println!(
            "VALIDATION: PASS  (volume {:.2}%<= {:.1}%; uniform-residual {:.1e}<= {:.1e}; buoyancy {:.2}%<= {:.1}%)",
            100.0 * vol_err, 100.0 * valid.tol_volume, uniform_ratio, valid.tol_uniform_ratio, 100.0 * buoy_err, 100.0 * valid.tol_buoyancy
        );
    } else {
        println!("VALIDATION: FAIL  (volume_ok={pass_vol}  uniform_ok={pass_uniform}  buoyancy_ok={pass_buoy})");
        std::process::exit(1);
    }
}
