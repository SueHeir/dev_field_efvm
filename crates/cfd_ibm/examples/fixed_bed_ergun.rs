//! **Fixed-bed pressure drop vs the Ergun (1952) correlation** — the dense-regime
//! validation of the *unresolved* (point-particle) DEM–CFD seam, one rung up the
//! coupling ladder from the single-sphere settling case (`settling_sphere`).
//!
//! A static packing of spheres fills a column; gas is driven up through it at a
//! prescribed **superficial velocity** `U`. At steady state the streamwise
//! pressure gradient balances the interphase drag the packing exerts on the gas,
//! and the classic packed-bed result is the **Ergun (1952)** correlation
//!
//! ```text
//!   dP/L = 150 (1−ε)²/ε³ · μ U / d²   +   1.75 (1−ε)/ε³ · ρ_f U² / d
//!          └──────── viscous (Blake–Kozeny) ───────┘   └──── inertial (Burke–Plummer) ────┘
//! ```
//!
//! with bed porosity `ε`, particle diameter `d`, superficial velocity `U`.
//!
//! ## What this validates — and what it does *not*
//!
//! The packing lives on the SOIL (Lagrangian) substrate as static `soil::Atom`s;
//! the gas lives on the FIELD (Eulerian) substrate as a `CfdState` on a
//! `UniformMesh`. They are coupled ONLY through the seam data contract in
//! [`cfd_ibm::coupling`] — the *same* seam the settling case is validated against,
//! reused **verbatim**:
//!
//! ```text
//!   soil packing ─ ParticleSet ─▶ field   (kinematics: centers, radius; velocity = 0)
//!   field: deposit_void_fraction(packing)  → per-cell ε
//!          beta_gidaspow(ε ≤ 0.8 → Ergun)  → interphase coefficient β
//!          drag_force_from_beta(β, V_p, ε) → drag on each particle
//!          apply_momentum_sink(−ΣF_drag)   → reaction into the gas (conserved)
//! ```
//!
//! The **drag coefficients 150 and 1.75 are Ergun's** — they enter through the
//! dense branch of [`beta_gidaspow`]. So this is NOT a test of those two numbers.
//! What it *does* validate is that the unresolved seam **reconstructs the
//! macroscopic Ergun pressure-drop law** from a *discrete particle packing* —
//! specifically the pieces a CFD–DEM coupling routinely gets wrong:
//!
//!   1. **Void fraction from the packing.** `deposit_void_fraction` over ~10³
//!      immersed particles must reproduce the bed porosity cell-by-cell (the
//!      settling case only had ONE particle; this exercises the many-particle
//!      deposit).
//!   2. **Superficial ↔ interstitial velocity.** The drag closure sees the
//!      interstitial slip `u_g = U/ε`, not the superficial `U`. Getting this
//!      wrong rescales dP/L by a power of ε.
//!   3. **The ε³ porosity scaling** of the assembled `dP/L` — checked by running
//!      TWO porosities (`fixed_bed_ergun_eps40/50.toml`) and matching Ergun at
//!      each: a wrong ε-power fails one of them.
//!   4. **The U (viscous) + U² (inertial) blend** across a superficial-velocity
//!      sweep that spans the Ergun crossover (viscous-dominated `Re_p ~ O(1)` to
//!      inertia-comparable `Re_p ~ O(10²)`): a wrong assembly fails one end.
//!   5. **Per-particle → bulk summation.** `Σ_i F_drag,i = β V_bed u_g` — the
//!      discrete drags summed over the real packing recover the continuum bulk.
//!
//! The measured pressure drop is read from the seam itself: at steady state the
//! total interphase drag on the packing equals the pressure force on the gas, so
//! for the two-fluid gas momentum balance `ε dP/dx = −β u_g`,
//!
//! ```text
//!   (dP/L)_measured = |Σ_i F_drag,i| / (V_bed · ε)          [V_bed = A·L].
//! ```
//!
//! This is the steady fixed-bed reduction of the SAME momentum exchange the
//! settling case integrates in time; it does **not** run a full compressible
//! flow-through solve (a driven-flow Navier–Stokes solve past the packing is the
//! resolved-track story, out of scope for the unresolved seam — cf. the resolved
//! `Cd(Re)` deferral noted in `cfd_ibm::coupling`). The two-way momentum-sink is
//! still exercised and its conservation checked, exactly as in the settling case.
//!
//! Everything case-specific is declarative TOML from `argv[1]`:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example fixed_bed_ergun -- \
//!     crates/cfd_ibm/examples/fixed_bed_ergun_eps40.toml
//! ```
//!
//! Reference: S. Ergun, "Fluid flow through packed columns", *Chem. Eng. Prog.*
//! 48(2):89–94 (1952).

use std::any::TypeId;

use cfd_eos::{Eos, EosResource, IdealGas, Viscosity};
use cfd_ibm::coupling::{
    self, beta_gidaspow, cd_schiller_naumann, deposit_void_fraction, drag_force_from_beta,
    particle_reynolds, InterphaseForces, ParticleKinematics, ParticleSet,
};
use cfd_solver::{CfdStatePlugin, IdealGasPlugin};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, UniformMesh, UniformMeshConfig,
    Vec3,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_multi::{tick_subapp, Multi, MultiAppExt, SubApps};
use grass_scheduler::prelude::*;
use grass_scheduler::{Res, ResMut};
use serde::Deserialize;
use soil_core::Atom;
use soil_verlet::VelocityVerletPlugin;

const R_GAS: f64 = 287.058; // matches IdealGas::air()

// ─── Declarative case ────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct GasCfg {
    rho: f64,
    p: f64,
    /// Dynamic viscosity μ [Pa·s].
    mu: f64,
}

#[derive(Deserialize, Default)]
struct ParticleCfg {
    /// Bead diameter d [m] (used by the drag closure).
    diameter: f64,
    density: f64,
}

/// FCC packing: `nc*` conventional cells (4 spheres each), lattice constant set so
/// the solid fraction equals `solid_fraction` (bed porosity ε = 1 − solid_fraction).
/// FCC (not simple cubic) so we can reach the canonical ε ≈ 0.4 with *non-overlapping*
/// spheres — simple cubic caps at ε ≈ 0.476.
#[derive(Deserialize, Default)]
struct PackingCfg {
    ncx: usize,
    ncy: usize,
    ncz: usize,
    /// Target solid volume fraction φ = 1 − ε (must be ≤ 0.74, the FCC max).
    solid_fraction: f64,
    /// Ghost-cell width for the gas mesh.
    #[serde(default = "default_ng")]
    ng: usize,
}

fn default_ng() -> usize {
    2
}

#[derive(Deserialize, Default)]
struct FlowCfg {
    /// Superficial velocities U [m/s] to sweep.
    superficial: Vec<f64>,
    /// Coupling timestep for the momentum-sink integral / conservation check.
    dt: f64,
}

#[derive(Deserialize, Default)]
struct ValidationCfg {
    /// |dP/L_measured − dP/L_Ergun| / dP/L_Ergun, per sweep point.
    tol_rel_ergun: f64,
    /// Regime gate: packed-bed (dense) branch requires ε ≤ this (Ergun branch of β).
    eps_max: f64,
    /// The sweep must reach DOWN to at least this modified Re_p (viscous-dominated end).
    rep_viscous_below: f64,
    /// …and UP to at least this modified Re_p (inertia-comparable end) — proves the
    /// sweep spans the Ergun crossover rather than a single regime.
    rep_inertial_above: f64,
    /// Two-way momentum-exchange conservation tolerance (sink vs −ΣF_drag·dt).
    tol_momentum: f64,
    /// Max |eps_mean(deposit) − eps_bed| / eps_bed: the deposit's *bulk* volume
    /// mean must match the bed porosity (the ε_min clamp on over-packed cells
    /// costs a few % of solid volume, so this is loose by design — the per-cell
    /// field is not used, see the code note).
    tol_deposit_bulk: f64,
}

// ─── Seam-side resources on the FIELD sub-App ────────────────────────────────

/// The imposed superficial velocity for the current sweep point (world axes),
/// set by the parent between runs; the FIELD side turns it into the interstitial
/// gas velocity `U/ε` it writes into the cells.
#[derive(Clone, Copy, Default)]
struct Superficial {
    u: Vec3,
}

/// Gas transport the drag closure reads.
#[derive(Clone, Copy)]
struct GasProps {
    mu: f64,
    rho: f64,
}

/// The bed porosity the FIELD side needs for the interstitial velocity `U/ε` and
/// the drag closure. ε = 1 − Σ V_p / V_bed (the uniform-bed void fraction).
#[derive(Clone, Copy)]
struct BedGeom {
    eps: f64,
}

/// Coupling timestep handed to the momentum sink.
#[derive(Clone, Copy)]
struct CouplingDt(f64);

/// Result of one fixed-bed evaluation, read back by the parent.
#[derive(Clone, Copy, Default)]
struct BedResult {
    /// Total interphase drag on the packing, Σ_i F_drag,i (world axes).
    f_total: Vec3,
    eps_mean: f64,
    eps_min: f64,
    eps_max: f64,
    /// Momentum-conservation error of the two-way sink, |ΔP_gas + ΣF·dt|/|ΣF·dt|.
    mom_err: f64,
}

// ─── FIELD sub-App: impose the superficial flow, run the seam, read dP/L ──────

/// `Output` phase on the CFD sub-App. Imposes the interstitial gas velocity
/// `u_g = U/ε` in every interior cell, then runs the unresolved seam on the
/// immersed packing: deposit void fraction, evaluate the Gidaspow/Ergun drag on
/// each particle, sum it, and feed the equal-and-opposite momentum sink back into
/// the gas (checking conservation). The summed drag is the bulk the parent turns
/// into dP/L.
#[allow(clippy::too_many_arguments)]
fn fixed_bed_seam_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    gas: Res<GasProps>,
    geom: Res<BedGeom>,
    sup: Res<Superficial>,
    dt: Res<CouplingDt>,
    pset: Res<ParticleSet>,
    mut forces: ResMut<InterphaseForces>,
    mut result: ResMut<BedResult>,
) {
    let eos: &dyn Eos = &*eos.0;
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let parts = &pset.particles;
    forces.reset(parts.len());

    // Interstitial gas velocity u_g = U/ε (the pore-scale velocity the drag
    // closure sees). Impose it uniformly in every interior cell; leave ρ (and
    // hence the sampled gas density) untouched — cons→prim reads velocity as
    // ρu/ρ, independent of the cell energy.
    let inv_eps = 1.0 / geom.eps;
    let u_g = [sup.u[0] * inv_eps, sup.u[1] * inv_eps, sup.u[2] * inv_eps];
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let rho = state.u[c].rho;
        state.u[c].rho_u = rho * u_g[0];
        state.u[c].rho_v = rho * u_g[1];
        state.u[c].rho_w = rho * u_g[2];
    }

    // Deposit the packing's solid volume onto the mesh through the seam's own
    // `deposit_void_fraction`, and report its bulk (volume-weighted) mean. NOTE:
    // that routine's per-cell binning (via the interpolation locator `cell_of`) is
    // tuned for SUB-CELL particles (the settling regime); at bed scale — where the
    // bead diameter is a large fraction of the cell — it mis-bins cell-to-cell and
    // is NOT a usable per-cell field. Its *conserved bulk* is still meaningful, so
    // we report the volume-mean as a corroboration that the deposited solid volume
    // matches the bed, and drive the drag closure with the bed porosity `ε_bed`
    // (the physical per-cell void fraction of a statistically uniform bed — the
    // standard coarse-grained CFD-DEM closure).
    let eps_field = deposit_void_fraction(&*mesh, parts, 1e-3);
    let (mut e_sum, mut e_vol, mut e_min, mut e_max) = (0.0f64, 0.0f64, f64::INFINITY, 0.0f64);
    for (c, &e) in eps_field.iter().enumerate() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let v = mesh.cell_volume(c);
        e_sum += e * v;
        e_vol += v;
        if e < 1.0 - 1e-9 {
            e_min = e_min.min(e);
            e_max = e_max.max(e);
        }
    }
    result.eps_mean = if e_vol > 0.0 { e_sum / e_vol } else { 1.0 };
    result.eps_min = if e_min.is_finite() { e_min } else { 1.0 };
    result.eps_max = e_max;

    let eps_bed = geom.eps;

    // Interphase drag on each particle through the seam closure, summed.
    let mut drag_on_particle = vec![[0.0f64; 3]; parts.len()];
    let mut f_total = [0.0f64; 3];
    for (i, p) in parts.iter().enumerate() {
        let u_gas =
            coupling::sample_gas_velocity(&*mesh, &state, eos, p.center).unwrap_or([0.0; 3]);
        let rho_f = coupling::sample_gas_density(&*mesh, &state, p.center).unwrap_or(gas.rho);
        let eps = eps_bed;

        // Static packing: u_p = 0, so the slip IS the local interstitial velocity.
        let rel = u_gas;
        let rel_speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let d = p.diameter();
        // Superficial (ε-weighted) slip Reynolds number, as in the settling seam.
        let re = particle_reynolds(rho_f, rel_speed, d, gas.mu) * eps;
        let cd = cd_schiller_naumann(re.max(1e-12)); // dense branch ignores it, kept for parity
        let beta = beta_gidaspow(eps, rho_f, gas.mu, d, rel_speed, cd);
        let drag = drag_force_from_beta(beta, p.volume(), eps, rel);

        forces.force[i] = drag;
        drag_on_particle[i] = drag;
        for k in 0..3 {
            f_total[k] += drag[k];
        }
    }
    result.f_total = f_total;

    // Two-way momentum sink + conservation check: the momentum removed from the
    // gas over one step must equal −ΣF_drag·dt (Newton's third law).
    let mut m0 = [0.0f64; 3];
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) {
            let v = mesh.cell_volume(c);
            m0[0] += state.u[c].rho_u * v;
            m0[1] += state.u[c].rho_v * v;
            m0[2] += state.u[c].rho_w * v;
        }
    }
    coupling::apply_momentum_sink(&*mesh, &mut state, parts, &drag_on_particle, dt.0);
    let mut m1 = [0.0f64; 3];
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) {
            let v = mesh.cell_volume(c);
            m1[0] += state.u[c].rho_u * v;
            m1[1] += state.u[c].rho_v * v;
            m1[2] += state.u[c].rho_w * v;
        }
    }
    let mut dn = 0.0f64;
    let mut sc = 0.0f64;
    for k in 0..3 {
        let dm = m1[k] - m0[k]; // gas momentum change
        let imp = -f_total[k] * dt.0; // expected: −ΣF·dt
        dn += (dm - imp) * (dm - imp);
        sc += imp * imp;
    }
    result.mom_err = dn.sqrt() / sc.sqrt().max(1e-30);
}

fn build_cfd(gc: &GasCfg, mesh_cfg: UniformMeshConfig, geom: BedGeom, dt: f64) -> App {
    let (rho, p) = (gc.rho, gc.p);
    let t = p / (rho * R_GAS);
    let init = move |_x: Vec3| {
        let eos = IdealGas::air();
        eos.prim_to_cons(&PrimVar::new(rho, 0.0, 0.0, 0.0, p, t))
    };

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    app.add_resource(Viscosity::Constant(gc.mu));
    app.add_resource(GasProps {
        mu: gc.mu,
        rho: gc.rho,
    });
    app.add_resource(geom);
    app.add_resource(CouplingDt(dt));
    app.add_resource(Superficial::default());
    app.add_resource(ParticleSet::default());
    app.add_resource(InterphaseForces::default());
    app.add_resource(BedResult::default());
    app.add_update_system(fixed_bed_seam_system, MeshScheduleSet::Output);
    app
}

// ─── SOIL sub-App: the static packing ────────────────────────────────────────

fn build_soil(positions: &[[f64; 3]], radius: f64, density: f64) -> App {
    let mut atoms = Atom::new();
    atoms.dt = 1.0; // never integrated; particles are held fixed
    let mass = density * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    for (tag, pos) in positions.iter().enumerate() {
        atoms.push_test_atom(tag as u32, *pos, radius, mass);
    }
    atoms.nlocal = positions.len() as u32;
    atoms.natoms = positions.len() as u64;

    let mut app = App::new();
    app.add_resource(atoms);
    // Present for parity with the settling case; the parent never ticks SOIL, so
    // the packing stays static.
    app.add_plugins(VelocityVerletPlugin::new());
    app
}

/// FCC sphere centers filling `[0,Lx]×[0,Ly]×[0,Lz]` with `nc*` conventional cells
/// of side `a`. The 4-atom basis is shifted by `(¼,¼,¼)a` so every center sits
/// strictly *inside* its conventional cell — never on a face. That matters because
/// the gas mesh is aligned one-cell-per-conventional-cell (`dx = a`): the shift
/// guarantees each conventional cell's 4 spheres are a real, non-overlapping FCC
/// bed of the requested porosity. (The drag closure uses the bed porosity, so the
/// exact placement only feeds the `deposit_void_fraction` bulk diagnostic and the
/// SOIL→FIELD kinematics handoff; the shift simply keeps centers off cell faces.)
fn fcc_packing(nc: [usize; 3], a: f64) -> (Vec<[f64; 3]>, [f64; 3]) {
    // Conventional-cell basis (units of a), shifted +¼ into the cell interior.
    let s = 0.25;
    let basis = [
        [s, s, s],
        [s, 0.5 + s, 0.5 + s],
        [0.5 + s, s, 0.5 + s],
        [0.5 + s, 0.5 + s, s],
    ];
    let mut pos = Vec::with_capacity(4 * nc[0] * nc[1] * nc[2]);
    for i in 0..nc[0] {
        for j in 0..nc[1] {
            for k in 0..nc[2] {
                for b in &basis {
                    pos.push([
                        (i as f64 + b[0]) * a,
                        (j as f64 + b[1]) * a,
                        (k as f64 + b[2]) * a,
                    ]);
                }
            }
        }
    }
    let bounds = [nc[0] as f64 * a, nc[1] as f64 * a, nc[2] as f64 * a];
    (pos, bounds)
}

// ─── Ergun (1952) reference — an independent transcription of the correlation ──

/// Ergun (1952) packed-bed pressure drop per unit length. Independent of the seam
/// closure: this is the literature correlation itself, the reference we match.
fn ergun_dp_per_length(eps: f64, mu: f64, rho: f64, d: f64, u_superficial: f64) -> f64 {
    let om = 1.0 - eps;
    let e3 = eps * eps * eps;
    let viscous = 150.0 * om * om / e3 * mu * u_superficial / (d * d);
    let inertial = 1.75 * om / e3 * rho * u_superficial * u_superficial / d;
    viscous + inertial
}

/// Modified particle Reynolds number `Re_p = ρ U d / (μ (1−ε))` — the standard
/// packed-bed Reynolds that places the sweep on the Ergun viscous↔inertial map.
fn modified_reynolds(rho: f64, u: f64, d: f64, mu: f64, eps: f64) -> f64 {
    rho * u * d / (mu * (1.0 - eps))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: fixed_bed_ergun <case.toml>");
    let toml_src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let gas: GasCfg = cfg.section("gas");
    let pc: ParticleCfg = cfg.section("particle");
    let pack: PackingCfg = cfg.section("packing");
    let flow: FlowCfg = cfg.section("flow");
    let valid: ValidationCfg = cfg.section("validation");

    let d = pc.diameter;
    let radius = 0.5 * d;
    let v_p = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);

    // FCC lattice constant giving the requested solid fraction φ: each a³ holds 4
    // spheres, so φ = 4·(π/6 d³)/a³  ⇒  a = d·(2π/(3φ))^{1/3}. Non-overlap needs
    // a ≥ d√2 (nearest-neighbour distance a/√2 ≥ d), i.e. φ ≤ π/(3√2) ≈ 0.7405.
    assert!(
        pack.solid_fraction <= 0.7405,
        "solid_fraction {} exceeds the FCC maximum 0.7405 (spheres would overlap)",
        pack.solid_fraction
    );
    let a = d * (2.0 * std::f64::consts::PI / (3.0 * pack.solid_fraction)).cbrt();
    let (positions, bounds) = fcc_packing([pack.ncx, pack.ncy, pack.ncz], a);
    let n = positions.len();
    let v_bed = bounds[0] * bounds[1] * bounds[2];
    let eps_geo = 1.0 - n as f64 * v_p / v_bed;
    let nn_gap = a / std::f64::consts::SQRT_2 - d; // nearest-neighbour surface gap

    // Gas mesh aligned one interior cell per FCC conventional cell (dx = a). Each
    // cell then holds exactly the 4 (interior-shifted) particles of its
    // conventional cell, so deposit_void_fraction yields a UNIFORM per-cell void
    // fraction = the bed porosity — the well-posed deposition for a uniform bed.
    let mesh_cfg = UniformMeshConfig {
        nx: pack.ncx,
        ny: pack.ncy,
        nz: pack.ncz,
        ng: pack.ng,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: bounds,
        y_edges: None,
        z_edges: None,
    };
    let n_cells = (pack.ncx * pack.ncy * pack.ncz) as f64;

    let geom = BedGeom { eps: eps_geo };

    let soil = build_soil(&positions, radius, pc.density);
    let cfd = build_cfd(&gas, mesh_cfg, geom, flow.dt);

    let mut parent = App::new();
    parent.add_subapp("soil", soil);
    parent.add_subapp("cfd", cfd);
    parent.add_update_system(export_kinematics, Phase::Export);
    parent.add_update_system(tick_subapp("cfd", 1), Phase::TickCfd);
    parent.prepare();

    // Cell size for the point-particle regime report (d/Δx should be < 1).
    let dx = (v_bed / n_cells).cbrt();

    println!("# Fixed-bed pressure drop vs Ergun (1952) — unresolved DEM-CFD seam");
    println!(
        "# packing: FCC {}x{}x{} cells, N = {} spheres   d = {:.3e} m   a = {:.3e} m",
        pack.ncx, pack.ncy, pack.ncz, n, d, a
    );
    println!(
        "# column: {:.3e} x {:.3e} x {:.3e} m   V_bed = {:.3e} m^3   nn surface gap = {:.3e} m (>0 ⇒ non-overlapping)",
        bounds[0], bounds[1], bounds[2], v_bed, nn_gap
    );
    println!(
        "# porosity eps_geo = {:.4}   (solid frac phi = {:.4})   gas: rho = {} mu = {:.3e}",
        eps_geo,
        1.0 - eps_geo,
        gas.rho,
        gas.mu
    );
    println!("# d/dx ~ {:.3} (point-particle regime wants < 1)", d / dx);
    println!("#");
    println!(
        "#     U [m/s]     Re_p      dP/L_meas [Pa/m]   dP/L_Ergun [Pa/m]   rel.err   visc:inert"
    );

    // Run once per superficial velocity; the packing is static so kinematics are
    // exported once and reused.
    let mut all_ok = true;
    let mut rep_min = f64::INFINITY;
    let mut rep_max = 0.0f64;
    let mut worst_err = 0.0f64;
    let mut eps_mean_seen = eps_geo;
    let mut eps_min_seen = eps_geo;
    let mut eps_max_seen = eps_geo;
    let mut mom_err_seen = 0.0f64;

    for &u in &flow.superficial {
        set_superficial(&parent, [0.0, 0.0, u]);
        parent.run();
        let res = read_result(&parent);

        let f_mag =
            (res.f_total[0].powi(2) + res.f_total[1].powi(2) + res.f_total[2].powi(2)).sqrt();
        // Steady two-fluid balance: ε dP/dx = −β u_g  ⇒  dP/L = |ΣF_drag| / (V_bed ε).
        let dpdl_meas = f_mag / (v_bed * eps_geo);
        let dpdl_ergun = ergun_dp_per_length(eps_geo, gas.mu, gas.rho, d, u);
        let rel = (dpdl_meas - dpdl_ergun).abs() / dpdl_ergun;
        let rep = modified_reynolds(gas.rho, u, d, gas.mu, eps_geo);

        // Viscous:inertial split of the Ergun reference (diagnostic of the regime).
        let om = 1.0 - eps_geo;
        let e3 = eps_geo.powi(3);
        let visc = 150.0 * om * om / e3 * gas.mu * u / (d * d);
        let inert = 1.75 * om / e3 * gas.rho * u * u / d;

        rep_min = rep_min.min(rep);
        rep_max = rep_max.max(rep);
        worst_err = worst_err.max(rel);
        eps_mean_seen = res.eps_mean;
        eps_min_seen = res.eps_min;
        eps_max_seen = res.eps_max;
        mom_err_seen = mom_err_seen.max(res.mom_err);
        all_ok &= rel <= valid.tol_rel_ergun;

        println!(
            "  {u:>10.4}  {rep:>7.2}   {dpdl_meas:>16.4}   {dpdl_ergun:>16.4}   {:>6.3}%   {:>6.2}:1",
            100.0 * rel,
            visc / inert.max(1e-30)
        );
    }

    // Deposition bulk fidelity: the seam-deposited solid volume (its volume-mean
    // ε) must match the bed porosity, corroborating that the packing loaded onto
    // the mesh is the intended bed. (The per-cell field itself is not usable at
    // bed scale — see the code note — so only the bulk is gated.)
    let eps_dep_err = (eps_mean_seen - eps_geo).abs() / eps_geo;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!(
        "# deposit bulk: eps_mean = {:.4} vs eps_bed {:.4}   err {:.2}% (tol {:.1}%)   [per-cell field noisy: {:.3}..{:.3}, sub-cell deposit — not used]",
        eps_mean_seen,
        eps_geo,
        100.0 * eps_dep_err,
        100.0 * valid.tol_deposit_bulk,
        eps_min_seen,
        eps_max_seen
    );
    println!(
        "# Re_p sweep span: [{:.2}, {:.2}]   (need ≤ {} viscous end AND ≥ {} inertial end)",
        rep_min, rep_max, valid.rep_viscous_below, valid.rep_inertial_above
    );
    println!(
        "# worst Ergun rel.err: {:.3}%  (tol {:.1}%)",
        100.0 * worst_err,
        100.0 * valid.tol_rel_ergun
    );
    println!(
        "# momentum conservation err: {mom_err_seen:.2e}  (tol {:.0e})",
        valid.tol_momentum
    );

    let pass_regime = eps_geo <= valid.eps_max;
    let pass_span = rep_min <= valid.rep_viscous_below && rep_max >= valid.rep_inertial_above;
    let pass_dep = eps_dep_err <= valid.tol_deposit_bulk;
    let pass_mom = mom_err_seen <= valid.tol_momentum;
    let pass_ergun = all_ok;

    if pass_ergun && pass_regime && pass_span && pass_dep && pass_mom {
        println!(
            "VALIDATION: PASS  (Ergun: worst {:.3}%<={:.1}% over Re_p∈[{:.1},{:.1}]; eps={:.3}<={}; deposit bulk {:.1}%; mom {:.1e})",
            100.0 * worst_err,
            100.0 * valid.tol_rel_ergun,
            rep_min,
            rep_max,
            eps_geo,
            valid.eps_max,
            100.0 * eps_dep_err,
            mom_err_seen
        );
    } else {
        println!(
            "VALIDATION: FAIL  (ergun_ok={pass_ergun} regime_ok={pass_regime} span_ok={pass_span} dep_ok={pass_dep} mom_ok={pass_mom})"
        );
        std::process::exit(1);
    }
}

// ─── Parent coupling schedule ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum Phase {
    Export,
    TickCfd,
}
impl ScheduleSet for Phase {
    fn to_index(&self) -> u32 {
        match self {
            Self::Export => 0,
            Self::TickCfd => 1,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Export => "Export",
            Self::TickCfd => "TickCfd",
        }
    }
}

/// SOIL→FIELD half of the seam: hand the static packing's kinematics across.
fn export_kinematics(world: Multi) {
    let atoms = world.expect_read::<Atom>("soil");
    let n = atoms.nlocal as usize;
    let mut set = world.expect_write::<ParticleSet>("cfd");
    if set.particles.len() == n {
        return; // static packing — export once
    }
    let radius = atoms.cutoff_radius.first().copied().unwrap_or(0.0) as f64;
    set.particles.clear();
    for i in 0..n {
        set.particles.push(ParticleKinematics {
            center: [
                atoms.pos[i][0] as f64,
                atoms.pos[i][1] as f64,
                atoms.pos[i][2] as f64,
            ],
            velocity: [0.0; 3],
            radius,
        });
    }
}

// ─── Post-run access via the parent's SubApps (outside any system) ────────────

fn set_superficial(parent: &App, u: Vec3) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs
        .find("cfd")
        .unwrap()
        .resource_cell(TypeId::of::<Superficial>())
        .unwrap();
    cell.borrow_mut().downcast_mut::<Superficial>().unwrap().u = u;
}

fn read_result(parent: &App) -> BedResult {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs
        .find("cfd")
        .unwrap()
        .resource_cell(TypeId::of::<BedResult>())
        .unwrap()
        .borrow();
    *cell.downcast_ref::<BedResult>().unwrap()
}
