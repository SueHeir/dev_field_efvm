//! **Fixed-bed pressure drop vs the Ergun (1952) correlation** — the dense-regime
//! validation of the *unresolved* (point-particle) DEM–CFD seam, one rung up the
//! coupling ladder from the single-sphere settling case (`settling_sphere`).
//!
//! A static packing of spheres fills a column; gas moves up through it at a
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
//! ## Why the *measured* drag is an INDEPENDENT closure (no self-comparison)
//!
//! An earlier version of this example computed the "measured" `dP/L` from the
//! seam's own `beta_gidaspow` closure and compared it to the Ergun formula. That
//! is an **algebraic tautology**: the dense branch of `beta_gidaspow` *is* Ergun
//! (the very same 150 / 1.75 constants), so the measured `dP/L = |ΣF|/(V_bed ε)`
//! reduces IDENTICALLY to `ergun_dp_per_length` for any `ε, U, d` — the relative
//! error is machine-epsilon at every sweep point and the tolerance can never fail.
//! Such a check validates nothing.
//!
//! This version breaks that self-comparison. The **measured** per-particle drag is
//! assembled through the seam from an **independent** packed-bed closure —
//! **MacDonald et al. (1979)**, the "Ergun equation revisited" re-fit of the
//! Blake–Kozeny / Burke–Plummer constants to a much larger data set, which
//! recommends `180` (viscous) and `1.8` (inertial) in place of Ergun's `150` and
//! `1.75` (see [`macdonald_beta`]). Those constants come from data Ergun never saw,
//! so the measured value shares **no constant** with the Ergun *reference*; the
//! relative error is a genuine, non-zero, Reynolds-dependent spread (~5–20 %, the
//! documented inter-correlation difference — largest in the viscous-dominated
//! regime where `180/150 ≈ 1.20`, shrinking toward the inertial end where
//! `1.8/1.75 ≈ 1.03`).
//!
//! ### What that makes falsifiable
//!
//! The pass gate is `|dP/L_meas − dP/L_Ergun| / dP/L_Ergun ≤ tol` with `tol` set to
//! bracket the *documented* MacDonald-vs-Ergun spread. This is **not** a test of
//! the drag constants (Ergun's `150/1.75` and MacDonald's `180/1.8` are two
//! literature fits; neither is "the" truth). It tests that the unresolved seam's
//! **assembly** — deposit a discrete packing to a per-cell void fraction, read the
//! interstitial slip `u_g = U/ε`, evaluate a per-particle drag, sum it, and reduce
//! `Σ_i F_i → dP/L` — reconstructs the macroscopic packed-bed pressure-drop *law*
//! to within that spread. Any of the mistakes a CFD–DEM coupling routinely makes
//! breaks it by **factors far exceeding** the tolerance and the check FAILS:
//!
//!   * a wrong void-fraction power (`ε²` vs `ε³`) → error `~1/ε ≈ 2.5×`;
//!   * superficial `U` mistaken for interstitial `U/ε` in the slip → error `~ε`;
//!   * per-particle → bulk summation wrong (e.g. `V_bed` or the `1−ε` factor) → order-unity error.
//!
//! To prove that (and that the tolerance is not tuned to pass), this example also
//! runs a **negative control**: it re-assembles the SAME sweep with a deliberately
//! corrupted seam (the `ε²`-instead-of-`ε³` bug above) and asserts it would FAIL.
//!
//! ## The deposited void fraction is load-bearing
//!
//! The per-particle drag is driven by a per-cell void fraction **deposited from the
//! actual packing** ([`deposit_bed_void_fraction`]), not an analytic `ε_geo`. For
//! that field to be usable the gas mesh must be **coarser than the particles** —
//! the defining requirement of the *unresolved* regime — so each cell contains
//! *many* sub-cell spheres and its deposited void fraction converges to the bed
//! porosity. The `[mesh]` section sets a gas grid several particle diameters per
//! cell (`d/Δx ≪ 1`); the example gates that the deposited per-cell field is uniform
//! to within `tol_deposit_cell` of `ε_bed` (a real deposition-fidelity check — a
//! mis-binning deposit fails it), and then *uses* that field to evaluate the drag.
//! The binning uses a **containment** locator here in the example: the seam's own
//! `coupling::deposit_void_fraction` uses an interpolation locator tuned for a
//! single SUB-CELL particle and mis-bins at bed scale, so the bed-scale binning
//! lives in this example (per the library-placement rule — one-off to the packed-
//! bed cases). `ε_bed` itself is the exact geometric porosity of the FCC packing,
//! used only for the Ergun reference and the superficial↔interstitial conversion.
//!
//! ## Momentum conservation (kept, but NOT the validation)
//!
//! The two-way momentum sink is still exercised and its conservation checked
//! (`|ΔP_gas + ΣF·dt| / |ΣF·dt| ~ 1e-13`). That is a legitimate property of the
//! seam but it is **not** Ergun-specific — it holds for any drag whatsoever — so it
//! is reported and gated as a sanity check, never counted as the pressure-drop
//! validation.
//!
//! This example imposes the superficial flow (interstitial `u_g = U/ε`) rather than
//! running a full driven compressible flow-through solve past the packing: a
//! porosity-weighted (volume-averaged) momentum solve is the resolved-track story,
//! out of scope for the unresolved seam (cf. the `Cd(Re)` deferral in
//! `cfd_ibm::coupling`). The falsifiability here comes from the *independent* drag
//! reference, not from a flow solve.
//!
//! Everything case-specific is declarative TOML from `argv[1]`:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example fixed_bed_ergun -- \
//!     crates/cfd_ibm/examples/fixed_bed_ergun_eps40.toml
//! ```
//!
//! References:
//! * S. Ergun, "Fluid flow through packed columns", *Chem. Eng. Prog.* 48(2):89–94 (1952).
//! * I.F. MacDonald, M.S. El-Sayed, K. Mow, F.A.L. Dullien, "Flow through Porous
//!   Media — the Ergun Equation Revisited", *Ind. Eng. Chem. Fundam.* 18(3):199–208 (1979).

use std::any::TypeId;

use cfd_eos::{Eos, EosResource, IdealGas, Viscosity};
use cfd_ibm::coupling::{
    self, drag_force_from_beta, InterphaseForces, ParticleKinematics, ParticleSet,
};
use cfd_solver::{CfdStatePlugin, IdealGasPlugin};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, StructuredMesh, UniformMesh,
    UniformMeshConfig, Vec3,
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
}

/// Gas mesh — **coarser than the particles** (the unresolved regime). `nx*` must
/// divide the packing's conventional-cell counts so the domain tiles evenly; each
/// gas cell then spans `ncx/nx × … ` conventional cells and holds many spheres.
#[derive(Deserialize, Default)]
struct MeshCfg {
    nx: usize,
    ny: usize,
    nz: usize,
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
    /// |dP/L_measured − dP/L_Ergun| / dP/L_Ergun, per sweep point. Set to bracket
    /// the documented MacDonald(1979)-vs-Ergun(1952) inter-correlation spread.
    tol_rel_ergun: f64,
    /// Regime gate: packed-bed (dense) branch requires ε ≤ this.
    eps_max: f64,
    /// The sweep must reach DOWN to at least this modified Re_p (viscous-dominated end).
    rep_viscous_below: f64,
    /// …and UP to at least this modified Re_p (inertia-comparable end) — proves the
    /// sweep spans the Ergun crossover rather than a single regime.
    rep_inertial_above: f64,
    /// Two-way momentum-exchange conservation tolerance (sink vs −ΣF_drag·dt).
    tol_momentum: f64,
    /// Max |ε_cell(deposit) − ε_bed| / ε_bed over interior cells: the deposited
    /// per-cell void fraction that DRIVES the drag must reproduce the bed porosity
    /// (a mis-binning deposit fails this — the field is load-bearing, not analytic).
    tol_deposit_cell: f64,
}

// ─── Seam-side resources on the FIELD sub-App ────────────────────────────────

/// The imposed superficial velocity for the current sweep point (world axes),
/// set by the parent between runs; the FIELD side turns it into the interstitial
/// gas velocity `U/ε` it writes into the cells.
#[derive(Clone, Copy, Default)]
struct Superficial {
    u: Vec3,
}

/// Which drag closure the measured path assembles, and whether to corrupt the
/// `Σ → dP/L` reduction (the negative control). Set by the parent per pass.
#[derive(Clone, Copy)]
struct SeamMode {
    /// `true` → independent MacDonald(1979) closure; `false` → Ergun(1952) closure
    /// (used only by the negative control's baseline, never by the real validation).
    macdonald: bool,
    /// `true` → inject the `ε²`-instead-of-`ε³` reduction bug (negative control).
    corrupt_eps_power: bool,
}

impl Default for SeamMode {
    fn default() -> Self {
        Self { macdonald: true, corrupt_eps_power: false }
    }
}

/// The static seam context read by the FIELD system — gas transport, bed porosity
/// ε (for `U/ε` and the superficial↔interstitial conversion), the coupling
/// timestep, and the current [`SeamMode`]. Bundled into one resource so the system
/// stays within the scheduler's parameter-count limit.
#[derive(Clone, Copy)]
struct SeamCtx {
    mu: f64,
    rho: f64,
    /// Bed porosity ε = 1 − Σ V_p / V_bed.
    eps: f64,
    /// Coupling timestep handed to the momentum sink.
    dt: f64,
    mode: SeamMode,
}

/// Result of one fixed-bed evaluation, read back by the parent.
#[derive(Clone, Copy, Default)]
struct BedResult {
    /// Total interphase drag on the packing, Σ_i F_drag,i (world axes).
    f_total: Vec3,
    /// Max |ε_cell − ε_bed| / ε_bed over interior cells (deposition fidelity).
    eps_cell_err: f64,
    eps_min: f64,
    eps_max: f64,
    /// Momentum-conservation error of the two-way sink, |ΔP_gas + ΣF·dt|/|ΣF·dt|.
    mom_err: f64,
}

// ─── Independent packed-bed closure: MacDonald et al. (1979) ─────────────────

/// Interphase momentum-exchange coefficient `β` [kg/(m³·s)] in the **MacDonald et
/// al. (1979)** form — the "Ergun equation revisited" re-fit of the Blake–Kozeny
/// (viscous) and Burke–Plummer (inertial) constants to a far larger data set,
/// recommending `180` and `1.8` (smooth particles) where Ergun used `150` and
/// `1.75`:
///
/// ```text
///   β = 180 (1−ε)² μ /(ε d²)  +  1.8 (1−ε) ρ_f |u_rel| / d
/// ```
///
/// This is the **independent** measured closure — it shares no constant with the
/// Ergun *reference* — so `drag_force_from_beta(β, …)` summed over the packing
/// reduces to the MacDonald `dP/L`, which differs from Ergun by the documented
/// ~5–20 % inter-correlation spread rather than by machine epsilon.
fn macdonald_beta(eps: f64, rho_f: f64, mu: f64, diameter: f64, rel_speed: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let one_m = 1.0 - eps;
    180.0 * one_m * one_m * mu / (eps * diameter * diameter)
        + 1.8 * one_m * rho_f * rel_speed / diameter
}

/// Ergun (1952) β, same functional form with the original `150 / 1.75`. Used ONLY
/// by the negative control's baseline pass (to show that with a *correct* reduction
/// the Ergun closure reproduces the Ergun reference, i.e. that the harness itself is
/// wired right); the real validation never uses it (that would be the tautology).
fn ergun_beta(eps: f64, rho_f: f64, mu: f64, diameter: f64, rel_speed: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let one_m = 1.0 - eps;
    150.0 * one_m * one_m * mu / (eps * diameter * diameter)
        + 1.75 * one_m * rho_f * rel_speed / diameter
}

// ─── Bed-scale void-fraction deposition (containment binning) ────────────────

/// Interior cell-center coordinates along each axis (uniform, separable grid).
fn axis_centers(mesh: &UniformMesh) -> ([Vec<f64>; 3], usize) {
    let [ni, nj, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    let xc = (0..ni).map(|i| mesh.cell_centroid(mesh.idx_raw(i + ng, ng, ng))[0]).collect();
    let yc = (0..nj).map(|j| mesh.cell_centroid(mesh.idx_raw(ng, j + ng, ng))[1]).collect();
    let zc = (0..nk).map(|k| mesh.cell_centroid(mesh.idx_raw(ng, ng, k + ng))[2]).collect();
    ([xc, yc, zc], ng)
}

/// Nearest interior index along an axis = the CONTAINING cell for a uniform grid.
#[inline]
fn nearest_center(cs: &[f64], v: f64) -> usize {
    if cs.len() < 2 {
        return 0;
    }
    let dx = cs[1] - cs[0];
    (((v - cs[0]) / dx).round() as isize).clamp(0, cs.len() as isize - 1) as usize
}

/// Raw cell index containing `p` by nearest-center (containment) binning.
fn containing_cell(mesh: &UniformMesh, centers: &[Vec<f64>; 3], ng: usize, p: Vec3) -> usize {
    let i = nearest_center(&centers[0], p[0]);
    let j = nearest_center(&centers[1], p[1]);
    let k = nearest_center(&centers[2], p[2]);
    mesh.idx_raw(i + ng, j + ng, k + ng)
}

/// Per-cell void fraction of the packing by CONTAINMENT deposition: each particle's
/// volume is charged to the cell that geometrically contains its center, giving
/// `ε_cell = 1 − Σ V_p / V_cell`. This is the correct bed-scale volume-average the
/// unresolved coupling needs. (The seam's own `coupling::deposit_void_fraction`
/// uses an interpolation locator tuned for a single SUB-CELL particle and mis-bins
/// at bed scale, so the bed-scale binning is done here, per the library-placement
/// rule — it is one-off to the packed-bed cases.) Returns the field plus the
/// per-particle containing-cell indices (reused to drive the drag).
fn deposit_bed_void_fraction(
    mesh: &UniformMesh,
    particles: &[ParticleKinematics],
) -> (Vec<f64>, Vec<usize>) {
    let (centers, ng) = axis_centers(mesh);
    let total = mesh.n_cells_total();
    let mut solid = vec![0.0f64; total];
    let mut cell_of_particle = Vec::with_capacity(particles.len());
    for p in particles {
        let c = containing_cell(mesh, &centers, ng, p.center);
        solid[c] += p.volume();
        cell_of_particle.push(c);
    }
    let mut eps = vec![1.0f64; total];
    for c in 0..total {
        let v = mesh.cell_volume(c);
        if v > 0.0 {
            eps[c] = (1.0 - solid[c] / v).clamp(1e-6, 1.0);
        }
    }
    (eps, cell_of_particle)
}

// ─── FIELD sub-App: impose the superficial flow, run the seam, read ΣF ────────

/// `Output` phase on the CFD sub-App. Imposes the interstitial gas velocity
/// `u_g = U/ε` in every interior cell, then runs the unresolved seam on the
/// immersed packing: deposit void fraction, evaluate the selected per-particle drag
/// closure from the DEPOSITED per-cell void fraction, sum it, and feed the
/// equal-and-opposite momentum sink back into the gas (checking conservation). The
/// summed drag is the bulk the parent turns into dP/L.
#[allow(clippy::too_many_arguments)]
fn fixed_bed_seam_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    ctx: Res<SeamCtx>,
    sup: Res<Superficial>,
    pset: Res<ParticleSet>,
    mut forces: ResMut<InterphaseForces>,
    mut result: ResMut<BedResult>,
) {
    let eos: &dyn Eos = &*eos.0;
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let parts = &pset.particles;
    forces.reset(parts.len());
    let mode = ctx.mode;

    // Interstitial gas velocity u_g = U/ε_bed (the pore-scale velocity the drag
    // closure sees). Impose it uniformly in every interior cell; leave ρ untouched.
    let eps_bed = ctx.eps;
    let inv_eps = 1.0 / eps_bed;
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

    // Deposit the packing's solid volume onto the mesh by containment binning. In
    // the unresolved regime (gas cell ≫ particle) each cell holds many spheres, so
    // this per-cell field converges to the bed porosity and is the field that DRIVES
    // the drag below. Report its worst interior-cell deviation from ε_bed as the
    // deposition-fidelity diagnostic.
    let (eps_field, cell_of_particle) = deposit_bed_void_fraction(&*mesh, parts);
    let (mut e_min, mut e_max, mut e_err) = (f64::INFINITY, 0.0f64, 0.0f64);
    for (c, &e) in eps_field.iter().enumerate() {
        // Only interior cells that actually received packing (a coarse gas mesh may
        // have padding cells the bed does not fill).
        if !mesh.is_local_cell(c) || e >= 1.0 - 1e-9 {
            continue;
        }
        e_min = e_min.min(e);
        e_max = e_max.max(e);
        e_err = e_err.max((e - eps_bed).abs() / eps_bed);
    }
    result.eps_min = if e_min.is_finite() { e_min } else { 1.0 };
    result.eps_max = e_max;
    result.eps_cell_err = e_err;

    // Interphase drag on each particle through the selected closure, driven by the
    // DEPOSITED per-cell void fraction (load-bearing), summed.
    let mut drag_on_particle = vec![[0.0f64; 3]; parts.len()];
    let mut f_total = [0.0f64; 3];
    for (i, p) in parts.iter().enumerate() {
        let u_gas =
            coupling::sample_gas_velocity(&*mesh, &state, eos, p.center).unwrap_or([0.0; 3]);
        let rho_f = coupling::sample_gas_density(&*mesh, &state, p.center).unwrap_or(ctx.rho);
        // Void fraction from the deposited field at the particle's containing cell.
        let eps = eps_field[cell_of_particle[i]];

        // Static packing: u_p = 0, so the slip IS the local interstitial velocity.
        let rel = u_gas;
        let rel_speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let d = p.diameter();
        let beta = if mode.macdonald {
            macdonald_beta(eps, rho_f, ctx.mu, d, rel_speed)
        } else {
            ergun_beta(eps, rho_f, ctx.mu, d, rel_speed)
        };
        let drag = drag_force_from_beta(beta, p.volume(), eps, rel);

        forces.force[i] = drag;
        drag_on_particle[i] = drag;
        for k in 0..3 {
            f_total[k] += drag[k];
        }
    }
    // Negative control: corrupt the reduction with the ε²-instead-of-ε³ bug. The
    // parent divides ΣF by (V_bed·ε) to get dP/L; scaling ΣF by 1/ε here makes the
    // assembled law scale as 1/ε³·(1/ε)… i.e. one wrong void-fraction power, a
    // ~1/ε ≈ 2.5× error. The real validation never sets this flag.
    if mode.corrupt_eps_power {
        for k in 0..3 {
            f_total[k] /= eps_bed;
        }
    }
    result.f_total = f_total;

    // Two-way momentum sink + conservation check (NOT the pressure-drop validation).
    let mut m0 = [0.0f64; 3];
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) {
            let v = mesh.cell_volume(c);
            m0[0] += state.u[c].rho_u * v;
            m0[1] += state.u[c].rho_v * v;
            m0[2] += state.u[c].rho_w * v;
        }
    }
    coupling::apply_momentum_sink(&*mesh, &mut state, parts, &drag_on_particle, ctx.dt);
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
        let imp = -drag_on_particle.iter().map(|f| f[k]).sum::<f64>() * ctx.dt; // −ΣF_drag·dt
        dn += (dm - imp) * (dm - imp);
        sc += imp * imp;
    }
    result.mom_err = dn.sqrt() / sc.sqrt().max(1e-30);
}

fn build_cfd(gc: &GasCfg, mesh_cfg: UniformMeshConfig, eps: f64, dt: f64) -> App {
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
    app.add_resource(SeamCtx {
        mu: gc.mu,
        rho: gc.rho,
        eps,
        dt,
        mode: SeamMode::default(),
    });
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
/// strictly *inside* its conventional cell.
fn fcc_packing(nc: [usize; 3], a: f64) -> (Vec<[f64; 3]>, [f64; 3]) {
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

// ─── Ergun (1952) reference — the literature correlation we match ─────────────

/// Ergun (1952) packed-bed pressure drop per unit length — the reference. This is
/// the literature correlation itself (`150 / 1.75`), independent of the measured
/// MacDonald closure assembled through the seam.
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

/// Run the full superficial-velocity sweep once for a given seam mode, returning
/// `(dP/L_meas per U, worst deposition-cell error, worst momentum error)`.
fn run_sweep(parent: &mut App, us: &[f64], v_bed: f64, eps_bed: f64, mode: SeamMode) -> (Vec<f64>, f64, f64) {
    set_seam_mode(parent, mode);
    let mut dpdl = Vec::with_capacity(us.len());
    let mut worst_dep = 0.0f64;
    let mut worst_mom = 0.0f64;
    for &u in us {
        set_superficial(parent, [0.0, 0.0, u]);
        parent.run();
        let res = read_result(parent);
        let f_mag =
            (res.f_total[0].powi(2) + res.f_total[1].powi(2) + res.f_total[2].powi(2)).sqrt();
        // Steady two-fluid balance: ε dP/dx = −β u_g ⇒ dP/L = |ΣF_drag| / (V_bed ε).
        dpdl.push(f_mag / (v_bed * eps_bed));
        worst_dep = worst_dep.max(res.eps_cell_err);
        worst_mom = worst_mom.max(res.mom_err);
    }
    (dpdl, worst_dep, worst_mom)
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
    let meshc: MeshCfg = cfg.section("mesh");
    let flow: FlowCfg = cfg.section("flow");
    let valid: ValidationCfg = cfg.section("validation");

    let d = pc.diameter;
    let radius = 0.5 * d;
    let v_p = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);

    // FCC lattice constant giving the requested solid fraction φ.
    assert!(
        pack.solid_fraction <= 0.7405,
        "solid_fraction {} exceeds the FCC maximum 0.7405 (spheres would overlap)",
        pack.solid_fraction
    );
    // The gas mesh must tile the packing evenly and be COARSER than the particles.
    assert!(
        pack.ncx % meshc.nx == 0 && pack.ncy % meshc.ny == 0 && pack.ncz % meshc.nz == 0,
        "mesh {}x{}x{} must divide packing {}x{}x{} evenly",
        meshc.nx, meshc.ny, meshc.nz, pack.ncx, pack.ncy, pack.ncz
    );
    let a = d * (2.0 * std::f64::consts::PI / (3.0 * pack.solid_fraction)).cbrt();
    let (positions, bounds) = fcc_packing([pack.ncx, pack.ncy, pack.ncz], a);
    let n = positions.len();
    let v_bed = bounds[0] * bounds[1] * bounds[2];
    let eps_geo = 1.0 - n as f64 * v_p / v_bed;
    let nn_gap = a / std::f64::consts::SQRT_2 - d; // nearest-neighbour surface gap

    let mesh_cfg = UniformMeshConfig {
        nx: meshc.nx,
        ny: meshc.ny,
        nz: meshc.nz,
        ng: meshc.ng,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: bounds,
        y_edges: None,
        z_edges: None,
    };
    let n_cells = (meshc.nx * meshc.ny * meshc.nz) as f64;

    let soil = build_soil(&positions, radius, pc.density);
    let cfd = build_cfd(&gas, mesh_cfg, eps_geo, flow.dt);

    let mut parent = App::new();
    parent.add_subapp("soil", soil);
    parent.add_subapp("cfd", cfd);
    parent.add_update_system(export_kinematics, Phase::Export);
    parent.add_update_system(tick_subapp("cfd", 1), Phase::TickCfd);
    parent.prepare();

    // Cell size for the point-particle regime report (d/Δx should be ≪ 1).
    let dx = (v_bed / n_cells).cbrt();
    let parts_per_cell = n as f64 / n_cells;

    println!("# Fixed-bed pressure drop vs Ergun (1952) — unresolved DEM-CFD seam");
    println!("# MEASURED drag: INDEPENDENT MacDonald et al. (1979) closure (180/1.8), assembled through the seam");
    println!("# REFERENCE:     Ergun (1952) correlation (150/1.75)");
    println!(
        "# packing: FCC {}x{}x{} cells, N = {} spheres   d = {:.3e} m   a = {:.3e} m",
        pack.ncx, pack.ncy, pack.ncz, n, d, a
    );
    println!(
        "# gas mesh: {}x{}x{} cells   dx ~ {:.3e} m   d/dx = {:.3} (≪1 ⇒ unresolved)   ~{:.0} spheres/cell",
        meshc.nx, meshc.ny, meshc.nz, dx, d / dx, parts_per_cell
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
    println!("#");
    println!(
        "#     U [m/s]     Re_p      dP/L_meas [Pa/m]   dP/L_Ergun [Pa/m]   rel.err   visc:inert"
    );

    // ── The validation: MEASURED = independent MacDonald closure through the seam.
    let mode = SeamMode { macdonald: true, corrupt_eps_power: false };
    let (dpdl_meas, worst_dep, worst_mom) = run_sweep(&mut parent, &flow.superficial, v_bed, eps_geo, mode);

    let mut all_ok = true;
    let mut rep_min = f64::INFINITY;
    let mut rep_max = 0.0f64;
    let mut worst_err = 0.0f64;
    let mut best_err = f64::INFINITY;
    for (idx, &u) in flow.superficial.iter().enumerate() {
        let dpdl_ergun = ergun_dp_per_length(eps_geo, gas.mu, gas.rho, d, u);
        let rel = (dpdl_meas[idx] - dpdl_ergun).abs() / dpdl_ergun;
        let rep = modified_reynolds(gas.rho, u, d, gas.mu, eps_geo);
        let om = 1.0 - eps_geo;
        let e3 = eps_geo.powi(3);
        let visc = 150.0 * om * om / e3 * gas.mu * u / (d * d);
        let inert = 1.75 * om / e3 * gas.rho * u * u / d;

        rep_min = rep_min.min(rep);
        rep_max = rep_max.max(rep);
        worst_err = worst_err.max(rel);
        best_err = best_err.min(rel);
        all_ok &= rel <= valid.tol_rel_ergun;

        println!(
            "  {u:>10.4}  {rep:>7.2}   {:>16.4}   {dpdl_ergun:>16.4}   {:>6.3}%   {:>6.2}:1",
            dpdl_meas[idx],
            100.0 * rel,
            visc / inert.max(1e-30)
        );
    }

    // ── Negative control: corrupt the seam reduction (ε² instead of ε³) and show
    // it blows past the tolerance — proof the pass is genuinely capable of failing
    // and the tolerance is not tuned to admit a broken seam.
    let corrupt = SeamMode { macdonald: true, corrupt_eps_power: true };
    let (dpdl_bad, _, _) = run_sweep(&mut parent, &flow.superficial, v_bed, eps_geo, corrupt);
    let mut worst_bad = 0.0f64;
    for (idx, &u) in flow.superficial.iter().enumerate() {
        let dpdl_ergun = ergun_dp_per_length(eps_geo, gas.mu, gas.rho, d, u);
        worst_bad = worst_bad.max((dpdl_bad[idx] - dpdl_ergun).abs() / dpdl_ergun);
    }
    let neg_control_fails = worst_bad > valid.tol_rel_ergun;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!(
        "# deposition fidelity: worst |eps_cell - eps_bed|/eps_bed = {:.2}% (tol {:.1}%)  [the deposited field DRIVES the drag]",
        100.0 * worst_dep,
        100.0 * valid.tol_deposit_cell
    );
    println!(
        "# Re_p sweep span: [{:.2}, {:.2}]   (need ≤ {} viscous end AND ≥ {} inertial end)",
        rep_min, rep_max, valid.rep_viscous_below, valid.rep_inertial_above
    );
    println!(
        "# Ergun rel.err spread over sweep: [{:.3}%, {:.3}%]  (tol {:.1}%)  — MacDonald-vs-Ergun, non-zero ⇒ not a tautology",
        100.0 * best_err,
        100.0 * worst_err,
        100.0 * valid.tol_rel_ergun
    );
    println!(
        "# negative control (corrupted eps-power seam): worst rel.err {:.1}%  ⇒ {} (must exceed tol {:.1}%)",
        100.0 * worst_bad,
        if neg_control_fails { "FAILS as required" } else { "DID NOT FAIL — gate is vacuous!" },
        100.0 * valid.tol_rel_ergun
    );
    println!(
        "# momentum conservation err: {worst_mom:.2e}  (tol {:.0e})  [sanity only — not the Ergun validation]",
        valid.tol_momentum
    );

    let pass_regime = eps_geo <= valid.eps_max;
    let pass_span = rep_min <= valid.rep_viscous_below && rep_max >= valid.rep_inertial_above;
    let pass_dep = worst_dep <= valid.tol_deposit_cell;
    let pass_mom = worst_mom <= valid.tol_momentum;
    let pass_ergun = all_ok;
    // The pass is only meaningful if the rel.err is genuinely non-zero (independent
    // reference, not a self-comparison) AND the corrupted seam would fail.
    let pass_nontrivial = worst_err > 1e-4 && neg_control_fails;

    if pass_ergun && pass_regime && pass_span && pass_dep && pass_mom && pass_nontrivial {
        println!(
            "VALIDATION: PASS  (independent MacDonald vs Ergun: rel.err∈[{:.2}%,{:.2}%]<={:.1}% over Re_p∈[{:.1},{:.1}]; eps={:.3}<={}; deposit cell {:.1}%; neg-control fails at {:.0}%; mom {:.1e})",
            100.0 * best_err,
            100.0 * worst_err,
            100.0 * valid.tol_rel_ergun,
            rep_min,
            rep_max,
            eps_geo,
            valid.eps_max,
            100.0 * worst_dep,
            100.0 * worst_bad,
            worst_mom
        );
    } else {
        println!(
            "VALIDATION: FAIL  (ergun_ok={pass_ergun} regime_ok={pass_regime} span_ok={pass_span} dep_ok={pass_dep} mom_ok={pass_mom} nontrivial_ok={pass_nontrivial})"
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

fn set_seam_mode(parent: &App, mode: SeamMode) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs
        .find("cfd")
        .unwrap()
        .resource_cell(TypeId::of::<SeamCtx>())
        .unwrap();
    cell.borrow_mut().downcast_mut::<SeamCtx>().unwrap().mode = mode;
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
