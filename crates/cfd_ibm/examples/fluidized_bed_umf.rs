//! **Minimum fluidization velocity `U_mf` via the unresolved DEM–CFD seam** — the
//! *dynamic* two-way-coupling capstone of the coupling ladder, one rung above the
//! static fixed-bed pressure drop (`fixed_bed_ergun`).
//!
//! A packing of spheres (the SOIL/Lagrangian bed) is immersed in an upward gas
//! stream (the FIELD/Eulerian gas). Gravity pulls the bed down; the interphase
//! force pushes it up. The **minimum fluidization velocity** `U_mf` is the
//! superficial gas velocity at which the total upward fluid force on the bed just
//! balances its net (buoyant) weight — below it the bed stays packed on the
//! distributor, above it the bed lifts and fluidizes. The classic result equates
//! the Ergun pressure drop to the buoyant weight per unit bed length,
//!
//! ```text
//!   (dP/L)|_{U_mf}  =  (1 − ε)(ρ_p − ρ_f) g,
//! ```
//!
//! and the reference used here is the **Wen & Yu (1966)** closed-form correlation
//!
//! ```text
//!   Re_mf = sqrt(33.7² + 0.0408 Ar) − 33.7,     Ar = ρ_f (ρ_p − ρ_f) g d³ / μ²,
//!   U_mf  = Re_mf · μ / (ρ_f d),
//! ```
//!
//! (`Ar` = Archimedes number). This is the standard CFD–DEM fluidization
//! validation, cf. Goniva et al. (2012) for CFDEM.
//!
//! ## Why this is DYNAMIC (and what it adds over `fixed_bed_ergun`)
//!
//! The fixed-bed rung imposes a **static** packing and checks a pressure drop. Here
//! the bed particles are **free to move**: each is a `soil::Atom` integrated by
//! `soil_verlet` under gravity + the fluid force handed across the seam. The whole
//! coupled system is stepped, and the bed's own **centre-of-mass acceleration**
//! `a_z(U)` is *measured from the integration*, not computed on paper. That curve
//! is the fluidization signature:
//!
//!   * `U < U_mf` → `a_z < 0`: net force is downward, the bed compacts onto the
//!     distributor and stays a packed fixed bed;
//!   * `U > U_mf` → `a_z > 0`: net force is upward, the bed lifts — it fluidizes;
//!   * the zero-crossing `a_z(U_mf) = 0` is incipient fluidization.
//!
//! `U_mf` is extracted two ways that must agree: (i) a bisection on the **measured**
//! net bed force read back through the live seam at rest, and (ii) the zero-crossing
//! of the **integrated** `a_z(U)` sweep. Both are then compared to Wen & Yu.
//!
//! ## The total fluid force on a bed particle — drag AND the ∇P (buoyancy) force
//!
//! A particle in a bed feels two fluid forces, and getting `U_mf` right requires
//! **both**:
//!
//!   1. **Interphase drag**, `F_drag = β V_p /(1−ε) · (u_g − u_p)` (the same
//!      [`coupling::drag_force_from_beta`] the settling and fixed-bed cases use).
//!      Summed over the packing this is only `ε · (dP/L) V_bed`.
//!   2. **The pressure-gradient (generalized buoyancy) force**, `−V_p ∇P`. In the
//!      imposed-flow (porous-medium) model the gas-phase momentum balance fixes the
//!      streamwise gradient at `ε ∇P = −β (u_g − u_p)`, so the force on each
//!      particle is `+V_p β (u_g − u_p)/ε`. Summed this is the remaining
//!      `(1−ε) · (dP/L) V_bed`. This is the standard unresolved-CFD–DEM "pressure
//!      gradient force" (e.g. Kafui et al. 2002; Zhou et al. 2010); it is the
//!      bed-scale generalization of the tiny `−ρ_f V g` hydrostatic buoyancy the
//!      settling case carries (recovered as the `∇P → ρ_f g`, no-flow limit).
//!
//! Only `F_drag + F_∇P = (dP/L) V_bed` balances the buoyant weight at the correct
//! velocity. **Dropping the ∇P force is a real, commonly-made CFD–DEM mistake**; it
//! shifts `U_mf` by a factor `~1/ε` (here ≈ +80 %), so this example runs it as a
//! **negative control** and asserts the Wen-&-Yu gate then FAILS. A second negative
//! control injects the fixed-bed's `ε²`-instead-of-`ε³` reduction bug (≈ −53 %).
//!
//! ## Non-tautology (independent measured closure)
//!
//! Exactly as in `fixed_bed_ergun`, the **measured** drag is assembled from the
//! *independent* **MacDonald et al. (1979)** packed-bed closure (`180 / 1.8`), which
//! shares **no constant** with the Ergun-based Wen & Yu (1966) reference
//! (`150 / 1.75` → `33.7 / 0.0408`). The resulting `U_mf` therefore differs from
//! Wen & Yu by a genuine, non-zero spread (≈ 4.5 % here) — the documented
//! inter-correlation difference, well inside Wen & Yu's own ~34 % reported scatter —
//! not by construction. For orientation the example also reports the crossover
//! obtained with the *exact* Ergun (1952) constants, which brackets Wen & Yu from
//! the other side (≈ +4 %).
//!
//! The gas flow is **imposed** (interstitial `u_g = U/ε`), not solved — a porosity-
//! weighted momentum solve is the resolved-track story, out of scope for the
//! unresolved seam (cf. the `Cd(Re)` deferral in `cfd_ibm::coupling`); the
//! falsifiability here comes from the independent drag reference and the dynamic
//! force balance, not from a flow solve. Everything case-specific is declarative
//! TOML from `argv[1]`:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example fluidized_bed_umf -- \
//!     crates/cfd_ibm/examples/fluidized_bed_umf_sand.toml
//! ```
//!
//! References:
//! * C.Y. Wen & Y.H. Yu, "A generalized method for predicting the minimum
//!   fluidization velocity", *AIChE J.* 12(3):610–612 (1966).
//! * S. Ergun, "Fluid flow through packed columns", *Chem. Eng. Prog.* 48(2):89 (1952).
//! * I.F. MacDonald, M.S. El-Sayed, K. Mow, F.A.L. Dullien, "Flow through Porous
//!   Media — the Ergun Equation Revisited", *Ind. Eng. Chem. Fundam.* 18(3):199 (1979).
//! * C. Goniva et al., "Influence of rolling friction on single spout fluidized bed
//!   simulation", *Particuology* 10(5):582–591 (2012) — CFDEM CFD–DEM validation.

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
use soil_core::{Atom, ParticleSimScheduleSet};
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
    /// Bead diameter d [m].
    diameter: f64,
    density: f64,
}

/// FCC packing, identical construction to `fixed_bed_ergun`: `nc*` conventional
/// cells (4 spheres each), lattice constant set so the solid fraction equals
/// `solid_fraction` (bed porosity ε = 1 − solid_fraction), non-overlapping.
#[derive(Deserialize, Default)]
struct PackingCfg {
    ncx: usize,
    ncy: usize,
    ncz: usize,
    /// Target solid volume fraction φ = 1 − ε (must be ≤ 0.74, the FCC max).
    solid_fraction: f64,
}

/// Gas mesh — **coarser than the particles** (unresolved regime). `nx*` must divide
/// the packing's conventional-cell counts so the domain tiles evenly.
#[derive(Deserialize, Default)]
struct MeshCfg {
    nx: usize,
    ny: usize,
    nz: usize,
    #[serde(default = "default_ng")]
    ng: usize,
}
fn default_ng() -> usize {
    2
}

#[derive(Deserialize, Default)]
struct GravityCfg {
    gz: f64,
}

#[derive(Deserialize, Default)]
struct FlowCfg {
    /// Superficial velocities U [m/s] to sweep (must bracket U_mf: some below, some above).
    superficial: Vec<f64>,
    /// Coupling / integration timestep.
    dt: f64,
    /// Number of coupled steps integrated from rest to MEASURE the bed's
    /// centre-of-mass acceleration a_z(U) at each sweep point.
    settle_steps: usize,
}

#[derive(Deserialize, Default)]
struct ValidationCfg {
    /// |U_mf_meas − U_mf_WenYu| / U_mf_WenYu. Brackets the documented MacDonald(1979)-
    /// vs-Wen&Yu(1966) closure spread; well inside Wen&Yu's own ~34% scatter.
    tol_rel_umf: f64,
    /// The measured rel.err must be at least this (non-tautology floor: an
    /// independent closure, not a self-comparison, so the error is genuinely > 0).
    umf_err_floor: f64,
    /// |a_z_measured(integrated) − a_z_force(net seam force / M_bed)| / g, per sweep
    /// point — the two-way handoff: the integrator must deliver the seam force to
    /// the freely-moving DEM bed.
    tol_handoff: f64,
    /// Regime gate: packed-bed (dense Ergun/MacDonald) branch requires ε ≤ this.
    eps_max: f64,
    /// Two-way momentum-exchange conservation tolerance (sink vs −ΣF_drag·dt).
    tol_momentum: f64,
    /// Worst |ε_cell(deposit) − ε_bed| / ε_bed over interior cells (deposition
    /// fidelity — the deposited void fraction DRIVES the drag).
    tol_deposit_cell: f64,
}

// ─── Independent packed-bed closures ─────────────────────────────────────────

/// MacDonald et al. (1979) interphase coefficient β (the "Ergun revisited" 180/1.8
/// re-fit) — the INDEPENDENT measured closure (shares no constant with Wen & Yu).
fn macdonald_beta(eps: f64, rho_f: f64, mu: f64, d: f64, rel_speed: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let om = 1.0 - eps;
    180.0 * om * om * mu / (eps * d * d) + 1.8 * om * rho_f * rel_speed / d
}

/// Ergun (1952) β (150/1.75) — reported only for the exact-Ergun crossover that
/// brackets Wen & Yu; never the measured validation closure (that would share the
/// reference's constants).
fn ergun_beta(eps: f64, rho_f: f64, mu: f64, d: f64, rel_speed: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let om = 1.0 - eps;
    150.0 * om * om * mu / (eps * d * d) + 1.75 * om * rho_f * rel_speed / d
}

/// Which β closure the seam assembles, and whether to inject a fault (negative
/// controls). Set by the parent per pass.
#[derive(Clone, Copy)]
struct SeamMode {
    /// `true` → MacDonald(1979) measured closure; `false` → Ergun(1952) constants
    /// (used only for the reported bracket, never the pass gate).
    macdonald: bool,
    /// Negative control A: drop the ∇P (pressure-gradient buoyancy) force — a real
    /// CFD–DEM mistake that shifts U_mf by ~1/ε.
    omit_pressure_grad: bool,
    /// Negative control B: the fixed-bed's ε²-instead-of-ε³ reduction bug (scale
    /// the assembled force by 1/ε).
    corrupt_eps_power: bool,
}
impl Default for SeamMode {
    fn default() -> Self {
        Self { macdonald: true, omit_pressure_grad: false, corrupt_eps_power: false }
    }
}

fn beta_for(mode: SeamMode, eps: f64, rho_f: f64, mu: f64, d: f64, rel_speed: f64) -> f64 {
    if mode.macdonald {
        macdonald_beta(eps, rho_f, mu, d, rel_speed)
    } else {
        ergun_beta(eps, rho_f, mu, d, rel_speed)
    }
}

// ─── Wen & Yu (1966) reference + exact-Ergun bracket ─────────────────────────

/// Archimedes number `Ar = ρ_f (ρ_p − ρ_f) g d³ / μ²`.
fn archimedes(rho_f: f64, rho_p: f64, g: f64, d: f64, mu: f64) -> f64 {
    rho_f * (rho_p - rho_f) * g.abs() * d.powi(3) / (mu * mu)
}

/// Wen & Yu (1966) minimum fluidization velocity: `Re_mf = sqrt(33.7² + 0.0408 Ar) − 33.7`.
fn u_mf_wen_yu(rho_f: f64, rho_p: f64, g: f64, d: f64, mu: f64) -> f64 {
    let ar = archimedes(rho_f, rho_p, g, d, mu);
    let re_mf = (33.7f64 * 33.7 + 0.0408 * ar).sqrt() - 33.7;
    re_mf * mu / (rho_f * d)
}

/// Superficial velocity at which a packed-bed pressure drop (`C1` viscous, `C2`
/// inertial constants, porosity `eps`) equals the buoyant weight per unit length
/// `(1−ε)(ρ_p−ρ_f)g` — the incipient-fluidization criterion. Closed form of the
/// Ergun/MacDonald quadratic `a_inert U² + a_visc U − target = 0`. Used for the
/// analytic Ergun/MacDonald brackets reported alongside the SEAM-measured U_mf.
#[allow(clippy::too_many_arguments)]
fn u_mf_balance(c1: f64, c2: f64, eps: f64, rho_f: f64, rho_p: f64, g: f64, d: f64, mu: f64) -> f64 {
    let om = 1.0 - eps;
    let e3 = eps.powi(3);
    let a_visc = c1 * om / e3 * mu / (d * d); // × U
    let a_inert = c2 / e3 * rho_f / d; // × U²
    let target = (rho_p - rho_f) * g.abs(); // (dP/L)/(1−ε) at balance
    (-a_visc + (a_visc * a_visc + 4.0 * a_inert * target).sqrt()) / (2.0 * a_inert)
}

// ─── Bed-scale void-fraction deposition (containment binning) ────────────────
// Mirrors `fixed_bed_ergun`'s helper: the seam's own interpolation locator is
// tuned for a single SUB-CELL particle and mis-bins at bed scale, so bed-scale
// containment binning lives in the example (library-placement rule — one-off to
// the packed-bed cases).

fn axis_centers(mesh: &UniformMesh) -> ([Vec<f64>; 3], usize) {
    let [ni, nj, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    let xc = (0..ni).map(|i| mesh.cell_centroid(mesh.idx_raw(i + ng, ng, ng))[0]).collect();
    let yc = (0..nj).map(|j| mesh.cell_centroid(mesh.idx_raw(ng, j + ng, ng))[1]).collect();
    let zc = (0..nk).map(|k| mesh.cell_centroid(mesh.idx_raw(ng, ng, k + ng))[2]).collect();
    ([xc, yc, zc], ng)
}
#[inline]
fn nearest_center(cs: &[f64], v: f64) -> usize {
    if cs.len() < 2 {
        return 0;
    }
    let dx = cs[1] - cs[0];
    (((v - cs[0]) / dx).round() as isize).clamp(0, cs.len() as isize - 1) as usize
}
fn containing_cell(mesh: &UniformMesh, centers: &[Vec<f64>; 3], ng: usize, p: Vec3) -> usize {
    let i = nearest_center(&centers[0], p[0]);
    let j = nearest_center(&centers[1], p[1]);
    let k = nearest_center(&centers[2], p[2]);
    mesh.idx_raw(i + ng, j + ng, k + ng)
}
/// Per-cell void fraction `ε = 1 − ΣV_p/V_cell` by containment deposition, plus the
/// per-particle containing-cell index (reused to drive the drag).
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

// ─── Seam resources shared across the namespace boundary ─────────────────────

/// Per-particle total fluid force (drag + ∇P buoyancy + hydrostatic), FIELD→SOIL.
#[derive(Clone, Debug, Default)]
struct FluidForces {
    f: Vec<Vec3>,
}
/// Gravity applied to the bed atoms (SOIL side).
#[derive(Clone, Copy)]
struct BodyAccel {
    g: Vec3,
}
/// Imposed superficial velocity for the current sweep point (world axes).
#[derive(Clone, Copy, Default)]
struct Superficial {
    u: Vec3,
}
/// Static seam context on the FIELD sub-App.
#[derive(Clone, Copy)]
struct SeamCtx {
    mu: f64,
    rho: f64,
    /// Bed porosity ε used for the superficial↔interstitial conversion `u_g = U/ε`.
    eps: f64,
    /// Gravity (for the hydrostatic buoyancy term).
    g: Vec3,
    dt: f64,
    mode: SeamMode,
}
/// Read back by the parent each coupled step.
#[derive(Clone, Copy, Default)]
struct BedResult {
    /// Σ drag force on the packing (world axes) — used for the momentum check.
    f_drag_total: Vec3,
    /// Σ TOTAL fluid force (drag + ∇P buoyancy + hydrostatic) — the force that
    /// balances the bed weight at U_mf.
    f_fluid_total: Vec3,
    eps_cell_err: f64,
    mom_err: f64,
}

// ─── FIELD sub-App: impose the flow, run the seam, deliver per-particle force ─

#[allow(clippy::too_many_arguments)]
fn fluidized_seam_system(
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

    // Impose the interstitial gas velocity u_g = U/ε_bed in every interior cell.
    let inv_eps = 1.0 / ctx.eps;
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

    // Deposit the packing's void fraction (drives the drag) and report fidelity.
    let (eps_field, cell_of_particle) = deposit_bed_void_fraction(&*mesh, parts);
    let mut e_err = 0.0f64;
    for (c, &e) in eps_field.iter().enumerate() {
        if !mesh.is_local_cell(c) || e >= 1.0 - 1e-9 {
            continue;
        }
        e_err = e_err.max((e - ctx.eps).abs() / ctx.eps);
    }
    result.eps_cell_err = e_err;

    let mut drag_on_particle = vec![[0.0f64; 3]; parts.len()];
    let mut f_drag = [0.0f64; 3];
    let mut f_fluid = [0.0f64; 3];
    for (i, p) in parts.iter().enumerate() {
        let u_gas =
            coupling::sample_gas_velocity(&*mesh, &state, eos, p.center).unwrap_or(u_g);
        let rho_f = coupling::sample_gas_density(&*mesh, &state, p.center).unwrap_or(ctx.rho);
        let eps = eps_field[cell_of_particle[i]];

        // Slip = local interstitial gas velocity − particle velocity.
        let rel = [
            u_gas[0] - p.velocity[0],
            u_gas[1] - p.velocity[1],
            u_gas[2] - p.velocity[2],
        ];
        let rel_speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let d = p.diameter();
        let beta = beta_for(mode, eps, rho_f, ctx.mu, d, rel_speed);
        let v_p = p.volume();

        // (1) Interphase drag F = β V_p/(1−ε) · u_rel  (Σ = ε (dP/L) V_bed).
        let drag = drag_force_from_beta(beta, v_p, eps, rel);

        // (2) Pressure-gradient (generalized buoyancy) force −V_p ∇P. The gas-phase
        //     momentum balance in the imposed-flow model fixes ε ∇P = −β u_rel, so
        //     the force on the particle is +V_p β u_rel / ε  (Σ = (1−ε)(dP/L) V_bed).
        //     Dropped by the omit_pressure_grad negative control.
        let pg_coeff = if mode.omit_pressure_grad { 0.0 } else { v_p * beta / eps };
        let f_pg = [pg_coeff * rel[0], pg_coeff * rel[1], pg_coeff * rel[2]];

        // (3) Hydrostatic buoyancy −ρ_f V g (tiny in gas; the ∇P→ρ_f g no-flow limit).
        let buoy = [
            -rho_f * v_p * ctx.g[0],
            -rho_f * v_p * ctx.g[1],
            -rho_f * v_p * ctx.g[2],
        ];

        let mut total = [
            drag[0] + f_pg[0] + buoy[0],
            drag[1] + f_pg[1] + buoy[1],
            drag[2] + f_pg[2] + buoy[2],
        ];
        // Negative control B: ε²-instead-of-ε³ reduction bug (scale total by 1/ε).
        if mode.corrupt_eps_power {
            for k in 0..3 {
                total[k] /= ctx.eps;
            }
        }

        forces.force[i] = total;
        drag_on_particle[i] = drag;
        for k in 0..3 {
            f_drag[k] += drag[k];
            f_fluid[k] += total[k];
        }
    }
    result.f_drag_total = f_drag;
    result.f_fluid_total = f_fluid;

    // Two-way momentum sink (reaction of the DRAG part) + conservation check.
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
    let (mut dn, mut sc) = (0.0f64, 0.0f64);
    for k in 0..3 {
        let dm = m1[k] - m0[k];
        let imp = -drag_on_particle.iter().map(|f| f[k]).sum::<f64>() * ctx.dt;
        dn += (dm - imp) * (dm - imp);
        sc += imp * imp;
    }
    result.mom_err = dn.sqrt() / sc.sqrt().max(1e-30);
}

fn build_cfd(gc: &GasCfg, mesh_cfg: UniformMeshConfig, eps: f64, gz: f64, dt: f64) -> App {
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
        g: [0.0, 0.0, gz],
        dt,
        mode: SeamMode::default(),
    });
    app.add_resource(Superficial::default());
    app.add_resource(ParticleSet::default());
    app.add_resource(InterphaseForces::default());
    app.add_resource(BedResult::default());
    app.add_update_system(fluidized_seam_system, MeshScheduleSet::Output);
    app
}

// ─── SOIL sub-App: the freely-moving bed ─────────────────────────────────────

/// `Force` phase: force = gravity + the per-particle fluid load from the seam.
fn bed_force(mut atoms: ResMut<Atom>, ff: Res<FluidForces>, body: Res<BodyAccel>) {
    let n = atoms.nlocal as usize;
    for i in 0..n {
        let m = atoms.mass[i] as f64;
        let f = ff.f.get(i).copied().unwrap_or([0.0; 3]);
        atoms.force[i] = [
            (m * body.g[0] + f[0]) as _,
            (m * body.g[1] + f[1]) as _,
            (m * body.g[2] + f[2]) as _,
        ];
    }
}

fn build_soil(positions: &[[f64; 3]], radius: f64, density: f64, gz: f64, dt: f64) -> App {
    let mut atoms = Atom::new();
    atoms.dt = dt;
    let mass = density * 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    for (tag, pos) in positions.iter().enumerate() {
        atoms.push_test_atom(tag as u32, *pos, radius, mass);
    }
    atoms.nlocal = positions.len() as u32;
    atoms.natoms = positions.len() as u64;

    let mut app = App::new();
    app.add_resource(atoms);
    app.add_resource(FluidForces::default());
    app.add_resource(BodyAccel { g: [0.0, 0.0, gz] });
    app.add_update_system(bed_force, ParticleSimScheduleSet::Force);
    app.add_plugins(VelocityVerletPlugin::new());
    app
}

/// FCC sphere centers filling `[0,Lx]×[0,Ly]×[0,Lz]` (identical to `fixed_bed_ergun`).
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

// ─── Parent coupling schedule ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum Phase {
    Export,
    TickCfd,
    Import,
    TickSoil,
}
impl ScheduleSet for Phase {
    fn to_index(&self) -> u32 {
        match self {
            Self::Export => 0,
            Self::TickCfd => 1,
            Self::Import => 2,
            Self::TickSoil => 3,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Export => "Export",
            Self::TickCfd => "TickCfd",
            Self::Import => "Import",
            Self::TickSoil => "TickSoil",
        }
    }
}

#[derive(Clone, Copy)]
struct ParticleSpec {
    radius: f64,
}

/// SOIL→FIELD: hand the (moving) bed kinematics across.
fn export_kinematics(world: Multi, spec: Res<ParticleSpec>) {
    let atoms = world.expect_read::<Atom>("soil");
    let n = atoms.nlocal as usize;
    let mut set = world.expect_write::<ParticleSet>("cfd");
    set.particles.clear();
    for i in 0..n {
        set.particles.push(ParticleKinematics {
            center: [atoms.pos[i][0] as f64, atoms.pos[i][1] as f64, atoms.pos[i][2] as f64],
            velocity: [atoms.vel[i][0] as f64, atoms.vel[i][1] as f64, atoms.vel[i][2] as f64],
            radius: spec.radius,
        });
    }
}

/// FIELD→SOIL: copy the per-particle total fluid force to the bed.
fn import_force(world: Multi) {
    let forces = world.expect_read::<InterphaseForces>("cfd");
    let v = forces.force.clone();
    drop(forces);
    world.expect_write::<FluidForces>("soil").f = v;
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: fluidized_bed_umf <case.toml>");
    let toml_src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let gas: GasCfg = cfg.section("gas");
    let pc: ParticleCfg = cfg.section("particle");
    let pack: PackingCfg = cfg.section("packing");
    let meshc: MeshCfg = cfg.section("mesh");
    let grav: GravityCfg = cfg.section("gravity");
    let flow: FlowCfg = cfg.section("flow");
    let valid: ValidationCfg = cfg.section("validation");

    let d = pc.diameter;
    let radius = 0.5 * d;
    let v_p = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    let g = grav.gz.abs();

    assert!(
        pack.solid_fraction <= 0.7405,
        "solid_fraction {} exceeds FCC max 0.7405",
        pack.solid_fraction
    );
    assert!(
        pack.ncx % meshc.nx == 0 && pack.ncy % meshc.ny == 0 && pack.ncz % meshc.nz == 0,
        "mesh {}x{}x{} must divide packing {}x{}x{} evenly",
        meshc.nx, meshc.ny, meshc.nz, pack.ncx, pack.ncy, pack.ncz
    );
    let a = d * (2.0 * std::f64::consts::PI / (3.0 * pack.solid_fraction)).cbrt();
    let (positions, bounds) = fcc_packing([pack.ncx, pack.ncy, pack.ncz], a);
    let n = positions.len();
    let v_bed = bounds[0] * bounds[1] * bounds[2];
    let eps = 1.0 - n as f64 * v_p / v_bed;
    let nn_gap = a / std::f64::consts::SQRT_2 - d;

    let mass = pc.density * v_p;
    let m_bed = mass * n as f64;
    let w_buoy = (pc.density - gas.rho) * (1.0 - eps) * v_bed * g;

    // Analytic references.
    let ar = archimedes(gas.rho, pc.density, g, d, gas.mu);
    let u_wy = u_mf_wen_yu(gas.rho, pc.density, g, d, gas.mu);
    let u_erg = u_mf_balance(150.0, 1.75, eps, gas.rho, pc.density, g, d, gas.mu);
    let u_mac = u_mf_balance(180.0, 1.8, eps, gas.rho, pc.density, g, d, gas.mu);
    let re_wy = gas.rho * u_wy * d / gas.mu;

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
    let dx = (v_bed / n_cells).cbrt();

    let soil = build_soil(&positions, radius, pc.density, grav.gz, flow.dt);
    let cfd = build_cfd(&gas, mesh_cfg, eps, grav.gz, flow.dt);
    let mut parent = App::new();
    parent.add_subapp("soil", soil);
    parent.add_subapp("cfd", cfd);
    parent.add_resource(ParticleSpec { radius });
    parent.add_update_system(export_kinematics, Phase::Export);
    parent.add_update_system(tick_subapp("cfd", 1), Phase::TickCfd);
    parent.add_update_system(import_force, Phase::Import);
    parent.add_update_system(tick_subapp("soil", 1), Phase::TickSoil);
    parent.prepare();

    println!("# Minimum fluidization velocity U_mf — DYNAMIC unresolved DEM-CFD seam");
    println!("# MEASURED drag: INDEPENDENT MacDonald et al. (1979) closure (180/1.8), assembled through the seam");
    println!("# REFERENCE:     Wen & Yu (1966) correlation  Re_mf = sqrt(33.7^2 + 0.0408 Ar) - 33.7");
    println!(
        "# packing: FCC {}x{}x{} cells, N = {} spheres   d = {:.3e} m   a = {:.3e} m   nn gap = {:.2e} m",
        pack.ncx, pack.ncy, pack.ncz, n, d, a, nn_gap
    );
    println!(
        "# gas mesh: {}x{}x{} cells   dx ~ {:.3e} m   d/dx = {:.3} (<<1 => unresolved)   ~{:.0} spheres/cell",
        meshc.nx, meshc.ny, meshc.nz, dx, d / dx, n as f64 / n_cells
    );
    println!(
        "# porosity eps = {:.4}   rho_p = {}   rho_f = {}   mu = {:.3e}   g = {}",
        eps, pc.density, gas.rho, gas.mu, g
    );
    println!(
        "# M_bed = {:.4e} kg   W_buoy = (1-eps)(rho_p-rho_f)V_bed g = {:.4e} N",
        m_bed, w_buoy
    );
    println!("# Ar = {ar:.1}");
    println!(
        "# U_mf (Wen&Yu 1966, REFERENCE) = {u_wy:.5} m/s   (Re_mf = {re_wy:.3})",
    );
    println!(
        "# U_mf (Ergun 1952 exact balance, bracket) = {u_erg:.5} m/s  ({:+.2}% vs Wen&Yu)",
        100.0 * (u_erg / u_wy - 1.0)
    );
    println!(
        "# U_mf (MacDonald 1979 exact balance)       = {u_mac:.5} m/s  ({:+.2}% vs Wen&Yu)  <- the seam should measure this",
        100.0 * (u_mac / u_wy - 1.0)
    );
    println!("#");

    // ── (1) Measure U_mf through the LIVE seam: bisection on the net bed force
    // read back at rest (MacDonald closure, full physics). f_net(U) = ΣF_fluid_z − W_full.
    let mode = SeamMode::default();
    let mut worst_dep = 0.0f64;
    let mut worst_mom = 0.0f64;
    let net_force = |parent: &mut App, u: f64| -> f64 {
        reset_bed(parent, &positions);
        set_seam_mode(parent, mode);
        set_superficial(parent, [0.0, 0.0, u]);
        // One coupled step from rest → the seam force at v=0 (do not advance state we keep).
        export_and_seam(parent);
        let (fluid, ..) = read_result(parent);
        let w_full = (mass * g) * n as f64;
        // net upward force on the whole bed from rest.
        fluid[2] - w_full
    };
    // Bracket then bisect.
    let (mut lo, mut hi) = (1e-4, 10.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if net_force(&mut parent, mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let u_mf_seam = 0.5 * (lo + hi);
    let rel_umf = (u_mf_seam - u_wy).abs() / u_wy;

    // ── (2) Dynamic sweep: integrate the coupled bed from rest at each U and MEASURE
    // its centre-of-mass acceleration a_z(U); compare to the seam net force / M_bed.
    println!("#     U [m/s]     Re      a_z_meas [m/s^2]   a_z_force [m/s^2]   |Δ|/g     state");
    let mut sweep: Vec<(f64, f64)> = Vec::new(); // (U, a_z_meas)
    let mut worst_handoff = 0.0f64;
    for &u in &flow.superficial {
        // Analytic (net seam force at rest)/M_bed.
        let a_force = net_force(&mut parent, u) / m_bed;
        let (_f, _d, dep0, mom0) = read_result(&parent);
        worst_dep = worst_dep.max(dep0);
        worst_mom = worst_mom.max(mom0);

        // Integrated a_z from rest over settle_steps.
        reset_bed(&mut parent, &positions);
        set_seam_mode(&mut parent, mode);
        set_superficial(&mut parent, [0.0, 0.0, u]);
        for _ in 0..flow.settle_steps {
            parent.run();
        }
        let vz = bed_com_vz(&parent);
        let a_meas = vz / (flow.settle_steps as f64 * flow.dt);
        let handoff = (a_meas - a_force).abs() / g;
        worst_handoff = worst_handoff.max(handoff);
        sweep.push((u, a_meas));
        let re = gas.rho * u * d / gas.mu;
        let state = if a_meas > 0.0 { "FLUIDIZES (lifts)" } else { "packed (settles)" };
        println!(
            "  {u:>10.4}  {re:>6.2}   {a_meas:>15.4}   {a_force:>15.4}   {:>6.4}   {state}",
            handoff
        );
    }

    // Zero-crossing of the measured a_z(U) sweep (linear interpolation between the
    // last negative and first positive point) — the DYNAMIC fluidization onset.
    let u_mf_dyn = zero_crossing(&sweep);
    let dyn_matches = u_mf_dyn
        .map(|u| (u - u_mf_seam).abs() / u_mf_seam < 0.05)
        .unwrap_or(false);
    // Sign structure: strictly increasing through a single crossing (packed below,
    // fluidized above).
    let sign_ok = sweep.first().map(|x| x.1 < 0.0).unwrap_or(false)
        && sweep.last().map(|x| x.1 > 0.0).unwrap_or(false)
        && sweep.windows(2).all(|w| w[1].1 > w[0].1);

    // ── (3) Negative controls (RUN, not asserted on paper): each shifts the seam
    // U_mf far outside the Wen&Yu tolerance.
    let u_mf_nopg = bisect_umf(
        &mut parent,
        &positions,
        SeamMode { omit_pressure_grad: true, ..mode },
        mass,
        g,
        n,
    );
    let u_mf_epsbug = bisect_umf(
        &mut parent,
        &positions,
        SeamMode { corrupt_eps_power: true, ..mode },
        mass,
        g,
        n,
    );
    let err_nopg = (u_mf_nopg - u_wy).abs() / u_wy;
    let err_epsbug = (u_mf_epsbug - u_wy).abs() / u_wy;
    let neg_ok = err_nopg > valid.tol_rel_umf && err_epsbug > valid.tol_rel_umf;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!(
        "# U_mf MEASURED (seam, MacDonald via live net-force bisection): {u_mf_seam:.5} m/s",
    );
    println!(
        "# U_mf Wen&Yu (1966) REFERENCE:                                 {u_wy:.5} m/s   rel.err {:.2}%  (tol {:.1}%)",
        100.0 * rel_umf, 100.0 * valid.tol_rel_umf
    );
    match u_mf_dyn {
        Some(u) => println!(
            "# U_mf DYNAMIC (zero-crossing of integrated a_z sweep):         {u:.5} m/s   ({} seam within 5%)",
            if dyn_matches { "matches" } else { "DISAGREES with" }
        ),
        None => println!("# U_mf DYNAMIC: sweep did not bracket a zero crossing!"),
    }
    println!(
        "# non-tautology: measured rel.err {:.2}% is genuinely > {:.1}% floor (independent MacDonald closure, not self-comparison)",
        100.0 * rel_umf, 100.0 * valid.umf_err_floor
    );
    println!(
        "# negative controls: omit-∇P U_mf {u_mf_nopg:.4} ({:+.1}%)  eps-power-bug U_mf {u_mf_epsbug:.4} ({:+.1}%)  => {} (must exceed tol {:.1}%)",
        100.0 * (u_mf_nopg / u_wy - 1.0),
        100.0 * (u_mf_epsbug / u_wy - 1.0),
        if neg_ok { "both FAIL as required" } else { "a control DID NOT FAIL — gate vacuous!" },
        100.0 * valid.tol_rel_umf
    );
    println!(
        "# two-way handoff: worst |a_z_meas − a_z_force|/g = {:.4}  (tol {:.3})  [integrator delivers seam force to the moving bed]",
        worst_handoff, valid.tol_handoff
    );
    println!(
        "# deposition fidelity: worst |eps_cell − eps_bed|/eps_bed = {:.2}%  (tol {:.1}%)",
        100.0 * worst_dep, 100.0 * valid.tol_deposit_cell
    );
    println!(
        "# momentum conservation err: {worst_mom:.2e}  (tol {:.0e})  [sanity only]",
        valid.tol_momentum
    );

    let pass_umf = rel_umf <= valid.tol_rel_umf;
    let pass_nontrivial = rel_umf > valid.umf_err_floor && neg_ok;
    let pass_dyn = dyn_matches && sign_ok;
    let pass_handoff = worst_handoff <= valid.tol_handoff;
    let pass_regime = eps <= valid.eps_max;
    let pass_dep = worst_dep <= valid.tol_deposit_cell;
    let pass_mom = worst_mom <= valid.tol_momentum;

    if pass_umf && pass_nontrivial && pass_dyn && pass_handoff && pass_regime && pass_dep && pass_mom
    {
        println!(
            "VALIDATION: PASS  (U_mf seam {u_mf_seam:.4} vs Wen&Yu {u_wy:.4}, {:.2}%<={:.1}%; dynamic onset matches & sign-monotone; handoff {:.3}<={:.3}; neg-controls fail at {:+.0}%/{:+.0}%; eps={eps:.3}<={}; dep {:.1}%; mom {worst_mom:.1e})",
            100.0 * rel_umf,
            100.0 * valid.tol_rel_umf,
            worst_handoff,
            valid.tol_handoff,
            100.0 * (u_mf_nopg / u_wy - 1.0),
            100.0 * (u_mf_epsbug / u_wy - 1.0),
            valid.eps_max,
            100.0 * worst_dep,
        );
    } else {
        println!(
            "VALIDATION: FAIL  (umf_ok={pass_umf} nontrivial_ok={pass_nontrivial} dynamic_ok={pass_dyn} handoff_ok={pass_handoff} regime_ok={pass_regime} dep_ok={pass_dep} mom_ok={pass_mom})"
        );
        std::process::exit(1);
    }
}

// ─── Live-seam helpers (outside any system, via the parent's SubApps) ─────────

/// Step the coupled system once from the current (rest) bed state and record the
/// seam force. `BedResult` is written by the FIELD seam in the `TickCfd` phase —
/// i.e. from the bed kinematics exported at the START of the step (rest) — before
/// `TickSoil` advances the particles, so the recorded force is the rest-state force
/// regardless of the trailing integration (which the caller discards via `reset_bed`).
fn export_and_seam(parent: &mut App) {
    parent.run();
}

/// Reset the bed atoms to their initial positions and zero velocity.
fn reset_bed(parent: &App, positions: &[[f64; 3]]) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs.find("soil").unwrap().resource_cell(TypeId::of::<Atom>()).unwrap();
    let mut b = cell.borrow_mut();
    let atoms = b.downcast_mut::<Atom>().unwrap();
    let n = atoms.nlocal as usize;
    for i in 0..n {
        atoms.pos[i] = [positions[i][0], positions[i][1], positions[i][2]];
        atoms.vel[i] = [0.0, 0.0, 0.0];
        atoms.force[i] = [0.0, 0.0, 0.0];
    }
}

fn set_superficial(parent: &App, u: Vec3) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell =
        subs.find("cfd").unwrap().resource_cell(TypeId::of::<Superficial>()).unwrap();
    cell.borrow_mut().downcast_mut::<Superficial>().unwrap().u = u;
}

fn set_seam_mode(parent: &App, mode: SeamMode) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs.find("cfd").unwrap().resource_cell(TypeId::of::<SeamCtx>()).unwrap();
    cell.borrow_mut().downcast_mut::<SeamCtx>().unwrap().mode = mode;
}

/// (Σ total fluid force, Σ drag force, deposition err, momentum err).
fn read_result(parent: &App) -> (Vec3, Vec3, f64, f64) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs.find("cfd").unwrap().resource_cell(TypeId::of::<BedResult>()).unwrap().borrow();
    let r = cell.downcast_ref::<BedResult>().unwrap();
    (r.f_fluid_total, r.f_drag_total, r.eps_cell_err, r.mom_err)
}

/// Bed centre-of-mass vertical velocity (uniform-mass bed → mean v_z).
fn bed_com_vz(parent: &App) -> f64 {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cell = subs.find("soil").unwrap().resource_cell(TypeId::of::<Atom>()).unwrap().borrow();
    let atoms = cell.downcast_ref::<Atom>().unwrap();
    let n = atoms.nlocal as usize;
    let mut sum = 0.0f64;
    for i in 0..n {
        sum += atoms.vel[i][2] as f64;
    }
    sum / n.max(1) as f64
}

/// Bisect the superficial velocity where the seam's net bed force (from rest)
/// crosses zero, for a given (possibly corrupted) seam mode.
fn bisect_umf(
    parent: &mut App,
    positions: &[[f64; 3]],
    mode: SeamMode,
    mass: f64,
    g: f64,
    n: usize,
) -> f64 {
    let w_full = mass * g * n as f64;
    let net = |parent: &mut App, u: f64| -> f64 {
        reset_bed(parent, positions);
        set_seam_mode(parent, mode);
        set_superficial(parent, [0.0, 0.0, u]);
        export_and_seam(parent);
        let (fluid, ..) = read_result(parent);
        fluid[2] - w_full
    };
    let (mut lo, mut hi) = (1e-4, 20.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if net(parent, mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Linear-interpolated zero crossing of a monotone-increasing (U, a_z) sweep.
fn zero_crossing(sweep: &[(f64, f64)]) -> Option<f64> {
    for w in sweep.windows(2) {
        let (u0, a0) = w[0];
        let (u1, a1) = w[1];
        if a0 <= 0.0 && a1 > 0.0 {
            return Some(u0 + (u1 - u0) * (-a0) / (a1 - a0));
        }
    }
    None
}
