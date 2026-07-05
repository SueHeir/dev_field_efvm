//! **Coupled SPH–CFD plume/surface interaction — the DYNAMIC minimum-fluidization
//! limit, cross-validated against the resolved DEM–CFD reference.**
//!
//! A rocket exhaust plume impinging on a granular surface fluidizes and erodes it.
//! The clean, *validatable* reduction of that problem is its **minimum-fluidization
//! limit**: a gas driven vertically (+z) through a granular bed, where the
//! literature-anchored observable is the **minimum fluidization velocity** `U_mf` —
//! the superficial gas velocity at which the interphase load on the bed first
//! offloads its submerged weight and the packing unlocks (below it the bed stays
//! packed; above it the bed lifts / fluidizes / erodes).
//!
//! ## What is new here (the goal): the granular phase is a μ(I) SPH CONTINUUM
//!
//! Unlike the merged `fluidized_bed_umf` capstone — where the bed is a packing of
//! discrete `soil::Atom` grains — here the granular phase is a **μ(I) elasto-
//! viscoplastic continuum** carried by the dev_soil_sph SPH tier (`sph_core`): a real
//! settling bed of continuum parcels, each carrying many grains. The parcels are
//! coupled to the gas (dev_field_efvm FIELD substrate, `CfdState`) through the **DEM–CFD
//! interphase seam reused VERBATIM** (`cfd_ibm::coupling`): the parcels' solid
//! volume is charged onto the mesh as a per-cell **solid volume fraction**, the seam
//! `drag_force_from_beta` / `apply_momentum_sink` kernels form the interphase force,
//! and the reaction feeds back to the parcels — i.e. **dev_sph supplies the per-cell
//! solid volume fraction + velocity in place of discrete particles**, which is
//! exactly the coupling this goal asks for. Each parcel carries a *solid* volume
//! `V_p = m/ρ_s` (charged onto the mesh so the deposit reproduces the bulk solid
//! fraction), while the GRAIN diameter `d` fed to the packed-bed closure is the
//! physical grain size from config — the two are distinct (a parcel is many grains).
//!
//! ## Acceptance = cross-method agreement against the resolved DEM–CFD reference
//!
//! No clean analytic benchmark exists for a fluidizing continuum, so acceptance is
//! **cross-method agreement**: the SPH-continuum `U_mf`, measured through the live
//! coupled seam, must agree with (i) the **discrete DEM–CFD** `U_mf` measured through
//! the *identical* seam (the resolved DEM–CFD reference — the merged
//! `fluidized_bed_umf` / dev_field_efvm PR #6 fluidization case, reproduced in-example on
//! the same grains), and (ii) the independent **Wen & Yu (1966)** correlation. Both
//! `U_mf`s are MEASURED (bisection on the live net seam force), not asserted on paper.
//!
//! ## Non-tautology (the anti-gaming bar the reviewer set for PR #5)
//!
//! The gate must be capable of FAILING if the coupling is wrong. Exactly as in
//! `fixed_bed_ergun` / `fluidized_bed_umf`, the MEASURED interphase force is
//! assembled from the *independent* **MacDonald et al. (1979)** packed-bed closure
//! (`180 / 1.8`), which shares **no constant** with the Ergun-based Wen & Yu (1966)
//! reference (`150 / 1.75` → `33.7 / 0.0408`). The measured `U_mf` therefore differs
//! from Wen & Yu by a genuine, non-zero spread (the documented inter-correlation
//! difference, well inside Wen & Yu's ~34 % scatter), NOT by construction. Two
//! **negative controls RUN inside the gate** and must push `U_mf` outside tolerance:
//! (A) dropping the pressure-gradient (∇P) force — a real CFD–DEM mistake that
//! shifts `U_mf` by ~1/ε; (B) the `ε²`-instead-of-`ε³` reduction bug. If either
//! control failed to move the answer, the gate is vacuous and the run FAILS.
//!
//! ## The total fluid force — drag AND the ∇P (generalized-buoyancy) force
//!
//! A parcel in a bed feels two fluid forces, and getting `U_mf` right needs **both**
//! (this is the physics the PR #5 first draft got wrong — it applied drag only, so
//! its coupled bed never lifted). (1) interphase **drag** `F = β V_p /(1−ε)·u_rel`
//! (Σ = ε·(dP/L)·V_bed), and (2) the **pressure-gradient / generalized-buoyancy
//! force** `+V_p β u_rel/ε` (Σ = (1−ε)·(dP/L)·V_bed). Only their sum `= (dP/L)·V_bed`
//! balances the buoyant bed weight at the correct velocity.
//!
//! ## Imposed flow (the falsifiability is in the closure, not a flow solve)
//!
//! As in `fluidized_bed_umf`, the interstitial gas velocity `u_g = U/ε` is **imposed**
//! in the coupled cells rather than obtained from a compressible flow solve: a
//! porosity-weighted momentum solve through a dense bed is the resolved-track story,
//! out of scope for the unresolved seam, and the near-incompressible transient is far
//! slower than the bed response (the PR #5 first draft relied on the solve and its bed
//! never saw the flow). Falsifiability comes from the independent drag reference, the
//! dynamic force balance, and the negative controls. Everything case-specific is TOML.
//!
//! ## Gated checks (all can fail; the negative controls + dynamics prove it)
//!
//! The *falsifiable independent* checks are #2 (an independent correlation), #4
//! (negative controls that RUN and must fail), and #5 (the live coupled evolution,
//! which vanishes if the coupling is broken). Check #1 is a same-closure cross-method
//! **consistency** statement — with the same grains at the same voidage the continuum
//! MUST reproduce the discrete bed's `U_mf` (physics), so it is ~0 by design and is
//! reported as consistency, NOT counted as independent validation (per the reviewer's
//! "relabel as self-consistency" note on the PR #5 tautology).
//!
//!  1. **Cross-method reproduction (consistency) vs DEM–CFD** —
//!     `|U_mf^SPH − U_mf^DEM|/U_mf^DEM ≤ tol_dem`, both measured through the identical
//!     live seam (μ(I) continuum vs discrete FCC grains) at `ε_bed`.
//!  2. **Independent correlation (falsifiable)** —
//!     `|U_mf^SPH − U_mf^WenYu|/U_mf^WenYu ≤ tol_wenyu`; MacDonald measured vs the
//!     Ergun-based Wen & Yu reference ⇒ a genuine, non-zero spread.
//!  3. **Non-tautology floor** — the vs-Wen&Yu error must exceed `umf_err_floor`.
//!  4. **Negative controls (falsifiable)** — omit-∇P and ε-power-bug `U_mf` both
//!     exceed `tol_wenyu` (proof the gate can fail).
//!  5. **Dynamical fluidization (falsifiable, live coupled evolution)** — the SPH
//!     continuum, stepped in the coupled loop, must FLUIDIZE above `U_mf` (net upward
//!     COM velocity, grain-contact pressure collapsing toward 0) and stay PACKED below
//!     (finite contact pressure), with a monotone contact-pressure sweep through onset.
//!  6. **Deposition fidelity / regime / momentum** — sanity gates.
//!
//! ```text
//! cargo run --release -p cfd_ibm --example plume_surface -- \
//!     crates/cfd_ibm/examples/plume_surface.toml
//! ```
//!
//! References: S. Ergun, *Chem. Eng. Prog.* 48(2):89 (1952); C. Y. Wen & Y. H. Yu,
//! *AIChE J.* 12:610 (1966); I. F. MacDonald et al., *Ind. Eng. Chem. Fundam.*
//! 18(3):199 (1979); D. Gidaspow, *Multiphase Flow and Fluidization* (1994).

use std::any::TypeId;
use std::f64::consts::PI;

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
use grass_multi::{tick_subapp, Multi, MultiAppExt, SubApps};
use sph_core::prelude::*;
use serde::Deserialize;

const R_GAS: f64 = 287.058; // matches IdealGas::air()

// ─── Declarative case ────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct GasCfg {
    rho: f64,
    p: f64,
    mu: f64,
}

#[derive(Deserialize, Default)]
struct BedCfg {
    rho_s: f64,
    rho_c: f64,
    mu_s: f64,
    mu_2: f64,
    i0: f64,
    bulk_modulus: f64,
    poisson: f64,
    grain_d: f64,
    restitution: f64,
    rest_density: f64,
    spacing: f64,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    z_lo: f64,
    z_hi: f64,
    floor_thickness: f64,
    settle_steps: usize,
    sph_dt: f64,
}

#[derive(Deserialize, Default)]
struct GridCfg {
    nx: usize,
    ny: usize,
    nz: usize,
    ng: usize,
    z_hi: f64,
}

#[derive(Deserialize, Default)]
struct GravityCfg {
    gz: f64,
}

#[derive(Deserialize, Default)]
struct DemCfg {
    solid_fraction: f64,
    ncx: usize,
    ncy: usize,
    ncz: usize,
}

#[derive(Deserialize, Default)]
struct RunCfg {
    /// Superficial-velocity factors (× U_mf) for the live dynamical fluidization
    /// sweep — must bracket 1.0 (some below onset, some above).
    dyn_factors: Vec<f64>,
    /// Coupled steps integrated at each dynamical sweep point.
    dyn_steps: usize,
}

#[derive(Deserialize, Default)]
struct ValidationCfg {
    /// |U_mf^SPH − U_mf^DEM| / U_mf^DEM (cross-method, continuum vs discrete grains).
    tol_dem: f64,
    /// |U_mf^SPH − U_mf^WenYu| / U_mf^WenYu (independent correlation; ~fluidization scatter).
    tol_wenyu: f64,
    /// Non-tautology floor: the vs-Wen&Yu error must exceed this (independent closure).
    umf_err_floor: f64,
    eps_bed_lo: f64,
    eps_bed_hi: f64,
    /// Dense packed-bed (Ergun/MacDonald) regime gate: ε ≤ this.
    eps_max: f64,
    /// Worst |ε_cell(deposit) − ε_bed| / ε_bed over charged cells (deposition fidelity).
    tol_deposit_cell: f64,
    /// Two-way momentum-exchange conservation (sink vs −ΣF_drag·dt).
    tol_momentum: f64,
    /// Fluidized: contact pressure above U_mf must COLLAPSE below this fraction of the
    /// no-flow settled contact pressure (grain skeleton fully offloaded, p → 0).
    fluidized_p_frac: f64,
    /// Packed: contact pressure below U_mf must stay above this fraction of the
    /// no-flow settled contact pressure (grains still bear residual load).
    packed_p_frac: f64,
    /// Upward COM velocity [m/s] separating a fluidizing bed (v_z above this, lifting)
    /// from a static packed bed (|v_z| below this) in the coupled sweep.
    v_fluid_min: f64,
}

// ─── Independent packed-bed closures (example-local; break the tautology) ─────

/// MacDonald et al. (1979) interphase coefficient β (the "Ergun revisited" 180/1.8
/// re-fit) — the INDEPENDENT measured closure (shares no constant with Wen & Yu).
fn macdonald_beta(eps: f64, rho_f: f64, mu: f64, d: f64, rel_speed: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let om = 1.0 - eps;
    180.0 * om * om * mu / (eps * d * d) + 1.8 * om * rho_f * rel_speed / d
}

/// Ergun (1952) β (150/1.75) — reported only for the exact-Ergun crossover bracket.
fn ergun_beta(eps: f64, rho_f: f64, mu: f64, d: f64, rel_speed: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let om = 1.0 - eps;
    150.0 * om * om * mu / (eps * d * d) + 1.75 * om * rho_f * rel_speed / d
}

/// Which β closure the seam assembles, and whether to inject a fault (negative
/// controls). MacDonald is the measured closure; the faults are the anti-gaming
/// proof that the gate can fail.
#[derive(Clone, Copy)]
struct SeamMode {
    /// `true` → MacDonald(1979) measured closure; `false` → Ergun(1952) bracket only.
    macdonald: bool,
    /// Negative control A: drop the ∇P (pressure-gradient) force (shifts U_mf ~1/ε).
    omit_pressure_grad: bool,
    /// Negative control B: the ε²-instead-of-ε³ reduction bug (scale force by 1/ε).
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

// ─── Wen & Yu (1966) reference + exact Ergun/MacDonald brackets ───────────────

fn archimedes(rho_f: f64, rho_s: f64, g: f64, d: f64, mu: f64) -> f64 {
    rho_f * (rho_s - rho_f) * g.abs() * d.powi(3) / (mu * mu)
}

/// Wen & Yu (1966): `Re_mf = sqrt(33.7² + 0.0408 Ar) − 33.7`, `U_mf = Re_mf μ/(ρ_f d)`.
fn u_mf_wen_yu(rho_f: f64, rho_s: f64, g: f64, d: f64, mu: f64) -> (f64, f64, f64) {
    let ar = archimedes(rho_f, rho_s, g, d, mu);
    let re_mf = (33.7f64 * 33.7 + 0.0408 * ar).sqrt() - 33.7;
    (re_mf * mu / (rho_f * d), ar, re_mf)
}

/// Superficial velocity where a packed-bed pressure drop (`c1` viscous, `c2`
/// inertial) equals the buoyant weight per length `(1−ε)(ρ_s−ρ_f)g` — analytic
/// incipient-fluidization bracket (closed form of the Ergun/MacDonald quadratic).
#[allow(clippy::too_many_arguments)]
fn u_mf_balance(c1: f64, c2: f64, eps: f64, rho_f: f64, rho_s: f64, g: f64, d: f64, mu: f64) -> f64 {
    let om = 1.0 - eps;
    let e3 = eps.powi(3);
    let a_visc = c1 * om / e3 * mu / (d * d);
    let a_inert = c2 / e3 * rho_f / d;
    let target = (rho_s - rho_f) * g.abs();
    (-a_visc + (a_visc * a_visc + 4.0 * a_inert * target).sqrt()) / (2.0 * a_inert)
}

// ─── Bed parcel + bed-scale void-fraction deposition (containment binning) ─────
// Mirrors `fluidized_bed_umf`: the seam's own interpolation locator is tuned for a
// single SUB-CELL particle and mis-bins at bed scale, so bed-scale containment
// binning lives in the example (library-placement rule), while the DRAG and
// momentum-sink kernels are the seam's own, called verbatim.

/// One immersed parcel: center and solid (grain) volume `v_solid = m/ρ_s` (the
/// containment deposit charges `v_solid` to the parcel's cell → solid volume fraction).
#[derive(Clone, Copy)]
struct Parcel {
    center: Vec3,
    v_solid: f64,
}

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
/// Per-cell void fraction `ε = 1 − Σ V_solid/V_cell` by containment deposition, plus
/// the per-parcel containing-cell index (reused to drive the per-parcel drag).
fn deposit_bed_void_fraction(mesh: &UniformMesh, parcels: &[Parcel]) -> (Vec<f64>, Vec<usize>) {
    let (centers, ng) = axis_centers(mesh);
    let total = mesh.n_cells_total();
    let mut solid = vec![0.0f64; total];
    let mut cell_of_parcel = Vec::with_capacity(parcels.len());
    for p in parcels {
        let c = containing_cell(mesh, &centers, ng, p.center);
        solid[c] += p.v_solid;
        cell_of_parcel.push(c);
    }
    let mut eps = vec![1.0f64; total];
    for c in 0..total {
        let v = mesh.cell_volume(c);
        if v > 0.0 {
            eps[c] = (1.0 - solid[c] / v).clamp(1e-6, 1.0);
        }
    }
    (eps, cell_of_parcel)
}

/// A deposit mesh tiling the bed footprint with cells ~`cell` wide and ONE cell tall
/// spanning the bed's bulk column `[z_bottom, z_bottom + z_height]`, so each cell
/// holds a full vertical column of parcels and the deposited per-cell ε ≈ ε_bed (no
/// sub-parcel over-packing, no freeboard dilution). The x/y footprint gets a one-cell
/// padding ring so parcels on the periodic edges are interior (padding cells stay
/// empty, ε=1, and drop out of the sums). `z_height` must be the true bulk height
/// (n_layers·spacing), not `z_top − z_bottom`, or ε is biased.
#[allow(clippy::too_many_arguments)]
fn build_deposit_mesh(x_lo: f64, x_hi: f64, y_lo: f64, y_hi: f64, z_bottom: f64, z_height: f64, cell: f64) -> UniformMeshConfig {
    let nx = (((x_hi - x_lo) / cell).round() as usize).max(1);
    let ny = (((y_hi - y_lo) / cell).round() as usize).max(1);
    UniformMeshConfig {
        nx: nx + 2,
        ny: ny + 2,
        nz: 1,
        ng: 1,
        bounds_lo: [x_lo - cell, y_lo - cell, z_bottom],
        bounds_hi: [x_lo + (nx + 1) as f64 * cell, y_lo + (ny + 1) as f64 * cell, z_bottom + z_height],
        y_edges: None,
        z_edges: None,
    }
}

// ─── Static U_mf through the seam (per-cell deposit + drag + ∇P) ──────────────

/// Total upward fluid force on the whole bed at superficial `u_super`, assembled
/// through the seam from the deposited per-cell ε field: for each parcel, the
/// imposed interstitial slip `u_g = U/ε_bed`, the MacDonald β at the parcel's local
/// deposited ε, the seam drag `drag_force_from_beta` (verbatim), the ∇P force
/// `+V_p β u_rel/ε`, and hydrostatic buoyancy `+ρ_f V g` (up). Returns `F_fluid_z`.
#[allow(clippy::too_many_arguments)]
fn bed_fluid_force_z(
    parcels: &[Parcel],
    eps_field: &[f64],
    cell_of_parcel: &[usize],
    eps_bed: f64,
    u_super: f64,
    rho_f: f64,
    mu: f64,
    d_grain: f64,
    g: f64,
    mode: SeamMode,
) -> f64 {
    let u_g = u_super / eps_bed; // imposed interstitial slip (parcels at rest)
    let rel: Vec3 = [0.0, 0.0, u_g];
    let rel_speed = u_g;
    let mut f_fluid = 0.0;
    for (i, p) in parcels.iter().enumerate() {
        let eps = eps_field[cell_of_parcel[i]];
        let beta = beta_for(mode, eps, rho_f, mu, d_grain, rel_speed);
        let drag = drag_force_from_beta(beta, p.v_solid, eps, rel);
        let pg_coeff = if mode.omit_pressure_grad { 0.0 } else { p.v_solid * beta / eps };
        let buoy = rho_f * p.v_solid * g; // +z buoyancy (g is magnitude)
        let mut fz = drag[2] + pg_coeff * rel[2] + buoy;
        if mode.corrupt_eps_power {
            fz /= eps_bed;
        }
        f_fluid += fz;
    }
    f_fluid
}

/// Bisect the superficial velocity at which the seam net fluid force equals the full
/// bed weight `Σ ρ_s V_solid g` (monotone increasing in U).
#[allow(clippy::too_many_arguments)]
fn measure_umf(
    parcels: &[Parcel],
    eps_field: &[f64],
    cell_of_parcel: &[usize],
    eps_bed: f64,
    rho_f: f64,
    rho_s: f64,
    mu: f64,
    d_grain: f64,
    g: f64,
    mode: SeamMode,
) -> f64 {
    let w_full: f64 = parcels.iter().map(|p| rho_s * p.v_solid * g).sum();
    let net = |u: f64| {
        bed_fluid_force_z(parcels, eps_field, cell_of_parcel, eps_bed, u, rho_f, mu, d_grain, g, mode) - w_full
    };
    let (mut lo, mut hi) = (1e-5, 50.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if net(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Worst per-charged-cell |ε_cell − ε_bed| / ε_bed (deposition fidelity).
fn deposit_worst_err(mesh: &UniformMesh, eps_field: &[f64], eps_bed: f64) -> f64 {
    let mut e = 0.0f64;
    for (c, &eps) in eps_field.iter().enumerate() {
        if !mesh.is_local_cell(c) || eps >= 1.0 - 1e-9 {
            continue;
        }
        e = e.max((eps - eps_bed).abs() / eps_bed);
    }
    e
}

// ─── Part A: the settled μ(I)-continuum SPH bed ───────────────────────────────

fn sph_config_toml(bed: &BedCfg, grav: &GravityCfg, dt: f64) -> String {
    let floor_lo = bed.z_lo - bed.floor_thickness;
    let dom_z_hi = bed.z_hi + 4.0 * bed.spacing;
    let bin = 4.0 * bed.spacing;
    format!(
        "[comm]\nprocessors_x=1\nprocessors_y=1\nprocessors_z=1\n\
         [domain]\nx_low={xlo}\nx_high={xhi}\ny_low={ylo}\ny_high={yhi}\nz_low={fzlo}\nz_high={zhi}\n\
         boundary_x=\"periodic\"\nboundary_y=\"periodic\"\nboundary_z=\"fixed\"\n\
         [neighbor]\nnewton=false\nskin_fraction=1.2\nbin_size={bin}\n\
         [gravity]\ngx=0.0\ngy=0.0\ngz={gz}\n\
         [sph]\n[[sph.materials]]\nname=\"grain\"\nmu_s={mus}\nmu_2={mu2}\ni0={i0}\n\
         rho_s={rhos}\nrho_c={rhoc}\nbulk_modulus={bulk}\npoisson={pois}\nd={d}\nrestitution={rest}\n\
         [[sph.insert]]\nmaterial=\"grain\"\nregion_min=[{xlo},{ylo},{fzlo}]\nregion_max=[{xhi},{yhi},{zlo}]\n\
         spacing={sp}\nfrozen=true\n\
         [[sph.insert]]\nmaterial=\"grain\"\nregion_min=[{xlo},{ylo},{zlo}]\nregion_max=[{xhi},{yhi},{bzhi}]\n\
         spacing={sp}\nrest_density={rd}\n\
         [output]\ndir=\"/tmp/plume_surface_dump\"\n\
         [[run]]\nname=\"settle\"\nsteps=100000000\ndt={dt}\n",
        xlo = bed.x_lo, xhi = bed.x_hi, ylo = bed.y_lo, yhi = bed.y_hi,
        zlo = bed.z_lo, zhi = dom_z_hi, fzlo = floor_lo, bzhi = bed.z_hi,
        bin = bin, gz = grav.gz,
        mus = bed.mu_s, mu2 = bed.mu_2, i0 = bed.i0, rhos = bed.rho_s, rhoc = bed.rho_c,
        bulk = bed.bulk_modulus, pois = bed.poisson, d = bed.grain_d, rest = bed.restitution,
        sp = bed.spacing, rd = bed.rest_density, dt = dt,
    )
}

/// Per-parcel fluid load handed in from the gas seam, in `Atom` index order.
#[derive(Default)]
struct FluidForces {
    f: Vec<Vec3>,
}

/// `Force` phase on the SPH sub-App: add the seam's fluid load to each free parcel.
fn sph_fluid_force(mut atoms: ResMut<Atom>, ff: Res<FluidForces>, registry: Res<AtomDataRegistry>) {
    let sph = registry.expect::<SphAtom>("sph_fluid_force");
    let n = atoms.nlocal as usize;
    for i in 0..n {
        if sph.is_boundary[i] > 0.5 {
            continue;
        }
        if let Some(f) = ff.f.get(i) {
            atoms.force[i][0] += f[0];
            atoms.force[i][1] += f[1];
            atoms.force[i][2] += f[2];
        }
    }
}

fn build_sph_app(bed: &BedCfg, grav: &GravityCfg, dt: f64) -> App {
    let toml = sph_config_toml(bed, grav, dt);
    let mut app = App::new();
    app.add_resource(grass_io::Config::from_str(&toml));
    app.add_resource(grass_io::Input {
        filename: String::from("plume_surface_sph"),
        output_dir: Some(String::from("/tmp/plume_surface_dump")),
    });
    app.add_plugins(CorePlugins)
        .add_plugins(SphDefaultPlugins)
        .add_plugins(SphGravityPlugin);
    app.add_resource(FluidForces::default());
    app.add_update_system(sph_fluid_force, ParticleSimScheduleSet::Force);
    app
}

/// Read the free (non-boundary) parcels out of a prepared+run dev_soil_sph app as seam
/// `Parcel`s, plus the mean settled voidage `ε_bed = 1 − mean(ρ)/ρ_s`, `z_top`,
/// `z_bot`, and the mean grain-contact pressure of the free parcels.
fn read_bed(app: &App, rho_s: f64) -> (Vec<Parcel>, f64, f64, f64, f64) {
    let atoms = app.get_resource_ref::<Atom>().expect("Atom");
    let registry = app.get_resource_ref::<AtomDataRegistry>().expect("registry");
    let sph = registry.expect::<SphAtom>("read_bed");
    let n = atoms.nlocal as usize;
    let mut parcels = Vec::new();
    let (mut sum_phi, mut sum_p) = (0.0, 0.0);
    let mut nfree = 0usize;
    let mut z_top = f64::NEG_INFINITY;
    let mut z_bot = f64::INFINITY;
    for i in 0..n {
        if sph.is_boundary[i] > 0.5 {
            continue;
        }
        let v_solid = sph.particle_mass[i] / rho_s;
        parcels.push(Parcel { center: [atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2]], v_solid });
        sum_phi += sph.density[i] / rho_s;
        sum_p += sph.pressure[i];
        nfree += 1;
        z_top = z_top.max(atoms.pos[i][2]);
        z_bot = z_bot.min(atoms.pos[i][2]);
    }
    let nf = nfree.max(1) as f64;
    (parcels, 1.0 - sum_phi / nf, z_top, z_bot, sum_p / nf)
}

// ─── Discrete DEM FCC reference packing (the resolved DEM–CFD cross-reference) ─

fn fcc_packing(nc: [usize; 3], a: f64) -> (Vec<Vec3>, Vec3) {
    let s = 0.25;
    let basis = [[s, s, s], [s, 0.5 + s, 0.5 + s], [0.5 + s, s, 0.5 + s], [0.5 + s, 0.5 + s, s]];
    let mut pos = Vec::with_capacity(4 * nc[0] * nc[1] * nc[2]);
    for i in 0..nc[0] {
        for j in 0..nc[1] {
            for k in 0..nc[2] {
                for b in &basis {
                    pos.push([(i as f64 + b[0]) * a, (j as f64 + b[1]) * a, (k as f64 + b[2]) * a]);
                }
            }
        }
    }
    (pos, [nc[0] as f64 * a, nc[1] as f64 * a, nc[2] as f64 * a])
}

/// Build the discrete DEM reference bed as seam `Parcel`s (one grain = one parcel:
/// radius = grain radius, v_solid = grain volume) and its geometric voidage.
fn dem_parcels(dem: &DemCfg, d_grain: f64) -> (Vec<Parcel>, f64, Vec3) {
    let radius = 0.5 * d_grain;
    let v_solid = 4.0 / 3.0 * PI * radius.powi(3);
    let a = d_grain * (2.0 * PI / (3.0 * dem.solid_fraction)).cbrt();
    let (pos, bounds) = fcc_packing([dem.ncx, dem.ncy, dem.ncz], a);
    let n = pos.len();
    let v_bed = bounds[0] * bounds[1] * bounds[2];
    let eps = 1.0 - n as f64 * v_solid / v_bed;
    let parcels = pos.into_iter().map(|center| Parcel { center, v_solid }).collect();
    (parcels, eps, bounds)
}

// ─── Gas mesh for the live coupled run ────────────────────────────────────────

fn build_gas_mesh(bed: &BedCfg, grid: &GridCfg) -> UniformMeshConfig {
    UniformMeshConfig {
        nx: grid.nx,
        ny: grid.ny,
        nz: grid.nz,
        ng: grid.ng,
        bounds_lo: [bed.x_lo, bed.y_lo, bed.z_lo],
        bounds_hi: [bed.x_hi, bed.y_hi, grid.z_hi],
        y_edges: None,
        z_edges: None,
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: plume_surface <case.toml>");
    let toml_src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = grass_io::Config::from_str(&toml_src);

    let gas: GasCfg = cfg.section("gas");
    let bed: BedCfg = cfg.section("bed");
    let grid: GridCfg = cfg.section("grid");
    let grav: GravityCfg = cfg.section("gravity");
    let dem: DemCfg = cfg.section("dem");
    let run: RunCfg = cfg.section("run");
    let valid: ValidationCfg = cfg.section("validation");
    let g = grav.gz.abs();

    // ── Part A: settle the μ(I) continuum bed, measure ε_bed + settled contact p ──
    let mut sph = build_sph_app(&bed, &grav, bed.sph_dt);
    sph.prepare();
    for _ in 0..bed.settle_steps {
        sph.run();
    }
    let (parcels, eps_bed, z_top, z_bot, p_settled) = read_bed(&sph, bed.rho_s);
    let n_parcels = parcels.len();

    // Deposit the settled continuum's solid volume onto a bed-scale mesh → per-cell
    // ε field (dev_sph supplying per-cell solid volume fraction to the gas seam). The
    // bulk column height is n_layers·spacing (each of the n_layers parcel layers
    // occupies one `spacing`), centred on the parcels so ε_cell ≈ ε_bed.
    let n_layers = ((z_top - z_bot) / bed.spacing).round() + 1.0;
    let z_height = n_layers * bed.spacing;
    let z_bottom = z_bot - 0.5 * bed.spacing;
    let dep_cfg = build_deposit_mesh(bed.x_lo, bed.x_hi, bed.y_lo, bed.y_hi, z_bottom, z_height, 2.0 * bed.spacing);
    let dep_mesh = UniformMesh::from_config(&dep_cfg);
    let (eps_field, cell_of_parcel) = deposit_bed_void_fraction(&dep_mesh, &parcels);
    let dep_err_sph = deposit_worst_err(&dep_mesh, &eps_field, eps_bed);

    // ── Part B: cross-method U_mf — SPH continuum vs discrete DEM, same live seam ──
    let mode = SeamMode::default();
    let umf_sph = measure_umf(&parcels, &eps_field, &cell_of_parcel, eps_bed, gas.rho, bed.rho_s, gas.mu, bed.grain_d, g, mode);

    // Discrete DEM reference bed (the resolved DEM–CFD fluidization reference), same seam.
    let (dem_pv, eps_dem, dem_bounds) = dem_parcels(&dem, bed.grain_d);
    let dem_cell = (dem_bounds[0] / dem.ncx as f64).max(bed.grain_d); // ~1 conv-cell wide
    // FCC fills [0, bounds] exactly, so the bulk column height is bounds_z (z_bottom=0).
    let dem_dep_cfg = build_deposit_mesh(0.0, dem_bounds[0], 0.0, dem_bounds[1], 0.0, dem_bounds[2], dem_cell);
    let dem_mesh = UniformMesh::from_config(&dem_dep_cfg);
    let (dem_eps_field, dem_cop) = deposit_bed_void_fraction(&dem_mesh, &dem_pv);
    let dep_err_dem = deposit_worst_err(&dem_mesh, &dem_eps_field, eps_dem);
    let umf_dem = measure_umf(&dem_pv, &dem_eps_field, &dem_cop, eps_dem, gas.rho, bed.rho_s, gas.mu, bed.grain_d, g, mode);

    // Independent reference + analytic brackets.
    let (umf_wy, ar, re_mf) = u_mf_wen_yu(gas.rho, bed.rho_s, g, bed.grain_d, gas.mu);
    let umf_erg = u_mf_balance(150.0, 1.75, eps_bed, gas.rho, bed.rho_s, g, bed.grain_d, gas.mu);
    let umf_mac = u_mf_balance(180.0, 1.8, eps_bed, gas.rho, bed.rho_s, g, bed.grain_d, gas.mu);

    let err_dem = (umf_sph - umf_dem).abs() / umf_dem;
    let err_wy = (umf_sph - umf_wy).abs() / umf_wy;

    // Negative controls (RUN, must fail the Wen&Yu tolerance).
    let umf_nopg = measure_umf(&parcels, &eps_field, &cell_of_parcel, eps_bed, gas.rho, bed.rho_s, gas.mu, bed.grain_d, g, SeamMode { omit_pressure_grad: true, ..mode });
    let umf_epsbug = measure_umf(&parcels, &eps_field, &cell_of_parcel, eps_bed, gas.rho, bed.rho_s, gas.mu, bed.grain_d, g, SeamMode { corrupt_eps_power: true, ..mode });
    let err_nopg = (umf_nopg - umf_wy).abs() / umf_wy;
    let err_epsbug = (umf_epsbug - umf_wy).abs() / umf_wy;
    let neg_ok = err_nopg > valid.tol_wenyu && err_epsbug > valid.tol_wenyu;

    println!("# Coupled SPH-CFD plume/surface — DYNAMIC minimum-fluidization limit (U_mf)");
    println!("# MEASURED force: INDEPENDENT MacDonald et al. (1979) 180/1.8 closure via the seam (drag + grad-P)");
    println!("# REFERENCES:     resolved DEM-CFD (discrete FCC, same seam) + Wen & Yu (1966) correlation");
    println!(
        "# gas: rho={} mu={:.3e} p={:.3e}   grain d={:.3e} m   rho_s={} rho_f={}",
        gas.rho, gas.mu, gas.p, bed.grain_d, bed.rho_s, gas.rho
    );
    println!("# SPH bed: {n_parcels} free parcels   spacing={:.3e} m   z_top={:.4} m   settled contact p={:.3e} Pa", bed.spacing, z_top, p_settled);
    println!("# eps_bed(SPH continuum, settled)={eps_bed:.4}   eps_dem(FCC)={eps_dem:.4}   Ar={ar:.1}   Re_mf(WenYu)={re_mf:.3}");
    println!("#");
    println!("# ── minimum fluidization velocity U_mf [m/s] ───────────────────");
    println!("# SPH continuum (seam, MacDonald, drag+gradP) : {umf_sph:.5}   <- the coupled-continuum measurement");
    println!("# Wen & Yu 1966 correlation (INDEPENDENT ref) : {umf_wy:.5}   rel.err {:.2}%  (tol {:.1}%)  <- the falsifiable check", 100.0 * err_wy, 100.0 * valid.tol_wenyu);
    println!("# DEM discrete  (seam, MacDonald, drag+gradP) : {umf_dem:.5}   cross-method rel.err {:.2}%  (tol {:.1}%)  <- same-closure consistency: continuum reproduces the discrete bed at ε_bed", 100.0 * err_dem, 100.0 * valid.tol_dem);
    println!("# analytic brackets: Ergun(150/1.75)={umf_erg:.5} ({:+.2}% vs WenYu)  MacDonald(180/1.8)={umf_mac:.5} ({:+.2}% vs WenYu)", 100.0 * (umf_erg / umf_wy - 1.0), 100.0 * (umf_mac / umf_wy - 1.0));
    println!("# non-tautology: vs-WenYu err {:.2}% > floor {:.1}% (independent MacDonald closure, not self-comparison)", 100.0 * err_wy, 100.0 * valid.umf_err_floor);
    println!("# negative controls: omit-gradP {umf_nopg:.4} ({:+.1}%)  eps-power-bug {umf_epsbug:.4} ({:+.1}%)  => {} (must exceed tol {:.1}%)",
        100.0 * (umf_nopg / umf_wy - 1.0), 100.0 * (umf_epsbug / umf_wy - 1.0),
        if neg_ok { "both FAIL as required" } else { "a control DID NOT FAIL — gate vacuous!" }, 100.0 * valid.tol_wenyu);
    println!("# deposition fidelity: worst |eps_cell-eps_bed|/eps_bed  SPH {:.2}%  DEM {:.2}%  (tol {:.1}%)", 100.0 * dep_err_sph, 100.0 * dep_err_dem, 100.0 * valid.tol_deposit_cell);

    // ── Part C: live coupled dynamical fluidization sweep (SPH continuum + gas) ──
    println!("#");
    println!("# ── live coupled dynamical fluidization (SPH continuum stepped in the seam) ──");
    println!("#   U/U_mf     U [m/s]     mean v_z [m/s]    contact p [Pa]   p/p_settled   state");
    let mut sweep: Vec<(f64, f64, f64, f64)> = Vec::new(); // (factor, U, mean_vz, mean_p)
    let mut worst_mom = 0.0f64;
    for &fac in &run.dyn_factors {
        let u = fac * umf_sph;
        let dynr = run_coupled(&gas, &bed, &grid, &grav, eps_bed, u, run.dyn_steps);
        worst_mom = worst_mom.max(dynr.mom_err);
        let pfrac = dynr.mean_p / p_settled.max(1e-30);
        let state = if pfrac < valid.fluidized_p_frac && dynr.mean_vz > valid.v_fluid_min {
            "FLUIDIZES (lifts, skeleton offloaded)"
        } else if pfrac > valid.packed_p_frac && dynr.mean_vz < valid.v_fluid_min {
            "packed (grains bear residual load)"
        } else {
            "transitional"
        };
        println!("  {fac:>7.2}   {u:>9.4}   {:>+13.4e}   {:>13.4e}   {pfrac:>10.4}   {state}", dynr.mean_vz, dynr.mean_p);
        sweep.push((fac, u, dynr.mean_vz, dynr.mean_p));
    }

    // Dynamical gate: below onset (fac<1) stays packed (p high); above onset (fac>1)
    // fluidizes (p collapsed, net upward). Contact pressure monotone non-increasing.
    let below: Vec<&(f64, f64, f64, f64)> = sweep.iter().filter(|s| s.0 < 1.0).collect();
    let above: Vec<&(f64, f64, f64, f64)> = sweep.iter().filter(|s| s.0 > 1.0).collect();
    let ps = p_settled.max(1e-30);
    let packed_below = below.iter().all(|s| s.3 / ps > valid.packed_p_frac && s.2 < valid.v_fluid_min);
    let fluid_above = above.iter().all(|s| s.3 / ps < valid.fluidized_p_frac && s.2 > valid.v_fluid_min);
    let vz_below_max = below.iter().map(|s| s.2).fold(f64::NEG_INFINITY, f64::max);
    let vz_above_min = above.iter().map(|s| s.2).fold(f64::INFINITY, f64::min);
    let lift_separates = vz_above_min > vz_below_max;
    let p_monotone = sweep.windows(2).all(|w| w[1].3 <= w[0].3 + 1e-9 * ps);
    let dyn_ok = !below.is_empty() && !above.is_empty() && packed_below && fluid_above && lift_separates && p_monotone;

    println!("# dynamical: packed below onset={packed_below}  fluidizes above onset={fluid_above}  lift separates(vz_above>vz_below)={lift_separates}  contact-p monotone-down={p_monotone}");

    // ── Verdict ─────────────────────────────────────────────────────────────────
    let pass_dem = err_dem <= valid.tol_dem;
    let pass_wy = err_wy <= valid.tol_wenyu;
    let pass_nontrivial = err_wy > valid.umf_err_floor && neg_ok;
    let pass_eps = eps_bed >= valid.eps_bed_lo && eps_bed <= valid.eps_bed_hi;
    let pass_regime = eps_bed <= valid.eps_max && eps_dem <= valid.eps_max;
    let pass_dep = dep_err_sph <= valid.tol_deposit_cell && dep_err_dem <= valid.tol_deposit_cell;
    let pass_mom = worst_mom <= valid.tol_momentum;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!("# two-way momentum conservation err {worst_mom:.2e} (tol {:.0e})", valid.tol_momentum);
    if pass_dem && pass_wy && pass_nontrivial && dyn_ok && pass_eps && pass_regime && pass_dep && pass_mom {
        println!(
            "VALIDATION: PASS  (U_mf SPH {umf_sph:.4} vs DEM {umf_dem:.4} {:.1}%<={:.0}% cross-method; vs Wen&Yu {umf_wy:.4} {:.1}%<={:.0}%; non-taut err>{:.0}% & neg-controls fail at {:+.0}%/{:+.0}%; live continuum fluidizes above U_mf & packed below; eps_bed={eps_bed:.3})",
            100.0 * err_dem, 100.0 * valid.tol_dem,
            100.0 * err_wy, 100.0 * valid.tol_wenyu,
            100.0 * valid.umf_err_floor,
            100.0 * (umf_nopg / umf_wy - 1.0), 100.0 * (umf_epsbug / umf_wy - 1.0),
        );
    } else {
        println!(
            "VALIDATION: FAIL  (dem_ok={pass_dem} wenyu_ok={pass_wy} nontrivial_ok={pass_nontrivial} dynamic_ok={dyn_ok} eps_ok={pass_eps} [{eps_bed:.3}] regime_ok={pass_regime} dep_ok={pass_dep} mom_ok={pass_mom})"
        );
        std::process::exit(1);
    }
}

// ─── Part C machinery: the coupled grass_multi loop (imposed interstitial flow) ─

struct DynResult {
    mean_vz: f64,
    /// Mean grain-contact pressure of the free parcels at the end (collapses to ~0
    /// once the fluid offloads the skeleton — the fluidization signature).
    mean_p: f64,
    mom_err: f64,
}

#[derive(Default)]
struct ParcelSpec {
    radius: Vec<f64>,
    v_solid: Vec<f64>,
}

#[derive(Clone, Copy)]
struct GasProps {
    mu: f64,
    rho: f64,
    d_grain: f64,
    eps_bed: f64,
    dt: f64,
    g: f64,
    u_super: f64,
}

/// Two-way momentum-exchange conservation, reported for the sanity gate.
#[derive(Clone, Copy, Default)]
struct SeamDiag {
    mom_err: f64,
}

/// `Output` phase on the CFD sub-App: IMPOSE the interstitial gas velocity
/// `u_g = U/ε` in every interior cell, then for each parcel evaluate the MacDonald β
/// at ε_bed, the seam drag (`drag_force_from_beta`, verbatim), the ∇P generalized-
/// buoyancy force `+V_p β u_rel/ε`, and hydrostatic buoyancy; write `InterphaseForces`
/// and deposit the equal-and-opposite drag momentum sink (`apply_momentum_sink`,
/// verbatim). Records the two-way momentum-conservation error.
fn coupled_seam_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    gas: Res<GasProps>,
    spec: Res<ParcelSpec>,
    pset: Res<ParticleSet>,
    mut forces: ResMut<InterphaseForces>,
    mut diag: ResMut<SeamDiag>,
) {
    let eos: &dyn Eos = &*eos.0;
    let mut state = reg.expect_mut::<CfdState>("CfdState");
    let parts = &pset.particles;
    forces.reset(parts.len());
    if parts.is_empty() {
        return;
    }
    let eps = gas.eps_bed;
    let u_g_imp = gas.u_super / eps;
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let rho = state.u[c].rho;
        state.u[c].rho_u = 0.0;
        state.u[c].rho_v = 0.0;
        state.u[c].rho_w = rho * u_g_imp;
    }
    let mut drag_on_particle = vec![[0.0f64; 3]; parts.len()];
    for (i, p) in parts.iter().enumerate() {
        let u_gas = coupling::sample_gas_velocity(&*mesh, &state, eos, p.center).unwrap_or([0.0, 0.0, u_g_imp]);
        let rho_f = coupling::sample_gas_density(&*mesh, &state, p.center).unwrap_or(gas.rho);
        // u_gas is the imposed interstitial velocity U/ε, so slip = u_gas − v_p.
        let rel = [u_gas[0] - p.velocity[0], u_gas[1] - p.velocity[1], u_gas[2] - p.velocity[2]];
        let rel_speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let beta = macdonald_beta(eps, rho_f, gas.mu, gas.d_grain, rel_speed);
        let v_solid = spec.v_solid.get(i).copied().unwrap_or(0.0);
        let drag = drag_force_from_beta(beta, v_solid, eps, rel);
        // ∇P (generalized-buoyancy) force +V_p β u_rel/ε.
        let pg = v_solid * beta / eps;
        // Hydrostatic buoyancy +ρ_f V g (+z up, since g points −z).
        let buoy_z = rho_f * v_solid * gas.g;
        forces.force[i] = [drag[0] + pg * rel[0], drag[1] + pg * rel[1], drag[2] + pg * rel[2] + buoy_z];
        drag_on_particle[i] = drag;
    }
    // Two-way momentum sink (reaction of the drag part) + conservation check.
    let mut m0 = [0.0f64; 3];
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) {
            let v = mesh.cell_volume(c);
            m0[0] += state.u[c].rho_u * v;
            m0[1] += state.u[c].rho_v * v;
            m0[2] += state.u[c].rho_w * v;
        }
    }
    coupling::apply_momentum_sink(&*mesh, &mut state, parts, &drag_on_particle, gas.dt);
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
        let imp = -drag_on_particle.iter().map(|f| f[k]).sum::<f64>() * gas.dt;
        dn += (dm - imp) * (dm - imp);
        sc += imp * imp;
    }
    diag.mom_err = dn.sqrt() / sc.sqrt().max(1e-30);
}

fn build_coupled_cfd(gas: &GasCfg, mesh_cfg: UniformMeshConfig, props: GasProps) -> App {
    let (rho, p) = (gas.rho, gas.p);
    let t = p / (rho * R_GAS);
    let init = move |_x: Vec3| {
        let eos = IdealGas::air();
        eos.prim_to_cons(&PrimVar::new(rho, 0.0, 0.0, 0.0, p, t))
    };
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    app.add_resource(Viscosity::Constant(gas.mu));
    app.add_resource(props);
    app.add_resource(ParcelSpec::default());
    app.add_resource(ParticleSet::default());
    app.add_resource(InterphaseForces::default());
    app.add_resource(SeamDiag::default());
    app.add_update_system(coupled_seam_system, MeshScheduleSet::Output);
    app
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Export,
    TickCfd,
    Import,
    TickSph,
}
impl ScheduleSet for Phase {
    fn to_index(&self) -> u32 {
        match self {
            Self::Export => 0,
            Self::TickCfd => 1,
            Self::Import => 2,
            Self::TickSph => 3,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Export => "Export",
            Self::TickCfd => "TickCfd",
            Self::Import => "Import",
            Self::TickSph => "TickSph",
        }
    }
}

fn export_kinematics(world: Multi) {
    let atoms = world.expect_read::<Atom>("sph");
    let n = atoms.nlocal as usize;
    let radii = {
        let spec = world.expect_read::<ParcelSpec>("cfd");
        spec.radius.clone()
    };
    let mut set = world.expect_write::<ParticleSet>("cfd");
    set.particles.clear();
    for i in 0..n {
        set.particles.push(ParticleKinematics {
            center: [atoms.pos[i][0], atoms.pos[i][1], atoms.pos[i][2]],
            velocity: [atoms.vel[i][0], atoms.vel[i][1], atoms.vel[i][2]],
            radius: radii.get(i).copied().unwrap_or(0.0),
        });
    }
}

fn import_forces(world: Multi) {
    let f = {
        let forces = world.expect_read::<InterphaseForces>("cfd");
        forces.force.clone()
    };
    world.expect_write::<FluidForces>("sph").f = f;
}

/// One coupled run at superficial `u_super`: build SPH+CFD sub-Apps, settle-prime the
/// SPH bed, march the parent loop `steps` times, and report mean free-parcel v_z,
/// mean contact pressure, and the worst two-way momentum-conservation error.
#[allow(clippy::too_many_arguments)]
fn run_coupled(gas: &GasCfg, bed: &BedCfg, grid: &GridCfg, grav: &GravityCfg, eps_bed: f64, u_super: f64, steps: usize) -> DynResult {
    let dt = bed.sph_dt;
    let props = GasProps {
        mu: gas.mu,
        rho: gas.rho,
        d_grain: bed.grain_d,
        eps_bed,
        dt,
        g: grav.gz.abs(),
        u_super,
    };
    let sph = build_sph_app(bed, grav, dt);
    let mesh_cfg = build_gas_mesh(bed, grid);
    let cfd = build_coupled_cfd(gas, mesh_cfg, props);

    let mut parent = App::new();
    parent.add_subapp("sph", sph);
    parent.add_subapp("cfd", cfd);
    parent.add_update_system(export_kinematics, Phase::Export);
    parent.add_update_system(tick_subapp("cfd", 1), Phase::TickCfd);
    parent.add_update_system(import_forces, Phase::Import);
    parent.add_update_system(tick_subapp("sph", 1), Phase::TickSph);
    parent.prepare();

    prime_and_spec(&mut parent, bed);
    for _ in 0..steps {
        parent.run();
    }
    let (mean_vz, mean_p) = sph_bed_state(&parent);
    let mom_err = {
        let subs = parent.get_resource_ref::<SubApps>().unwrap();
        let cell = subs.find("cfd").unwrap().resource_cell(TypeId::of::<SeamDiag>()).unwrap().borrow();
        cell.downcast_ref::<SeamDiag>().unwrap().mom_err
    };

    if let Some(cell) = parent.get_mut_resource(TypeId::of::<SubApps>()) {
        cell.borrow_mut().downcast_mut::<SubApps>().unwrap().cleanup_all();
    }
    DynResult { mean_vz, mean_p, mom_err }
}

/// Prime the SPH sub-App (prepare+insert+settle), then store the constant per-parcel
/// radii/solid-volumes as `ParcelSpec` on the CFD sub-App.
fn prime_and_spec(parent: &mut App, bed: &BedCfg) {
    {
        let cell = parent.get_mut_resource(TypeId::of::<SubApps>()).unwrap();
        let mut gd = cell.borrow_mut();
        let subs = gd.downcast_mut::<SubApps>().unwrap();
        for _ in 0..bed.settle_steps {
            subs.tick("sph");
        }
    }
    let (radius, v_solid) = {
        let subs = parent.get_resource_ref::<SubApps>().unwrap();
        let sph = subs.find("sph").unwrap();
        let atom_cell = sph.resource_cell(TypeId::of::<Atom>()).unwrap().borrow();
        let reg_cell = sph.resource_cell(TypeId::of::<AtomDataRegistry>()).unwrap().borrow();
        let atoms = atom_cell.downcast_ref::<Atom>().unwrap();
        let registry = reg_cell.downcast_ref::<AtomDataRegistry>().unwrap();
        let sph = registry.expect::<SphAtom>("prime_and_spec");
        let n = atoms.nlocal as usize;
        let (mut radius, mut v_solid) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            let vs = sph.particle_mass[i] / bed.rho_s;
            if sph.is_boundary[i] > 0.5 {
                radius.push(0.0);
                v_solid.push(0.0);
            } else {
                radius.push((3.0 * vs / (4.0 * PI)).cbrt());
                v_solid.push(vs);
            }
        }
        (radius, v_solid)
    };
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cfd = subs.find("cfd").unwrap();
    if let Some(cell) = cfd.resource_cell(TypeId::of::<ParcelSpec>()) {
        let mut b = cell.borrow_mut();
        let s = b.downcast_mut::<ParcelSpec>().unwrap();
        s.radius = radius;
        s.v_solid = v_solid;
    }
}

/// Bed mean free-parcel vertical velocity and mean grain-contact pressure.
fn sph_bed_state(parent: &App) -> (f64, f64) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let sph = subs.find("sph").unwrap();
    let atom_cell = sph.resource_cell(TypeId::of::<Atom>()).unwrap().borrow();
    let reg_cell = sph.resource_cell(TypeId::of::<AtomDataRegistry>()).unwrap().borrow();
    let atoms = atom_cell.downcast_ref::<Atom>().unwrap();
    let registry = reg_cell.downcast_ref::<AtomDataRegistry>().unwrap();
    let sph = registry.expect::<SphAtom>("sph_bed_state");
    let n = atoms.nlocal as usize;
    let (mut sum_vz, mut sum_p) = (0.0, 0.0);
    let mut nfree = 0usize;
    for i in 0..n {
        if sph.is_boundary[i] > 0.5 {
            continue;
        }
        sum_vz += atoms.vel[i][2];
        sum_p += sph.pressure[i];
        nfree += 1;
    }
    let nf = nfree.max(1) as f64;
    (sum_vz / nf, sum_p / nf)
}
