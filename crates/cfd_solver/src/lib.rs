//! The solver heart: RHS assembly over `FvMesh` faces, time integration, CFL
//! timestep, and the GRASS plugins that wire them into [`MeshScheduleSet`].
//!
//! This is the dev_field_efvm analog of DIRT's `dirt_granular` — the crate that owns
//! the per-step compute loop. It writes **no** mesh, storage, or halo code: it
//! reads the registered [`CfdState`], iterates the substrate's faces, and lets
//! `field_core` move ghosts. The inviscid flux loop is generic over any
//! [`FvMesh`]; CFL uses [`StructuredMesh`] (per-axis spacing).

use std::marker::PhantomData;
use std::sync::Arc;

use cfd_eos::{thermal_conductivity, Eos, EosResource, IdealGas, Viscosity};
use cfd_riemann::{FluxScheme, RiemannFlux};
use cfd_state::{CfdState, ConsVar, PrimVar, SolidMask};
use field_core::{
    halo_exchange_forward_all, CartesianMesh, FieldRegistry, FvMesh, MeshScheduleSet,
    StructuredMesh, Vec3,
};
use grass_app::prelude::*;
use grass_mpi::CommResource;
use grass_scheduler::{Res, ResMut};

mod integrate;
pub use integrate::{
    EulerIntegrator, Rk3Integrator, RkStage, SubstepResult, TimeIntegrator, TimeIntegratorResource,
};

// ─── Resources ───────────────────────────────────────────────────────────────
// `EosResource` lives in `cfd_eos` (shared with `cfd_boundary`).

/// The active numerical flux scheme.
pub struct RiemannFluxResource(pub Arc<dyn RiemannFlux>);

/// Initial-condition closure: maps a cell centroid to its conserved state.
pub struct InitState(pub Arc<dyn Fn(Vec3) -> ConsVar + Send + Sync>);

/// Static solver settings. Deserializable from a `[solver]` config section.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    /// CFL number (used when `fixed_dt` is `None`).
    pub cfl: f64,
    /// Override the CFL-computed timestep with a fixed value.
    pub fixed_dt: Option<f64>,
    /// Add the viscous (Navier–Stokes) terms. Bounds the solver to a
    /// `StructuredMesh` and tightens the CFL with the viscous spectral radius.
    pub viscous: bool,
    /// Use 2nd-order MUSCL reconstruction (minmod-limited slopes) instead of the
    /// 1st-order flux. Bounds the inviscid sweep to a `CartesianMesh`.
    pub muscl: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self { cfl: 0.4, fixed_dt: None, viscous: false, muscl: false }
    }
}

/// Per-step solver state advanced by the scheduler.
pub struct SolverState {
    pub dt: f64,
    pub time: f64,
    pub step: u64,
    pub rk_stage: RkStage,
}

impl Default for SolverState {
    fn default() -> Self {
        Self { dt: 0.0, time: 0.0, step: 0, rk_stage: RkStage::First }
    }
}

// ─── Systems ─────────────────────────────────────────────────────────────────

/// `Halo`: forward-exchange every registered field (the conserved state, and the
/// solid mask once IBM is registered) across rank boundaries. Serial = no-op.
pub fn halo_system<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>, comm: Res<CommResource>) {
    halo_exchange_forward_all(mesh.halo_plan(), &reg, &*comm.0);
}

/// `PreFlux`: reset the RHS accumulator (`#[zero]` columns) before any flux or
/// viscous system adds to it. Split out of the flux system so the inviscid and
/// viscous contributions can both *accumulate* in `Flux` in any order.
pub fn zero_rhs_system<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>) {
    reg.zero_all(mesh.n_cells_total());
}

/// `Flux`: assemble inviscid flux divergence over every face. The conservative
/// two-sided scatter: a face's normal flux leaves `owner` and (for an interior
/// face) enters `other`, each scaled by `area / volume`.
pub fn flux_system<M: FvMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    flux: Res<RiemannFluxResource>,
) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    let flux: &dyn RiemannFlux = &*flux.0;

    mesh.for_each_face(&mut |f| {
        let u_owner = state.u[f.owner];
        let u_other = state.u[f.other];
        let an = f.area_normal;
        let area = (an[0] * an[0] + an[1] * an[1] + an[2] * an[2]).sqrt();
        if area == 0.0 {
            return;
        }
        let nhat = [an[0] / area, an[1] / area, an[2] / area];
        let fn_ = flux.flux_normal(&u_owner, &u_other, eos, nhat);

        // `owner` is always a local interior cell. Scatter into `other` only when
        // it is an owned cell (an interior face's `other` may be a halo ghost
        // under MPI — the neighbor rank computes that face and owns that cell).
        state.rhs[f.owner] -= fn_ * (area / mesh.cell_volume(f.owner));
        if f.patch.is_none() && mesh.is_local_cell(f.other) {
            state.rhs[f.other] += fn_ * (area / mesh.cell_volume(f.other));
        }
    });
}

// ─── MUSCL 2nd-order reconstruction (generic over CartesianMesh) ──────────────

/// Per-(local)-cell limited slopes Δ in conserved variables, one per axis. A
/// face reconstructs `u_face = u_cell ± 0.5·Δ[axis]` (the half-cell extrapolation;
/// the cell size cancels in this uniform-stencil form). Recomputed every RK
/// substep by [`compute_slopes_system`].
///
/// **Accuracy caveat at refinement seams:** the slopes are built from
/// neighbour *cell-means* treated as one cell-width apart. Across a 2:1 coarse/
/// fine interface that spacing is not uniform, so the reconstruction is only
/// formally 2nd-order on uniform-spaced stencils; at a coarse/fine seam it
/// degrades toward 1st-order (still bounded by the limiter — stable, just less
/// accurate). A distance-weighted slope would recover full order there.
#[derive(Default)]
pub struct MusclSlopes(pub Vec<[ConsVar; 3]>);

/// Which TVD slope limiter [`compute_slopes_system`] applies to the forward/
/// backward differences. Both are 2nd-order TVD limiters; they differ in how much
/// they clip near extrema, which sets the scheme's *effective* order on smooth
/// flows:
/// - [`Minmod`](SlopeLimiter::Minmod) — the most diffusive TVD limiter: it takes
///   the smaller-magnitude one-sided difference and **zeros the slope at every
///   smooth extremum**, so on a smooth field it degrades the formal 2nd order to
///   ≈1.6 in L1 (the extremum is clipped to piecewise-constant). Most robust near
///   strong shocks.
/// - [`MonotonizedCentral`](SlopeLimiter::MonotonizedCentral) — van Leer's MC
///   limiter: the central difference bounded by twice each one-sided difference.
///   Still TVD, but it only clips where a genuine over/undershoot would occur, so
///   it preserves the reconstruction's **full 2nd-order** accuracy on smooth flows
///   (≈2.0 in L1 on the isentropic vortex — see the order gate).
///
/// Default is [`Minmod`](SlopeLimiter::Minmod), preserving the historical scheme;
/// choose `MonotonizedCentral` when 2nd-order accuracy on smooth flow matters more
/// than maximal shock robustness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SlopeLimiter {
    #[default]
    Minmod,
    MonotonizedCentral,
}

/// Two-argument minmod: 0 if the one-sided differences disagree in sign, else the
/// smaller-magnitude one. Bounds the slope (keeps the scheme TVD); it does not
/// re-check the reconstructed face value, so it is stable rather than provably
/// monotone.
#[inline]
fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

/// Monotonized-central (MC) limiter of the two one-sided differences: the central
/// slope `(a+b)/2` clamped to `±2·min(|a|,|b|)`. Zero when the differences disagree
/// in sign (same TVD guard as minmod), but away from extrema it keeps the full
/// central slope, so it is 2nd-order accurate at smooth extrema where minmod clips.
#[inline]
fn monotonized_central(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else {
        let s = if a > 0.0 { 1.0 } else { -1.0 };
        s * (2.0 * a.abs()).min(2.0 * b.abs()).min(0.5 * (a + b).abs())
    }
}

/// Apply `limiter` to the forward/backward differences of every conserved
/// component. Limiting on *conserved* (not characteristic) variables — the usual
/// robust heuristic — so it is strictly TVD only for scalar advection; on the
/// Euler system it can mildly smear contacts near strong shocks.
#[inline]
fn limited_slope_cons(fwd: ConsVar, bwd: ConsVar, limiter: SlopeLimiter) -> ConsVar {
    let lim: fn(f64, f64) -> f64 = match limiter {
        SlopeLimiter::Minmod => minmod,
        SlopeLimiter::MonotonizedCentral => monotonized_central,
    };
    ConsVar::new(
        lim(fwd.rho, bwd.rho),
        lim(fwd.rho_u, bwd.rho_u),
        lim(fwd.rho_v, bwd.rho_v),
        lim(fwd.rho_w, bwd.rho_w),
        lim(fwd.rho_e, bwd.rho_e),
    )
}

/// Mean of the conserved states of the neighbours across one axis face (one
/// for a same/coarser neighbour, four averaged for a finer side). `None` if the
/// neighbour is across a non-local boundary (empty slice ⇒ first order).
#[inline]
fn neighbor_mean(idx: &[usize], u: &[ConsVar]) -> Option<ConsVar> {
    if idx.is_empty() {
        return None;
    }
    let mut s = ConsVar::ZERO;
    for &i in idx {
        s += u[i];
    }
    Some(s * (1.0 / idx.len() as f64))
}

/// `PreFlux` (after boundary ghosts are filled): minmod-limited central-difference
/// slopes per cell, in conserved variables. When an axis is missing a neighbour
/// (a non-local cross-rank side), only **that axis** falls back to first order;
/// the other axes keep their slopes (toy-cfd zeroed all three — this is tighter
/// and recovers 2nd-order in the partition-interior directions).
pub fn compute_slopes_system<M: CartesianMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    limiter: Res<SlopeLimiter>,
    mut slopes: ResMut<MusclSlopes>,
) {
    let limiter = *limiter;
    let state = reg.expect_mut::<CfdState>("CfdState not registered");
    // Size + index the slope buffer by the *full* cell array (locals + ghosts), so
    // it is addressable by any face's `owner`/`other` regardless of how the mesh
    // packs local cells. `ForestMesh` packs locals contiguously at the front
    // (indices < n_local); `UniformMesh` interleaves them with ghost layers in a
    // padded raw array — both are handled by keying on the cell index directly and
    // skipping non-local cells via `is_local_cell`.
    let n = mesh.n_cells_total();
    slopes.0.resize(n, [ConsVar::ZERO; 3]);

    for c in 0..n {
        if !mesh.is_local_cell(c) {
            slopes.0[c] = [ConsVar::ZERO; 3];
            continue;
        }
        let u_self = state.u[c];
        let mut s = [ConsVar::ZERO; 3];
        for (axis, slot) in s.iter_mut().enumerate() {
            let pos = neighbor_mean(mesh.axis_neighbors(c, axis, true), &state.u);
            let neg = neighbor_mean(mesh.axis_neighbors(c, axis, false), &state.u);
            // Missing a neighbour on this axis ⇒ stay 1st-order on this axis only
            // (slot keeps its zero); the other axes are unaffected.
            if let (Some(up), Some(un)) = (pos, neg) {
                *slot = limited_slope_cons(up - u_self, u_self - un, limiter);
            }
        }
        slopes.0[c] = s;
    }
}

/// Reconstruct a cell's conserved state at `face_centroid` from its limited
/// slopes (half-cell extrapolation, geometry-signed per axis).
#[inline]
fn reconstruct_to_face(u_self: ConsVar, slope: &[ConsVar; 3], face_centroid: Vec3, cc: Vec3) -> ConsVar {
    let mut u = u_self;
    for (axis, sl) in slope.iter().enumerate() {
        let frac = if face_centroid[axis] > cc[axis] {
            0.5
        } else if face_centroid[axis] < cc[axis] {
            -0.5
        } else {
            0.0
        };
        u += *sl * frac;
    }
    u
}

/// `Flux`: the **MUSCL** inviscid sweep. Identical conservative two-sided scatter
/// as [`flux_system`], but each face's left/right states are reconstructed to the
/// face by half-cell extrapolation of the limited slopes — 2nd-order accurate and
/// far steadier across coarse/fine interfaces than the 1st-order flux. Cells
/// without slopes (boundary/halo ghosts) fall back to their cell-centre value.
pub fn muscl_flux_system<M: CartesianMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    flux: Res<RiemannFluxResource>,
    slopes: Res<MusclSlopes>,
) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    let flux: &dyn RiemannFlux = &*flux.0;
    let slopes = &slopes.0;

    mesh.for_each_face(&mut |f| {
        let an = f.area_normal;
        let area = (an[0] * an[0] + an[1] * an[1] + an[2] * an[2]).sqrt();
        if area == 0.0 {
            return;
        }
        let nhat = [an[0] / area, an[1] / area, an[2] / area];

        // Physical-boundary faces stay 1st-order: the wall ghost is a reflection
        // of the cell-*centre* state, so reconstructing the owner there would
        // break the symmetry that zeroes the wall mass flux (and lose strict
        // conservation in a closed box). Interior faces get full MUSCL.
        let interior = f.patch.is_none();
        let u_l = if interior {
            reconstruct_to_face(state.u[f.owner], &slopes[f.owner], f.centroid, mesh.cell_centroid(f.owner))
        } else {
            state.u[f.owner]
        };
        // Reconstruct the other side only if it is a real local leaf with slopes;
        // boundary/halo ghosts use their BC-filled cell value (`is_local_cell` is
        // the packing-agnostic test — ForestMesh ghosts sit at index ≥ n_local,
        // UniformMesh ghosts sit in the padded raw band).
        let u_r = if interior && mesh.is_local_cell(f.other) {
            reconstruct_to_face(state.u[f.other], &slopes[f.other], f.centroid, mesh.cell_centroid(f.other))
        } else {
            state.u[f.other]
        };

        let fn_ = flux.flux_normal(&u_l, &u_r, eos, nhat);
        state.rhs[f.owner] -= fn_ * (area / mesh.cell_volume(f.owner));
        if f.patch.is_none() && mesh.is_local_cell(f.other) {
            state.rhs[f.other] += fn_ * (area / mesh.cell_volume(f.other));
        }
    });
}

/// Per-substep cache of primitives + sound speeds, so the structured fast path
/// computes them once per cell instead of redundantly per face.
#[derive(Default)]
pub struct SolverScratch {
    pub prim: Vec<PrimVar>,
    pub sound: Vec<f64>,
}

impl SolverScratch {
    fn refresh(&mut self, u: &[ConsVar], eos: &dyn Eos, n: usize) {
        self.prim.resize(n, PrimVar::default());
        self.sound.resize(n, 0.0);
        for (c, uc) in u.iter().enumerate().take(n) {
            if uc.rho > 0.0 {
                self.prim[c] = eos.cons_to_prim(uc);
                self.sound[c] = eos.sound_speed(uc);
            } else {
                self.prim[c] = PrimVar::default();
                self.sound[c] = 0.0;
            }
        }
    }
}

#[inline(always)]
fn cached_flux<F: FluxScheme>(
    scheme: &F,
    u: &[ConsVar],
    s: &SolverScratch,
    l: usize,
    r: usize,
    eos: &dyn Eos,
    dir: usize,
) -> ConsVar {
    scheme.flux_scheme_cached(&u[l], &u[r], &s.prim[l], &s.prim[r], s.sound[l], s.sound[r], eos, dir)
}

/// `Flux`: the **structured fast path** — toy-cfd's fused `(i,j,k)` sweep with a
/// per-substep [`SolverScratch`] and `flux_scheme_cached` (skips the redundant
/// sound-speed sqrts the generic `flux_system` redoes per face). Monomorphized
/// over the concrete flux scheme `F` (so the flux body inlines), and bounded to
/// [`StructuredMesh`]. Produces the same RHS as `flux_system`; the
/// `fast_path_matches_generic` test pins them together.
///
/// **Serial only.** This sweep decides boundary vs. interior faces by *index*
/// edge (`i == 0`, `i+1 < ni`), not by cell *ownership*. Under MPI a subdomain
/// edge `i == 0` is a halo face, not a physical boundary — so this would compute
/// wrong fluxes at every rank boundary. Use the ownership-guarded `flux_system`
/// under MPI. The `debug_assert` below fails loudly if it is ever wired into a
/// decomposed run.
pub fn flux_fast_system<M: StructuredMesh, F: FluxScheme + Send + Sync + 'static>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    scheme: Res<F>,
    mut scratch: ResMut<SolverScratch>,
) {
    debug_assert!(
        mesh.halo_plan().is_serial(),
        "flux_fast_system is serial-only (index-edge boundary test); use flux_system under MPI",
    );
    let eos: &dyn Eos = &*eos.0;
    let scheme = &*scheme;
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let n = mesh.n_cells_total();
    scratch.refresh(&state.u, eos, n);

    let [ni, nj, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let owner = mesh.idx_raw(ir, jr, kr);
                let vol = mesh.cell_volume(owner);
                let (dx, dy, dz) = (mesh.spacing(0, i), mesh.spacing(1, j), mesh.spacing(2, k));

                // One positive face per axis, plus the low-boundary face at index 0.
                // Each axis: (dir, face area, +neighbour raw idx, has-interior-neighbour,
                //             at-low-edge, −neighbour raw idx).
                let axes = [
                    (0usize, dy * dz, mesh.idx_raw(ir + 1, jr, kr), i + 1 < ni, i == 0, mesh.idx_raw(ir - 1, jr, kr)),
                    (1, dx * dz, mesh.idx_raw(ir, jr + 1, kr), j + 1 < nj, j == 0, mesh.idx_raw(ir, jr - 1, kr)),
                    (2, dx * dy, mesh.idx_raw(ir, jr, kr + 1), k + 1 < nk, k == 0, mesh.idx_raw(ir, jr, kr - 1)),
                ];
                for (dir, area, hi, hi_interior, at_lo, lo) in axes {
                    let inv = area / vol;
                    // +axis face: F·n̂ = flux(owner, other). Out of owner; into other
                    // only if the neighbour is a local interior cell.
                    let f = cached_flux(scheme, &state.u, &scratch, owner, hi, eos, dir);
                    state.rhs[owner] -= f * inv;
                    if hi_interior {
                        state.rhs[hi] += f * (area / mesh.cell_volume(hi));
                    }
                    // −axis face at the low edge: outward normal flips, owner sits on
                    // the + side ⇒ contribution is +flux(ghost, owner).
                    if at_lo {
                        let fb = cached_flux(scheme, &state.u, &scratch, lo, owner, eos, dir);
                        state.rhs[owner] += fb * inv;
                    }
                }
            }
        }
    }
}

/// 3-point non-uniform second difference on pre-inverted spacings (multiply-only).
#[inline(always)]
fn d2_pre(um: f64, u0: f64, up: f64, inv_hm: f64, inv_hp: f64, inv_hs2: f64) -> f64 {
    ((up - u0) * inv_hp - (u0 - um) * inv_hm) * inv_hs2
}

/// `Flux` (after the inviscid sweep): add the viscous Navier–Stokes terms
/// `μ∇²u` and the conductive `κ∇²T`, via 2nd-order central differences. Wide
/// directional stencil ⇒ bounded to [`StructuredMesh`]; reaches into ghost layers
/// with `idx_raw`, so (unlike toy-cfd) there is no `i==0` special-casing. Ported
/// from toy-cfd's `compute_viscous_rhs`, including its solid-mask coupling: when a
/// [`SolidMask`] is registered, solid neighbours contribute no-slip (`u=0`) and
/// adiabatic (`T=T_self`) wall values to the stencil, and solid cells are skipped.
/// Transport is ideal-gas + Sutherland.
pub fn viscous_system<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    gas: Res<IdealGas>,
    visc: Res<Viscosity>,
) {
    let gas = *gas;
    let visc = *visc;
    let mask = reg.get::<SolidMask>(); // None when no IBM is registered
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");

    // Primitive cache over all cells (ghosts included — their BC/halo-filled
    // state supplies the wall/inflow gradients).
    let prim: Vec<PrimVar> = state.u.iter().map(|u| gas.cons_to_prim(u)).collect();

    let [ni, nj, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    let dx = mesh.spacing(0, 0);
    let dx_inv2 = 1.0 / (dx * dx);

    // Pre-invert non-uniform y/z 3-point weights from cell-center distances.
    let yc = |jr: usize| mesh.cell_centroid(mesh.idx_raw(ng, jr, ng))[1];
    let zc = |kr: usize| mesh.cell_centroid(mesh.idx_raw(ng, ng, kr))[2];
    let (mut y_im, mut y_ip, mut y_is) = (vec![0.0; nj], vec![0.0; nj], vec![0.0; nj]);
    for j in 0..nj {
        let jr = j + ng;
        let (hm, hp) = (yc(jr) - yc(jr - 1), yc(jr + 1) - yc(jr));
        y_im[j] = 1.0 / hm;
        y_ip[j] = 1.0 / hp;
        y_is[j] = 2.0 / (hm + hp);
    }
    let (mut z_im, mut z_ip, mut z_is) = (vec![0.0; nk], vec![0.0; nk], vec![0.0; nk]);
    for k in 0..nk {
        let kr = k + ng;
        let (hm, hp) = (zc(kr) - zc(kr - 1), zc(kr + 1) - zc(kr));
        z_im[k] = 1.0 / hm;
        z_ip[k] = 1.0 / hp;
        z_is[k] = 2.0 / (hm + hp);
    }

    // Neighbour velocity+temperature, substituting no-slip/adiabatic at a solid.
    let is_solid = |c: usize| mask.as_ref().is_some_and(|m| m.solid[c]);
    let nb = |c: usize, self_t: f64| -> (f64, f64, f64, f64) {
        if is_solid(c) {
            (0.0, 0.0, 0.0, self_t)
        } else {
            let p = &prim[c];
            (p.u, p.v, p.w, p.t)
        }
    };

    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let c = mesh.idx_raw(ir, jr, kr);
                if is_solid(c) {
                    continue; // solid cells never receive viscous RHS
                }
                let ps = prim[c];
                let mu = visc.mu(ps.t);
                let kappa = thermal_conductivity(mu, gas.gamma, gas.r_gas);

                let xp = nb(mesh.idx_raw(ir + 1, jr, kr), ps.t);
                let xm = nb(mesh.idx_raw(ir - 1, jr, kr), ps.t);
                let yp = nb(mesh.idx_raw(ir, jr + 1, kr), ps.t);
                let ym = nb(mesh.idx_raw(ir, jr - 1, kr), ps.t);
                let zp = nb(mesh.idx_raw(ir, jr, kr + 1), ps.t);
                let zm = nb(mesh.idx_raw(ir, jr, kr - 1), ps.t);

                let vel = [ps.u, ps.v, ps.w];
                let d2_x = |xm_c: f64, xp_c: f64, vc: f64| (xp_c - 2.0 * vc + xm_c) * dx_inv2;
                let d2_y = |ym_c: f64, yp_c: f64, vc: f64| {
                    d2_pre(ym_c, vc, yp_c, y_im[j], y_ip[j], y_is[j])
                };
                let d2_z = |zm_c: f64, zp_c: f64, vc: f64| {
                    d2_pre(zm_c, vc, zp_c, z_im[k], z_ip[k], z_is[k])
                };

                // (.0,.1,.2,.3) = (u, v, w, t)
                let visc_fx = mu
                    * (d2_x(xm.0, xp.0, vel[0]) + d2_y(ym.0, yp.0, vel[0]) + d2_z(zm.0, zp.0, vel[0]));
                let visc_fy = mu
                    * (d2_x(xm.1, xp.1, vel[1]) + d2_y(ym.1, yp.1, vel[1]) + d2_z(zm.1, zp.1, vel[1]));
                let visc_fz = mu
                    * (d2_x(xm.2, xp.2, vel[2]) + d2_y(ym.2, yp.2, vel[2]) + d2_z(zm.2, zp.2, vel[2]));

                let d2t = d2_x(xm.3, xp.3, ps.t) + d2_y(ym.3, yp.3, ps.t) + d2_z(zm.3, zp.3, ps.t);
                let visc_energy =
                    vel[0] * visc_fx + vel[1] * visc_fy + vel[2] * visc_fz + kappa * d2t;

                state.rhs[c] += ConsVar::new(0.0, visc_fx, visc_fy, visc_fz, visc_energy);
            }
        }
    }
}

/// `Setup`: CFL timestep for a **general** (possibly unstructured / adaptive)
/// `FvMesh` — uses `cell_volume^(1/3)` as the length scale instead of per-axis
/// spacing, so it works on a `ForestMesh` where `StructuredMesh` is unavailable.
/// (Inviscid only; the viscous term needs structured spacing.)
pub fn cfl_generic_system<M: FvMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    config: Res<SolverConfig>,
    comm: Res<CommResource>,
    mut sstate: ResMut<SolverState>,
) {
    if sstate.rk_stage != RkStage::First {
        return;
    }
    if let Some(dt) = config.fixed_dt {
        sstate.dt = dt;
        return;
    }
    let state = reg.expect::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    let mut lambda_max = 0.0f64;
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let u = &state.u[c];
        if u.rho <= 0.0 {
            continue;
        }
        let h = mesh.cell_volume(c).cbrt();
        let vel = u.velocity();
        let speed = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
        lambda_max = lambda_max.max(3.0 * (speed + eos.sound_speed(u)) / h);
    }
    let dt_local = if lambda_max > 0.0 { config.cfl / lambda_max } else { f64::INFINITY };
    sstate.dt = comm.all_reduce_min_f64(dt_local);
}

/// `Update`: advance the conserved state by one integrator substep, stepping the
/// RK state machine and (on a completed step) the clock.
pub fn update_system(
    reg: Res<FieldRegistry>,
    integ: Res<TimeIntegratorResource>,
    mut sstate: ResMut<SolverState>,
) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let result = integ.integrator().substep(&mut state, sstate.dt, sstate.rk_stage);
    sstate.rk_stage = result.next_stage;
    if result.completed_step {
        sstate.step += 1;
        sstate.time += sstate.dt;
    }
}

/// `Setup`: CFL-limited global timestep. Computed once per full step (on the
/// first substage) over owned cells, then min-reduced across ranks. Structured-
/// only (needs per-axis spacing); a generic-mesh CFL is future work.
pub fn cfl_system<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    config: Res<SolverConfig>,
    comm: Res<CommResource>,
    visc: Res<Viscosity>,
    mut sstate: ResMut<SolverState>,
) {
    if sstate.rk_stage != RkStage::First {
        return; // dt is held constant across a step's substages
    }
    if let Some(dt) = config.fixed_dt {
        sstate.dt = dt;
        return;
    }

    let state = reg.expect::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    let [ni, nj, nk] = mesh.dims();
    let mut lambda_max = 0.0f64;
    for i in 0..ni {
        let dx = mesh.spacing(0, i);
        for j in 0..nj {
            let dy = mesh.spacing(1, j);
            for k in 0..nk {
                let dz = mesh.spacing(2, k);
                let u = &state.u[mesh.idx(i, j, k)];
                if u.rho <= 0.0 {
                    continue;
                }
                let vel = u.velocity();
                let a = eos.sound_speed(u);
                let mut lambda = (vel[0].abs() + a) / dx
                    + (vel[1].abs() + a) / dy
                    + (vel[2].abs() + a) / dz;
                if config.viscous {
                    // Viscous spectral radius 2ν(1/dx²+1/dy²+1/dz²), ν = μ/ρ.
                    let nu = visc.mu(eos.cons_to_prim(u).t) / u.rho;
                    lambda += 2.0 * nu * (1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz));
                }
                lambda_max = lambda_max.max(lambda);
            }
        }
    }
    let dt_local = if lambda_max > 0.0 {
        config.cfl / lambda_max
    } else {
        f64::INFINITY
    };
    sstate.dt = comm.all_reduce_min_f64(dt_local);
}

/// `PostSetup` (one-shot): fill every cell's conserved state from the initial-
/// condition closure. Runs after FIELD's field resize (which is in `Setup`).
fn init_state_system<M: FvMesh>(mesh: Res<M>, reg: Res<FieldRegistry>, init: Res<InitState>) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    for c in 0..mesh.n_cells_total() {
        state.u[c] = (init.0)(mesh.cell_centroid(c));
    }
}

// ─── Plugins ─────────────────────────────────────────────────────────────────

/// Registers [`CfdState`] and initializes it from a closure on the cell centroid.
pub struct CfdStatePlugin {
    init: Arc<dyn Fn(Vec3) -> ConsVar + Send + Sync>,
}

impl CfdStatePlugin {
    pub fn new(init: impl Fn(Vec3) -> ConsVar + Send + Sync + 'static) -> Self {
        Self { init: Arc::new(init) }
    }
    /// Construct from an already-shared init closure (used by the bundle).
    pub fn from_arc(init: Arc<dyn Fn(Vec3) -> ConsVar + Send + Sync>) -> Self {
        Self { init }
    }
}

impl Plugin for CfdStatePlugin {
    fn build(&self, app: &mut App) {
        field_core::register_field_data!(app, CfdState::default());
        app.add_resource(InitState(self.init.clone()));
        app.add_setup_system(init_state_system::<field_core::UniformMesh>, ScheduleSetupSet::PostSetup);
    }

    fn provides(&self) -> Vec<&str> {
        vec!["cfd_state"]
    }

    fn requires(&self) -> Vec<&str> {
        vec!["field_registry", "mesh"]
    }
}

/// Adds the [`EosResource`] (calorically-perfect air by default).
pub struct IdealGasPlugin;

impl Plugin for IdealGasPlugin {
    fn build(&self, app: &mut App) {
        let gas = IdealGas::air();
        app.add_resource(gas); // concrete, for the viscous transport coefficients
        app.add_resource(EosResource(Arc::new(gas))); // dyn, for flux / boundary
        // Default transport = air's Sutherland μ(T). A case that targets a chosen
        // Reynolds number overrides this with `Viscosity::Constant(μ)` after adding
        // the gas plugin (add_resource replaces by type).
        app.add_resource(Viscosity::default());
    }
    fn provides(&self) -> Vec<&str> {
        vec!["eos"]
    }
}

/// Adds the [`RiemannFluxResource`] and the generic 1st-order inviscid sweep.
/// Defaults to Rusanov; `.hllc()` selects HLLC.
///
/// Generic over the mesh, defaulting to `UniformMesh` so `FluxPlugin::rusanov()`
/// is unchanged; a forest app uses `FluxPlugin::<ForestMesh>::rusanov()`.
pub struct FluxPlugin<M: FvMesh = field_core::UniformMesh> {
    flux: Arc<dyn RiemannFlux>,
    _marker: PhantomData<fn() -> M>,
}

impl<M: FvMesh> Default for FluxPlugin<M> {
    fn default() -> Self {
        Self::rusanov()
    }
}

impl<M: FvMesh> FluxPlugin<M> {
    pub fn rusanov() -> Self {
        Self { flux: Arc::new(cfd_riemann::Rusanov), _marker: PhantomData }
    }
    pub fn hllc() -> Self {
        Self { flux: Arc::new(cfd_riemann::Hllc), _marker: PhantomData }
    }
}

impl<M: FvMesh> Plugin for FluxPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(RiemannFluxResource(self.flux.clone()));
        // The generic (mesh-agnostic) inviscid sweep. The fast path is the
        // alternative `FastFluxPlugin`.
        app.add_update_system(flux_system::<M>, MeshScheduleSet::Flux);
    }
    fn provides(&self) -> Vec<&str> {
        vec!["riemann_flux"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "mesh"]
    }
}

/// Alternative to [`FluxPlugin`]: 2nd-order **MUSCL** reconstruction. Registers
/// the slope buffer ([`MusclSlopes`]), the slope [`SlopeLimiter`], the limited
/// slope pass (`PreFlux`, after the boundary ghosts are filled), and the
/// reconstructing flux (`Flux`). Generic over the [`CartesianMesh`], so the same
/// plugin serves the uniform grid and the p4est forest. Use *instead of*
/// [`FluxPlugin`]. The limiter defaults to [`SlopeLimiter::Minmod`]; select
/// another with [`with_limiter`](MusclPlugin::with_limiter).
pub struct MusclPlugin<M: CartesianMesh> {
    flux: Arc<dyn RiemannFlux>,
    limiter: SlopeLimiter,
    _marker: PhantomData<fn() -> M>,
}

impl<M: CartesianMesh> MusclPlugin<M> {
    pub fn rusanov() -> Self {
        Self { flux: Arc::new(cfd_riemann::Rusanov), limiter: SlopeLimiter::default(), _marker: PhantomData }
    }
    pub fn hllc() -> Self {
        Self { flux: Arc::new(cfd_riemann::Hllc), limiter: SlopeLimiter::default(), _marker: PhantomData }
    }
    /// Select the slope limiter (default [`SlopeLimiter::Minmod`]). Use
    /// [`SlopeLimiter::MonotonizedCentral`] for full 2nd-order accuracy on smooth
    /// flow.
    pub fn with_limiter(mut self, limiter: SlopeLimiter) -> Self {
        self.limiter = limiter;
        self
    }
}

impl<M: CartesianMesh> Plugin for MusclPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(RiemannFluxResource(self.flux.clone()));
        app.add_resource(MusclSlopes::default());
        app.add_resource(self.limiter);
        app.add_update_system(compute_slopes_system::<M>, MeshScheduleSet::PreFlux);
        app.add_update_system(muscl_flux_system::<M>, MeshScheduleSet::Flux);
    }
    fn provides(&self) -> Vec<&str> {
        vec!["riemann_flux"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "mesh"]
    }
}

/// Alternative to [`FluxPlugin`]: the **structured fast path** for a concrete flux
/// scheme `F` (Rusanov / HLLC). Same physics as `FluxPlugin`, monomorphized + cached.
pub struct FastFluxPlugin<F> {
    scheme: F,
}

impl<F: FluxScheme + Clone + Send + Sync + 'static> FastFluxPlugin<F> {
    pub fn new(scheme: F) -> Self {
        Self { scheme }
    }
}

impl<F: FluxScheme + RiemannFlux + Clone + Send + Sync + 'static> Plugin for FastFluxPlugin<F> {
    fn build(&self, app: &mut App) {
        // The flux scheme is also exposed as a `dyn RiemannFlux` resource so the
        // boundary/other consumers that take `RiemannFluxResource` still work.
        app.add_resource(RiemannFluxResource(Arc::new(self.scheme.clone())));
        app.add_resource(self.scheme.clone());
        app.add_resource(SolverScratch::default());
        app.add_update_system(
            flux_fast_system::<field_core::UniformMesh, F>,
            MeshScheduleSet::Flux,
        );
    }
    fn provides(&self) -> Vec<&str> {
        vec!["riemann_flux"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "structured_mesh"]
    }
}

/// Adds the [`TimeIntegratorResource`] + [`SolverState`]. Euler by default.
pub struct IntegratorPlugin {
    integrator: Arc<dyn TimeIntegrator>,
}

impl IntegratorPlugin {
    pub fn euler() -> Self {
        Self { integrator: Arc::new(EulerIntegrator) }
    }
    pub fn rk3() -> Self {
        Self { integrator: Arc::new(Rk3Integrator) }
    }
}

impl Plugin for IntegratorPlugin {
    fn build(&self, app: &mut App) {
        app.add_resource(TimeIntegratorResource(self.integrator.clone()));
        app.add_resource(SolverState::default());
    }
    fn provides(&self) -> Vec<&str> {
        vec!["time_integrator"]
    }
}

/// Wires the inviscid solver systems into [`MeshScheduleSet`], generic over the
/// (structured) mesh type. The phase decomposition that replaces toy-cfd's fused
/// `do_full_step`: CFL in `Setup`, halo in `Halo`, BC in `Boundary`, flux in
/// `Flux`, update in `Update`.
pub struct SolverPlugin<M: StructuredMesh> {
    config: SolverConfig,
    _marker: PhantomData<fn() -> M>,
}

impl<M: StructuredMesh> SolverPlugin<M> {
    pub fn new(config: SolverConfig) -> Self {
        Self { config, _marker: PhantomData }
    }
}

impl<M: StructuredMesh> Plugin for SolverPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(self.config);
        app.add_update_system(cfl_system::<M>, MeshScheduleSet::Setup);
        app.add_update_system(halo_system::<M>, MeshScheduleSet::Halo);
        app.add_update_system(zero_rhs_system::<M>, MeshScheduleSet::PreFlux);
        // The inviscid flux system is registered by the flux plugin (`FluxPlugin`
        // or `FastFluxPlugin`); SolverPlugin only requires `riemann_flux`.
        if self.config.viscous {
            app.add_update_system(viscous_system::<M>, MeshScheduleSet::Flux);
        }
        app.add_update_system(update_system, MeshScheduleSet::Update);
    }

    fn provides(&self) -> Vec<&str> {
        vec!["solver"]
    }

    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "riemann_flux", "time_integrator", "mesh", "boundary_conditions"]
    }
}

/// **Mesh-generic** inviscid core, bounded only to [`FvMesh`] — the assembly-layer
/// counterpart of [`SolverPlugin`] for meshes that are *not* `StructuredMesh`
/// (octree/forest AMR). It registers everything the generic `FvMesh` path needs:
/// field resize, [`CfdState`] + its initial condition, generic CFL, halo, zero-rhs,
/// and the update. Pair it with a flux plugin (`FluxPlugin::<M>` or
/// `MusclPlugin::<M>`), [`IdealGasPlugin`], [`IntegratorPlugin`], and
/// `BoundaryPlugin::<M>`.
///
/// This is what lets a forest app assemble through plugins instead of hand-wiring
/// systems. (It does **not** add the viscous terms or the structured fast path —
/// those require `StructuredMesh` and stay on [`SolverPlugin`].)
pub struct CfdCorePlugin<M: FvMesh> {
    config: SolverConfig,
    init: Arc<dyn Fn(Vec3) -> ConsVar + Send + Sync>,
    _marker: PhantomData<fn() -> M>,
}

impl<M: FvMesh> CfdCorePlugin<M> {
    pub fn new(config: SolverConfig, init: impl Fn(Vec3) -> ConsVar + Send + Sync + 'static) -> Self {
        Self { config, init: Arc::new(init), _marker: PhantomData }
    }
}

impl<M: FvMesh> Plugin for CfdCorePlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(self.config);
        field_core::register_field_data!(app, CfdState::default());
        app.add_resource(InitState(self.init.clone()));
        // Size fields to the mesh, then fill the IC, then the per-step systems.
        app.add_setup_system(field_core::resize_fields::<M>, ScheduleSetupSet::Setup);
        app.add_setup_system(init_state_system::<M>, ScheduleSetupSet::PostSetup);
        app.add_update_system(cfl_generic_system::<M>, MeshScheduleSet::Setup);
        app.add_update_system(halo_system::<M>, MeshScheduleSet::Halo);
        app.add_update_system(zero_rhs_system::<M>, MeshScheduleSet::PreFlux);
        app.add_update_system(update_system, MeshScheduleSet::Update);
    }
    fn provides(&self) -> Vec<&str> {
        vec!["cfd_state", "solver"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["field_registry", "mesh", "eos", "riemann_flux", "time_integrator", "boundary_conditions"]
    }
}

#[cfg(test)]
mod tests;
