//! Numerical flux functions for the compressible Euler equations.
//!
//! Defines the [`RiemannFlux`] trait and ships [`Rusanov`] as the default
//! implementation. Adding a new scheme (HLLC, exact Godunov, AUSM+, …) is a
//! matter of writing one struct that implements [`RiemannFlux`] and exposing a
//! plugin that registers it in the [`RiemannFluxResource`].

// The cached fast-path flux (`flux_scheme_cached`) intrinsically takes L/R
// states, their primitives, both sound speeds, the EOS, and a direction — the
// whole point is to pass the per-substep cache in. Ported verbatim from toy-cfd.
#![allow(clippy::too_many_arguments)]

use cfd_eos::Eos;
use cfd_state::{ConsVar, PrimVar};

// ─── Physical (Euler) flux ───────────────────────────────────────────────────

/// Physical (Euler) flux in direction `dir` (0=x, 1=y, 2=z). Generic over EOS
/// so the compiler can monomorphize and inline `eos.pressure(...)` at the
/// call site (concrete EOS) or fall back to vtable dispatch (`dyn Eos`).
#[inline]
pub fn physical_flux_inline<E: Eos + ?Sized>(u: &ConsVar, eos: &E, dir: usize) -> ConsVar {
    let p = eos.pressure(u);
    let vel = u.velocity();
    let v_n = vel[dir];

    let mut f = ConsVar {
        rho: u.rho * v_n,
        rho_u: u.rho_u * v_n,
        rho_v: u.rho_v * v_n,
        rho_w: u.rho_w * v_n,
        rho_e: (u.rho_e + p) * v_n,
    };

    match dir {
        0 => f.rho_u += p,
        1 => f.rho_v += p,
        2 => f.rho_w += p,
        _ => unreachable!(),
    }

    f
}

/// Convenience alias preserving the original `&dyn Eos` API.
pub fn physical_flux(u: &ConsVar, eos: &dyn Eos, dir: usize) -> ConsVar {
    physical_flux_inline(u, eos, dir)
}

// ─── RiemannFlux trait ───────────────────────────────────────────────────────

/// A numerical flux function for an interface between two cells.
pub trait RiemannFlux: Send + Sync {
    /// Compute the upwind flux across a face given left and right states.
    fn flux(&self, left: &ConsVar, right: &ConsVar, eos: &dyn Eos, dir: usize) -> ConsVar;

    /// Numerical flux **dotted with a face's outward unit normal**, for FIELD's
    /// mesh-agnostic FVM loop. `owner`/`other` are the two cell states a
    /// `field_core::Face` connects; `normal` is the unit normal pointing
    /// `owner → other`. The result is `F · n̂_owner` (per unit area): the flux
    /// loop scales it by `area / cell_volume` and subtracts it from `owner`'s
    /// RHS (and adds it to `other`'s, for an interior face).
    ///
    /// The default implementation works for any normal. On an **axis-aligned**
    /// normal (every face of a structured Cartesian mesh) it recovers the axis
    /// `dir` and sign and reuses the direction-indexed [`flux`](Self::flux), so
    /// the numerics are bit-for-bit those of toy-cfd's fused sweep. A `+axis`
    /// face is `flux(owner, other, dir)`; a `-axis` face is
    /// `-flux(other, owner, dir)` (owner sits on the `+` side of the interface,
    /// so the upwind ordering flips and the outward normal negates the result).
    ///
    /// On any other normal (unstructured or rotated faces) it uses the rotational
    /// invariance of the Euler equations: rotate both states' momentum into the
    /// face frame `(n̂, t̂₁, t̂₂)`, solve the 1-D problem along x, and rotate the
    /// momentum flux back. Every scheme that implements `flux` gets this for free.
    fn flux_normal(
        &self,
        owner: &ConsVar,
        other: &ConsVar,
        eos: &dyn Eos,
        normal: [f64; 3],
    ) -> ConsVar {
        if let Some((dir, sign)) = axis_aligned(&normal) {
            return if sign >= 0.0 {
                self.flux(owner, other, eos, dir)
            } else {
                self.flux(other, owner, eos, dir) * -1.0
            };
        }
        let frame = face_frame(normal);
        let f = self.flux(&to_frame(owner, &frame), &to_frame(other, &frame), eos, 0);
        from_frame(&f, &frame)
    }

    fn name(&self) -> &'static str { "<unnamed>" }
}

/// `Some((axis, sign))` when `n` is axis-aligned (to a relative 1e-12), else
/// `None`.
#[inline]
pub fn axis_aligned(n: &[f64; 3]) -> Option<(usize, f64)> {
    let (a, b, c) = (n[0].abs(), n[1].abs(), n[2].abs());
    let max = a.max(b).max(c);
    if max > 0.0 && (a + b + c - max) <= 1e-12 * max {
        Some(axis_and_sign(n))
    } else {
        None
    }
}

/// An orthonormal frame `[n̂, t̂₁, t̂₂]` (rows) whose first axis is the unit
/// vector `n`.
pub fn face_frame(n: [f64; 3]) -> [[f64; 3]; 3] {
    // Pick the coordinate axis least aligned with n to build t̂₁.
    let a = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    };
    let t1 = cross(n, a);
    let l = (t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]).sqrt();
    let t1 = [t1[0] / l, t1[1] / l, t1[2] / l];
    [n, t1, cross(n, t1)]
}

/// Rotate a state's momentum into `frame` (rows are the new axes).
pub fn to_frame(u: &ConsVar, frame: &[[f64; 3]; 3]) -> ConsVar {
    let m = [u.rho_u, u.rho_v, u.rho_w];
    let r = frame.map(|e| e[0] * m[0] + e[1] * m[1] + e[2] * m[2]);
    ConsVar { rho: u.rho, rho_u: r[0], rho_v: r[1], rho_w: r[2], rho_e: u.rho_e }
}

/// Rotate a flux's momentum components back out of `frame`.
pub fn from_frame(f: &ConsVar, frame: &[[f64; 3]; 3]) -> ConsVar {
    let m = [f.rho_u, f.rho_v, f.rho_w];
    let back = |d: usize| frame[0][d] * m[0] + frame[1][d] * m[1] + frame[2][d] * m[2];
    ConsVar { rho: f.rho, rho_u: back(0), rho_v: back(1), rho_w: back(2), rho_e: f.rho_e }
}

/// Recover the dominant axis (0=x, 1=y, 2=z) and its sign (`+1`/`-1`) from a
/// face normal. For an axis-aligned mesh exactly one component is nonzero.
#[inline]
pub fn axis_and_sign(n: &[f64; 3]) -> (usize, f64) {
    debug_assert!(
        {
            let (a, b, c) = (n[0].abs(), n[1].abs(), n[2].abs());
            let max = a.max(b).max(c);
            max > 0.0 && (a + b + c - max) <= 1e-9 * max
        },
        "flux_normal default handles axis-aligned normals only; got {n:?}"
    );
    let mut dir = 0;
    let mut maxabs = n[0].abs();
    if n[1].abs() > maxabs {
        dir = 1;
        maxabs = n[1].abs();
    }
    if n[2].abs() > maxabs {
        dir = 2;
    }
    (dir, if n[dir] >= 0.0 { 1.0 } else { -1.0 })
}

// ─── FluxScheme: monomorphizable companion to RiemannFlux ────────────────────

/// Marker trait for inline-callable Riemann fluxes. Companion to
/// [`RiemannFlux`] (which is dyn-dispatched and object-safe). `FluxScheme`
/// has a generic method, which makes it **not** object-safe — its sole
/// purpose is to bound generic functions like `compute_inviscid_rhs<F:
/// FluxScheme>` so the EOS calls inside the flux body monomorphize at the
/// call site instead of going through a vtable.
pub trait FluxScheme {
    fn flux_scheme<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar;

    /// Fast path with pre-computed primitives + sound speeds for the L/R
    /// states. Caller (`cfd_solver::compute_inviscid_rhs`) populates these
    /// once per substep into `SolverScratch` and passes them per face,
    /// avoiding the ~2 sound-speed sqrts a naive scheme would redo per face.
    ///
    /// Default impl falls back to [`Self::flux_scheme`]; schemes override
    /// to actually use the cache (HLLC, Rusanov).
    #[inline]
    fn flux_scheme_cached<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        _left_prim: &PrimVar,
        _right_prim: &PrimVar,
        _a_l: f64,
        _a_r: f64,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        self.flux_scheme(left, right, eos, dir)
    }
}

// ─── Rusanov (Local Lax-Friedrichs) ──────────────────────────────────────────

/// Rusanov / Local Lax-Friedrichs flux:
/// `F = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)`.
#[derive(Clone, Copy, Default)]
pub struct Rusanov;

impl Rusanov {
    /// Generic, monomorphizable flux. Hot-path callers (`compute_inviscid_rhs`)
    /// take `&Rusanov` directly and call this method, letting the compiler
    /// inline the entire flux body and devirtualize all EOS calls. The
    /// `RiemannFlux` trait impl below delegates here for `&dyn` callers.
    #[inline]
    pub fn flux_inline<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        let f_l = physical_flux_inline(left, eos, dir);
        let f_r = physical_flux_inline(right, eos, dir);

        let vel_l = left.velocity();
        let vel_r = right.velocity();
        let a_l = eos.sound_speed(left);
        let a_r = eos.sound_speed(right);

        let lambda = f64::max(vel_l[dir].abs() + a_l, vel_r[dir].abs() + a_r);

        0.5 * (f_l + f_r) - 0.5 * lambda * (*right - *left)
    }
}

impl RiemannFlux for Rusanov {
    fn flux(&self, left: &ConsVar, right: &ConsVar, eos: &dyn Eos, dir: usize) -> ConsVar {
        // Generic over `dyn Eos` here — keeps the trait-object API working,
        // pays the vtable cost for callers that need the abstraction. Hot-path
        // callers should use `flux_inline` instead.
        self.flux_inline(left, right, eos, dir)
    }

    fn name(&self) -> &'static str { "Rusanov" }
}

impl FluxScheme for Rusanov {
    #[inline(always)]
    fn flux_scheme<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        self.flux_inline(left, right, eos, dir)
    }

    /// Cached fast path: skips the two `eos.sound_speed(...)` sqrts inside
    /// [`Self::flux_inline`] by reading from caller's pre-computed cache.
    /// `physical_flux_inline` still needs the conservative state for the
    /// flux vector (uses ρ, ρu, ρe directly), so we keep `left` / `right`.
    #[inline(always)]
    fn flux_scheme_cached<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        left_prim: &PrimVar,
        right_prim: &PrimVar,
        a_l: f64,
        a_r: f64,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        let f_l = physical_flux_inline(left, eos, dir);
        let f_r = physical_flux_inline(right, eos, dir);

        let v_l = [left_prim.u, left_prim.v, left_prim.w][dir];
        let v_r = [right_prim.u, right_prim.v, right_prim.w][dir];
        let lambda = f64::max(v_l.abs() + a_l, v_r.abs() + a_r);

        0.5 * (f_l + f_r) - 0.5 * lambda * (*right - *left)
    }
}

// ─── HLLC (Harten-Lax-van Leer-Contact) ──────────────────────────────────────

/// HLLC three-wave Riemann solver with Thornber low-Mach correction.
///
/// Resolves the contact discontinuity (which Rusanov / LLF smears by an
/// O(λ_max) dissipation term), so flow features like wakes, boundary layers,
/// and surface pressure gradients stay sharp. Standard recommendation in
/// compressible CFD when shocks + contact discontinuities both matter.
///
/// Algorithm: Toro, *Riemann Solvers and Numerical Methods for Fluid
/// Dynamics*, ch. 10. Wave-speed estimate: simple Davis bound
/// `S_L = min(u_L - a_L, u_R - a_R)`, `S_R = max(u_L + a_L, u_R + a_R)` —
/// robust for non-vacuum cases. Contact speed `S_*` from the
/// pressure-velocity jump relations. Star-region flux from
/// `F_*K = F_K + S_K (U_*K - U_K)` with `U_*K` reconstructed from the
/// contact-state algebra.
///
/// **Low-Mach correction** (Thornber, Mosedale, Drikakis & Youngs,
/// JCP 2008): the velocity jump `u_L - u_R` used in the reconstruction is
/// attenuated by `z = min(M_local, 1)`, where `M_local` is the larger of
/// `|v_L|/a_max`, `|v_R|/a_max`. At high Mach (`z=1`) this is a no-op; at
/// low Mach (`z<1`) it removes the `O(1/M)` excess kinetic-energy
/// dissipation that thickens boundary layers and depresses skin-friction
/// drag. Internal energy (and hence pressure) is preserved exactly through
/// the rebuild, so the modification only affects the kinetic part of the
/// upwind reconstruction.
#[derive(Clone, Copy, Default)]
pub struct Hllc;

/// Textbook HLLC — same wave structure as [`Hllc`] but without the
/// Thornber low-Mach correction. Used as a diagnostic when Thornber's
/// reduced dissipation is suspected of producing nonphysical
/// steady states (e.g., the leading-edge wrong-attractor problem in
/// the cylinder cut-cell case).
#[derive(Clone, Copy, Default)]
pub struct HllcTextbook;

impl HllcTextbook {
    #[inline]
    pub fn flux_inline<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        Hllc::flux_with_thornber(left, right, eos, dir, false)
    }
}

impl RiemannFlux for HllcTextbook {
    fn flux(&self, left: &ConsVar, right: &ConsVar, eos: &dyn Eos, dir: usize) -> ConsVar {
        self.flux_inline(left, right, eos, dir)
    }
    fn name(&self) -> &'static str { "HllcTextbook" }
}

impl FluxScheme for HllcTextbook {
    #[inline(always)]
    fn flux_scheme<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        self.flux_inline(left, right, eos, dir)
    }

    #[inline(always)]
    fn flux_scheme_cached<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        _left_prim: &PrimVar,
        _right_prim: &PrimVar,
        _a_l: f64,
        _a_r: f64,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        self.flux_inline(left, right, eos, dir)
    }
}

impl Hllc {
    #[inline]
    pub fn flux_inline<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        Self::flux_with_thornber(left, right, eos, dir, true)
    }

    /// HLLC with Thornber low-Mach correction toggleable. `thornber=false`
    /// gives textbook HLLC (full upwinding velocity jump). Useful as a
    /// diagnostic when low-Mach undersmoothing is suspected of producing
    /// wrong-attractor states near walls.
    #[inline]
    pub fn flux_with_thornber<E: Eos + ?Sized>(
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
        thornber: bool,
    ) -> ConsVar {
        let (left_c, right_c, a_l, a_r) = if thornber {
            thornber_reconstruct(left, right, eos)
        } else {
            // Textbook HLLC: no velocity-jump attenuation. Returns the
            // input states unchanged plus their sound speeds.
            let a_l = eos.sound_speed(left);
            let a_r = eos.sound_speed(right);
            (*left, *right, a_l, a_r)
        };

        let vel_l = left_c.velocity();
        let vel_r = right_c.velocity();
        let p_l = eos.pressure(&left_c);
        let p_r = eos.pressure(&right_c);
        let un_l = vel_l[dir];
        let un_r = vel_r[dir];

        // Wave-speed estimates (Davis).
        let s_l = (un_l - a_l).min(un_r - a_r);
        let s_r = (un_l + a_l).max(un_r + a_r);

        // Contact-wave speed from pressure-velocity jump.
        let factor_l = left_c.rho * (s_l - un_l);
        let factor_r = right_c.rho * (s_r - un_r);
        let s_star = ((p_r - p_l) + factor_l * un_l - factor_r * un_r) / (factor_l - factor_r);

        // Region-based flux selection.
        if s_l >= 0.0 {
            physical_flux_inline(&left_c, eos, dir)
        } else if s_star >= 0.0 {
            let u_star_l = star_state(&left_c, &vel_l, p_l, un_l, s_l, s_star, dir);
            physical_flux_inline(&left_c, eos, dir) + s_l * (u_star_l - left_c)
        } else if s_r >= 0.0 {
            let u_star_r = star_state(&right_c, &vel_r, p_r, un_r, s_r, s_star, dir);
            physical_flux_inline(&right_c, eos, dir) + s_r * (u_star_r - right_c)
        } else {
            physical_flux_inline(&right_c, eos, dir)
        }
    }
}

/// Thornber low-Mach reconstruction. Attenuates the velocity jump by
/// `z = min(M_local, 1)` where `M_local = max(|v_L|, |v_R|) / max(a_L, a_R)`.
/// Returns the modified L/R states with rebuilt momentum + total energy
/// (internal energy preserved exactly so pressure is unchanged), and the
/// L/R sound speeds — Thornber preserves ρ + p, so `sound_speed(left_c) ==
/// sound_speed(left)` and the caller can reuse these instead of paying two
/// extra sqrts.
///
/// Two sqrt micro-optimizations vs the textbook formulation:
/// 1. `max(|v_L|, |v_R|) = sqrt(max(v_L², v_R²))` — one sqrt instead of two.
/// 2. `M_local ≥ 1 ⇔ v_max² ≥ a_max²` — when supersonic we set z = 1
///    directly, skipping the sqrt + division entirely.
#[inline]
fn thornber_reconstruct<E: Eos + ?Sized>(
    left: &ConsVar,
    right: &ConsVar,
    eos: &E,
) -> (ConsVar, ConsVar, f64, f64) {
    let vel_l = left.velocity();
    let vel_r = right.velocity();
    let v_l_sq = vel_l[0] * vel_l[0] + vel_l[1] * vel_l[1] + vel_l[2] * vel_l[2];
    let v_r_sq = vel_r[0] * vel_r[0] + vel_r[1] * vel_r[1] + vel_r[2] * vel_r[2];
    let a_l = eos.sound_speed(left);
    let a_r = eos.sound_speed(right);

    let v_max_sq = v_l_sq.max(v_r_sq);
    let a_max = a_l.max(a_r);
    let z = if v_max_sq >= a_max * a_max {
        1.0
    } else {
        v_max_sq.sqrt() / a_max
    };

    // Half-sum + attenuated half-jump.
    let half = 0.5;
    let avg0 = half * (vel_l[0] + vel_r[0]);
    let avg1 = half * (vel_l[1] + vel_r[1]);
    let avg2 = half * (vel_l[2] + vel_r[2]);
    let dvel0 = half * z * (vel_l[0] - vel_r[0]);
    let dvel1 = half * z * (vel_l[1] - vel_r[1]);
    let dvel2 = half * z * (vel_l[2] - vel_r[2]);
    let vel_lc = [avg0 + dvel0, avg1 + dvel1, avg2 + dvel2];
    let vel_rc = [avg0 - dvel0, avg1 - dvel1, avg2 - dvel2];

    let v_lc_sq = vel_lc[0] * vel_lc[0] + vel_lc[1] * vel_lc[1] + vel_lc[2] * vel_lc[2];
    let v_rc_sq = vel_rc[0] * vel_rc[0] + vel_rc[1] * vel_rc[1] + vel_rc[2] * vel_rc[2];

    // Preserve internal energy: rho_e_new = rho_e + 0.5*rho*(|v_new|² - |v|²).
    let left_c = ConsVar {
        rho: left.rho,
        rho_u: left.rho * vel_lc[0],
        rho_v: left.rho * vel_lc[1],
        rho_w: left.rho * vel_lc[2],
        rho_e: left.rho_e + 0.5 * left.rho * (v_lc_sq - v_l_sq),
    };
    let right_c = ConsVar {
        rho: right.rho,
        rho_u: right.rho * vel_rc[0],
        rho_v: right.rho * vel_rc[1],
        rho_w: right.rho * vel_rc[2],
        rho_e: right.rho_e + 0.5 * right.rho * (v_rc_sq - v_r_sq),
    };
    (left_c, right_c, a_l, a_r)
}

impl RiemannFlux for Hllc {
    fn flux(&self, left: &ConsVar, right: &ConsVar, eos: &dyn Eos, dir: usize) -> ConsVar {
        self.flux_inline(left, right, eos, dir)
    }
    fn name(&self) -> &'static str { "HLLC" }
}

impl FluxScheme for Hllc {
    #[inline(always)]
    fn flux_scheme<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        self.flux_inline(left, right, eos, dir)
    }

    /// Cached fast path: takes pre-computed `left_prim` / `right_prim` /
    /// `a_l` / `a_r` from the caller's substep cache. Internally still does
    /// Thornber's velocity blend (it depends on `a_l`/`a_r` already) but
    /// skips the four `cons_to_prim` + four `sound_speed` calls a fresh
    /// `flux_inline(...)` would redo.
    #[inline(always)]
    fn flux_scheme_cached<E: Eos + ?Sized>(
        &self,
        left: &ConsVar,
        right: &ConsVar,
        left_prim: &PrimVar,
        right_prim: &PrimVar,
        a_l: f64,
        a_r: f64,
        eos: &E,
        dir: usize,
    ) -> ConsVar {
        // Thornber blend, given cached velocities + sound speeds.
        let vel_l = [left_prim.u, left_prim.v, left_prim.w];
        let vel_r = [right_prim.u, right_prim.v, right_prim.w];
        let v_l_sq = vel_l[0] * vel_l[0] + vel_l[1] * vel_l[1] + vel_l[2] * vel_l[2];
        let v_r_sq = vel_r[0] * vel_r[0] + vel_r[1] * vel_r[1] + vel_r[2] * vel_r[2];
        let v_max_sq = v_l_sq.max(v_r_sq);
        let a_max = a_l.max(a_r);
        let z = if v_max_sq >= a_max * a_max { 1.0 } else { v_max_sq.sqrt() / a_max };

        let half = 0.5;
        let avg = [
            half * (vel_l[0] + vel_r[0]),
            half * (vel_l[1] + vel_r[1]),
            half * (vel_l[2] + vel_r[2]),
        ];
        let dvel = [
            half * z * (vel_l[0] - vel_r[0]),
            half * z * (vel_l[1] - vel_r[1]),
            half * z * (vel_l[2] - vel_r[2]),
        ];
        let vel_lc = [avg[0] + dvel[0], avg[1] + dvel[1], avg[2] + dvel[2]];
        let vel_rc = [avg[0] - dvel[0], avg[1] - dvel[1], avg[2] - dvel[2]];
        let v_lc_sq = vel_lc[0] * vel_lc[0] + vel_lc[1] * vel_lc[1] + vel_lc[2] * vel_lc[2];
        let v_rc_sq = vel_rc[0] * vel_rc[0] + vel_rc[1] * vel_rc[1] + vel_rc[2] * vel_rc[2];

        // Rebuild conservative states with reconstructed velocities (preserves
        // ρ + p, hence sound speed — so a_l / a_r still apply).
        let left_c = ConsVar {
            rho: left.rho,
            rho_u: left.rho * vel_lc[0],
            rho_v: left.rho * vel_lc[1],
            rho_w: left.rho * vel_lc[2],
            rho_e: left.rho_e + 0.5 * left.rho * (v_lc_sq - v_l_sq),
        };
        let right_c = ConsVar {
            rho: right.rho,
            rho_u: right.rho * vel_rc[0],
            rho_v: right.rho * vel_rc[1],
            rho_w: right.rho * vel_rc[2],
            rho_e: right.rho_e + 0.5 * right.rho * (v_rc_sq - v_r_sq),
        };

        // HLLC body — using cached pressure (Thornber preserves p) and the
        // already-known a_l / a_r.
        let p_l = left_prim.p;
        let p_r = right_prim.p;
        let un_l = vel_lc[dir];
        let un_r = vel_rc[dir];

        let s_l = (un_l - a_l).min(un_r - a_r);
        let s_r = (un_l + a_l).max(un_r + a_r);

        let factor_l = left_c.rho * (s_l - un_l);
        let factor_r = right_c.rho * (s_r - un_r);
        let s_star = ((p_r - p_l) + factor_l * un_l - factor_r * un_r) / (factor_l - factor_r);

        if s_l >= 0.0 {
            physical_flux_inline(&left_c, eos, dir)
        } else if s_star >= 0.0 {
            let u_star_l = star_state(&left_c, &vel_lc, p_l, un_l, s_l, s_star, dir);
            physical_flux_inline(&left_c, eos, dir) + s_l * (u_star_l - left_c)
        } else if s_r >= 0.0 {
            let u_star_r = star_state(&right_c, &vel_rc, p_r, un_r, s_r, s_star, dir);
            physical_flux_inline(&right_c, eos, dir) + s_r * (u_star_r - right_c)
        } else {
            physical_flux_inline(&right_c, eos, dir)
        }
    }
}

/// Reconstruct the HLLC star-region conservative state on side `K` ∈ {L, R}.
/// Tangential velocity components carry through unchanged; the normal
/// component becomes `S_*`; density and energy follow from the contact-state
/// algebra. See Toro §10.4.
#[inline]
fn star_state(
    u: &ConsVar,
    vel: &[f64; 3],
    p: f64,
    un: f64,
    s: f64,
    s_star: f64,
    dir: usize,
) -> ConsVar {
    let rho_star = u.rho * (s - un) / (s - s_star);
    let energy_per_mass = u.rho_e / u.rho;
    let e_star = energy_per_mass + (s_star - un) * (s_star + p / (u.rho * (s - un)));

    // Normal momentum becomes ρ_* · S_*; tangential stay as ρ_* · v_K.
    let rho_u = if dir == 0 { rho_star * s_star } else { rho_star * vel[0] };
    let rho_v = if dir == 1 { rho_star * s_star } else { rho_star * vel[1] };
    let rho_w = if dir == 2 { rho_star * s_star } else { rho_star * vel[2] };

    ConsVar {
        rho: rho_star,
        rho_u,
        rho_v,
        rho_w,
        rho_e: rho_star * e_star,
    }
}

/// Free-function form of the Rusanov flux. Kept for direct callers; new code
/// should go through [`RiemannFlux`] / [`RiemannFluxResource`].
pub fn rusanov_flux(left: &ConsVar, right: &ConsVar, eos: &dyn Eos, dir: usize) -> ConsVar {
    Rusanov.flux(left, right, eos, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_eos::IdealGas;
    use cfd_state::PrimVar;

    #[test]
    fn flux_normal_is_rotation_invariant_for_slanted_faces() {
        let eos = IdealGas::air();
        // A 1-D problem along x, then the same problem rotated so x maps to n.
        let (c, s) = (0.6f64, 0.8f64);
        let rot = |v: [f64; 3]| [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
        for (va, vb) in [([30.0, 5.0, -2.0], [10.0, -4.0, 1.0]), ([-80.0, 0.0, 0.0], [120.0, 3.0, 0.0])] {
            let a = eos.prim_to_cons(&PrimVar::new(1.0, va[0], va[1], va[2], 100000.0, 348.4));
            let b = eos.prim_to_cons(&PrimVar::new(0.5, vb[0], vb[1], vb[2], 50000.0, 348.4));
            let (ra, rb) = (rot(va), rot(vb));
            let a2 = eos.prim_to_cons(&PrimVar::new(1.0, ra[0], ra[1], ra[2], 100000.0, 348.4));
            let b2 = eos.prim_to_cons(&PrimVar::new(0.5, rb[0], rb[1], rb[2], 50000.0, 348.4));
            let n = rot([1.0, 0.0, 0.0]);
            let schemes: [&dyn RiemannFlux; 3] = [&Rusanov, &HllcTextbook, &Hllc];
            for scheme in schemes {
                let f = scheme.flux(&a, &b, &eos, 0);
                let g = scheme.flux_normal(&a2, &b2, &eos, n);
                let fm = rot([f.rho_u, f.rho_v, f.rho_w]);
                let tol = 1e-9 * (f.rho_e.abs() + 1.0);
                assert!((g.rho - f.rho).abs() < tol, "{}", scheme.name());
                assert!((g.rho_e - f.rho_e).abs() < tol, "{}", scheme.name());
                assert!((g.rho_u - fm[0]).abs() < tol && (g.rho_v - fm[1]).abs() < tol, "{}", scheme.name());
                assert!((g.rho_w - fm[2]).abs() < tol, "{}", scheme.name());
            }
        }
    }

    #[test]
    fn flux_normal_matches_directional_flux() {
        let eos = IdealGas::air();
        let a = eos.prim_to_cons(&PrimVar::new(1.0, 30.0, 0.0, 0.0, 100000.0, 348.4));
        let b = eos.prim_to_cons(&PrimVar::new(0.5, 10.0, 0.0, 0.0, 50000.0, 348.4));

        // +x face: flux_normal(owner=a, other=b, +x) == flux(a, b, dir=0)
        let fn_px = Rusanov.flux_normal(&a, &b, &eos, [1.0, 0.0, 0.0]);
        let f_dir = Rusanov.flux(&a, &b, &eos, 0);
        assert!((fn_px.rho - f_dir.rho).abs() < 1e-12);
        assert!((fn_px.rho_u - f_dir.rho_u).abs() < 1e-12);

        // -x face: owner sits on the + side, so flux_normal == -flux(other, owner, 0)
        let fn_mx = Rusanov.flux_normal(&a, &b, &eos, [-1.0, 0.0, 0.0]);
        let f_flip = Rusanov.flux(&b, &a, &eos, 0) * -1.0;
        assert!((fn_mx.rho - f_flip.rho).abs() < 1e-12);
        assert!((fn_mx.rho_e - f_flip.rho_e).abs() < 1e-12);
    }

    #[test]
    fn flux_normal_conservative_between_two_cells() {
        // The flux a +x face scatters out of `owner` must equal what the same
        // interface scatters into `other` — i.e. owner's outward flux is the
        // negative of other's outward flux across the shared face.
        let eos = IdealGas::air();
        let a = eos.prim_to_cons(&PrimVar::new(1.0, 50.0, 0.0, 0.0, 100000.0, 348.4));
        let b = eos.prim_to_cons(&PrimVar::new(0.4, -20.0, 0.0, 0.0, 40000.0, 348.4));
        // owner=a sees +x normal; other=b sees the same face as its -x normal.
        let from_owner = Rusanov.flux_normal(&a, &b, &eos, [1.0, 0.0, 0.0]);
        let from_other = Rusanov.flux_normal(&b, &a, &eos, [-1.0, 0.0, 0.0]);
        // Conservation: rhs[owner] -= from_owner, rhs[other] -= from_other, and
        // the two must cancel (mass leaving owner == mass entering other).
        assert!((from_owner.rho + from_other.rho).abs() < 1e-12);
        assert!((from_owner.rho_e + from_other.rho_e).abs() < 1e-12);
    }

    #[test]
    fn flux_symmetry() {
        let eos = IdealGas::air();
        let prim = PrimVar::new(1.225, 100.0, 0.0, 0.0, 101325.0, 288.15);
        let cons = eos.prim_to_cons(&prim);

        let f_rusanov = Rusanov.flux(&cons, &cons, &eos, 0);
        let f_phys = physical_flux(&cons, &eos, 0);

        assert!((f_rusanov.rho - f_phys.rho).abs() < 1e-10);
        assert!((f_rusanov.rho_u - f_phys.rho_u).abs() < 1e-6);
        assert!((f_rusanov.rho_e - f_phys.rho_e).abs() < 1e-4);
    }

    #[test]
    fn rusanov_dissipation() {
        let eos = IdealGas::air();
        let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100000.0, 348.4));
        let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10000.0, 278.7));

        let f = Rusanov.flux(&left, &right, &eos, 0);
        assert!(f.rho.is_finite());
        assert!(f.rho_e.is_finite());
    }

    // ── HLLC tests ────────────────────────────────────────────────────────

    /// Uniform state: HLLC must reduce to the physical flux (no jump → no
    /// dissipation contribution from the contact wave).
    #[test]
    fn hllc_uniform_state() {
        let eos = IdealGas::air();
        let prim = PrimVar::new(1.225, 100.0, 0.0, 0.0, 101325.0, 288.15);
        let cons = eos.prim_to_cons(&prim);

        let f_hllc = Hllc.flux(&cons, &cons, &eos, 0);
        let f_phys = physical_flux(&cons, &eos, 0);

        assert!((f_hllc.rho - f_phys.rho).abs() < 1e-10, "rho diff = {}", f_hllc.rho - f_phys.rho);
        assert!((f_hllc.rho_u - f_phys.rho_u).abs() < 1e-6);
        assert!((f_hllc.rho_v - f_phys.rho_v).abs() < 1e-10);
        assert!((f_hllc.rho_w - f_phys.rho_w).abs() < 1e-10);
        assert!((f_hllc.rho_e - f_phys.rho_e).abs() < 1e-4);
    }

    /// Sod shock tube initial state: HLLC must produce a finite, sensible
    /// flux (this is a common test problem in compressible CFD).
    #[test]
    fn hllc_sod_shock_finite() {
        let eos = IdealGas::air();
        let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100000.0, 348.4));
        let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10000.0, 278.7));

        let f = Hllc.flux(&left, &right, &eos, 0);
        assert!(f.rho.is_finite());
        assert!(f.rho_u.is_finite());
        assert!(f.rho_e.is_finite());
        // For a stationary discontinuity with higher-pressure left, flux
        // should advect mass to the right (positive rho).
        assert!(f.rho > 0.0, "expected positive mass flux, got {}", f.rho);
    }

    /// Tangential velocity should be exactly preserved across all 3 directions
    /// — i.e. choosing dir=0 vs dir=1 vs dir=2 with equivalent rotated states
    /// gives equivalent rotated fluxes.
    #[test]
    fn hllc_direction_consistency() {
        let eos = IdealGas::air();
        // Same physical state, but velocity oriented along x (for dir=0) vs y (dir=1).
        let prim_x = PrimVar::new(1.0, 100.0, 50.0, 25.0, 101325.0, 288.15);
        let prim_y = PrimVar::new(1.0, 25.0, 100.0, 50.0, 101325.0, 288.15); // velocity rotated
        let cons_x = eos.prim_to_cons(&prim_x);
        let cons_y = eos.prim_to_cons(&prim_y);

        let f_x = Hllc.flux(&cons_x, &cons_x, &eos, 0);
        let f_y = Hllc.flux(&cons_y, &cons_y, &eos, 1);

        // For uniform state, both reduce to physical flux. The "normal" momentum
        // component should be the same magnitude after the rotation.
        assert!((f_x.rho - f_y.rho).abs() < 1e-10);
        // The pressure contribution adds to rho_u in dir=0 and rho_v in dir=1.
        assert!((f_x.rho_u - f_y.rho_v).abs() < 1e-6);
    }
}
