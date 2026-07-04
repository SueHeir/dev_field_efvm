//! The DEM–CFD interphase **seam**: the data contract and drag closures that
//! couple immersed particles (the SOIL/Lagrangian side) to the gas (the
//! FIELD/Eulerian side) across a `grass_multi` namespace boundary.
//!
//! FIELD is the sole mesh owner; particles are *immersed* and never own cells.
//! Each outer coupling iteration the two substrates exchange exactly two small
//! POD payloads — this is the whole contract, deliberately independent of either
//! substrate's internals so the *same* seam serves both coupling regimes:
//!
//! ```text
//!   soil ── ParticleSet     ──▶ field   (kinematics: center, velocity, radius)
//!   field ── InterphaseForces ──▶ soil   (net fluid force on each particle)
//! ```
//!
//! and, for the unresolved regime, field additionally carries a per-cell **void
//! fraction** `ε` and a **momentum sink** (the equal-and-opposite reaction to the
//! particle drag), so the two-way exchange conserves momentum.
//!
//! ## Two regimes, one contract
//!
//! - **Resolved (IBM / cut-cell).** The sphere is geometrically resolved on the
//!   mesh (via [`super::Sphere`] + the ghost-cell / cut-cell machinery); the net
//!   fluid force is the **surface-load integral** (pressure + viscous traction)
//!   over the body. Valid when `D/Δx ≳ 10`. The force handoff is validated in the
//!   `sphere_drag_resolved` example against exact references (Archimedes buoyancy
//!   and the Stokes drag `Cd = 24/Re`).
//!
//! - **Unresolved (point-particle).** The sphere is smaller than a cell; the net
//!   force is a **drag closure** ([`beta_gidaspow`] / [`sphere_drag_force`]) of the
//!   local slip velocity and void fraction, with the reaction fed back to the gas
//!   as a momentum sink. Validated in the `settling_sphere` example: a sphere
//!   released in gas settles to its terminal velocity.
//!
//! ## Drag references
//!
//! - Schiller & Naumann (1935) standard-drag correlation — [`cd_schiller_naumann`].
//! - Clift, Grace & Weber (1978) standard-drag curve — [`cd_clift_grace_weber`]
//!   (an *independent* correlation used to cross-check the above).
//! - Wen & Yu (1966) / Ergun (1952), blended per Gidaspow (1994) — [`beta_gidaspow`].
//! - Stokes (1851) terminal velocity — the `Re → 0` limit checked in the examples.

use cfd_eos::Eos;
use cfd_state::CfdState;
use field_core::{StructuredMesh, Vec3};

// ─── Seam data contract ──────────────────────────────────────────────────────

/// Kinematics of one immersed particle — the SOIL→FIELD half of the seam.
///
/// This is deliberately *not* a `soil::Atom`: the seam speaks in geometry +
/// kinematics only, so the FIELD side never depends on the particle method.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParticleKinematics {
    /// Particle center in world coordinates.
    pub center: Vec3,
    /// Particle velocity.
    pub velocity: Vec3,
    /// Particle radius.
    pub radius: f64,
}

impl ParticleKinematics {
    /// Sphere volume `4/3 π R³`.
    #[inline]
    pub fn volume(&self) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * self.radius.powi(3)
    }
    /// Frontal (projected) area `π R²`.
    #[inline]
    pub fn frontal_area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    /// Diameter `2R`.
    #[inline]
    pub fn diameter(&self) -> f64 {
        2.0 * self.radius
    }
}

/// The set of immersed particles crossing the seam SOIL→FIELD each iteration.
#[derive(Clone, Debug, Default)]
pub struct ParticleSet {
    pub particles: Vec<ParticleKinematics>,
}

/// Net fluid force on each particle, FIELD→SOIL each iteration. `force[i]` is the
/// **total** fluid load on `ParticleSet.particles[i]` — generalized buoyancy (the
/// undisturbed pressure-gradient force) plus drag — expressed in world axes.
#[derive(Clone, Debug, Default)]
pub struct InterphaseForces {
    pub force: Vec<Vec3>,
}

impl InterphaseForces {
    /// Resize to `n` particles, zeroing.
    pub fn reset(&mut self, n: usize) {
        self.force.clear();
        self.force.resize(n, [0.0; 3]);
    }
}

// ─── Small vector helpers ────────────────────────────────────────────────────

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn norm(a: Vec3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

// ─── Standard-drag correlations Cd(Re) ───────────────────────────────────────

/// Reynolds number of a sphere: `Re = ρ_f |u_rel| d / μ`.
#[inline]
pub fn particle_reynolds(rho_f: f64, rel_speed: f64, diameter: f64, mu: f64) -> f64 {
    rho_f * rel_speed * diameter / mu
}

/// Schiller & Naumann (1935) standard-drag correlation for a rigid sphere,
/// accurate to a few percent for `Re ≲ 800`:
///
/// ```text
///   Cd = 24/Re · (1 + 0.15 Re^0.687)
/// ```
///
/// Reduces to the Stokes drag `Cd = 24/Re` as `Re → 0`.
#[inline]
pub fn cd_schiller_naumann(re: f64) -> f64 {
    if re <= 0.0 {
        return f64::INFINITY;
    }
    24.0 / re * (1.0 + 0.15 * re.powf(0.687))
}

/// Clift, Grace & Weber (1978) standard-drag curve for a rigid sphere — an
/// *independent* piecewise correlation (their Table 5.2), used to cross-check
/// [`cd_schiller_naumann`]. Branches implemented here cover `Re ≲ 260`, which
/// spans the low/intermediate-Re coupling cases:
///
/// ```text
///   Re < 0.01 :  Cd = 24/Re · (1 + 3/16 Re)
///   Re < 20   :  Cd = 24/Re · (1 + 0.1315 Re^(0.82 − 0.05 w)),  w = log10 Re
///   Re < 260  :  Cd = 24/Re · (1 + 0.1935 Re^0.6305)
/// ```
#[inline]
pub fn cd_clift_grace_weber(re: f64) -> f64 {
    if re <= 0.0 {
        return f64::INFINITY;
    }
    if re < 0.01 {
        24.0 / re * (1.0 + 3.0 / 16.0 * re)
    } else if re < 20.0 {
        let w = re.log10();
        24.0 / re * (1.0 + 0.1315 * re.powf(0.82 - 0.05 * w))
    } else if re < 260.0 {
        24.0 / re * (1.0 + 0.1935 * re.powf(0.6305))
    } else {
        // Outside the branches used by the coupling cases; fall back to the
        // Schiller–Naumann fit (valid to ~800) rather than extrapolate CGW.
        cd_schiller_naumann(re)
    }
}

// ─── Single-sphere drag force (the ε → 1 unresolved limit) ───────────────────

/// Drag force on an isolated sphere in fluid moving at `u_f` relative to a
/// particle at `u_p`:
///
/// ```text
///   F = ½ Cd(Re) ρ_f (π R²) |u_f − u_p| (u_f − u_p)
/// ```
///
/// with `Cd` from `cd_fn` and `Re = ρ_f |u_f − u_p| (2R) / μ`. This is the ε → 1
/// limit of the Wen & Yu closure ([`beta_gidaspow`]); the point-particle
/// settling case exercises exactly this expression.
pub fn sphere_drag_force(
    u_fluid: Vec3,
    u_particle: Vec3,
    radius: f64,
    rho_f: f64,
    mu: f64,
    cd_fn: impl Fn(f64) -> f64,
) -> Vec3 {
    let rel = sub(u_fluid, u_particle);
    let speed = norm(rel);
    if speed <= 0.0 {
        return [0.0; 3];
    }
    let re = particle_reynolds(rho_f, speed, 2.0 * radius, mu);
    let cd = cd_fn(re);
    let area = std::f64::consts::PI * radius * radius;
    let mag = 0.5 * cd * rho_f * area * speed; // F = coeff · rel
    [mag * rel[0], mag * rel[1], mag * rel[2]]
}

// ─── Unresolved interphase momentum-exchange coefficient β ───────────────────

/// Interphase momentum-exchange coefficient `β` [kg/(m³·s)] blended per Gidaspow
/// (1994): Wen & Yu (1966) in the dilute regime (`ε > 0.8`) and Ergun (1952) in
/// the dense regime (`ε ≤ 0.8`). `cd` is the single-sphere drag coefficient at
/// the local slip Reynolds number.
///
/// ```text
///   ε > 0.8 (Wen–Yu):  β = ¾ Cd (ρ_f ε (1−ε) |u_rel| / d) ε^(−2.65)
///   ε ≤ 0.8 (Ergun) :  β = 150 (1−ε)² μ /(ε d²) + 1.75 (1−ε) ρ_f |u_rel| / d
/// ```
///
/// The drag force on ONE particle of volume `V_p` in a cell with void fraction
/// `ε` is `F = β V_p /(1−ε) · (u_f − u_p)`. As `ε → 1` this recovers the isolated
/// single-sphere drag of [`sphere_drag_force`].
pub fn beta_gidaspow(eps: f64, rho_f: f64, mu: f64, diameter: f64, rel_speed: f64, cd: f64) -> f64 {
    let eps = eps.clamp(1e-6, 1.0);
    let one_m = 1.0 - eps;
    if eps > 0.8 {
        0.75 * cd * rho_f * eps * one_m * rel_speed / diameter * eps.powf(-2.65)
    } else {
        150.0 * one_m * one_m * mu / (eps * diameter * diameter)
            + 1.75 * one_m * rho_f * rel_speed / diameter
    }
}

/// Drag force on a single particle of volume `v_p` from the Gidaspow `β`
/// (`F = β v_p /(1−ε) · u_rel`). Kept alongside [`beta_gidaspow`] so fixed-bed
/// and fluidized-bed follow-ups reuse the exact same closure the single-particle
/// case is validated against.
pub fn drag_force_from_beta(beta: f64, v_p: f64, eps: f64, u_rel: Vec3) -> Vec3 {
    let one_m = (1.0 - eps).max(1e-6);
    let c = beta * v_p / one_m;
    [c * u_rel[0], c * u_rel[1], c * u_rel[2]]
}

// ─── Terminal velocity (steady force balance) ────────────────────────────────

/// Terminal settling speed of a sphere from the steady 1-D force balance
///
/// ```text
///   (ρ_p − ρ_f) V g = ½ Cd(Re) ρ_f (π R²) v²,   Re = ρ_f v (2R)/μ
/// ```
///
/// solved by bisection with the supplied `cd_fn`. In the Stokes limit
/// (`Cd = 24/Re`) this has the closed form `v_t = (2/9)(ρ_p − ρ_f) g R²/μ`
/// (Stokes 1851), which the examples use as an independent reference.
pub fn terminal_velocity(
    rho_p: f64,
    rho_f: f64,
    radius: f64,
    g: f64,
    mu: f64,
    cd_fn: impl Fn(f64) -> f64,
) -> f64 {
    let vol = 4.0 / 3.0 * std::f64::consts::PI * radius.powi(3);
    let area = std::f64::consts::PI * radius * radius;
    let d = 2.0 * radius;
    let net = (rho_p - rho_f) * vol * g.abs(); // driving weight (buoyant)
    // residual(v) = net − ½ Cd(Re) ρ_f A v²   (monotone decreasing in v)
    let residual = |v: f64| {
        if v <= 0.0 {
            return net;
        }
        let re = particle_reynolds(rho_f, v, d, mu);
        net - 0.5 * cd_fn(re) * rho_f * area * v * v
    };
    // Bracket: v_lo = 0 (residual = net > 0). Grow v_hi until residual < 0.
    let mut v_hi = 1e-6;
    for _ in 0..200 {
        if residual(v_hi) < 0.0 {
            break;
        }
        v_hi *= 2.0;
    }
    let mut lo = 0.0;
    let mut hi = v_hi;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if residual(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Closed-form Stokes (1851) terminal velocity `v_t = (2/9)(ρ_p − ρ_f) g R²/μ`,
/// the `Re → 0` limit and the independent reference for the low-Re settling case.
#[inline]
pub fn terminal_velocity_stokes(rho_p: f64, rho_f: f64, radius: f64, g: f64, mu: f64) -> f64 {
    2.0 / 9.0 * (rho_p - rho_f) * g.abs() * radius * radius / mu
}

// ─── Void fraction + momentum sink on the FIELD mesh (unresolved regime) ─────

/// Locate the mesh cell (raw, ghost-inclusive index) containing `p`, or `None`
/// if `p` is outside the interior.
fn cell_of<M: StructuredMesh>(mesh: &M, p: Vec3) -> Option<usize> {
    let [ni, nj, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    // Nearest interior cell by centroid scan on each axis is overkill; use the
    // mesh's own point locate and clamp to interior.
    let (lo, _t) = mesh.locate(p)?;
    // `locate` returns the low corner of the bracketing cell block in raw
    // indices; clamp into the interior range so a particle sitting on a cell
    // face still lands in a real interior cell.
    let clamp = |v: usize, n: usize| v.max(ng).min(ng + n - 1);
    Some(mesh.idx_raw(clamp(lo[0], ni), clamp(lo[1], nj), clamp(lo[2], nk)))
}

/// Per-cell void fraction field `ε` for a point-particle deposit: each particle's
/// volume is charged wholly to the cell containing its center, giving
/// `ε = 1 − Σ V_p / V_cell` (clamped to `[ε_min, 1]`). Returned as a dense vector
/// indexed like `CfdState.u`. A single sub-cell sphere yields `ε ≈ 1`; this is the
/// dilute-limit deposition the settling case runs in, and the same routine the
/// fixed-bed / fluidized follow-ups reuse at higher loadings.
pub fn deposit_void_fraction<M: StructuredMesh>(
    mesh: &M,
    particles: &[ParticleKinematics],
    eps_min: f64,
) -> Vec<f64> {
    let total = mesh.n_cells_total();
    let mut solid_vol = vec![0.0f64; total];
    for p in particles {
        if let Some(c) = cell_of(mesh, p.center) {
            solid_vol[c] += p.volume();
        }
    }
    let mut eps = vec![1.0f64; total];
    for c in 0..total {
        let v = mesh.cell_volume(c);
        if v > 0.0 {
            eps[c] = (1.0 - solid_vol[c] / v).clamp(eps_min, 1.0);
        }
    }
    eps
}

/// Add a momentum sink to the gas: the equal-and-opposite reaction to the drag
/// forces the fluid exerts on the particles. `sink_force[i]` is the force the gas
/// exerts on particle `i` (world axes); Newton's third law puts `−sink_force[i]`
/// on the fluid, applied as a momentum source in the containing cell over one
/// coupling step `dt`. Returns the total momentum removed from the gas (for a
/// conservation check). Only the drag part should be passed here — the buoyancy
/// part is an undisturbed-field pressure force that carries no net momentum
/// exchange with the resolved gas state.
pub fn apply_momentum_sink<M: StructuredMesh>(
    mesh: &M,
    state: &mut CfdState,
    particles: &[ParticleKinematics],
    drag_on_particle: &[Vec3],
    dt: f64,
) -> Vec3 {
    let mut total = [0.0f64; 3];
    for (p, f) in particles.iter().zip(drag_on_particle) {
        let Some(c) = cell_of(mesh, p.center) else {
            continue;
        };
        let v = mesh.cell_volume(c);
        if v <= 0.0 {
            continue;
        }
        // Reaction on the fluid = −f, distributed over the cell volume as a
        // momentum density increment Δ(ρu) = −f/V_cell · dt.
        let inv = dt / v;
        state.u[c].rho_u -= f[0] * inv;
        state.u[c].rho_v -= f[1] * inv;
        state.u[c].rho_w -= f[2] * inv;
        for d in 0..3 {
            total[d] += f[d];
        }
    }
    total
}

/// Sample the gas velocity `(u,v,w)` at the cell containing `p` (nearest-cell
/// interpolation — the point-particle regime resolves the particle within one
/// cell). `None` if the point is outside the interior or the cell is non-physical.
pub fn sample_gas_velocity<M: StructuredMesh>(
    mesh: &M,
    state: &CfdState,
    eos: &dyn Eos,
    p: Vec3,
) -> Option<Vec3> {
    let c = cell_of(mesh, p)?;
    let u = &state.u[c];
    if !u.rho.is_finite() || u.rho <= 0.0 {
        return None;
    }
    let prim = eos.cons_to_prim(u);
    Some([prim.u, prim.v, prim.w])
}

/// Gas density at the cell containing `p`.
pub fn sample_gas_density<M: StructuredMesh>(mesh: &M, state: &CfdState, p: Vec3) -> Option<f64> {
    let c = cell_of(mesh, p)?;
    let rho = state.u[c].rho;
    (rho.is_finite() && rho > 0.0).then_some(rho)
}

/// Sample a per-cell void-fraction field (as produced by [`deposit_void_fraction`])
/// at the cell containing `p`.
pub fn sample_void_fraction<M: StructuredMesh>(mesh: &M, eps: &[f64], p: Vec3) -> Option<f64> {
    let c = cell_of(mesh, p)?;
    eps.get(c).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schiller_naumann_reduces_to_stokes_at_low_re() {
        // At Re = 1e-4 the (1 + 0.15 Re^0.687) correction is ~1e-3, so Cd ≈ 24/Re.
        let re = 1e-4;
        let cd = cd_schiller_naumann(re);
        let stokes = 24.0 / re;
        assert!((cd / stokes - 1.0).abs() < 5e-3, "cd {cd} vs stokes {stokes}");
    }

    #[test]
    fn schiller_naumann_and_cgw_agree_within_a_few_percent() {
        // Two independent standard-drag correlations should track each other
        // across the low/intermediate-Re range.
        for &re in &[0.1, 1.0, 10.0, 50.0, 100.0] {
            let a = cd_schiller_naumann(re);
            let b = cd_clift_grace_weber(re);
            let rel = (a - b).abs() / b;
            assert!(rel < 0.06, "Re={re}: SN {a} vs CGW {b} rel {rel}");
        }
    }

    #[test]
    fn wen_yu_recovers_single_sphere_drag_as_eps_to_one() {
        // ε → 1: F = β V_p/(1−ε)·u_rel must equal ½ Cd ρ_f (πR²)|u_rel|u_rel.
        let (rho_f, mu, r) = (1.2, 1.8e-5, 5e-4);
        let u_rel = [0.3, 0.0, 0.0];
        let speed = norm(u_rel);
        let d = 2.0 * r;
        let re = particle_reynolds(rho_f, speed, d, mu);
        let cd = cd_schiller_naumann(re);
        let eps = 1.0 - 1e-6;
        let v_p = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let beta = beta_gidaspow(eps, rho_f, mu, d, speed, cd);
        let f_beta = drag_force_from_beta(beta, v_p, eps, u_rel);
        let f_direct = sphere_drag_force([0.3, 0.0, 0.0], [0.0; 3], r, rho_f, mu, cd_schiller_naumann);
        let rel = (f_beta[0] - f_direct[0]).abs() / f_direct[0];
        assert!(rel < 1e-3, "wen-yu {:?} vs direct {:?} rel {rel}", f_beta, f_direct);
    }

    #[test]
    fn terminal_velocity_matches_stokes_in_the_stokes_limit() {
        // A tiny dense sphere in a light gas: Re should be << 1 so the bisection
        // (Schiller–Naumann) result matches the closed-form Stokes velocity.
        let (rho_p, rho_f, mu, g) = (2500.0, 1.2, 1.8e-5, 9.81);
        let r = 1.5e-5; // 30 µm sphere
        let v_num = terminal_velocity(rho_p, rho_f, r, g, mu, cd_schiller_naumann);
        let v_stokes = terminal_velocity_stokes(rho_p, rho_f, r, g, mu);
        let re = particle_reynolds(rho_f, v_num, 2.0 * r, mu);
        assert!(re < 0.5, "expected low Re, got {re}");
        assert!(
            (v_num / v_stokes - 1.0).abs() < 0.05,
            "v_num {v_num} vs stokes {v_stokes} (Re={re})"
        );
    }
}
