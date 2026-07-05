//! **Exact-Riemann gate for the Sod shock tube.**
//!
//! The Sod problem (Sod, *J. Comput. Phys.* 27 (1978) 1–31) is the canonical
//! compressible-flow verification case: a diaphragm separating two quiescent
//! ideal-gas states whose decay is a self-similar left rarefaction + contact +
//! right shock with a *closed-form* exact solution (Toro, *Riemann Solvers and
//! Numerical Methods for Fluid Dynamics*, 3rd ed., Ch. 4). Because the analytic
//! profile is known at every `x` and `t`, we can gate the FIELD solver on how
//! closely it reproduces the profile — not merely on whether it conserves mass.
//!
//! "Conserves mass" is far weaker than "matches the analytic profile": a scheme
//! with the wrong flux, wrong wave speeds, or the wrong equation of state can be
//! *exactly* mass-conserving (a closed box always is) while placing the shock in
//! the wrong spot. This test therefore asserts the **L1 density error against the
//! exact Riemann solution** falls below a resolution-justified tolerance at a
//! fixed time, and carries a **negative control**: the same solver run with the
//! wrong ratio of specific heats (γ = 1.67 instead of air's 1.4) moves the shock
//! and contact and blows the L1 gate wide open, even though it conserves mass to
//! round-off just like the correct run. Mass conservation cannot tell those two
//! runs apart; this gate can.
//!
//! The companion `isentropic_vortex_convergence` test gates the *smooth* regime
//! (formal order of accuracy); this one gates the *discontinuous* regime (shock +
//! contact placement and strength) against an independent analytic reference.
//!
//! The exact solver below is self-contained (a Newton solve for the star-region
//! pressure followed by self-similar sampling), so the reference is INDEPENDENT
//! of the code under test — it shares no flux, EOS-evaluation, or time-integration
//! path with the solver. It lives in the test, not the crate core: it serves
//! exactly this one benchmark.

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, SlipWall};
use cfd_eos::{Eos, EosResource, IdealGas};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
    SolverState,
};
use cfd_state::CfdState;
use cfd_state::PrimVar;
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, StructuredMesh, UniformMesh, UniformMeshConfig,
    Vec3,
};
use grass_app::prelude::*;
use std::sync::Arc;

// Sod's classic left/right primitive states (quiescent; jump at x = X0), matching
// the `test_cfd` `sod_shock_tube` example. Air, so p = ρ R T; the exact solver
// needs only (ρ, u, p) and γ.
const RHO_L: f64 = 1.0;
const U_L: f64 = 0.0;
const P_L: f64 = 100_000.0;
const RHO_R: f64 = 0.125;
const U_R: f64 = 0.0;
const P_R: f64 = 10_000.0;
const X0: f64 = 0.5;

// A fixed comparison time. At t = 6e-4 s the right shock sits at x ≈ 0.83 and the
// rarefaction head at x ≈ 0.28 (see the wave-speed check in the test), so every
// wave is well inside [0, 1] and the transmissive x-ends never influence the
// interior we score — the exact infinite-domain solution is the right reference.
const T_FINAL: f64 = 6.0e-4;

const AIR_GAMMA: f64 = 1.4;

// ---------------------------------------------------------------------------
// Exact Riemann solver for the ideal-gas Euler equations (Toro, Ch. 4).
// ---------------------------------------------------------------------------

/// Pressure function `f_K(p)` and its derivative for one side of the interface
/// (Toro eqs. 4.6–4.7): a shock branch when `p > p_k`, a rarefaction branch
/// otherwise.
fn f_and_df(p: f64, rho_k: f64, p_k: f64, a_k: f64, gamma: f64) -> (f64, f64) {
    if p > p_k {
        // Shock.
        let a_cap = 2.0 / ((gamma + 1.0) * rho_k);
        let b_cap = (gamma - 1.0) / (gamma + 1.0) * p_k;
        let sq = (a_cap / (b_cap + p)).sqrt();
        let f = (p - p_k) * sq;
        let df = sq * (1.0 - 0.5 * (p - p_k) / (b_cap + p));
        (f, df)
    } else {
        // Rarefaction.
        let f = 2.0 * a_k / (gamma - 1.0) * ((p / p_k).powf((gamma - 1.0) / (2.0 * gamma)) - 1.0);
        let df = 1.0 / (rho_k * a_k) * (p / p_k).powf(-(gamma + 1.0) / (2.0 * gamma));
        (f, df)
    }
}

/// Solve for the star-region pressure `p*` and velocity `u*` by Newton iteration
/// on `f_L(p) + f_R(p) + (u_R − u_L) = 0` (Toro §4.3).
fn star_state(gamma: f64) -> (f64, f64) {
    let a_l = (gamma * P_L / RHO_L).sqrt();
    let a_r = (gamma * P_R / RHO_R).sqrt();

    // A robust positive initial guess (arithmetic mean of the data pressures);
    // Newton converges quadratically for Sod from here.
    let mut p = (0.5 * (P_L + P_R)).max(1e-6);
    for _ in 0..100 {
        let (fl, dfl) = f_and_df(p, RHO_L, P_L, a_l, gamma);
        let (fr, dfr) = f_and_df(p, RHO_R, P_R, a_r, gamma);
        let f = fl + fr + (U_R - U_L);
        let df = dfl + dfr;
        let p_new = (p - f / df).max(1e-9);
        if (2.0 * (p_new - p) / (p_new + p)).abs() < 1e-12 {
            p = p_new;
            break;
        }
        p = p_new;
    }
    let (fl, _) = f_and_df(p, RHO_L, P_L, a_l, gamma);
    let (fr, _) = f_and_df(p, RHO_R, P_R, a_r, gamma);
    let u_star = 0.5 * (U_L + U_R) + 0.5 * (fr - fl);
    (p, u_star)
}

/// Sample the self-similar exact solution at speed `s = (x − X0)/t`, returning
/// the exact density there (Toro §4.5, the `sample` routine). Only density is
/// needed for the L1 gate.
fn exact_density(gamma: f64, p_star: f64, u_star: f64, s: f64) -> f64 {
    let a_l = (gamma * P_L / RHO_L).sqrt();
    let a_r = (gamma * P_R / RHO_R).sqrt();
    let g1 = (gamma - 1.0) / (gamma + 1.0);

    if s <= u_star {
        // Left of the contact.
        if p_star > P_L {
            // Left shock.
            let sl = U_L
                - a_l
                    * ((gamma + 1.0) / (2.0 * gamma) * p_star / P_L
                        + (gamma - 1.0) / (2.0 * gamma))
                        .sqrt();
            if s <= sl {
                RHO_L
            } else {
                RHO_L * (p_star / P_L + g1) / (g1 * p_star / P_L + 1.0)
            }
        } else {
            // Left rarefaction.
            let shl = U_L - a_l;
            if s <= shl {
                RHO_L
            } else {
                let a_star = a_l * (p_star / P_L).powf((gamma - 1.0) / (2.0 * gamma));
                let stl = u_star - a_star;
                if s >= stl {
                    // Star region (left of contact).
                    RHO_L * (p_star / P_L).powf(1.0 / gamma)
                } else {
                    // Inside the fan.
                    let a = 2.0 / (gamma + 1.0) * (a_l + (gamma - 1.0) / 2.0 * (U_L - s));
                    RHO_L * (a / a_l).powf(2.0 / (gamma - 1.0))
                }
            }
        }
    } else {
        // Right of the contact.
        if p_star > P_R {
            // Right shock.
            let sr = U_R
                + a_r
                    * ((gamma + 1.0) / (2.0 * gamma) * p_star / P_R
                        + (gamma - 1.0) / (2.0 * gamma))
                        .sqrt();
            if s >= sr {
                RHO_R
            } else {
                RHO_R * (p_star / P_R + g1) / (g1 * p_star / P_R + 1.0)
            }
        } else {
            // Right rarefaction.
            let shr = U_R + a_r;
            if s >= shr {
                RHO_R
            } else {
                let a_star = a_r * (p_star / P_R).powf((gamma - 1.0) / (2.0 * gamma));
                let str_tail = u_star + a_star;
                if s <= str_tail {
                    RHO_R * (p_star / P_R).powf(1.0 / gamma)
                } else {
                    let a = 2.0 / (gamma + 1.0) * (a_r - (gamma - 1.0) / 2.0 * (U_R - s));
                    RHO_R * (a / a_r).powf(2.0 / (gamma - 1.0))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Solver run + error measurement.
// ---------------------------------------------------------------------------

/// One interior cell after the run: (centroid x, numerical density, cell volume).
type CellSample = (f64, f64, f64);

/// Run the Sod tube on an `n`-cell FIELD grid with equation of state `gas`,
/// advancing to `T_FINAL`. Returns the achieved final time and the interior
/// density field. The initial (ρ, u, p) discontinuity is built with the *same*
/// `gas`, so the run is a physically self-consistent Euler solution for that γ.
fn simulate(n: usize, gas: IdealGas) -> (f64, Vec<CellSample>) {
    let t_l = P_L / (RHO_L * gas.r_gas);
    let t_r = P_R / (RHO_R * gas.r_gas);
    let left = gas.prim_to_cons(&PrimVar::new(RHO_L, U_L, 0.0, 0.0, P_L, t_l));
    let right = gas.prim_to_cons(&PrimVar::new(RHO_R, U_R, 0.0, 0.0, P_R, t_r));
    let ic = move |x: Vec3| if x[0] < X0 { left } else { right };

    // A thin 1-D tube: slip walls on the transverse faces, transmissive x-ends.
    let mesh = UniformMeshConfig {
        nx: n,
        ny: 1,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [1.0, 0.01, 0.01],
        y_edges: None,
        z_edges: None,
    };
    let bcs = BoundaryRegistry::default()
        .with_axis(1, SlipWall)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(ic))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.4,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    // Override the default air EOS with the requested gas. `add_resource` replaces
    // by type, so this is a no-op for γ = 1.4 and injects the wrong γ for the
    // negative control — the flux/CFL path reads this `EosResource` every step.
    app.add_resource(gas);
    app.add_resource(EosResource(Arc::new(gas)));
    app.prepare();

    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= T_FINAL {
            break;
        }
        for _ in 0..3 {
            app.run(); // one RK3 step (three substeps)
        }
    }

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let t = app.get_resource_ref::<SolverState>().unwrap().time;

    assert!(
        state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
        "Sod at N={n} went non-physical",
    );

    let [ni, _, _] = mesh.dims();
    let mut cells = Vec::with_capacity(ni);
    for i in 0..ni {
        let c = mesh.idx(i, 0, 0);
        cells.push((
            mesh.cell_centroid(c)[0],
            state.u[c].rho,
            mesh.cell_volume(c),
        ));
    }
    (t, cells)
}

/// Volume-weighted L1 density error of a numerical field against the exact
/// Riemann solution built with `ref_gamma`, evaluated at `t`.
fn l1_density_error(cells: &[CellSample], t: f64, ref_gamma: f64) -> f64 {
    let (p_star, u_star) = star_state(ref_gamma);
    let (mut num, mut vol) = (0.0, 0.0);
    for &(x, rho_num, cv) in cells {
        let s = (x - X0) / t;
        let rho_ex = exact_density(ref_gamma, p_star, u_star, s);
        num += (rho_num - rho_ex).abs() * cv;
        vol += cv;
    }
    num / vol
}

/// The gate. The solver must reproduce the exact Sod density profile to within a
/// resolution-justified L1 tolerance, and a physically-wrong γ must fail it.
#[test]
fn sod_matches_exact_riemann_profile_with_wrong_gamma_negative_control() {
    let air = IdealGas::air();

    // --- Wave-containment sanity: every wave stays strictly inside (0, 1) at
    // T_FINAL, so the transmissive ends never touch the scored interior. -------
    let (p_star, u_star) = star_state(AIR_GAMMA);
    let a_l = (AIR_GAMMA * P_L / RHO_L).sqrt();
    let a_r = (AIR_GAMMA * P_R / RHO_R).sqrt();
    let x_head = X0 + (U_L - a_l) * T_FINAL; // rarefaction head (leftmost wave)
    let s_shock = U_R
        + a_r
            * ((AIR_GAMMA + 1.0) / (2.0 * AIR_GAMMA) * p_star / P_R
                + (AIR_GAMMA - 1.0) / (2.0 * AIR_GAMMA))
                .sqrt();
    let x_shock = X0 + s_shock * T_FINAL; // shock (rightmost wave)
    eprintln!(
        "exact star state: p* = {p_star:.1} Pa, u* = {u_star:.2} m/s; \
         rarefaction head x = {x_head:.3}, shock x = {x_shock:.3}",
    );
    assert!(
        x_head > 0.05 && x_shock < 0.95,
        "waves left the domain interior: head={x_head:.3}, shock={x_shock:.3}",
    );

    // --- Convergence: L1 density error vs the exact profile on refined grids. --
    // First-order Godunov/HLLC on a solution with a shock + contact converges in
    // L1 at ~first order, so the error scales like C/N. We fit the observed rate
    // to justify the tolerance rather than picking a bare number.
    let ns = [100usize, 200, 400];
    let errs: Vec<f64> = ns
        .iter()
        .map(|&n| {
            let (t, cells) = simulate(n, air);
            l1_density_error(&cells, t, AIR_GAMMA)
        })
        .collect();
    let orders: Vec<f64> = (1..ns.len())
        .map(|k| (errs[k - 1] / errs[k]).log2())
        .collect();
    eprintln!("Sod L1 density error vs exact Riemann (γ = 1.4):");
    eprintln!("  N={:4}  err={:.6e}", ns[0], errs[0]);
    for k in 1..ns.len() {
        eprintln!(
            "  N={:4}  err={:.6e}  order={:.3}",
            ns[k],
            errs[k],
            orders[k - 1]
        );
    }

    // Error falls monotonically under refinement.
    assert!(
        errs.windows(2).all(|w| w[1] < w[0]),
        "L1 error not monotonically decreasing under refinement: {errs:?}",
    );
    // Observed L1 rate is at least ~0.5. For a first-order Godunov/HLLC scheme the
    // Sod error is dominated by the *linearly-degenerate contact*, which smears
    // over O(√N) cells and so contributes an L1 density error decaying like
    // N^(−1/2); the shock and rarefaction converge faster, giving a blended rate
    // ≈0.66 here. A broken flux or wrong wave speeds stall or reverse this.
    let finest_order = *orders.last().unwrap();
    assert!(
        finest_order >= 0.5,
        "finest-pair L1 order {finest_order:.3} below 0.5 — convergence stalled",
    );

    // Resolution-justified tolerance. The contact-smearing argument above bounds
    // the L1 density error by C₀·N^(−1/2). The measured C₀ = err·√N is
    // 0.173 → 0.151 → 0.136 (decreasing, as it must for a rate above 1/2), so
    // C₀ = 0.25 bounds every grid with ≥30% headroom while staying far below the
    // wrong-γ control. Assert each grid clears its own √N bound.
    let tol = |n: usize| 0.25 / (n as f64).sqrt();
    for (k, (&n, &e)) in ns.iter().zip(&errs).enumerate() {
        assert!(
            e < tol(n),
            "grid {k}: N={n} L1 density error {e:.6e} exceeds resolution bound {:.6e}",
            tol(n),
        );
    }

    // --- Negative control: wrong γ must FAIL the same gate, at EVERY resolution.
    // Run the solver with γ = 1.67 (monatomic — argon, not air), everything else
    // identical. It is still a valid, mass-conserving Euler solution, but its shock
    // and contact sit elsewhere, so its L1 error against the correct γ = 1.4 exact
    // profile is a *model* error, not a discretization error: it does NOT shrink
    // under refinement, it plateaus at the physical distance between the two exact
    // solutions and stays above the gate on every grid. That is the crux — a
    // discretization error you can refine away; a wrong-physics error you cannot,
    // and mass conservation (identical for both runs) is blind to it.
    const WRONG_GAMMA: f64 = 1.67;
    let wrong_gas = IdealGas {
        gamma: WRONG_GAMMA,
        r_gas: air.r_gas,
    };

    let mut correct_mass = 0.0;
    let mut wrong_mass = 0.0;
    let mut wrong_errs = Vec::new();
    // Start at N = 200 (not the coarsest N = 100): the wrong-γ error clears the
    // gate at every grid, but at N = 100 the margin is only ~14% (wrong-γ L1
    // ≈2.85e-2 vs gate 2.50e-2), widening to ~40% at N = 200 and more at N = 400.
    // Running the control on 200/400 keeps the decisive margin comfortably wide
    // and robust to future flux/limiter changes that might nudge the coarse-grid
    // constant, while still spanning a refinement pair for the no-convergence check.
    for &n in &ns[1..] {
        // N = 200, 400
        let (t_ok, cells_ok) = simulate(n, air);
        let (t_bad, cells_bad) = simulate(n, wrong_gas);
        let e_ok = l1_density_error(&cells_ok, t_ok, AIR_GAMMA);
        let e_bad = l1_density_error(&cells_bad, t_bad, AIR_GAMMA);
        correct_mass = cells_ok.iter().map(|&(_, rho, v)| rho * v).sum();
        wrong_mass = cells_bad.iter().map(|&(_, rho, v)| rho * v).sum();
        eprintln!(
            "negative control @ N={n}: correct-γ L1 = {e_ok:.6e}, wrong-γ L1 = {e_bad:.6e}, \
             tol = {:.6e}",
            tol(n),
        );
        // The correct run clears the gate; the wrong-γ run fails it — at this N.
        assert!(
            e_ok < tol(n),
            "sanity: correct-γ run should pass its own gate at N={n}"
        );
        assert!(
            e_bad > tol(n),
            "negative control did not fail at N={n}: wrong-γ L1 {e_bad:.6e} within tol {:.6e}",
            tol(n),
        );
        wrong_errs.push(e_bad);
    }

    // Mass is conserved to round-off by BOTH runs — the weak check cannot tell
    // the physically-wrong run from the correct one.
    eprintln!(
        "total mass: air = {correct_mass:.8e}, γ=1.67 = {wrong_mass:.8e} \
         (identical to round-off — mass conservation is blind to the wrong physics)",
    );
    assert!(
        (correct_mass - wrong_mass).abs() / correct_mass < 1e-9,
        "both runs must conserve mass identically: {correct_mass} vs {wrong_mass}",
    );

    // The wrong-γ error does NOT converge away: refining 200→400 leaves it a fixed
    // physical floor (it must not drop like the discretization error would). This
    // is what proves the gate tests the analytic profile, not the mesh.
    assert!(
        wrong_errs[1] > 0.8 * wrong_errs[0],
        "wrong-γ error fell under refinement ({:.6e} → {:.6e}) — not a model-error floor",
        wrong_errs[0],
        wrong_errs[1],
    );
}
