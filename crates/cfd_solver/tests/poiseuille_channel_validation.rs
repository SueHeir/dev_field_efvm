//! **External-reference validation of the viscous (Navier–Stokes) path:**
//! steady planar **Poiseuille flow** in a plane channel, compared cell-by-cell
//! against the closed-form analytic parabola.
//!
//! # The reference (independent of this code)
//! For fully-developed, incompressible, constant-property laminar flow between two
//! stationary parallel plates a distance `H` apart, driven by a uniform body force
//! `f_x = ρ g` per unit volume (equivalently a constant streamwise pressure
//! gradient `−dp/dx = ρ g`), the steady streamwise velocity is the classic
//! parabola
//!
//! ```text
//!     u(y) = (ρ g) / (2 μ) · y (H − y),      0 ≤ y ≤ H,
//! ```
//!
//! with no-slip `u(0) = u(H) = 0` and centreline maximum `u_max = ρ g H² / (8 μ)`.
//! This is a textbook exact solution of the Navier–Stokes equations — e.g.
//! F. M. White, *Viscous Fluid Flow*, 3rd ed., §3-2.3 (plane Poiseuille flow), or
//! Batchelor, *An Introduction to Fluid Dynamics*, §4.2. It is **external** to the
//! solver: the reference comes from the analytic formula, not from a second run or
//! a refined-grid self-comparison.
//!
//! # Why this is an honest test of the viscous operator
//! For this flow `v = w = 0` and `∂/∂x = ∂/∂z = 0`, so the full viscous stress
//! divergence collapses **exactly** to `μ ∂²u/∂y²` (the dilatation term `∇(∇·u)`
//! vanishes because `∇·u = 0`). The solver's `viscous_system` adds precisely
//! `μ ∇²u`, so at steady state the discrete balance is `0 = μ ∂²u/∂y² + ρ g`,
//! whose continuous solution is the parabola above. A wrong viscous coefficient, a
//! missing `1/μ`, a mis-scaled Laplacian, or a broken no-slip wall all move the
//! numerical profile off the analytic curve and blow up the L2 error.
//!
//! We pin a **constant** viscosity (`Viscosity::Constant`) so `μ` is a known
//! number rather than Sutherland's `μ(T)`; the Mach number is ~0.006 and viscous
//! heating raises the temperature by <0.1 K over the run, so the incompressible,
//! constant-property reference applies to ~ppm.
//!
//! # Flux scheme: HLLC, not Rusanov
//! The inviscid Riemann solver must **not** smear tangential momentum, or its
//! numerical viscosity `μ_num ≈ ½ ρ a Δy` would add to the physical `μ` and the
//! steady profile would report an *effective* viscosity `μ + μ_num`. At these
//! parameters `μ_num ≈ 5.4`, i.e. ~0.5·μ — so Rusanov yields a profile that
//! matches a parabola for `1.5 μ`, not `μ` (measured directly: Rusanov gives a
//! 26 % L2 error against the true-μ parabola). HLLC restores the contact wave, so
//! the tangential (x) velocity is advected with essentially zero added
//! dissipation and the steady profile matches the true-μ analytic parabola to
//! <0.1 %. Using HLLC here is not tuning — it is the standard, more accurate
//! solver already used by the isentropic-vortex order-of-accuracy test; Rusanov's
//! excess dissipation is simply too large to leave the physical `μ` observable.
//!
//! # Checks
//! 1. **Positive:** the steady numerical profile matches the analytic parabola in
//!    a volume-weighted relative L2 norm below `TOL`.
//! 2. **Negative control (wrong μ ⇒ fails):** the same numerical profile compared
//!    against a parabola built with a deliberately wrong viscosity (`1.5 μ`) gives
//!    an L2 error an order of magnitude above `TOL` — i.e. the metric genuinely
//!    resolves the viscosity and would reject a mis-set `μ`.

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, NoSlipWall, SlipWall};
use cfd_eos::{Eos, IdealGas, Viscosity};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
    SolverState,
};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, StructuredMesh, UniformMesh,
    UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;
use grass_scheduler::Res;

// ── Case definition (declarative constants) ──────────────────────────────────
const H: f64 = 1.0; // channel height (wall-normal extent, y ∈ [0, H])
const RHO0: f64 = 1.0; // nominal density [kg/m³]
const T0: f64 = 300.0; // nominal temperature [K] ⇒ p0 = ρ R T0, a ≈ 347 m/s
const MU: f64 = 10.0; // constant dynamic viscosity [Pa·s] (pins Re, incompressible limit)
const GX: f64 = 160.0; // streamwise body-force acceleration [m/s²]; u_max = ρ g H²/(8μ) = 2 m/s

const NY: usize = 32; // wall-normal resolution (the only resolved direction)
const NX: usize = 3; // streamwise: fully-developed ⇒ x-invariant, few cells suffice
const NZ: usize = 1; // spanwise: single cell with symmetry (slip) walls

// Diffusive relaxation to steady state has slowest-mode time τ = ρH²/(π²μ) ≈ 0.0101 s.
// Run to ~8τ so the transient residual (~e⁻⁸ ≈ 3e-4·u_max) is far below the spatial
// discretization error we are measuring.
const T_FINAL: f64 = 0.081;

// Measured relative-L2 error at these settings is ~7.6e-4; a 0.5 % gate keeps a
// ~6× margin against transient/round-off drift while still tightly binding the
// viscous coefficient (the negative control below sits at ~37 %).
const TOL: f64 = 0.005;

/// Analytic plane-Poiseuille velocity at height `y` for viscosity `mu`.
fn poiseuille_u(y: f64, mu: f64) -> f64 {
    RHO0 * GX / (2.0 * mu) * y * (H - y)
}

/// Uniform body force `f_x = ρ g` in `+x`, added to the RHS every substep in the
/// `PostFlux` phase (after the inviscid + viscous fluxes). Also injects the
/// force's rate of work `f_x · u = g · (ρu)` into the energy RHS so the energy
/// balance stays consistent. Example-local: it needs no new solver API.
#[derive(Clone, Copy)]
struct BodyForceX(f64);

fn body_force_system(mesh: Res<UniformMesh>, reg: Res<FieldRegistry>, force: Res<BodyForceX>) {
    let gx = force.0;
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let [ni, nj, nk] = mesh.dims();
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let c = mesh.idx(i, j, k);
                let rho = state.u[c].rho;
                let rho_u = state.u[c].rho_u;
                state.rhs[c].rho_u += rho * gx;
                state.rhs[c].rho_e += gx * rho_u; // f_x · u = g · (ρu)
            }
        }
    }
}

/// Build the channel, drive it with the body force to `T_FINAL`, and return the
/// steady per-cell `(y, u)` samples at the mid-x, mid-z column.
fn run_channel() -> Vec<(f64, f64)> {
    let eos = IdealGas::air();
    let p0 = RHO0 * eos.r_gas * T0;
    // Quiescent, uniform initial state; the body force spins up the profile.
    let init = move |_x: Vec3| eos.prim_to_cons(&PrimVar::new(RHO0, 0.0, 0.0, 0.0, p0, T0));

    let mesh = UniformMeshConfig {
        nx: NX,
        ny: NY,
        nz: NZ,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [1.0, H, 1.0],
        y_edges: None,
        z_edges: None,
    };

    // x: zero-gradient (default SupersonicOutflow) — consistent with ∂/∂x = 0.
    // y: no-slip adiabatic walls (the physical channel walls).
    // z: symmetry (slip) — a single spanwise cell with no z-variation.
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::YLo, NoSlipWall)
        .with(BoundarySide::YHi, NoSlipWall)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.3,
            fixed_dt: None,
            viscous: true,
            muscl: false,
        }));
    // Pin a constant viscosity (overrides the default Sutherland μ(T)) and register
    // the streamwise body force in PostFlux.
    app.add_resource(Viscosity::Constant(MU));
    app.add_resource(BodyForceX(GX));
    app.add_update_system(body_force_system, field_core::MeshScheduleSet::PostFlux);
    app.prepare();

    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= T_FINAL {
            break;
        }
        app.run();
    }

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    assert!(
        state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
        "channel went non-physical",
    );

    let [_, nj, _] = mesh.dims();
    let (im, km) = (NX / 2, NZ / 2);
    (0..nj)
        .map(|j| {
            let c = mesh.idx(im, j, km);
            let y = mesh.cell_centroid(c)[1];
            let u = state.u[c].rho_u / state.u[c].rho;
            (y, u)
        })
        .collect()
}

/// Volume-weighted relative L2 error of the sampled profile against the analytic
/// parabola for viscosity `mu` (cells are uniform in y, so plain averaging is the
/// volume weighting). Normalised by the analytic centreline speed.
fn rel_l2(profile: &[(f64, f64)], mu: f64) -> f64 {
    let u_max = poiseuille_u(0.5 * H, mu);
    let mut num = 0.0;
    for &(y, u) in profile {
        let e = u - poiseuille_u(y, mu);
        num += e * e;
    }
    (num / profile.len() as f64).sqrt() / u_max
}

#[test]
fn poiseuille_profile_matches_analytic_parabola() {
    let profile = run_channel();

    // Positive check: steady numerical profile vs the analytic parabola (true μ).
    let err_true = rel_l2(&profile, MU);

    // Negative control: same numerical profile vs a parabola built with the WRONG
    // viscosity (1.5 μ). If the solver's μ were mis-set this is what the check
    // would see — it must be far outside tolerance.
    let mu_wrong = 1.5 * MU;
    let err_wrong = rel_l2(&profile, mu_wrong);

    eprintln!("plane Poiseuille — {NY} wall-normal cells, t_final = {T_FINAL} s");
    eprintln!("  u_max (analytic) = {:.4} m/s", poiseuille_u(0.5 * H, MU));
    eprintln!("  y/H      u_num       u_exact");
    for &(y, u) in &profile {
        eprintln!("  {:5.3}  {:9.5}  {:9.5}", y / H, u, poiseuille_u(y, MU));
    }
    eprintln!("  relative L2 (true  μ={MU})     = {err_true:.3e}  (tol {TOL})");
    eprintln!("  relative L2 (wrong μ={mu_wrong}) = {err_wrong:.3e}  (negative control)");

    // (1) The viscous path reproduces the external analytic reference.
    assert!(
        err_true < TOL,
        "Poiseuille profile L2 error {err_true:.3e} exceeds tolerance {TOL} — \
         viscous operator does not match the analytic parabola",
    );
    // (2) The metric resolves μ: the wrong-viscosity reference is clearly rejected.
    assert!(
        err_wrong > 10.0 * TOL,
        "negative control too weak: wrong-μ L2 error {err_wrong:.3e} is not far \
         above tolerance {TOL} — the check would not catch a mis-set viscosity",
    );
}
