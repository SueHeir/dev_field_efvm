//! **Order-of-accuracy regression for the 2D isentropic vortex.**
//!
//! The isentropic (Shu) vortex — Shu, ICASE Report 97-65 (1998), §4.1 — is a
//! smooth, exact solution of the 2D compressible Euler equations that advects
//! rigidly with the mean flow. Because the exact state is known at every point
//! and every time, refining the mesh must drive the L1 error down at the scheme's
//! *formal* spatial order. Unlike Sod (whose shock/contact clamp every scheme to
//! 1st order and mask reconstruction/flux errors), the vortex is smooth, so a
//! wrong flux or a miswired reconstruction surfaces as an order deficit.
//!
//! On the uniform grid the inviscid flux is piecewise-constant Godunov (HLLC),
//! i.e. **formally 1st order**; TVD-RK3 keeps the temporal error subdominant. The
//! test sweeps three grids, fits the observed order, and asserts it matches the
//! 1st-order design order within tolerance.
//!
//! (2nd-order MUSCL reconstruction exists in `cfd_solver` but its
//! `compute_slopes`/`muscl_flux` systems are wired for `field_amr::ForestMesh`'s
//! contiguous local-cell packing via `CartesianMesh`, which `UniformMesh` does
//! not implement — so the uniform-grid design order is 1. See the PR notes.)

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, SlipWall, SupersonicOutflow};
use cfd_eos::{Eos, IdealGas};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
    SolverState,
};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, StructuredMesh, UniformMesh,
    UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;

const GAMMA: f64 = 1.4;
const BETA: f64 = 5.0;
const DOMAIN: f64 = 10.0;
const MEAN_U: f64 = 0.5;
const MEAN_V: f64 = 0.5;
const T_FINAL: f64 = 1.0;

/// Shu isentropic vortex at time `t`: the vortex centered at the domain center in
/// the initial field, translated by the mean flow. Nondimensional (ρ∞=p∞=T∞=1).
fn vortex_state(t: f64, x: Vec3) -> ConsVar {
    let eos = IdealGas { gamma: GAMMA, r_gas: 1.0 };
    let cx = 0.5 * DOMAIN + MEAN_U * t;
    let cy = 0.5 * DOMAIN + MEAN_V * t;
    let dx = x[0] - cx;
    let dy = x[1] - cy;
    let r2 = dx * dx + dy * dy;
    let f = (0.5 * (1.0 - r2)).exp();
    let du = -BETA / (2.0 * std::f64::consts::PI) * dy * f;
    let dv = BETA / (2.0 * std::f64::consts::PI) * dx * f;
    let dtemp = -(GAMMA - 1.0) * BETA * BETA / (8.0 * GAMMA * std::f64::consts::PI.powi(2))
        * (1.0 - r2).exp();
    let temp = 1.0 + dtemp;
    let rho = temp.powf(1.0 / (GAMMA - 1.0));
    let p = rho.powf(GAMMA);
    eos.prim_to_cons(&PrimVar::new(rho, MEAN_U + du, MEAN_V + dv, 0.0, p, temp))
}

/// Advect the vortex to `T_FINAL` on an `n × n` grid; return the mesh-averaged L1
/// density error against the exact translated vortex.
fn l1_density_error(n: usize) -> f64 {
    let ic = |x: Vec3| vortex_state(0.0, x);
    let mesh = UniformMeshConfig {
        nx: n,
        ny: n,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [DOMAIN, DOMAIN, DOMAIN / n as f64],
        y_edges: None,
        z_edges: None,
    };
    // Vortex stays ≳4 core radii from every wall for the whole run, so the
    // boundary state is the mean flow to ~1e-8 and zero-gradient faces are exact.
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, SupersonicOutflow)
        .with(BoundarySide::XHi, SupersonicOutflow)
        .with(BoundarySide::YLo, SupersonicOutflow)
        .with(BoundarySide::YHi, SupersonicOutflow)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(ic))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.3,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();

    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= T_FINAL {
            break;
        }
        for _ in 0..3 {
            app.run(); // one RK3 step
        }
    }

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let t = app.get_resource_ref::<SolverState>().unwrap().time;
    let [ni, nj, _] = mesh.dims();
    let (mut num, mut vol) = (0.0, 0.0);
    for j in 0..nj {
        for i in 0..ni {
            let c = mesh.idx(i, j, 0);
            let exact = vortex_state(t, mesh.cell_centroid(c));
            let cv = mesh.cell_volume(c);
            num += (state.u[c].rho - exact.rho).abs() * cv;
            vol += cv;
        }
    }
    assert!(
        state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
        "vortex at N={n} went non-physical",
    );
    num / vol
}

/// The vortex is smooth ⇒ 1st-order Godunov/HLLC on the uniform grid must show a
/// clean 1st-order L1 convergence rate. A wrong flux would break the rate even
/// though it maintains a shock (Sod) just fine — that is the point of this case.
#[test]
fn isentropic_vortex_is_first_order_accurate() {
    let ns = [40usize, 80, 160];
    let errs: Vec<f64> = ns.iter().map(|&n| l1_density_error(n)).collect();

    // Pairwise observed orders (grids differ by exactly 2×).
    let orders: Vec<f64> = (1..ns.len())
        .map(|k| (errs[k - 1] / errs[k]).log2())
        .collect();

    eprintln!("isentropic vortex convergence (L1 density error):");
    eprintln!("  N={:4}  err={:.6e}", ns[0], errs[0]);
    for k in 1..ns.len() {
        eprintln!("  N={:4}  err={:.6e}  order={:.3}", ns[k], errs[k], orders[k - 1]);
    }

    // Least-squares fit of ln(err) vs ln(n): slope is -order.
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let m = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    let fitted = -(m * sxy - sx * sy) / (m * sxx - sx * sx);
    eprintln!("  fitted order = {fitted:.3} (design 1.0)");

    // Error must fall monotonically under refinement.
    assert!(
        errs.windows(2).all(|w| w[1] < w[0]),
        "L1 error not monotonically decreasing: {errs:?}",
    );
    // Design order is 1; accept the fitted rate within ±0.2 (a 1st-order scheme
    // approaches its asymptotic rate from below on a nonlinear smooth feature —
    // here 0.86 → 0.92 → 1.0). A miswired flux collapses this well outside band.
    assert!(
        (0.8..=1.2).contains(&fitted),
        "fitted order {fitted:.3} not within 1.0 ± 0.2 — flux/scheme is not 1st-order accurate",
    );
    // The finest, most-asymptotic pair must be clearly first-order.
    let finest = *orders.last().unwrap();
    assert!(
        finest >= 0.85,
        "finest-pair order {finest:.3} below 0.85 — convergence stalled",
    );
    // Sanity: the error is genuinely converging (not merely a favorable slope).
    assert!(errs[ns.len() - 1] < 2.0e-3, "finest-grid L1 error too large: {}", errs[ns.len() - 1]);
}
