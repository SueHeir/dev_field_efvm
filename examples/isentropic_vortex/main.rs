//! **2D isentropic vortex advection — spatial order-of-accuracy harness.**
//!
//! The isentropic (Shu) vortex is a *smooth, exact* solution of the 2D
//! compressible Euler equations: a compact vortex superimposed on a uniform mean
//! flow simply advects at that mean velocity without changing shape. Because the
//! exact solution at any time is just the initial vortex translated by
//! `(u∞·t, v∞·t)`, the discretization error is known pointwise, and refining the
//! mesh must drive the error down at the scheme's *formal* spatial order.
//!
//! This is the classic verification case (Shu, *Essentially Non-Oscillatory and
//! Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation
//! Laws*, ICASE Report 97-65, 1998, §4.1): unlike Sod — whose discontinuities cap
//! every scheme at 1st order and hide reconstruction/flux bugs behind limiters —
//! the vortex is smooth, so a wrong reconstruction or a miswired flux shows up
//! immediately as an order deficit.
//!
//! The whole run is **declarative**: grid, mean flow, vortex strength, flux
//! scheme, integrator, the refinement sequence, and the expected design order all
//! come from a TOML config. The harness sweeps the resolutions, fits the observed
//! order, prints the convergence table, writes a VTK of the finest run, and exits
//! non-zero if the measured order misses the declared design order.
//!
//! ```text
//! cargo run --release -p dev_field_efvm --bin isentropic_vortex -- \
//!     examples/isentropic_vortex/config.toml
//! ```
//!
//! ## Where this lives
//! The scheme assembled here (`FieldDefaultPlugins` + the `cfd_solver`/
//! `cfd_boundary` plugins) is exactly what `test_cfd`'s `CfdDefaultPlugins`
//! bundles — this example wires the same plugins directly so it depends only on
//! the `field_core`-backed physics crates (no `field_p4est`), and therefore
//! builds and validates without a p4est install.

use cfd_boundary::{BoundaryRegistry, SlipWall, SupersonicOutflow};
use cfd_eos::{Eos, IdealGas};
use cfd_output::write_vtk;
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
use serde::Deserialize;

// ─── Declarative config ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
struct Config {
    vortex: VortexCfg,
    run: RunCfg,
    convergence: ConvergenceCfg,
}

#[derive(Debug, Clone, Deserialize)]
struct VortexCfg {
    /// Square domain side length `L`; the vortex is centered at `(L/2, L/2)`.
    domain: f64,
    /// Vortex strength `β` (Shu uses 5).
    beta: f64,
    /// Ratio of specific heats.
    gamma: f64,
    /// Uniform mean-flow velocity the vortex advects with.
    mean_u: f64,
    mean_v: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RunCfg {
    t_final: f64,
    cfl: f64,
    /// "hllc" or "rusanov".
    scheme: String,
    /// "rk3" or "euler".
    integrator: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ConvergenceCfg {
    /// Cells per side for each grid in the refinement sequence (increasing).
    resolutions: Vec<usize>,
    /// Formal spatial order the scheme is expected to exhibit.
    design_order: f64,
    /// Allowed deviation of the fitted order from `design_order`.
    order_tol: f64,
}

// ─── The isentropic vortex exact solution ────────────────────────────────────

/// The Shu isentropic vortex at time `t`: the vortex centered at `(cx, cy)` in
/// the initial field, translated by the mean flow. Returns the conserved state
/// at a point. Nondimensional (ρ∞ = p∞ = T∞ = 1); only γ enters the dynamics.
fn vortex_state(v: &VortexCfg, t: f64, x: Vec3) -> ConsVar {
    let eos = IdealGas {
        gamma: v.gamma,
        r_gas: 1.0,
    };
    let cx = 0.5 * v.domain + v.mean_u * t;
    let cy = 0.5 * v.domain + v.mean_v * t;
    let dx = x[0] - cx;
    let dy = x[1] - cy;
    let r2 = dx * dx + dy * dy;
    let f = (0.5 * (1.0 - r2)).exp();
    let du = -v.beta / (2.0 * std::f64::consts::PI) * dy * f;
    let dv = v.beta / (2.0 * std::f64::consts::PI) * dx * f;
    // δT = -(γ-1)β²/(8γπ²) · exp(1-r²);  ρ = T^{1/(γ-1)};  p = ρ^γ  (S ≡ 1).
    let dtemp = -(v.gamma - 1.0) * v.beta * v.beta / (8.0 * v.gamma * std::f64::consts::PI.powi(2))
        * (1.0 - r2).exp();
    let temp = 1.0 + dtemp;
    let rho = temp.powf(1.0 / (v.gamma - 1.0));
    let p = rho.powf(v.gamma);
    eos.prim_to_cons(&PrimVar::new(
        rho,
        v.mean_u + du,
        v.mean_v + dv,
        0.0,
        p,
        temp,
    ))
}

/// One converged run at `n × n` cells; returns `(mesh-averaged L1 density error,
/// app)` — the app is returned so the caller can dump a VTK of the finest grid.
fn run_case(cfg: &Config, n: usize) -> (f64, App) {
    let l = cfg.vortex.domain;
    let vcfg = cfg.vortex.clone();
    let ic = move |x: Vec3| vortex_state(&vcfg, 0.0, x);

    let mesh = UniformMeshConfig {
        nx: n,
        ny: n,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [l, l, l / n as f64],
        y_edges: None,
        z_edges: None,
    };
    // The vortex perturbation is Gaussian-compact; with the vortex kept ≳4 core
    // radii from every wall the boundary state is the mean flow to ~1e-8, so
    // zero-gradient (transmissive) faces introduce no measurable error.
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, SupersonicOutflow)
        .with(BoundarySide::XHi, SupersonicOutflow)
        .with(BoundarySide::YLo, SupersonicOutflow)
        .with(BoundarySide::YHi, SupersonicOutflow)
        .with_axis(2, SlipWall);

    let flux = match cfg.run.scheme.as_str() {
        "rusanov" => FluxPlugin::<UniformMesh>::rusanov(),
        "hllc" => FluxPlugin::<UniformMesh>::hllc(),
        s => panic!("unknown flux scheme {s:?} (use \"hllc\" or \"rusanov\")"),
    };
    let (integrator, n_sub) = match cfg.run.integrator.as_str() {
        "euler" => (IntegratorPlugin::euler(), 1),
        "rk3" => (IntegratorPlugin::rk3(), 3),
        s => panic!("unknown integrator {s:?} (use \"rk3\" or \"euler\")"),
    };

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(ic))
        .add_plugins(IdealGasPlugin)
        .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(flux)
        .add_plugins(integrator)
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: cfg.run.cfl,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();

    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= cfg.run.t_final {
            break;
        }
        for _ in 0..n_sub {
            app.run();
        }
    }

    let err = {
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        let [ni, nj, _] = mesh.dims();
        let (mut num, mut vol) = (0.0, 0.0);
        for j in 0..nj {
            for i in 0..ni {
                let c = mesh.idx(i, j, 0);
                let exact = vortex_state(&cfg.vortex, t, mesh.cell_centroid(c));
                let cv = mesh.cell_volume(c);
                num += (state.u[c].rho - exact.rho).abs() * cv;
                vol += cv;
            }
        }
        assert!(
            state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
            "vortex run at N={n} went non-physical",
        );
        num / vol
    };
    (err, app)
}

/// Least-squares slope of `log(err)` vs `log(1/h)` — the fitted convergence order.
fn fitted_order(ns: &[usize], errs: &[f64]) -> f64 {
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let m = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    // err ~ h^p = (1/n)^p ⇒ ln err = -p ln n + c ⇒ slope is -p.
    -(m * sxy - sx * sy) / (m * sxx - sx * sx)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: isentropic_vortex <config.toml>");
    let text = std::fs::read_to_string(&path).expect("read config");
    let cfg: Config = toml::from_str(&text).expect("parse config");

    let mut ns = cfg.convergence.resolutions.clone();
    ns.sort_unstable();
    assert!(ns.len() >= 2, "need ≥2 resolutions to measure an order");

    println!(
        "2D isentropic vortex (Shu 1998) — {} + {}, β={}, mean=({},{}), t_final={}",
        cfg.run.scheme,
        cfg.run.integrator,
        cfg.vortex.beta,
        cfg.vortex.mean_u,
        cfg.vortex.mean_v,
        cfg.run.t_final,
    );
    println!("   N     L1(rho)        order");

    let mut errs: Vec<f64> = Vec::new();
    let mut last_app = None;
    for (k, &n) in ns.iter().enumerate() {
        let (e, app) = run_case(&cfg, n);
        if k == 0 {
            println!("{n:5}   {e:.6e}      —");
        } else {
            let p = (errs[k - 1] / e).log2(); // successive grids differ by 2×
            println!("{n:5}   {e:.6e}   {p:.3}");
        }
        errs.push(e);
        last_app = Some((n, app));
    }

    let order = fitted_order(&ns, &errs);
    let lo = cfg.convergence.design_order - cfg.convergence.order_tol;
    let hi = cfg.convergence.design_order + cfg.convergence.order_tol;
    let pass = errs.windows(2).all(|w| w[1] < w[0]) && (lo..=hi).contains(&order);

    println!(
        "\nfitted order = {:.3}  (design {} ± {})  → {}",
        order,
        cfg.convergence.design_order,
        cfg.convergence.order_tol,
        if pass { "PASS" } else { "FAIL" },
    );

    if let Some((n, app)) = last_app {
        let eos = IdealGas {
            gamma: cfg.vortex.gamma,
            r_gas: 1.0,
        };
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        match write_vtk("isentropic_vortex.vtk", &*mesh, &state, &eos) {
            Ok(()) => println!("wrote isentropic_vortex.vtk (finest grid N={n})"),
            Err(e) => eprintln!("VTK write failed: {e}"),
        }
    }

    if !pass {
        std::process::exit(1);
    }
}
