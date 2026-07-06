//! Periodic contact-wave advection gate.
//!
//! A smooth density pulse with constant pressure and velocity is an exact Euler
//! entropy/contact wave: it advects at the uniform velocity without changing
//! shape. The case sends the pulse across the high-x periodic face, checks mass
//! conservation, and measures the expected first-order accuracy of the
//! piecewise-constant HLLC finite-volume scheme.

use cfd_boundary::{BoundaryRegistry, Periodic, SlipWall, StructuredBoundaryPlugin};
use cfd_eos::{Eos, IdealGas};
use cfd_output::write_vtk;
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
    SolverState,
};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, StructuredMesh, UniformMesh, UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct Config {
    pulse: PulseCfg,
    run: RunCfg,
    convergence: ConvergenceCfg,
    conservation: ConservationCfg,
}

#[derive(Debug, Clone, Deserialize)]
struct PulseCfg {
    domain: f64,
    center: f64,
    sigma: f64,
    amplitude: f64,
    gamma: f64,
    velocity: f64,
    pressure: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct RunCfg {
    t_final: f64,
    cfl: f64,
    scheme: String,
    integrator: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ConvergenceCfg {
    resolutions: Vec<usize>,
    design_order: f64,
    order_tol: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct ConservationCfg {
    mass_tol: f64,
}

fn pulse_state(p: &PulseCfg, t: f64, x: Vec3) -> ConsVar {
    let eos = IdealGas { gamma: p.gamma, r_gas: 1.0 };
    let center = (p.center + p.velocity * t).rem_euclid(p.domain);
    let dx = periodic_delta(x[0] - center, p.domain);
    let rho = 1.0 + p.amplitude * (-0.5 * (dx / p.sigma).powi(2)).exp();
    let temp = p.pressure / rho;
    eos.prim_to_cons(&PrimVar::new(rho, p.velocity, 0.0, 0.0, p.pressure, temp))
}

fn periodic_delta(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

fn run_case(cfg: &Config, n: usize) -> (f64, f64, App) {
    let l = cfg.pulse.domain;
    let pcfg = cfg.pulse.clone();
    let ic = move |x: Vec3| pulse_state(&pcfg, 0.0, x);
    let mesh = UniformMeshConfig {
        nx: n,
        ny: 1,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [l, l / n as f64, l / n as f64],
        y_edges: None,
        z_edges: None,
    };
    let bcs = BoundaryRegistry::default()
        .with_axis(0, Periodic)
        .with_axis(1, SlipWall)
        .with_axis(2, SlipWall);

    let flux = match cfg.run.scheme.as_str() {
        "hllc" => FluxPlugin::<UniformMesh>::hllc(),
        "rusanov" => FluxPlugin::<UniformMesh>::rusanov(),
        s => panic!("unknown flux scheme {s:?}"),
    };
    let (integrator, n_sub) = match cfg.run.integrator.as_str() {
        "rk3" => (IntegratorPlugin::rk3(), 3),
        "euler" => (IntegratorPlugin::euler(), 1),
        s => panic!("unknown integrator {s:?}"),
    };

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(ic))
        .add_plugins(IdealGasPlugin)
        .add_plugins(StructuredBoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(flux)
        .add_plugins(integrator)
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: cfg.run.cfl,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();
    let initial_mass = total_mass(&app);

    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= cfg.run.t_final {
            break;
        }
        for _ in 0..n_sub {
            app.run();
        }
    }

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let t = app.get_resource_ref::<SolverState>().unwrap().time;
    let (mut num, mut vol) = (0.0, 0.0);
    for i in 0..n {
        let c = mesh.idx(i, 0, 0);
        let exact = pulse_state(&cfg.pulse, t, mesh.cell_centroid(c));
        let cv = mesh.cell_volume(c);
        num += (state.u[c].rho - exact.rho).abs() * cv;
        vol += cv;
    }
    assert!(
        state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
        "periodic pulse run at N={n} went non-physical",
    );
    let err = num / vol;
    let mass_drift = ((total_mass(&app) - initial_mass) / initial_mass).abs();
    drop(state);
    drop(reg);
    drop(mesh);
    (err, mass_drift, app)
}

fn total_mass(app: &App) -> f64 {
    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    state
        .u
        .iter()
        .enumerate()
        .filter(|(c, _)| mesh.is_local_cell(*c))
        .map(|(c, u)| u.rho * mesh.cell_volume(c))
        .sum()
}

fn fitted_order(ns: &[usize], errs: &[f64]) -> f64 {
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let m = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    -(m * sxy - sx * sy) / (m * sxx - sx * sx)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: periodic_advected_pulse <config.toml>");
    let text = std::fs::read_to_string(&path).expect("read config");
    let cfg: Config = toml::from_str(&text).expect("parse config");
    let mut ns = cfg.convergence.resolutions.clone();
    ns.sort_unstable();
    assert!(ns.len() >= 2, "need at least two resolutions");

    println!(
        "periodic advected density pulse -- {} + {}, u={}, t_final={}",
        cfg.run.scheme, cfg.run.integrator, cfg.pulse.velocity, cfg.run.t_final
    );
    println!("   N     L1(rho)        order     mass drift");

    let mut errs: Vec<f64> = Vec::new();
    let mut mass_drifts: Vec<f64> = Vec::new();
    let mut last_app = None;
    for (k, &n) in ns.iter().enumerate() {
        let (e, dm, app) = run_case(&cfg, n);
        if k == 0 {
            println!("{n:5}   {e:.6e}      -       {dm:.3e}");
        } else {
            let p = (errs[k - 1] / e).log2();
            println!("{n:5}   {e:.6e}   {p:.3}     {dm:.3e}");
        }
        errs.push(e);
        mass_drifts.push(dm);
        last_app = Some((n, app));
    }

    let order = fitted_order(&ns, &errs);
    let lo = cfg.convergence.design_order - cfg.convergence.order_tol;
    let hi = cfg.convergence.design_order + cfg.convergence.order_tol;
    let mass_max = mass_drifts.iter().copied().fold(0.0, f64::max);
    let pass = errs.windows(2).all(|w| w[1] < w[0])
        && (lo..=hi).contains(&order)
        && mass_max <= cfg.conservation.mass_tol;
    println!(
        "\nfitted order = {:.3} (design {} +/- {}), max mass drift = {:.3e} (tol {:.1e}) -> {}",
        order,
        cfg.convergence.design_order,
        cfg.convergence.order_tol,
        mass_max,
        cfg.conservation.mass_tol,
        if pass { "PASS" } else { "FAIL" },
    );

    if let Some((n, app)) = last_app {
        let eos = IdealGas { gamma: cfg.pulse.gamma, r_gas: 1.0 };
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        match write_vtk("periodic_advected_pulse.vtk", &*mesh, &state, &eos) {
            Ok(()) => println!("wrote periodic_advected_pulse.vtk (finest grid N={n})"),
            Err(e) => eprintln!("VTK write failed: {e}"),
        }
    }

    if !pass {
        std::process::exit(1);
    }
}
