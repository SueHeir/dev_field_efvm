//! Order-of-accuracy gate for the Shu isentropic vortex on a statically refined
//! p4est AMR forest.
//!
//! The uniform-grid order gates prove the CFD physics on `UniformMesh`; this test
//! exercises the same smooth exact solution on the real `field_p4est::ForestMesh`
//! path. The forest is statically refined around the vortex trajectory, then the
//! base forest resolution is swept by 2x. The current generic p4est flux path is
//! piecewise-constant Rusanov, so its design order is 1; the fitted L1 density
//! rate must match that first-order design band.
//!
//! Reference solution: Shu, ICASE Report 97-65 (1998), section 4.1.

use cfd_boundary::{boundary_system, BoundaryRegistry, SlipWall, SupersonicOutflow};
use cfd_eos::{Eos, EosResource, IdealGas};
use cfd_riemann::Rusanov;
use cfd_solver::{
    cfl_generic_system, flux_system, halo_system, update_system, zero_rhs_system,
    RiemannFluxResource, Rk3Integrator, SolverConfig, SolverState, TimeIntegratorResource,
};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    register_field_data, BoundarySide, CommPlugin, FieldRegistry, FieldRegistryPlugin, FvMesh,
    MeshScheduleSet, Vec3,
};
use field_p4est::{ForestGrid, ForestLayout, ForestMesh};
use grass_app::prelude::*;
use grass_app::ScheduleSetupSet;
use grass_scheduler::Res;
use std::sync::Arc;

const GAMMA: f64 = 1.4;
const BETA: f64 = 5.0;
const DOMAIN: f64 = 10.0;
const MEAN_U: f64 = 0.5;
const MEAN_V: f64 = 0.5;
const T_FINAL: f64 = 1.0;

fn resize_forest(mesh: Res<ForestMesh>, reg: Res<FieldRegistry>) {
    reg.resize_all(mesh.n_cells_total());
}

fn vortex_state(t: f64, x: Vec3) -> ConsVar {
    let eos = IdealGas {
        gamma: GAMMA,
        r_gas: 1.0,
    };
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

fn static_vortex_forest(base_n: usize) -> ForestMesh {
    let mut grid = ForestGrid::new(ForestLayout {
        trees_x: base_n as i32,
        trees_y: base_n as i32,
        trees_z: 1,
        xmin: 0.0,
        xmax: DOMAIN,
        ymin: 0.0,
        ymax: DOMAIN,
        zmin: 0.0,
        // Keep cells cubic, so `cfl_generic_system`'s volume^(1/3) length scale
        // matches the x/y grid spacing for this 2D-in-a-thin-slab vortex.
        zmax: DOMAIN / base_n as f64,
        min_level: 0,
    })
    .expect("p4est forest");

    // Refine a fixed box around the whole vortex trajectory: from (5,5) to
    // (5.5,5.5), with a wide quiet-flow margin before the coarse/fine seam.
    grid.refine(1, |cx, cy, _cz, _h| {
        (2.5..=7.5).contains(&cx) && (2.5..=7.5).contains(&cy)
    });
    ForestMesh::new(grid)
}

fn build_app(mesh: ForestMesh) -> App {
    let eos = IdealGas {
        gamma: GAMMA,
        r_gas: 1.0,
    };
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, SupersonicOutflow)
        .with(BoundarySide::XHi, SupersonicOutflow)
        .with(BoundarySide::YLo, SupersonicOutflow)
        .with(BoundarySide::YHi, SupersonicOutflow)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(CommPlugin);
    app.add_plugins(FieldRegistryPlugin);
    app.add_resource(mesh);
    app.add_resource(EosResource(Arc::new(eos)));
    app.add_resource(RiemannFluxResource(Arc::new(Rusanov)));
    app.add_resource(TimeIntegratorResource(Arc::new(Rk3Integrator)));
    app.add_resource(SolverState::default());
    app.add_resource(SolverConfig {
        cfl: 0.3,
        fixed_dt: None,
        viscous: false,
        muscl: false,
    });
    app.add_resource(bcs);
    register_field_data!(app, CfdState::default());

    app.add_setup_system(resize_forest, ScheduleSetupSet::Setup);
    app.add_update_system(cfl_generic_system::<ForestMesh>, MeshScheduleSet::Setup);
    app.add_update_system(halo_system::<ForestMesh>, MeshScheduleSet::Halo);
    app.add_update_system(boundary_system::<ForestMesh>, MeshScheduleSet::Boundary);
    app.add_update_system(zero_rhs_system::<ForestMesh>, MeshScheduleSet::PreFlux);
    app.add_update_system(flux_system::<ForestMesh>, MeshScheduleSet::Flux);
    app.add_update_system(update_system, MeshScheduleSet::Update);
    app.prepare();
    app
}

fn l1_density_error(base_n: usize) -> f64 {
    let mut app = build_app(static_vortex_forest(base_n));
    {
        let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let mut state = reg.get_mut::<CfdState>().unwrap();
        for c in 0..mesh.n_cells_total() {
            state.u[c] = vortex_state(0.0, mesh.cell_centroid(c));
        }
    }

    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= T_FINAL {
            break;
        }
        for _ in 0..3 {
            app.run();
        }
    }

    let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let t = app.get_resource_ref::<SolverState>().unwrap().time;
    let (mut num, mut vol) = (0.0, 0.0);
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let exact = vortex_state(t, mesh.cell_centroid(c));
        let cv = mesh.cell_volume(c);
        num += (state.u[c].rho - exact.rho).abs() * cv;
        vol += cv;
    }
    assert!(
        state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
        "p4est vortex at base N={base_n} went non-physical",
    );
    num / vol
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

#[test]
fn p4est_static_amr_vortex_is_first_order_accurate() {
    let ns = [32usize, 64, 128];
    let errs: Vec<f64> = ns.iter().map(|&n| l1_density_error(n)).collect();
    let orders: Vec<f64> = (1..ns.len())
        .map(|k| (errs[k - 1] / errs[k]).log2())
        .collect();
    let fitted = fitted_order(&ns, &errs);

    eprintln!("p4est static-AMR isentropic vortex convergence (L1 density error):");
    eprintln!("  base N={:4}  err={:.6e}", ns[0], errs[0]);
    for k in 1..ns.len() {
        eprintln!(
            "  base N={:4}  err={:.6e}  order={:.3}",
            ns[k],
            errs[k],
            orders[k - 1]
        );
    }
    eprintln!("  fitted order = {fitted:.3} (design 1.0)");

    assert!(
        errs.windows(2).all(|w| w[1] < w[0]),
        "p4est AMR L1 error not monotonically decreasing: {errs:?}",
    );
    assert!(
        (0.8..=1.2).contains(&fitted),
        "p4est AMR fitted order {fitted:.3} not within 1.0 +/- 0.2",
    );
    let finest = *orders.last().unwrap();
    assert!(
        finest >= 0.85,
        "p4est AMR finest-pair order {finest:.3} below 0.85"
    );
    assert!(
        errs[ns.len() - 1] < 4.0e-3,
        "p4est AMR finest-grid L1 error too large: {}",
        errs[ns.len() - 1]
    );
}
