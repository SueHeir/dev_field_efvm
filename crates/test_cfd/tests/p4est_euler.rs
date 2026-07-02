//! End-to-end: run the inviscid Euler solver on a **real p4est forest mesh**
//! (`field_p4est::ForestMesh`). Proves the generic `FvMesh` physics path —
//! `cfl_generic`, halo, boundary, flux, update — runs unchanged on genuine,
//! arbitrary-level p4est AMR. Integration test (own process) for `MPI_Init`.

use cfd_boundary::{boundary_system, BoundaryRegistry};
use cfd_eos::{Eos, EosResource, IdealGas};
use cfd_riemann::Rusanov;
use cfd_solver::{
    cfl_generic_system, flux_system, halo_system, update_system, zero_rhs_system, EulerIntegrator,
    RiemannFluxResource, SolverConfig, SolverState, TimeIntegratorResource,
};
use cfd_state::{CfdState, PrimVar};
use field_core::{register_field_data, CommPlugin, FieldRegistry, FvMesh, MeshScheduleSet, FieldRegistryPlugin};
use field_p4est::{ForestGrid, ForestLayout, ForestMesh};
use grass_app::prelude::*;
use grass_app::ScheduleSetupSet;
use grass_scheduler::Res;
use std::sync::Arc;

fn resize_forest(mesh: Res<ForestMesh>, reg: Res<FieldRegistry>) {
    reg.resize_all(mesh.n_cells_total());
}

#[test]
fn inviscid_euler_runs_on_p4est_forest_and_preserves_uniform() {
    // Build + refine a real p4est forest, then wrap it.
    let layout = ForestLayout {
        trees_x: 4,
        trees_y: 4,
        trees_z: 1,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 0.25,
        min_level: 0,
    };
    let mut grid = ForestGrid::new(layout).expect("forest");
    grid.refine(1, |cx, cy, _cz, _h| ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt() < 0.22);
    let mesh = ForestMesh::new(grid);

    let eos = IdealGas::air();
    let u0 = eos.prim_to_cons(&PrimVar::new(1.0, 18.0, 0.0, 0.0, 100_000.0, 348.4));

    let mut app = App::new();
    app.add_plugins(CommPlugin);
    app.add_plugins(FieldRegistryPlugin);
    app.add_resource(mesh);
    app.add_resource(EosResource(Arc::new(eos)));
    app.add_resource(RiemannFluxResource(Arc::new(Rusanov)));
    app.add_resource(TimeIntegratorResource(Arc::new(EulerIntegrator)));
    app.add_resource(SolverState::default());
    app.add_resource(SolverConfig { cfl: 0.2, fixed_dt: None, viscous: false, muscl: false });
    app.add_resource(BoundaryRegistry::default()); // outflow
    register_field_data!(app, CfdState::default());

    app.add_setup_system(resize_forest, ScheduleSetupSet::Setup);
    app.add_update_system(cfl_generic_system::<ForestMesh>, MeshScheduleSet::Setup);
    app.add_update_system(halo_system::<ForestMesh>, MeshScheduleSet::Halo);
    app.add_update_system(boundary_system::<ForestMesh>, MeshScheduleSet::Boundary);
    app.add_update_system(zero_rhs_system::<ForestMesh>, MeshScheduleSet::PreFlux);
    app.add_update_system(flux_system::<ForestMesh>, MeshScheduleSet::Flux);
    app.add_update_system(update_system, MeshScheduleSet::Update);

    app.prepare();
    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let mut s = reg.get_mut::<CfdState>().unwrap();
        for u in s.u.iter_mut() {
            *u = u0;
        }
    }
    for _ in 0..15 {
        app.run();
    }

    // Uniform flow stays uniform on the p4est AMR mesh (exact RHS cancellation
    // across same-level and coarse/fine faces).
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    for (c, u) in state.u.iter().enumerate() {
        assert!(
            (u.rho - u0.rho).abs() < 1e-9
                && (u.rho_u - u0.rho_u).abs() < 1e-6
                && (u.rho_e - u0.rho_e).abs() < 1e-3,
            "p4est cell {c} drifted from uniform: {u:?}",
        );
    }
    let ss = app.get_resource_ref::<SolverState>().unwrap();
    assert_eq!(ss.step, 15);
    assert!(ss.dt.is_finite() && ss.dt > 0.0);
}
