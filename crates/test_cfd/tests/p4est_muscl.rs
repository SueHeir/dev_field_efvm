//! MUSCL 2nd-order reconstruction on a real p4est AMR forest. A pressure pulse
//! sloshes in a closed (slip-wall) box whose centre is refined, so the pulse
//! repeatedly crosses a coarse/fine interface — exactly where 1st-order Rusanov
//! struggled. We assert (1) mass is conserved (closed box) with MUSCL, and
//! (2) MUSCL is **less dissipative** than the 1st-order flux — it retains a
//! higher peak pressure after the same number of steps.
//! Integration test (own process) for `MPI_Init`.

use cfd_boundary::{boundary_system, BoundaryRegistry, SlipWall};
use cfd_eos::{Eos, EosResource, IdealGas};
use cfd_riemann::Rusanov;
use cfd_solver::{
    cfl_generic_system, compute_slopes_system, flux_system, halo_system, muscl_flux_system,
    update_system, zero_rhs_system, EulerIntegrator, MusclSlopes, RiemannFluxResource, SlopeLimiter,
    SolverConfig, SolverState, TimeIntegratorResource,
};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    register_field_data, BoundarySide, CommPlugin, FieldRegistry, FieldRegistryPlugin, FvMesh,
    MeshScheduleSet,
};
use field_p4est::{ForestGrid, ForestLayout, ForestMesh};
use grass_app::prelude::*;
use grass_app::ScheduleSetupSet;
use grass_scheduler::Res;
use std::sync::Arc;

fn resize_forest(mesh: Res<ForestMesh>, reg: Res<FieldRegistry>) {
    reg.resize_all(mesh.n_cells_total());
}

/// Run the closed-box pulse for `steps` and return (total mass, peak pressure).
fn run_box(muscl: bool, steps: usize) -> (f64, f64) {
    let layout = ForestLayout {
        trees_x: 8,
        trees_y: 8,
        trees_z: 1,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 0.125,
        min_level: 0,
    };
    let mut grid = ForestGrid::new(layout).expect("forest");
    // Refine the central band so the pulse crosses coarse/fine interfaces.
    grid.refine(1, |cx, cy, _cz, _h| (cx - 0.5).abs() < 0.25 && (cy - 0.5).abs() < 0.25);
    let mesh = ForestMesh::new(grid);

    let eos = IdealGas::air();
    let pulse = |x: field_core::Vec3| {
        let r2 = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
        let p = if r2 < 0.1 * 0.1 { 2.0e5 } else { 1.0e5 };
        eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, p, p / 287.058))
    };

    let mut app = App::new();
    app.add_plugins(CommPlugin);
    app.add_plugins(FieldRegistryPlugin);
    app.add_resource(mesh);
    app.add_resource(EosResource(Arc::new(eos)));
    app.add_resource(RiemannFluxResource(Arc::new(Rusanov)));
    app.add_resource(TimeIntegratorResource(Arc::new(EulerIntegrator)));
    app.add_resource(SolverState::default());
    app.add_resource(SolverConfig { cfl: 0.2, fixed_dt: None, viscous: false, muscl });
    app.add_resource(MusclSlopes::default());
    app.add_resource(SlopeLimiter::default()); // minmod — the historical MUSCL limiter
    // Closed box: slip walls on all six sides ⇒ no boundary mass flux.
    let mut bcs = BoundaryRegistry::default();
    for side in [
        BoundarySide::XLo,
        BoundarySide::XHi,
        BoundarySide::YLo,
        BoundarySide::YHi,
        BoundarySide::ZLo,
        BoundarySide::ZHi,
    ] {
        bcs.set(side, SlipWall);
    }
    app.add_resource(bcs);
    register_field_data!(app, CfdState::default());

    app.add_setup_system(resize_forest, ScheduleSetupSet::Setup);
    app.add_update_system(cfl_generic_system::<ForestMesh>, MeshScheduleSet::Setup);
    app.add_update_system(halo_system::<ForestMesh>, MeshScheduleSet::Halo);
    app.add_update_system(boundary_system::<ForestMesh>, MeshScheduleSet::Boundary);
    app.add_update_system(zero_rhs_system::<ForestMesh>, MeshScheduleSet::PreFlux);
    if muscl {
        app.add_update_system(compute_slopes_system::<ForestMesh>, MeshScheduleSet::PreFlux);
        app.add_update_system(muscl_flux_system::<ForestMesh>, MeshScheduleSet::Flux);
    } else {
        app.add_update_system(flux_system::<ForestMesh>, MeshScheduleSet::Flux);
    }
    app.add_update_system(update_system, MeshScheduleSet::Update);

    app.prepare();
    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
        let mut s = reg.get_mut::<CfdState>().unwrap();
        for c in 0..mesh.n_cells_total() {
            s.u[c] = pulse(mesh.cell_centroid(c));
        }
    }
    for _ in 0..steps {
        app.run();
    }

    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let mut mass = 0.0;
    let mut peak = 0.0f64;
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        mass += state.u[c].rho * mesh.cell_volume(c);
        peak = peak.max(IdealGas::air().pressure(&state.u[c]));
    }
    (mass, peak)
}

#[test]
fn muscl_preserves_uniform_flow_on_the_forest() {
    // Uniform field everywhere ⇒ every minmod slope is 0 ⇒ reconstruction is the
    // cell value ⇒ MUSCL must hold uniform flow exactly, just like 1st order.
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
    let u0 = eos.prim_to_cons(&PrimVar::new(1.0, 18.0, -7.0, 0.0, 100_000.0, 348.4));

    let mut app = App::new();
    app.add_plugins(CommPlugin);
    app.add_plugins(FieldRegistryPlugin);
    app.add_resource(mesh);
    app.add_resource(EosResource(Arc::new(eos)));
    app.add_resource(RiemannFluxResource(Arc::new(Rusanov)));
    app.add_resource(TimeIntegratorResource(Arc::new(EulerIntegrator)));
    app.add_resource(SolverState::default());
    app.add_resource(SolverConfig { cfl: 0.2, fixed_dt: None, viscous: false, muscl: true });
    app.add_resource(MusclSlopes::default());
    app.add_resource(SlopeLimiter::default()); // minmod — the historical MUSCL limiter
    app.add_resource(BoundaryRegistry::default()); // outflow
    register_field_data!(app, CfdState::default());

    app.add_setup_system(resize_forest, ScheduleSetupSet::Setup);
    app.add_update_system(cfl_generic_system::<ForestMesh>, MeshScheduleSet::Setup);
    app.add_update_system(halo_system::<ForestMesh>, MeshScheduleSet::Halo);
    app.add_update_system(boundary_system::<ForestMesh>, MeshScheduleSet::Boundary);
    app.add_update_system(zero_rhs_system::<ForestMesh>, MeshScheduleSet::PreFlux);
    app.add_update_system(compute_slopes_system::<ForestMesh>, MeshScheduleSet::PreFlux);
    app.add_update_system(muscl_flux_system::<ForestMesh>, MeshScheduleSet::Flux);
    app.add_update_system(update_system, MeshScheduleSet::Update);

    app.prepare();
    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let mut s = reg.get_mut::<CfdState>().unwrap();
        for u in s.u.iter_mut() {
            *u = u0;
        }
    }
    for _ in 0..12 {
        app.run();
    }

    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    for (c, u) in state.u.iter().enumerate() {
        assert!(
            (u.rho - u0.rho).abs() < 1e-9
                && (u.rho_u - u0.rho_u).abs() < 1e-6
                && (u.rho_e - u0.rho_e).abs() < 1e-3,
            "MUSCL drifted from uniform at cell {c}: {u:?}",
        );
    }
}

#[test]
fn muscl_conserves_mass_and_is_less_dissipative_than_first_order() {
    let steps = 40;
    let (mass0, _) = run_box(true, 0); // initial mass
    let (mass_muscl, peak_muscl) = run_box(true, steps);
    let (mass_fo, peak_fo) = run_box(false, steps);

    // Closed box ⇒ MUSCL conserves mass.
    assert!(
        (mass_muscl - mass0).abs() / mass0 < 1e-10,
        "MUSCL lost mass in the closed box: {mass0} -> {mass_muscl}",
    );
    // 1st-order conserves mass too (sanity that the comparison is fair).
    assert!((mass_fo - mass0).abs() / mass0 < 1e-10, "1st-order lost mass: {mass0} -> {mass_fo}");

    // Both have diffused below the initial 2e5 peak, but MUSCL — being 2nd-order
    // — retains a strictly higher peak than the 1st-order flux.
    assert!(peak_muscl < 2.0e5 && peak_fo < 2.0e5, "pulse should have diffused");
    assert!(
        peak_muscl > peak_fo * 1.001,
        "MUSCL should be less dissipative: peak {peak_muscl:.1} vs 1st-order {peak_fo:.1}",
    );
}
