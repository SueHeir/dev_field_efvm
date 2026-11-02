//! The solver tells a coupler when a whole RK step has finished, so a coupled
//! run can advance it by physical steps without knowing its integrator.

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, SlipWall};
use cfd_eos::{Eos, IdealGas};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
    SolverState,
};
use cfd_state::PrimVar;
use field_core::{FieldDefaultPlugins, UniformMesh, UniformMeshConfig, Vec3};
use grass_app::prelude::*;

fn gas_app(integrator: IntegratorPlugin) -> App {
    let gas = IdealGas::air();
    let u = gas.prim_to_cons(&PrimVar::new(1.0, 10.0, 0.0, 0.0, 1e5, 1e5 / gas.r_gas));
    let mesh = UniformMeshConfig {
        nx: 8,
        ny: 1,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0; 3],
        bounds_hi: [1.0, 0.1, 0.1],
        y_edges: None,
        z_edges: None,
    };
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(move |_: Vec3| u))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(
            BoundaryRegistry::default()
                .with_axis(1, SlipWall)
                .with_axis(2, SlipWall),
        ))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(integrator)
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.4,
            fixed_dt: Some(1e-6),
            viscous: false,
            muscl: false,
        }));
    app.prepare();
    app
}

fn boundaries(app: &mut App, runs: usize) -> Vec<bool> {
    (0..runs)
        .map(|_| {
            app.run();
            app.get_resource_ref::<StepBoundary>()
                .unwrap()
                .completed_full_step
        })
        .collect()
}

#[test]
fn rk3_reports_a_full_step_every_third_run() {
    let mut app = gas_app(IntegratorPlugin::rk3());
    assert_eq!(
        boundaries(&mut app, 6),
        vec![false, false, true, false, false, true]
    );
    assert_eq!(app.get_resource_ref::<SolverState>().unwrap().step, 2);
}

#[test]
fn euler_reports_a_full_step_every_run() {
    let mut app = gas_app(IntegratorPlugin::euler());
    assert_eq!(boundaries(&mut app, 3), vec![true, true, true]);
}
