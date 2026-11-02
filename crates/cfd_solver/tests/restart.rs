//! A run saved part-way with `grass_io::save_app` and restored into a fresh
//! App continues bit for bit: the conserved fields come back through FIELD's
//! registry and the solver's time, step and RK stage through its own entry.

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, SlipWall};
use cfd_eos::{Eos, IdealGas};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
    SolverState,
};
use cfd_state::{CfdState, PrimVar};
use field_core::{FieldDefaultPlugins, FieldRegistry, UniformMesh, UniformMeshConfig, Vec3};
use grass_app::prelude::*;
use grass_io::{restore_app, save_app};

/// A density and pressure pulse in a slip-walled box, with a CFL time step,
/// so both the fields and the step size change every step.
fn pulse_app() -> App {
    let gas = IdealGas::air();
    let mesh = UniformMeshConfig {
        nx: 24,
        ny: 1,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0; 3],
        bounds_hi: [1.0, 0.05, 0.05],
        y_edges: None,
        z_edges: None,
    };
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(move |x: Vec3| {
            let bump = (-((x[0] - 0.4) / 0.08).powi(2)).exp();
            let rho = 1.0 + 0.5 * bump;
            let p = 1e5 * (1.0 + 0.8 * bump);
            gas.prim_to_cons(&PrimVar::new(rho, 30.0, 0.0, 0.0, p, p / (rho * gas.r_gas)))
        }))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(
            BoundaryRegistry::default()
                .with_axis(0, SlipWall)
                .with_axis(1, SlipWall)
                .with_axis(2, SlipWall),
        ))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.4,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();
    app
}

/// Run whole RK steps.
fn steps(app: &mut App, n: usize) {
    for _ in 0..n {
        loop {
            app.run();
            if app
                .get_resource_ref::<StepBoundary>()
                .unwrap()
                .completed_full_step
            {
                break;
            }
        }
    }
}

fn state(app: &App) -> (Vec<[f64; 5]>, f64, u64) {
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let u = reg
        .expect::<CfdState>("test")
        .u
        .iter()
        .map(|c| [c.rho, c.rho_u, c.rho_v, c.rho_w, c.rho_e])
        .collect();
    let s = app.get_resource_ref::<SolverState>().unwrap();
    (u, s.time, s.step)
}

#[test]
fn restored_run_continues_bit_for_bit() {
    let mut reference = pulse_app();
    steps(&mut reference, 12);
    let saved = save_app(&reference);
    steps(&mut reference, 18);

    let mut restored = pulse_app();
    restore_app(&restored, &saved).unwrap();
    steps(&mut restored, 18);

    let (u_ref, t_ref, n_ref) = state(&reference);
    let (u_new, t_new, n_new) = state(&restored);
    assert_eq!(n_new, n_ref);
    assert_eq!(t_new.to_bits(), t_ref.to_bits());
    assert_eq!(u_new, u_ref, "restored fields drifted from the reference");

    // The pulse actually moved, so the comparison is not trivial.
    let (u0, _, _) = state(&pulse_app());
    assert_ne!(u0, u_ref);
}
