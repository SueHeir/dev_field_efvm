//! Boundary conditions can use where the face is, and outside states another
//! code supplies through `BoundaryInputs`, instead of only the interior cell.

use cfd_boundary::{
    BoundaryFace, BoundaryInputs, BoundaryPlugin, BoundaryRegistry, FnBoundary, FromOutside,
    SlipWall,
};
use cfd_eos::{Eos, IdealGas};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    BoundarySide, FieldDefaultPlugins, FieldRegistry, StructuredMesh, UniformMesh,
    UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;

fn gas() -> IdealGas {
    IdealGas::air()
}

fn state(rho: f64, u: f64) -> ConsVar {
    let p = 1e5;
    gas().prim_to_cons(&PrimVar::new(rho, u, 0.0, 0.0, p, p / (rho * gas().r_gas)))
}

/// 6 x 4 cells; `x_lo` and `x_hi` conditions under test, slip walls elsewhere.
fn app(bcs: BoundaryRegistry) -> App {
    let mesh = UniformMeshConfig {
        nx: 6,
        ny: 4,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0; 3],
        bounds_hi: [1.0, 1.0, 0.1],
        y_edges: None,
        z_edges: None,
    };
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(|_: Vec3| state(1.0, 0.0)))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(
            bcs.with_axis(1, SlipWall).with_axis(2, SlipWall),
        ))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::euler())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.4,
            fixed_dt: Some(1e-9),
            viscous: false,
            muscl: false,
        }));
    app.prepare();
    app
}

/// The first ghost layer beyond x-lo or x-hi, with each cell's y index.
fn x_ghosts(app: &App, hi: bool) -> Vec<(usize, usize)> {
    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let ng = mesh.n_ghost();
    let [ni, nj, _] = mesh.total_dims();
    let i = if hi { ni - ng } else { ng - 1 };
    (ng..nj - ng)
        .map(|j| (j - ng, mesh.idx_raw(i, j, ng)))
        .collect()
}

fn u_at(app: &App, c: usize) -> ConsVar {
    app.get_resource_ref::<FieldRegistry>()
        .unwrap()
        .expect::<CfdState>("test")
        .u[c]
}

#[test]
fn a_condition_can_depend_on_where_the_face_is() {
    // A parabolic inlet profile in y.
    let inlet = FnBoundary(|_: &ConsVar, f: &BoundaryFace, _: &dyn Eos| {
        let y = f.centroid[1];
        state(1.0, 400.0 * y * (1.0 - y))
    });
    let mut app = app(BoundaryRegistry::default().with(BoundarySide::XLo, inlet));
    app.run();
    let ghosts = x_ghosts(&app, false);
    let speeds: Vec<f64> = ghosts.iter().map(|&(_, c)| u_at(&app, c).rho_u).collect();
    for (&(j, _), u) in ghosts.iter().zip(&speeds) {
        let y = (j as f64 + 0.5) / 4.0;
        assert!((u - 400.0 * y * (1.0 - y)).abs() < 1e-12, "row {j}: {u}");
    }
    assert!(speeds[1] > speeds[0], "profile is not flat: {speeds:?}");
}

#[test]
fn another_code_can_supply_the_outside_state() {
    let mut app = app(BoundaryRegistry::default().with(BoundarySide::XHi, FromOutside::new()));
    let ghosts = x_ghosts(&app, true);
    // Supply rows 0 and 2 only; the others fall back to zero-gradient.
    let given = |j: usize| state(1.0 + 0.1 * j as f64, -50.0);
    {
        let mut inputs = app.get_resource_mut::<BoundaryInputs>().unwrap();
        for &(j, c) in &ghosts {
            if j % 2 == 0 {
                inputs.0.insert(c, given(j));
            }
        }
    }
    app.run();
    for &(j, c) in &ghosts {
        let u = u_at(&app, c);
        if j % 2 == 0 {
            assert_eq!(u, given(j), "row {j}");
        } else {
            assert_eq!(u.rho_u, 0.0, "row {j} should copy the interior");
        }
    }
}
