//! The CFD tier on an unstructured mesh with slanted walls.
//!
//! FIELD's `UnstructuredMesh` runs the mesh-generic CFD plugins
//! (`CfdCorePlugin`, `FluxPlugin`, `BoundaryPlugin`) unchanged. The mesh is a
//! jittered triangular-prism mesh sheared by `x ← x + k y`, so its XLo/XHi
//! walls are slanted. A uniform flow parallel to those walls must stay exactly
//! uniform with `SlipWall` there: the slip wall reflects momentum along the
//! face normal. The negative control reflects x-momentum instead (the old
//! box-side behaviour) and must disturb the flow.

use cfd_boundary::{BoundaryCondition, BoundaryFace, BoundaryPlugin, BoundaryRegistry, SlipWall};
use cfd_eos::{Eos, IdealGas};
use cfd_solver::{CfdCorePlugin, FluxPlugin, IntegratorPlugin, IdealGasPlugin, SolverConfig};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    BoundarySide, CommPlugin, FieldRegistry, FieldRegistryPlugin, FvMesh, PolyMesh, UnstructuredMesh,
    UnstructuredMeshPlugin,
};
use grass_app::prelude::*;
use std::sync::Arc;

const SHEAR: f64 = 0.6;

/// Apply `x ← x + k y` (determinant 1): centroids map by `F`, area normals by
/// `F⁻ᵀ`, volumes are unchanged.
fn sheared(mut m: PolyMesh, k: f64) -> PolyMesh {
    for c in &mut m.cells {
        c.centroid[0] += k * c.centroid[1];
    }
    for f in &mut m.faces {
        f.centroid[0] += k * f.centroid[1];
        f.area_normal[1] -= k * f.area_normal[0];
    }
    m
}

/// The pre-fix slip wall: flip x-momentum whatever the face normal is.
#[derive(Clone)]
struct AxisSlipWall;
impl BoundaryCondition for AxisSlipWall {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, _eos: &dyn Eos) -> ConsVar {
        ConsVar { rho_u: -interior.rho_u, ..*interior }
    }
}

fn free_stream() -> ConsVar {
    // Speed 50 m/s along the wall tangent (k, 1, 0).
    let n = (SHEAR * SHEAR + 1.0f64).sqrt();
    let (u, v) = (50.0 * SHEAR / n, 50.0 / n);
    IdealGas::air().prim_to_cons(&PrimVar::new(1.0, u, v, 0.0, 100_000.0, 348.4))
}

/// Largest relative deviation from the free stream after `steps` steps.
/// Both tests build an App with `CommPlugin`, and MPI calls from two test
/// threads at once intermittently abort (`free(): invalid size` inside
/// `MPI_Allreduce`). Run them one at a time.
static MPI_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run(walls: impl BoundaryCondition + Clone + 'static, steps: usize) -> f64 {
    let _mpi = MPI_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mesh = sheared(PolyMesh::triangle_prisms(16, 12, [0.0, 0.0], [1.0, 1.0], 0.1, 0.3), SHEAR);
    mesh.validate().expect("sheared mesh is closed");
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, walls.clone())
        .with(BoundarySide::XHi, walls)
        .with_axis(2, SlipWall);
    let u0 = free_stream();
    let mut app = App::new();
    app.add_plugins(CommPlugin)
        .add_plugins(FieldRegistryPlugin)
        .add_plugins(UnstructuredMeshPlugin { mesh: Arc::new(mesh) })
        .add_plugins(IdealGasPlugin)
        .add_plugins(IntegratorPlugin::euler())
        .add_plugins(BoundaryPlugin::<UnstructuredMesh>::new(bcs))
        .add_plugins(FluxPlugin::<UnstructuredMesh>::rusanov())
        .add_plugins(CfdCorePlugin::<UnstructuredMesh>::new(
            SolverConfig { cfl: 0.3, fixed_dt: None, viscous: false, muscl: false },
            move |_x| u0,
        ));
    app.prepare();
    for _ in 0..steps {
        app.run();
    }
    let mesh = app.get_resource_ref::<UnstructuredMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let s = reg.expect::<CfdState>("CfdState");
    let scale = u0.rho_e.abs();
    (0..mesh.n_cells_total())
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| {
            let d = s.u[c];
            [d.rho - u0.rho, d.rho_u - u0.rho_u, d.rho_v - u0.rho_v, d.rho_w - u0.rho_w, d.rho_e - u0.rho_e]
                .iter()
                .map(|x| x.abs() / scale)
                .fold(0.0, f64::max)
        })
        .fold(0.0, f64::max)
}

/// `SlipWall` itself, made `Clone` so both walls can share it.
#[derive(Clone)]
struct Slip;
impl BoundaryCondition for Slip {
    fn ghost(&self, interior: &ConsVar, side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        SlipWall.ghost(interior, side, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        SlipWall.ghost_at(interior, face, eos)
    }
}

#[test]
fn slip_walls_preserve_flow_along_slanted_walls() {
    let dev = run(Slip, 40);
    assert!(dev < 1e-12, "free stream drifted by {dev:e}");
}

#[test]
fn axis_reflection_disturbs_flow_along_slanted_walls() {
    let dev = run(AxisSlipWall, 40);
    assert!(dev > 1e-4, "negative control should disturb the flow, got {dev:e}");
}
