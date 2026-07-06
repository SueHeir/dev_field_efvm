//! Boundary conditions: trait, registry, stock implementations, and the
//! `Boundary`-phase plugin.
//!
//! Ported from toy-cfd's `cfd_boundary`, re-expressed against FIELD's `Face`
//! model. toy-cfd identified a face by `(dim, is_lo)` and reached into a `Grid`;
//! here a physical-boundary [`Face`](field_core::Face) already carries its
//! [`BoundarySide`], and the BC only needs the interior owner state. So the trait
//! shrinks to **owner-only**: `ghost(interior, side, eos) -> ConsVar`. Adding a BC
//! is still just one struct impl — no central enum, no framework edit.
//!
//! Periodic faces are handled as a FIELD-style forward halo wrap: registered
//! forward fields are packed from the opposite owned slab and unpacked into the
//! ghost slab before the owner-only physical BC fill runs.

use cfd_eos::{Eos, EosResource};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{BoundarySide, FieldRegistry, FvMesh, StructuredMesh};
use grass_app::prelude::*;
use grass_scheduler::Res;
use std::sync::Arc;

// ─── BoundaryCondition trait ─────────────────────────────────────────────────

/// A rule for the ghost state on one physical-boundary face, given the interior
/// owner state, which [`BoundarySide`] the face lies on, and the EOS.
pub trait BoundaryCondition: Send + Sync {
    fn ghost(&self, interior: &ConsVar, side: BoundarySide, eos: &dyn Eos) -> ConsVar;
    fn is_periodic(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "<unnamed>"
    }
}

// ─── Stock boundary conditions ───────────────────────────────────────────────

/// Supersonic outflow / zero-gradient: copy the interior state into the ghost.
pub struct SupersonicOutflow;
impl BoundaryCondition for SupersonicOutflow {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, _eos: &dyn Eos) -> ConsVar {
        *interior
    }
    fn name(&self) -> &'static str {
        "SupersonicOutflow"
    }
}

/// Periodic side marker.
///
/// Unlike owner-only physical BCs, this does not synthesize a state from the
/// adjacent owner cell. Use [`StructuredBoundaryPlugin`] (or call
/// [`periodic_wrap_system`] before [`boundary_system`]) so the opposite owned slab
/// is copied through FIELD's forward pack/unpack path into the ghost slab.
pub struct Periodic;
impl BoundaryCondition for Periodic {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, _eos: &dyn Eos) -> ConsVar {
        *interior
    }
    fn is_periodic(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "Periodic"
    }
}

/// Supersonic inflow: prescribed `(ρ, u, v, w, p)`. Temperature derived via
/// `T = p / (ρ R)` (air); use [`SubsonicInflow`] when an exact T matters.
pub struct SupersonicInflow {
    pub rho: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub p: f64,
}
impl BoundaryCondition for SupersonicInflow {
    fn ghost(&self, _interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        let t = self.p / (self.rho * 287.058);
        eos.prim_to_cons(&PrimVar::new(self.rho, self.u, self.v, self.w, self.p, t))
    }
    fn name(&self) -> &'static str {
        "SupersonicInflow"
    }
}

/// Inviscid slip wall: reflect the wall-normal momentum, copy everything else.
pub struct SlipWall;
impl BoundaryCondition for SlipWall {
    fn ghost(&self, interior: &ConsVar, side: BoundarySide, _eos: &dyn Eos) -> ConsVar {
        let mut state = *interior;
        match side.axis() {
            0 => state.rho_u = -state.rho_u,
            1 => state.rho_v = -state.rho_v,
            2 => state.rho_w = -state.rho_w,
            _ => unreachable!(),
        }
        state
    }
    fn name(&self) -> &'static str {
        "SlipWall"
    }
}

/// Viscous no-slip adiabatic wall: reflect *all* momentum so the wall-interpolated
/// velocity is zero (no-slip), keeping density and total energy (adiabatic).
pub struct NoSlipWall;
impl BoundaryCondition for NoSlipWall {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, _eos: &dyn Eos) -> ConsVar {
        let mut state = *interior;
        state.rho_u = -state.rho_u;
        state.rho_v = -state.rho_v;
        state.rho_w = -state.rho_w;
        state
    }
    fn name(&self) -> &'static str {
        "NoSlipWall"
    }
}

/// Symmetry plane: identical to [`SlipWall`] for the Euler equations.
pub struct Symmetry;
impl BoundaryCondition for Symmetry {
    fn ghost(&self, interior: &ConsVar, side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        SlipWall.ghost(interior, side, eos)
    }
    fn name(&self) -> &'static str {
        "Symmetry"
    }
}

/// Subsonic velocity inlet: prescribed `(ρ, u, v, w, T)`; pressure extrapolated
/// from the interior cell.
pub struct SubsonicInflow {
    pub rho: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub t: f64,
}
impl BoundaryCondition for SubsonicInflow {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        let p_interior = eos.pressure(interior);
        eos.prim_to_cons(&PrimVar::new(self.rho, self.u, self.v, self.w, p_interior, self.t))
    }
    fn name(&self) -> &'static str {
        "SubsonicInflow"
    }
}

/// Subsonic pressure outlet: prescribed back pressure; ρ and velocity
/// extrapolated from the interior; T recomputed for consistency.
pub struct PressureOutlet {
    pub p: f64,
}
impl BoundaryCondition for PressureOutlet {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        let mut prim = eos.cons_to_prim(interior);
        prim.p = self.p;
        prim.t = self.p / (prim.rho * 287.058);
        eos.prim_to_cons(&prim)
    }
    fn name(&self) -> &'static str {
        "PressureOutlet"
    }
}

// ─── BoundaryRegistry ────────────────────────────────────────────────────────

/// Per-side boundary conditions, indexed by [`BoundarySide`] (`0..6`).
#[derive(Clone)]
pub struct BoundaryRegistry {
    sides: [Arc<dyn BoundaryCondition>; 6],
}

impl Default for BoundaryRegistry {
    /// All six sides default to [`SupersonicOutflow`] (zero-gradient).
    fn default() -> Self {
        Self {
            sides: std::array::from_fn(|_| Arc::new(SupersonicOutflow) as Arc<dyn BoundaryCondition>),
        }
    }
}

impl BoundaryRegistry {
    /// Set the BC on one side.
    pub fn set<B: BoundaryCondition + 'static>(&mut self, side: BoundarySide, bc: B) -> &mut Self {
        self.sides[side as usize] = Arc::new(bc);
        self
    }

    /// Builder form of [`set`](Self::set).
    pub fn with<B: BoundaryCondition + 'static>(mut self, side: BoundarySide, bc: B) -> Self {
        self.set(side, bc);
        self
    }

    /// Set both sides of an axis (0=x, 1=y, 2=z) to the same BC.
    pub fn with_axis<B: BoundaryCondition + 'static>(mut self, axis: usize, bc: B) -> Self {
        let arc: Arc<dyn BoundaryCondition> = Arc::new(bc);
        self.sides[2 * axis] = Arc::clone(&arc);
        self.sides[2 * axis + 1] = arc;
        self
    }

    pub fn get(&self, side: BoundarySide) -> &dyn BoundaryCondition {
        &*self.sides[side as usize]
    }

    pub fn is_periodic(&self, side: BoundarySide) -> bool {
        self.get(side).is_periodic()
    }
}

// ─── Plugin + system ─────────────────────────────────────────────────────────

/// Installs the [`BoundaryRegistry`] and the `Boundary`-phase ghost-fill system.
///
/// Generic over the mesh, defaulting to `UniformMesh` so `BoundaryPlugin::new(..)`
/// keeps working unchanged; a forest app uses `BoundaryPlugin::<ForestMesh>::new(..)`.
pub struct BoundaryPlugin<M: field_core::FvMesh = field_core::UniformMesh> {
    registry: BoundaryRegistry,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M: field_core::FvMesh> Default for BoundaryPlugin<M> {
    fn default() -> Self {
        Self { registry: BoundaryRegistry::default(), _marker: std::marker::PhantomData }
    }
}

impl<M: field_core::FvMesh> BoundaryPlugin<M> {
    pub fn new(registry: BoundaryRegistry) -> Self {
        Self { registry, _marker: std::marker::PhantomData }
    }
}

impl<M: field_core::FvMesh> Plugin for BoundaryPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(self.registry.clone());
        app.add_update_system(boundary_system::<M>, field_core::MeshScheduleSet::Boundary);
    }

    fn provides(&self) -> Vec<&str> {
        vec!["boundary_conditions"]
    }

    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "mesh"]
    }
}

/// Structured-grid boundary plugin that supports [`Periodic`] sides.
///
/// It first fills periodic ghost slabs by wrapping all forward-registered FIELD
/// data, then applies the owner-only BC fill on non-periodic physical sides.
pub struct StructuredBoundaryPlugin<M: field_core::StructuredMesh = field_core::UniformMesh> {
    registry: BoundaryRegistry,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M: field_core::StructuredMesh> Default for StructuredBoundaryPlugin<M> {
    fn default() -> Self {
        Self { registry: BoundaryRegistry::default(), _marker: std::marker::PhantomData }
    }
}

impl<M: field_core::StructuredMesh> StructuredBoundaryPlugin<M> {
    pub fn new(registry: BoundaryRegistry) -> Self {
        Self { registry, _marker: std::marker::PhantomData }
    }
}

impl<M: field_core::StructuredMesh> Plugin for StructuredBoundaryPlugin<M> {
    fn build(&self, app: &mut App) {
        app.add_resource(self.registry.clone());
        app.add_update_system(periodic_wrap_system::<M>, field_core::MeshScheduleSet::Boundary);
        app.add_update_system(boundary_system::<M>, field_core::MeshScheduleSet::Boundary);
    }

    fn provides(&self) -> Vec<&str> {
        vec!["boundary_conditions"]
    }

    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "mesh"]
    }
}

/// Wrap periodic structured ghost slabs using FIELD's forward pack/unpack
/// semantics. Only side slabs are filled; face-based FVM fluxes and Cartesian
/// reconstruction read these axis ghosts, while corners stay outside the current
/// FIELD face-halo contract.
pub fn periodic_wrap_system<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    bcs: Res<BoundaryRegistry>,
) {
    if reg.forward_size() == 0 {
        return;
    }

    let dims = mesh.dims();
    let totals = mesh.total_dims();
    let ng = mesh.n_ghost();
    for axis in 0..3 {
        if dims[axis] == 0 {
            continue;
        }
        let lo = BoundarySide::from_index((2 * axis) as u32).expect("valid boundary side");
        let hi = BoundarySide::from_index((2 * axis + 1) as u32).expect("valid boundary side");
        if bcs.is_periodic(lo) {
            wrap_side(&*mesh, &*reg, axis, false, dims, totals, ng);
        }
        if bcs.is_periodic(hi) {
            wrap_side(&*mesh, &*reg, axis, true, dims, totals, ng);
        }
    }
}

fn wrap_side<M: StructuredMesh>(
    mesh: &M,
    reg: &FieldRegistry,
    axis: usize,
    high_side: bool,
    dims: [usize; 3],
    totals: [usize; 3],
    ng: usize,
) {
    let other_a = (axis + 1) % 3;
    let other_b = (axis + 2) % 3;
    let per = reg.forward_size();
    let mut raw = [0usize; 3];
    let mut src_raw = [0usize; 3];
    let mut buf = Vec::with_capacity(per);
    for g in 0..ng {
        if high_side {
            raw[axis] = ng + dims[axis] + g;
            src_raw[axis] = ng + (g % dims[axis]);
        } else {
            raw[axis] = g;
            src_raw[axis] = ng + ((dims[axis] + g).wrapping_sub(ng) % dims[axis]);
        }
        for a in ng..totals[other_a] - ng {
            raw[other_a] = a;
            src_raw[other_a] = a;
            for b in ng..totals[other_b] - ng {
                raw[other_b] = b;
                src_raw[other_b] = b;
                let src = mesh.idx_raw(src_raw[0], src_raw[1], src_raw[2]);
                let dst = mesh.idx_raw(raw[0], raw[1], raw[2]);
                buf.clear();
                reg.pack_forward_all(src, &mut buf);
                let consumed = reg.unpack_forward_all(dst, &buf);
                debug_assert_eq!(consumed, per, "periodic wrap size mismatch");
            }
        }
    }
}

/// `Boundary`: fill each physical-boundary face's ghost from its BC. Faces with
/// an MPI neighbor carry `patch == None` (filled by halo), so they are skipped.
pub fn boundary_system<M: FvMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    bcs: Res<BoundaryRegistry>,
    eos: Res<EosResource>,
) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    mesh.for_each_face(&mut |f| {
        if let Some(p) = f.patch {
            let side = BoundarySide::from_index(p).expect("valid boundary side");
            if bcs.is_periodic(side) {
                return;
            }
            let interior = state.u[f.owner];
            state.u[f.other] = bcs.get(side).ghost(&interior, side, eos);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_eos::IdealGas;

    #[test]
    fn outflow_copies_interior() {
        let eos = IdealGas::air();
        let interior = eos.prim_to_cons(&PrimVar::new(1.2, 100.0, 0.0, 0.0, 101325.0, 288.0));
        let g = SupersonicOutflow.ghost(&interior, BoundarySide::XLo, &eos);
        assert_eq!(g, interior);
    }

    #[test]
    fn slip_wall_reflects_only_normal_momentum() {
        let eos = IdealGas::air();
        let interior = eos.prim_to_cons(&PrimVar::new(1.2, 100.0, 50.0, 25.0, 101325.0, 288.0));
        let gx = SlipWall.ghost(&interior, BoundarySide::XHi, &eos);
        assert!((gx.rho_u + interior.rho_u).abs() < 1e-12); // x-momentum flipped
        assert_eq!(gx.rho_v, interior.rho_v); // tangential preserved
        assert_eq!(gx.rho_w, interior.rho_w);
        assert_eq!(gx.rho, interior.rho);
        let gy = SlipWall.ghost(&interior, BoundarySide::YLo, &eos);
        assert!((gy.rho_v + interior.rho_v).abs() < 1e-12); // y-momentum flipped
        assert_eq!(gy.rho_u, interior.rho_u);
    }

    #[test]
    fn supersonic_inflow_is_prescribed() {
        let eos = IdealGas::air();
        let interior = eos.prim_to_cons(&PrimVar::new(0.1, 0.0, 0.0, 0.0, 1000.0, 200.0));
        let bc = SupersonicInflow { rho: 1.0, u: 500.0, v: 0.0, w: 0.0, p: 2e5 };
        let g = bc.ghost(&interior, BoundarySide::XLo, &eos);
        let prim = eos.cons_to_prim(&g);
        assert!((prim.rho - 1.0).abs() < 1e-9);
        assert!((prim.u - 500.0).abs() < 1e-6);
        assert!((prim.p - 2e5).abs() < 1.0);
    }

    #[test]
    fn registry_dispatches_per_side() {
        let reg = BoundaryRegistry::default()
            .with(BoundarySide::XLo, SlipWall)
            .with_axis(1, Symmetry);
        assert_eq!(reg.get(BoundarySide::XLo).name(), "SlipWall");
        assert_eq!(reg.get(BoundarySide::XHi).name(), "SupersonicOutflow");
        assert_eq!(reg.get(BoundarySide::YLo).name(), "Symmetry");
        assert_eq!(reg.get(BoundarySide::YHi).name(), "Symmetry");
    }

    #[test]
    fn periodic_is_a_registry_marker() {
        let reg = BoundaryRegistry::default().with_axis(0, Periodic);
        assert!(reg.is_periodic(BoundarySide::XLo));
        assert!(reg.is_periodic(BoundarySide::XHi));
        assert_eq!(reg.get(BoundarySide::XLo).name(), "Periodic");
        assert!(!reg.is_periodic(BoundarySide::YLo));
    }

    #[test]
    fn periodic_wrap_copies_opposite_owned_slab_through_registry() {
        use field_core::{StructuredMesh, UniformMesh, UniformMeshConfig};

        let mesh = UniformMesh::from_config(&UniformMeshConfig {
            nx: 4,
            ny: 1,
            nz: 1,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [1.0, 1.0, 1.0],
            y_edges: None,
            z_edges: None,
        });
        let mut registry = FieldRegistry::new();
        registry.register(CfdState::default());
        registry.resize_all(mesh.n_cells_total());
        {
            let mut state = registry.get_mut::<CfdState>().unwrap();
            for i in 0..4 {
                let c = mesh.idx(i, 0, 0);
                state.u[c] = ConsVar::new((10 + i) as f64, 0.0, 0.0, 0.0, 0.0);
            }
        }

        let dims = mesh.dims();
        let totals = mesh.total_dims();
        let ng = mesh.n_ghost();
        wrap_side(&mesh, &registry, 0, false, dims, totals, ng);
        wrap_side(&mesh, &registry, 0, true, dims, totals, ng);

        let state = registry.get::<CfdState>().unwrap();
        let j = ng;
        let k = ng;
        assert_eq!(state.u[mesh.idx_raw(0, j, k)].rho, 12.0);
        assert_eq!(state.u[mesh.idx_raw(1, j, k)].rho, 13.0);
        assert_eq!(state.u[mesh.idx_raw(6, j, k)].rho, 10.0);
        assert_eq!(state.u[mesh.idx_raw(7, j, k)].rho, 11.0);
    }
}
