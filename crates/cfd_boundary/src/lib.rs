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
//! Periodic sides are FIELD halo links, not a boundary fill:
//! [`StructuredBoundaryPlugin`] turns every [`Periodic`] axis into a
//! [`field_core::MeshPeriodicity`] resource, the mesh plugin wraps those axes
//! when it decomposes the mesh, and the ordinary `Halo` phase then fills the
//! periodic ghosts, on one rank (a local copy) or many (a rank-wrap link).
//! The owner-only physical BC fill skips periodic sides.

use cfd_eos::{Eos, EosResource};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{BoundarySide, FieldRegistry, FvMesh, MeshPeriodicity, StructuredMesh};
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
/// adjacent owner cell. Use [`StructuredBoundaryPlugin`], which has FIELD wrap
/// the axis into the halo plan, so the `Halo` phase fills the ghost slab from
/// the opposite side of the domain (on this rank or another).
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

    /// Reject a half-periodic axis before a solver can step it.
    ///
    /// A periodic boundary is a paired topological identification: setting it
    /// on only one side would wrap one ghost slab and leave the opposite side
    /// to an unrelated physical BC. [`with_axis`](Self::with_axis) configures
    /// a valid pair atomically; this check makes the more general
    /// [`with`](Self::with)/[`set`](Self::set) API fail clearly when it is used
    /// to build an invalid pair.
    /// The periodic axes as FIELD's [`MeshPeriodicity`] resource (an axis is
    /// periodic when both its sides are; [`validate_periodic_pairs`](Self::validate_periodic_pairs)
    /// rejects half-periodic axes).
    pub fn mesh_periodicity(&self) -> MeshPeriodicity {
        MeshPeriodicity([0, 1, 2].map(|axis| {
            let lo = BoundarySide::from_index((2 * axis) as u32).expect("valid boundary side");
            let hi = BoundarySide::from_index((2 * axis + 1) as u32).expect("valid boundary side");
            self.is_periodic(lo) && self.is_periodic(hi)
        }))
    }

    pub fn validate_periodic_pairs(&self) -> Result<(), String> {
        for axis in 0..3 {
            let lo = BoundarySide::from_index((2 * axis) as u32).expect("valid boundary side");
            let hi = BoundarySide::from_index((2 * axis + 1) as u32).expect("valid boundary side");
            if self.is_periodic(lo) != self.is_periodic(hi) {
                return Err(format!(
                    "periodic boundary on axis {axis} requires both {lo:?} and {hi:?}; use BoundaryRegistry::with_axis(axis, Periodic)"
                ));
            }
        }
        Ok(())
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

/// Structured-grid boundary plugin that supports [`Periodic`] sides, serial or
/// MPI.
///
/// It records the periodic axes as a [`MeshPeriodicity`] resource so FIELD's
/// mesh plugin wraps them into the halo plan, checks once at setup that the
/// mesh really did (see [`validate_periodic_mesh`]), and applies the owner-only
/// BC fill on the non-periodic sides.
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
        self.registry
            .validate_periodic_pairs()
            .unwrap_or_else(|message| panic!("invalid periodic boundary configuration: {message}"));
        app.add_resource(self.registry.clone());
        app.add_resource(self.registry.mesh_periodicity());
        app.add_setup_system(check_periodic_mesh_system::<M>, ScheduleSetupSet::PostSetup);
        app.add_update_system(boundary_system::<M>, field_core::MeshScheduleSet::Boundary);
    }

    fn provides(&self) -> Vec<&str> {
        vec!["boundary_conditions"]
    }

    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "mesh"]
    }
}

/// `PostSetup`: fail before the first step if a periodic side was not wrapped
/// by the mesh (for example, a mesh resource built by hand instead of by the
/// FIELD mesh plugin).
pub fn check_periodic_mesh_system<M: StructuredMesh>(mesh: Res<M>, bcs: Res<BoundaryRegistry>) {
    validate_periodic_mesh(&*mesh, &bcs)
        .unwrap_or_else(|message| panic!("unsupported periodic boundary configuration: {message}"));
}

/// Check that every [`Periodic`] side is wrapped by the mesh's halo plan, i.e.
/// the mesh emits no physical-boundary face on it. FIELD's `UniformMesh` does
/// this for the axes named in [`MeshPeriodicity`]; a mesh that still has
/// boundary faces there would leave those ghosts unfilled.
pub fn validate_periodic_mesh<M: FvMesh>(mesh: &M, bcs: &BoundaryRegistry) -> Result<(), String> {
    let mut unwrapped = [false; 6];
    mesh.for_each_face(&mut |f| {
        if let Some(p) = f.patch {
            if let Some(side) = BoundarySide::from_index(p) {
                if bcs.is_periodic(side) {
                    unwrapped[side as usize] = true;
                }
            }
        }
    });
    let bad: Vec<String> = (0..6u32)
        .filter(|&i| unwrapped[i as usize])
        .map(|i| format!("{:?}", BoundarySide::from_index(i).expect("valid boundary side")))
        .collect();
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "periodic side(s) {} still have physical boundary faces; build the mesh with field_core::MeshPeriodicity (StructuredBoundaryPlugin adds it for the FIELD mesh plugin)",
            bad.join(", ")
        ))
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
    fn periodic_xlo_without_xhi_is_rejected() {
        let reg = BoundaryRegistry::default().with(BoundarySide::XLo, Periodic);
        let err = reg.validate_periodic_pairs().unwrap_err();
        assert!(err.contains("XLo"));
        assert!(err.contains("XHi"));
    }

    #[test]
    fn periodic_xhi_without_xlo_is_rejected() {
        let reg = BoundaryRegistry::default().with(BoundarySide::XHi, Periodic);
        let err = reg.validate_periodic_pairs().unwrap_err();
        assert!(err.contains("XLo"));
        assert!(err.contains("XHi"));
    }

    #[test]
    #[should_panic(expected = "invalid periodic boundary configuration")]
    fn structured_plugin_rejects_xlo_only_before_stepping() {
        let mut app = App::new();
        app.add_plugins(StructuredBoundaryPlugin::<field_core::UniformMesh>::new(
            BoundaryRegistry::default().with(BoundarySide::XLo, Periodic),
        ));
    }

    fn line_cfg(nx: usize) -> field_core::UniformMeshConfig {
        field_core::UniformMeshConfig {
            nx,
            ny: 1,
            nz: 1,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [1.0, 1.0, 1.0],
            y_edges: None,
            z_edges: None,
        }
    }

    #[test]
    fn registry_reports_periodic_axes_for_field() {
        let reg = BoundaryRegistry::default().with_axis(0, Periodic).with_axis(2, Periodic);
        assert_eq!(reg.mesh_periodicity(), MeshPeriodicity([true, false, true]));
    }

    #[test]
    fn periodic_mesh_passes_validation_serial_and_decomposed() {
        use field_core::UniformMesh;
        let bcs = BoundaryRegistry::default().with_axis(0, Periodic);
        let serial = UniformMesh::from_config_periodic(&line_cfg(8), [1, 1, 1], [0, 0, 0], [true, false, false]);
        validate_periodic_mesh(&serial, &bcs).unwrap();
        for pos in 0..2 {
            let part = UniformMesh::from_config_periodic(&line_cfg(8), [2, 1, 1], [pos, 0, 0], [true, false, false]);
            validate_periodic_mesh(&part, &bcs).unwrap();
        }
    }

    #[test]
    fn unwrapped_mesh_is_rejected_before_stepping() {
        use field_core::UniformMesh;
        let mesh = UniformMesh::from_config(&line_cfg(8));
        let err = validate_periodic_mesh(&mesh, &BoundaryRegistry::default().with_axis(0, Periodic))
            .unwrap_err();
        assert!(err.contains("XLo") && err.contains("XHi"));
        assert!(err.contains("MeshPeriodicity"));
    }

    #[test]
    fn halo_fills_periodic_ghosts_from_the_opposite_slab() {
        use field_core::{halo_exchange_forward_all, StructuredMesh, UniformMesh};
        let mesh = UniformMesh::from_config_periodic(&line_cfg(4), [1, 1, 1], [0, 0, 0], [true, false, false]);
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
        let comm = grass_mpi::SingleProcessComm::new();
        halo_exchange_forward_all(mesh.halo_plan(), &registry, &comm);

        let state = registry.get::<CfdState>().unwrap();
        let (j, k) = (mesh.n_ghost(), mesh.n_ghost());
        assert_eq!(state.u[mesh.idx_raw(0, j, k)].rho, 12.0);
        assert_eq!(state.u[mesh.idx_raw(1, j, k)].rho, 13.0);
        assert_eq!(state.u[mesh.idx_raw(6, j, k)].rho, 10.0);
        assert_eq!(state.u[mesh.idx_raw(7, j, k)].rho, 11.0);
    }
}
