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
use field_core::{
    BoundarySide, FieldRegistry, FvMesh, MeshPeriodicity, PatchId, StructuredMesh, Vec3,
};
use std::collections::HashMap;
use grass_app::prelude::*;
use grass_scheduler::Res;
use std::sync::Arc;

// ─── BoundaryCondition trait ─────────────────────────────────────────────────

/// A rule for the ghost state on one physical-boundary face, given the interior
/// owner state, which [`BoundarySide`] the face lies on, and the EOS.
pub trait BoundaryCondition: Send + Sync {
    fn ghost(&self, interior: &ConsVar, side: BoundarySide, eos: &dyn Eos) -> ConsVar;

    /// The ghost state given the whole face: where it is, and any outside
    /// state another code supplied for it through [`BoundaryInputs`]. The
    /// default ignores both and calls [`ghost`](Self::ghost), so an owner-only
    /// condition implements just that.
    ///
    /// A face on a patch that is not one of the six box sides (an
    /// unstructured mesh's own patches) has no [`BoundarySide`], so a condition
    /// used there must override this method.
    fn ghost_at(&self, interior: &ConsVar, face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        match face.side {
            Some(side) => self.ghost(interior, side, eos),
            None => panic!(
                "boundary condition {} is side-only; implement ghost_at to use it on patch {}",
                self.name(),
                face.patch
            ),
        }
    }
    fn is_periodic(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "<unnamed>"
    }
}

/// One physical-boundary face, as [`BoundaryCondition::ghost_at`] sees it.
pub struct BoundaryFace<'a> {
    /// The mesh patch the face belongs to.
    pub patch: PatchId,
    /// Which side of the box the face is on, when the patch is one of the six
    /// [`BoundarySide`]s.
    pub side: Option<BoundarySide>,
    /// Face centroid in physical space.
    pub centroid: Vec3,
    /// Area-weighted outward normal.
    pub area_normal: Vec3,
    /// The ghost cell being filled.
    pub ghost_cell: usize,
    /// The state another code supplied for this ghost cell, if any.
    pub outside: Option<&'a ConsVar>,
}

impl BoundaryFace<'_> {
    /// Unit outward normal.
    pub fn unit_normal(&self) -> Vec3 {
        let n = self.area_normal;
        let a = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        [n[0] / a, n[1] / a, n[2] / a]
    }
}

/// Anything that names a boundary patch: a structured [`BoundarySide`] or a raw
/// [`PatchId`] from an unstructured mesh.
pub trait PatchKey {
    /// The patch id.
    fn patch_id(self) -> PatchId;
}

impl PatchKey for BoundarySide {
    fn patch_id(self) -> PatchId {
        self as PatchId
    }
}

impl PatchKey for PatchId {
    fn patch_id(self) -> PatchId {
        self
    }
}

/// Outside states for ghost cells, by FIELD cell index, written by whatever
/// drives the solver (for example a coupler with another code) before the
/// `Boundary` phase. [`FromOutside`] imposes them; any condition can read
/// them through [`BoundaryFace::outside`]. Added by the boundary plugins.
#[derive(Default, Debug, Clone)]
pub struct BoundaryInputs(pub HashMap<usize, ConsVar>);

// ─── Stock boundary conditions ───────────────────────────────────────────────

/// The ghost state another code supplied in [`BoundaryInputs`]: the side of
/// a coupled interface. A ghost cell with no supplied state falls back to
/// `otherwise`.
pub struct FromOutside<B: BoundaryCondition = SupersonicOutflow> {
    pub otherwise: B,
}

impl FromOutside {
    /// Fall back to zero-gradient where no outside state was supplied.
    pub fn new() -> Self {
        Self { otherwise: SupersonicOutflow }
    }
}

impl Default for FromOutside {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: BoundaryCondition> BoundaryCondition for FromOutside<B> {
    fn ghost(&self, interior: &ConsVar, side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.otherwise.ghost(interior, side, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        match face.outside {
            Some(u) => *u,
            None => self.otherwise.ghost_at(interior, face, eos),
        }
    }
    fn name(&self) -> &'static str {
        "FromOutside"
    }
}

/// A condition written as a closure over the interior state, the face and the
/// EOS, for profiles that depend on position (an inlet profile, a moving lid).
pub struct FnBoundary<F>(pub F)
where
    F: Fn(&ConsVar, &BoundaryFace, &dyn Eos) -> ConsVar + Send + Sync;

impl<F> BoundaryCondition for FnBoundary<F>
where
    F: Fn(&ConsVar, &BoundaryFace, &dyn Eos) -> ConsVar + Send + Sync,
{
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, _eos: &dyn Eos) -> ConsVar {
        // Never called by the boundary system, which always has the face.
        *interior
    }
    fn ghost_at(&self, interior: &ConsVar, face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        (self.0)(interior, face, eos)
    }
    fn name(&self) -> &'static str {
        "FnBoundary"
    }
}

/// Supersonic outflow / zero-gradient: copy the interior state into the ghost.
pub struct SupersonicOutflow;
impl SupersonicOutflow {
    fn state(&self, interior: &ConsVar, _eos: &dyn Eos) -> ConsVar {
        *interior
    }
}
impl BoundaryCondition for SupersonicOutflow {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, _face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
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
impl Periodic {
    fn state(&self, interior: &ConsVar, _eos: &dyn Eos) -> ConsVar {
        *interior
    }
}
impl BoundaryCondition for Periodic {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn is_periodic(&self) -> bool {
        true
    }
    fn ghost_at(&self, interior: &ConsVar, _face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
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
impl SupersonicInflow {
    fn state(&self, _interior: &ConsVar, eos: &dyn Eos) -> ConsVar {
        let t = self.p / (self.rho * 287.058);
    eos.prim_to_cons(&PrimVar::new(self.rho, self.u, self.v, self.w, self.p, t))
    }
}
impl BoundaryCondition for SupersonicInflow {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, _face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
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
    fn ghost_at(&self, interior: &ConsVar, face: &BoundaryFace, _eos: &dyn Eos) -> ConsVar {
        // m' = m - 2 (m·n) n: the same sign flip as `ghost` on a box side, and
        // the correct mirror on a face of any orientation.
        let mut state = *interior;
        let n = face.unit_normal();
        let mn = state.rho_u * n[0] + state.rho_v * n[1] + state.rho_w * n[2];
        state.rho_u -= 2.0 * mn * n[0];
        state.rho_v -= 2.0 * mn * n[1];
        state.rho_w -= 2.0 * mn * n[2];
        state
    }
    fn name(&self) -> &'static str {
        "SlipWall"
    }
}

/// Viscous no-slip adiabatic wall: reflect *all* momentum so the wall-interpolated
/// velocity is zero (no-slip), keeping density and total energy (adiabatic).
pub struct NoSlipWall;
impl NoSlipWall {
    fn state(&self, interior: &ConsVar, _eos: &dyn Eos) -> ConsVar {
        let mut state = *interior;
    state.rho_u = -state.rho_u;
    state.rho_v = -state.rho_v;
    state.rho_w = -state.rho_w;
    state
    }
}
impl BoundaryCondition for NoSlipWall {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, _face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
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
    fn ghost_at(&self, interior: &ConsVar, face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        SlipWall.ghost_at(interior, face, eos)
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
impl SubsonicInflow {
    fn state(&self, interior: &ConsVar, eos: &dyn Eos) -> ConsVar {
        let p_interior = eos.pressure(interior);
    eos.prim_to_cons(&PrimVar::new(self.rho, self.u, self.v, self.w, p_interior, self.t))
    }
}
impl BoundaryCondition for SubsonicInflow {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, _face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
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
impl PressureOutlet {
    fn state(&self, interior: &ConsVar, eos: &dyn Eos) -> ConsVar {
        let mut prim = eos.cons_to_prim(interior);
    prim.p = self.p;
    prim.t = self.p / (prim.rho * 287.058);
    eos.prim_to_cons(&prim)
    }
}
impl BoundaryCondition for PressureOutlet {
    fn ghost(&self, interior: &ConsVar, _side: BoundarySide, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn ghost_at(&self, interior: &ConsVar, _face: &BoundaryFace, eos: &dyn Eos) -> ConsVar {
        self.state(interior, eos)
    }
    fn name(&self) -> &'static str {
        "PressureOutlet"
    }
}

// ─── BoundaryRegistry ────────────────────────────────────────────────────────

/// Boundary conditions per patch. Patches `0..6` are the structured
/// [`BoundarySide`]s; an unstructured mesh may use more. Any patch not set
/// explicitly gets [`SupersonicOutflow`] (zero-gradient).
#[derive(Clone)]
pub struct BoundaryRegistry {
    patches: Vec<Arc<dyn BoundaryCondition>>,
    fallback: Arc<dyn BoundaryCondition>,
}

impl Default for BoundaryRegistry {
    /// Every patch defaults to [`SupersonicOutflow`] (zero-gradient).
    fn default() -> Self {
        let fallback: Arc<dyn BoundaryCondition> = Arc::new(SupersonicOutflow);
        Self { patches: vec![Arc::clone(&fallback); 6], fallback }
    }
}

impl BoundaryRegistry {
    /// Set the BC on one patch (a [`BoundarySide`] or a [`PatchId`]).
    pub fn set<B: BoundaryCondition + 'static>(&mut self, patch: impl PatchKey, bc: B) -> &mut Self {
        let p = patch.patch_id() as usize;
        if p >= self.patches.len() {
            self.patches.resize(p + 1, Arc::clone(&self.fallback));
        }
        self.patches[p] = Arc::new(bc);
        self
    }

    /// Builder form of [`set`](Self::set).
    pub fn with<B: BoundaryCondition + 'static>(mut self, patch: impl PatchKey, bc: B) -> Self {
        self.set(patch, bc);
        self
    }

    /// Set both sides of an axis (0=x, 1=y, 2=z) to the same BC.
    pub fn with_axis<B: BoundaryCondition + 'static>(mut self, axis: usize, bc: B) -> Self {
        let arc: Arc<dyn BoundaryCondition> = Arc::new(bc);
        self.patches[2 * axis] = Arc::clone(&arc);
        self.patches[2 * axis + 1] = arc;
        self
    }

    /// The BC on a patch.
    pub fn get(&self, patch: impl PatchKey) -> &dyn BoundaryCondition {
        &**self.patches.get(patch.patch_id() as usize).unwrap_or(&self.fallback)
    }

    /// Whether a patch is marked [`Periodic`].
    pub fn is_periodic(&self, patch: impl PatchKey) -> bool {
        self.get(patch).is_periodic()
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
        app.add_resource(BoundaryInputs::default());
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
        app.add_resource(BoundaryInputs::default());
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
    inputs: Option<Res<BoundaryInputs>>,
) {
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let eos: &dyn Eos = &*eos.0;
    let outside = inputs.as_ref().map(|i| &i.0);
    mesh.for_each_face(&mut |f| {
        if let Some(p) = f.patch {
            if bcs.is_periodic(p) {
                return;
            }
            let interior = state.u[f.owner];
            let face = BoundaryFace {
                patch: p,
                side: BoundarySide::from_index(p),
                centroid: f.centroid,
                area_normal: f.area_normal,
                ghost_cell: f.other,
                outside: outside.and_then(|o| o.get(&f.other)),
            };
            state.u[f.other] = bcs.get(p).ghost_at(&interior, &face, eos);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_eos::IdealGas;

    /// A face on patch 7 (not a box side) with outward normal `n`.
    fn slanted_face(n: Vec3) -> BoundaryFace<'static> {
        BoundaryFace { patch: 7, side: None, centroid: [0.0; 3], area_normal: n, ghost_cell: 0, outside: None }
    }

    #[test]
    fn slip_wall_mirrors_about_a_slanted_normal() {
        let eos = IdealGas::air();
        let interior = eos.prim_to_cons(&PrimVar::new(1.2, 100.0, 50.0, 25.0, 101325.0, 288.0));
        let s = 0.5f64.sqrt();
        // Area-weighted: the BC must normalise it.
        let face = slanted_face([2.0 * s, 2.0 * s, 0.0]);
        let n = face.unit_normal();
        let g = SlipWall.ghost_at(&interior, &face, &eos);
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let (m, mg) = ([interior.rho_u, interior.rho_v, interior.rho_w], [g.rho_u, g.rho_v, g.rho_w]);
        // Normal momentum flips, tangential momentum is kept.
        assert!((dot(mg, n) + dot(m, n)).abs() < 1e-9);
        let t = [s, -s, 0.0];
        assert!((dot(mg, t) - dot(m, t)).abs() < 1e-9);
        assert!((g.rho_w - interior.rho_w).abs() < 1e-12);
        assert_eq!(g.rho, interior.rho);
        assert_eq!(g.rho_e, interior.rho_e);
    }

    #[test]
    fn slip_wall_ghost_at_matches_ghost_on_box_sides() {
        let eos = IdealGas::air();
        let interior = eos.prim_to_cons(&PrimVar::new(1.2, 100.0, 50.0, 25.0, 101325.0, 288.0));
        for p in 0..6u32 {
            let side = BoundarySide::from_index(p).unwrap();
            let mut n = [0.0; 3];
            n[side.axis()] = if p % 2 == 1 { 3.0 } else { -3.0 };
            let face = BoundaryFace { patch: p, side: Some(side), ..slanted_face(n) };
            assert_eq!(SlipWall.ghost_at(&interior, &face, &eos), SlipWall.ghost(&interior, side, &eos));
        }
    }

    #[test]
    fn stock_conditions_work_on_patches_beyond_the_box_sides() {
        let eos = IdealGas::air();
        let interior = eos.prim_to_cons(&PrimVar::new(1.2, 100.0, 50.0, 25.0, 101325.0, 288.0));
        let face = slanted_face([1.0, 0.0, 0.0]);
        let bcs: [&dyn BoundaryCondition; 6] = [
            &SupersonicOutflow,
            &NoSlipWall,
            &Symmetry,
            &SubsonicInflow { rho: 1.2, u: 10.0, v: 0.0, w: 0.0, t: 300.0 },
            &PressureOutlet { p: 1e5 },
            &FromOutside::new(),
        ];
        for bc in bcs {
            bc.ghost_at(&interior, &face, &eos);
        }
    }

    #[test]
    #[should_panic(expected = "implement ghost_at")]
    fn a_side_only_condition_names_itself_on_a_non_box_patch() {
        struct SideOnly;
        impl BoundaryCondition for SideOnly {
            fn ghost(&self, i: &ConsVar, _s: BoundarySide, _e: &dyn Eos) -> ConsVar {
                *i
            }
        }
        SideOnly.ghost_at(&ConsVar::default(), &slanted_face([1.0, 0.0, 0.0]), &IdealGas::air());
    }

    #[test]
    fn registry_accepts_patches_beyond_the_box_sides() {
        let reg = BoundaryRegistry::default().with(9u32, SlipWall);
        assert_eq!(reg.get(9u32).name(), "SlipWall");
        assert_eq!(reg.get(7u32).name(), "SupersonicOutflow");
        assert_eq!(reg.get(42u32).name(), "SupersonicOutflow");
        assert_eq!(reg.get(BoundarySide::XLo).name(), "SupersonicOutflow");
    }

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
