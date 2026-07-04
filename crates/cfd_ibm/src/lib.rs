//! Sharp-interface ghost-cell immersed boundary method (Mittal & Iaccarino 2005),
//! ported from `toy_cfd::ibm` onto the FIELD substrate.
//!
//! The solid is represented by a registered [`SolidMask`] (`#[forward] Vec<bool>`,
//! so it halo-exchanges like any other field). Each substep, for every solid cell
//! adjacent to fluid, the fluid state at the **image point** (the reflection of
//! the cell center across the surface) is trilinearly sampled and mirrored back so
//! the wall BC (no-slip, adiabatic, zero-pressure-gradient) holds at the surface
//! midpoint. The regular flux at fluid↔solid faces then reproduces the wall.
//!
//! This is the test-cfd analog of a DIRT wall/contact plugin: pure physics over
//! the substrate's mesh + fields. Geometry is pluggable; an analytic [`Cylinder`]
//! ships here (the `cylinder_drag_validation` case); FSI-coupled spheres from the
//! particle stack are the future `grass_multi` story.
//!
//! Scope (this pass): the core ghost-cell mechanism + mask + cylinder geometry.
//! Surface-marker drag integration (toy-cfd's `compute_*_wall_loadings`) is the
//! remaining diagnostic layer.

use cfd_eos::{Eos, EosResource, Viscosity};
use cfd_state::{CfdState, ConsVar, PrimVar, SolidMask};
use field_core::{register_field_data, FieldRegistry, MeshScheduleSet, StructuredMesh, UniformMesh, Vec3};
use grass_app::prelude::*;
use grass_scheduler::{Res, ResMut};

mod cutcell;
pub use cutcell::{Body, CutCellMetrics, Sphere};

pub mod coupling;
pub use coupling::{
    beta_gidaspow, cd_clift_grace_weber, cd_schiller_naumann, drag_force_from_beta,
    sphere_drag_force, terminal_velocity, terminal_velocity_stokes, InterphaseForces,
    ParticleKinematics, ParticleSet,
};

// ─── Ghost-cell metadata ─────────────────────────────────────────────────────

/// Per-ghost-cell IBM metadata, recomputed when the geometry moves.
#[derive(Debug, Clone, Copy, Default)]
pub struct IbmGhostInfo {
    pub cell_center: Vec3,
    pub image_point: Vec3,
    pub normal: Vec3,
    pub body_id: u32,
    /// Surface velocity (0 for a static body; `u_body + ω×r` for a moving one).
    pub wall_velocity: Vec3,
}

/// The active solid ghost cells: `(cell_index, info)`.
#[derive(Debug, Default, Clone)]
pub struct IbmGhostCells {
    pub cells: Vec<(usize, IbmGhostInfo)>,
}

// ─── Cylinder geometry ───────────────────────────────────────────────────────

/// Z-aligned cylinder; center in the xy-plane, infinite in z (trimmed by the mesh).
#[derive(Clone, Copy, Debug)]
pub struct Cylinder {
    pub center_xy: [f64; 2],
    pub radius: f64,
}

impl Cylinder {
    /// Signed distance to the surface (negative inside).
    #[inline]
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        let dx = p[0] - self.center_xy[0];
        let dy = p[1] - self.center_xy[1];
        (dx * dx + dy * dy).sqrt() - self.radius
    }
    /// Outward unit normal at the surface projection of `p`.
    #[inline]
    pub fn normal(&self, p: Vec3) -> Vec3 {
        let dx = p[0] - self.center_xy[0];
        let dy = p[1] - self.center_xy[1];
        let r = (dx * dx + dy * dy).sqrt();
        if r < 1e-12 {
            [1.0, 0.0, 0.0]
        } else {
            [dx / r, dy / r, 0.0]
        }
    }
}

/// All cylinders in the domain.
#[derive(Clone, Default, Debug)]
pub struct CylindersResource(pub Vec<Cylinder>);

/// The stagnant (zero-velocity) and ambient (freestream) states a cell takes when
/// it flips solid↔fluid.
#[derive(Clone, Copy, Debug)]
pub struct MaskConsts {
    pub stagnant: ConsVar,
    pub ambient: ConsVar,
}

// ─── Image-point sampling ────────────────────────────────────────────────────

/// Trilinear interpolation of primitives at `point` from the 8 bracketing cells,
/// dropping solid / zero-density cells and renormalizing. `None` if the point is
/// off-rank or every bracketing cell is unusable.
fn sample_trilinear<M: StructuredMesh>(
    mesh: &M,
    state: &CfdState,
    mask: &SolidMask,
    eos: &dyn Eos,
    point: Vec3,
) -> Option<PrimVar> {
    let (lo, t) = mesh.locate(point)?;
    let mut acc = [0.0f64; 6];
    let mut total_w = 0.0f64;
    for di in 0..2 {
        let wx = if di == 0 { 1.0 - t[0] } else { t[0] };
        for dj in 0..2 {
            let wy = if dj == 0 { 1.0 - t[1] } else { t[1] };
            for dk in 0..2 {
                let wz = if dk == 0 { 1.0 - t[2] } else { t[2] };
                let c = mesh.idx_raw(lo[0] + di, lo[1] + dj, lo[2] + dk);
                if mask.solid[c] {
                    continue;
                }
                let u = &state.u[c];
                // Skip uninitialized corner-halo cells (ρ=0 → cons_to_prim NaN)
                // and any non-finite state; treat them as solid for sampling.
                if !u.rho.is_finite() || u.rho <= 0.0 {
                    continue;
                }
                let w = wx * wy * wz;
                let p = eos.cons_to_prim(u);
                acc[0] += w * p.rho;
                acc[1] += w * p.u;
                acc[2] += w * p.v;
                acc[3] += w * p.w;
                acc[4] += w * p.p;
                acc[5] += w * p.t;
                total_w += w;
            }
        }
    }
    if total_w < 1e-12 {
        return None;
    }
    let inv = 1.0 / total_w;
    Some(PrimVar::new(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv, acc[4] * inv, acc[5] * inv))
}

// ─── Systems ─────────────────────────────────────────────────────────────────

/// `Setup`: mark cells inside any cylinder as solid (and flip back any that left),
/// seeding flipped cells with the stagnant/ambient state.
fn refresh_cylinder_mask<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    cyls: Res<CylindersResource>,
    consts: Res<MaskConsts>,
) {
    let mut mask = reg.expect_mut::<SolidMask>("SolidMask not registered");
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let [ni, nj, nk] = mesh.dims();
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let c = mesh.idx(i, j, k);
                let cc = mesh.cell_centroid(c);
                let inside = cyls.0.iter().any(|cy| cy.signed_distance(cc) < 0.0);
                let was = mask.solid[c];
                if inside && !was {
                    mask.solid[c] = true;
                    state.u[c] = consts.stagnant;
                    state.rhs[c] = ConsVar::ZERO;
                } else if !inside && was {
                    mask.solid[c] = false;
                    state.u[c] = consts.ambient;
                    state.rhs[c] = ConsVar::ZERO;
                }
            }
        }
    }
}

/// `Setup`: (re)build the ghost-cell list — solid cells with ≥1 fluid neighbor,
/// with the image point reflected across the nearest cylinder surface.
fn compute_cylinder_ghosts<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    cyls: Res<CylindersResource>,
    mut ghosts: ResMut<IbmGhostCells>,
) {
    ghosts.cells.clear();
    if cyls.0.is_empty() {
        return;
    }
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let ng = mesh.n_ghost();
    let [ni, nj, nk] = mesh.dims();
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let c = mesh.idx_raw(ir, jr, kr);
                if !mask.solid[c] {
                    continue;
                }
                let nbrs = [
                    mesh.idx_raw(ir + 1, jr, kr),
                    mesh.idx_raw(ir - 1, jr, kr),
                    mesh.idx_raw(ir, jr + 1, kr),
                    mesh.idx_raw(ir, jr - 1, kr),
                    mesh.idx_raw(ir, jr, kr + 1),
                    mesh.idx_raw(ir, jr, kr - 1),
                ];
                if !nbrs.iter().any(|&n| !mask.solid[n]) {
                    continue; // fully buried — no ghost needed
                }
                let cc = mesh.cell_centroid(c);
                let mut best_abs = f64::MAX;
                let mut best_signed = 0.0;
                let mut best_normal = [1.0, 0.0, 0.0];
                let mut best_body = 0u32;
                for (bid, cy) in cyls.0.iter().enumerate() {
                    let s = cy.signed_distance(cc);
                    if s.abs() < best_abs {
                        best_abs = s.abs();
                        best_signed = s;
                        best_normal = cy.normal(cc);
                        best_body = bid as u32;
                    }
                }
                let push = 2.0 * (-best_signed); // outward, into the fluid
                let image_point = [
                    cc[0] + push * best_normal[0],
                    cc[1] + push * best_normal[1],
                    cc[2] + push * best_normal[2],
                ];
                ghosts.cells.push((
                    c,
                    IbmGhostInfo {
                        cell_center: cc,
                        image_point,
                        normal: best_normal,
                        body_id: best_body,
                        wall_velocity: [0.0; 3],
                    },
                ));
            }
        }
    }
}

/// `PostHalo`: the per-substep ghost mirror. For each solid ghost cell, sample the
/// fluid at the image point and write the mirrored state (`u_g = 2 u_wall − u_i`).
fn apply_ibm_ghost_bc<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    ghosts: Res<IbmGhostCells>,
    eos: Res<EosResource>,
) {
    if ghosts.cells.is_empty() {
        return;
    }
    let eos: &dyn Eos = &*eos.0;
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    for (idx, info) in &ghosts.cells {
        let Some(image) = sample_trilinear(&*mesh, &state, &mask, eos, info.image_point) else {
            continue;
        };
        // Mirror: wall-midpoint velocity = u_wall ⇒ u_ghost = 2 u_wall − u_image;
        // adiabatic + zero-pressure-gradient keep T and p; density carries from the
        // image. `prim_to_cons` uses only (ρ, velocity, p), so no gas constant is
        // needed to rebuild the conserved ghost state.
        let uw = info.wall_velocity;
        let mirrored = PrimVar::new(
            image.rho,
            2.0 * uw[0] - image.u,
            2.0 * uw[1] - image.v,
            2.0 * uw[2] - image.w,
            image.p,
            image.t,
        );
        state.u[*idx] = eos.prim_to_cons(&mirrored);
    }
}

/// `PostFlux`: solid cells never advance — zero their RHS each substep so they
/// hold the mirrored value between applications.
fn zero_solid_rhs(reg: Res<FieldRegistry>) {
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    for c in 0..mask.solid.len() {
        if mask.solid[c] {
            state.rhs[c] = ConsVar::ZERO;
        }
    }
}

// ─── Wall forces / drag ──────────────────────────────────────────────────────

/// Pressure and viscous force on the immersed solid, summed over solid↔fluid
/// faces (the staircase wall-force estimate). Ported from toy-cfd's
/// `compute_wall_forces`. Pressure acts inward (`−p·n̂·A`); viscous wall shear
/// (`μ·u_tang/(½ wall-distance)·A`) acts along the fluid motion.
pub fn compute_wall_forces<M: StructuredMesh>(
    mesh: &M,
    state: &CfdState,
    mask: &SolidMask,
    eos: &dyn Eos,
    visc: Viscosity,
) -> (Vec3, Vec3) {
    let mut f_p = [0.0; 3];
    let mut f_v = [0.0; 3];
    let [ni, nj, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let c = mesh.idx_raw(ir, jr, kr);
                if !mask.solid[c] {
                    continue;
                }
                let (dx, dy, dz) = (mesh.spacing(0, i), mesh.spacing(1, j), mesh.spacing(2, k));
                let face_area = [dy * dz, dx * dz, dx * dy];
                let dn = [dx, dy, dz];
                let nbrs = [
                    (mesh.idx_raw(ir + 1, jr, kr), 0usize, 1.0),
                    (mesh.idx_raw(ir - 1, jr, kr), 0, -1.0),
                    (mesh.idx_raw(ir, jr + 1, kr), 1, 1.0),
                    (mesh.idx_raw(ir, jr - 1, kr), 1, -1.0),
                    (mesh.idx_raw(ir, jr, kr + 1), 2, 1.0),
                    (mesh.idx_raw(ir, jr, kr - 1), 2, -1.0),
                ];
                for (nbr, dim, sign) in nbrs {
                    if mask.solid[nbr] {
                        continue;
                    }
                    let u = &state.u[nbr];
                    // Pressure acts on the solid along −n̂_out.
                    f_p[dim] -= eos.pressure(u) * sign * face_area[dim];
                    // Viscous shear from no-slip gradient ≈ u_tang / (½ dn).
                    let prim = eos.cons_to_prim(u);
                    let mu = visc.mu(prim.t);
                    let vel = [prim.u, prim.v, prim.w];
                    for td in 0..3 {
                        if td != dim {
                            f_v[td] += mu * vel[td] / (0.5 * dn[dim]) * face_area[dim];
                        }
                    }
                }
            }
        }
    }
    (f_p, f_v)
}

/// Latest wall force on the immersed body (pressure + viscous), refreshed each
/// `Output` phase by [`GhostCellIbmPlugin`].
#[derive(Debug, Clone, Copy, Default)]
pub struct WallForce {
    pub pressure: Vec3,
    pub viscous: Vec3,
}

impl WallForce {
    pub fn total(&self) -> Vec3 {
        [
            self.pressure[0] + self.viscous[0],
            self.pressure[1] + self.viscous[1],
            self.pressure[2] + self.viscous[2],
        ]
    }
}

/// `Output`: refresh the [`WallForce`] diagnostic from the current field.
fn wall_force_system<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    visc: Res<Viscosity>,
    mut force: ResMut<WallForce>,
) {
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let state = reg.expect::<CfdState>("CfdState not registered");
    let (p, v) = compute_wall_forces(&*mesh, &state, &mask, &*eos.0, *visc);
    force.pressure = p;
    force.viscous = v;
}

// ─── Analytic surface markers (refinement-independent drag) ──────────────────

/// One analytic surface marker on a body: a point on the *true* surface with its
/// outward normal and the surface-element area it represents (`Σ area = total`).
#[derive(Debug, Clone, Copy)]
pub struct IbmMarker {
    pub body_id: u32,
    pub position: Vec3,
    pub normal: Vec3,
    pub area: f64,
}

/// All surface markers across all bodies, rebuilt per geometry update.
#[derive(Debug, Default, Clone)]
pub struct IbmMarkers {
    pub markers: Vec<IbmMarker>,
}

/// Net wall force from the marker integration (the refinement-independent drag,
/// vs the staircase [`WallForce`]). Refreshed each `Output` phase.
#[derive(Debug, Clone, Copy, Default)]
pub struct MarkerForce {
    pub pressure: Vec3,
    pub viscous: Vec3,
}

impl MarkerForce {
    pub fn total(&self) -> Vec3 {
        [
            self.pressure[0] + self.viscous[0],
            self.pressure[1] + self.viscous[1],
            self.pressure[2] + self.viscous[2],
        ]
    }
}

/// Integrate pressure + viscous traction over the markers. Each marker samples
/// the fluid one `sample_offset` (≈ a cell width) out along its normal: pressure
/// traction `−p·n̂·A`, viscous traction `μ·u_tangential/offset·A`. Marker areas are
/// analytic, so the result is insensitive to the staircase mask.
pub fn compute_marker_loadings<M: StructuredMesh>(
    mesh: &M,
    state: &CfdState,
    mask: &SolidMask,
    eos: &dyn Eos,
    markers: &IbmMarkers,
    sample_offset: f64,
    visc: Viscosity,
) -> (Vec3, Vec3) {
    let mut fp = [0.0; 3];
    let mut fv = [0.0; 3];
    for m in &markers.markers {
        let sp = [
            m.position[0] + sample_offset * m.normal[0],
            m.position[1] + sample_offset * m.normal[1],
            m.position[2] + sample_offset * m.normal[2],
        ];
        let Some(img) = sample_trilinear(mesh, state, mask, eos, sp) else {
            continue;
        };
        if ![img.p, img.t, img.u, img.v, img.w].iter().all(|x| x.is_finite()) {
            continue;
        }
        let us = [img.u, img.v, img.w];
        let udotn = us[0] * m.normal[0] + us[1] * m.normal[1] + us[2] * m.normal[2];
        let mu = visc.mu(img.t);
        for d in 0..3 {
            let ut = us[d] - udotn * m.normal[d]; // tangential velocity component
            fp[d] += -img.p * m.normal[d] * m.area;
            fv[d] += mu * ut / sample_offset * m.area;
        }
    }
    (fp, fv)
}

/// `Setup`: place markers uniformly around each cylinder (`n_theta × n_z`), with
/// analytic per-marker area `2π r L_z / (n_theta n_z)`.
fn compute_cylinder_markers<M: StructuredMesh>(
    mesh: Res<M>,
    cyls: Res<CylindersResource>,
    mut markers: ResMut<IbmMarkers>,
) {
    markers.markers.clear();
    if cyls.0.is_empty() {
        return;
    }
    let [_, _, nk] = mesh.dims();
    let ng = mesh.n_ghost();
    let z_lo = mesh.cell_centroid(mesh.idx_raw(ng, ng, ng))[2] - 0.5 * mesh.spacing(2, 0);
    let z_hi = mesh.cell_centroid(mesh.idx_raw(ng, ng, ng + nk - 1))[2] + 0.5 * mesh.spacing(2, nk - 1);
    let lz = z_hi - z_lo;
    let (n_theta, n_z) = (64usize, nk.max(1));
    for (bid, cy) in cyls.0.iter().enumerate() {
        let area = 2.0 * std::f64::consts::PI * cy.radius * lz / (n_theta * n_z) as f64;
        for it in 0..n_theta {
            let theta = 2.0 * std::f64::consts::PI * (it as f64 + 0.5) / n_theta as f64;
            let (nx, ny) = (theta.cos(), theta.sin());
            for iz in 0..n_z {
                let z = z_lo + (iz as f64 + 0.5) * lz / n_z as f64;
                markers.markers.push(IbmMarker {
                    body_id: bid as u32,
                    position: [cy.center_xy[0] + cy.radius * nx, cy.center_xy[1] + cy.radius * ny, z],
                    normal: [nx, ny, 0.0],
                    area,
                });
            }
        }
    }
}

/// `Output`: refresh the marker-integrated [`MarkerForce`].
fn marker_force_system<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    markers: Res<IbmMarkers>,
    visc: Res<Viscosity>,
    mut force: ResMut<MarkerForce>,
) {
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let state = reg.expect::<CfdState>("CfdState not registered");
    let dx = mesh.spacing(0, 0);
    let (p, v) = compute_marker_loadings(&*mesh, &state, &mask, &*eos.0, &markers, dx, *visc);
    force.pressure = p;
    force.viscous = v;
}

// ─── Plugins ─────────────────────────────────────────────────────────────────

/// Registers the [`SolidMask`] + [`IbmGhostCells`] and wires the per-substep
/// mirror (`PostHalo`) and solid-RHS zeroing (`PostFlux`). Geometry-agnostic.
pub struct GhostCellIbmPlugin;

impl Plugin for GhostCellIbmPlugin {
    fn build(&self, app: &mut App) {
        register_field_data!(app, SolidMask::default());
        app.add_resource(IbmGhostCells::default());
        app.add_resource(IbmMarkers::default());
        app.add_resource(WallForce::default());
        app.add_resource(MarkerForce::default());
        app.add_update_system(marker_force_system::<UniformMesh>, MeshScheduleSet::Output);
        // PreFlux (after Boundary) so the image point samples domain-edge ghosts
        // that the Boundary phase has already filled this substep — toy-cfd's
        // halo → BC → IBM-mirror → flux order. (PostHalo would read stale ghosts
        // for an immersed body straddling a domain boundary.)
        app.add_update_system(apply_ibm_ghost_bc::<UniformMesh>, MeshScheduleSet::PreFlux);
        app.add_update_system(zero_solid_rhs, MeshScheduleSet::PostFlux);
        app.add_update_system(wall_force_system::<UniformMesh>, MeshScheduleSet::Output);
    }
    fn provides(&self) -> Vec<&str> {
        vec!["ibm_ghost_cell"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["cfd_state", "eos", "structured_mesh"]
    }
}

/// Installs an analytic cylinder (or several) and the mask/ghost refresh systems.
pub struct CylinderIbmPlugin {
    cylinders: Vec<Cylinder>,
    freestream: ConsVar,
}

impl CylinderIbmPlugin {
    /// One cylinder immersed in a `freestream` state (used to seed flipped cells).
    pub fn new(cylinder: Cylinder, freestream: ConsVar) -> Self {
        Self { cylinders: vec![cylinder], freestream }
    }
    pub fn with_cylinders(cylinders: Vec<Cylinder>, freestream: ConsVar) -> Self {
        Self { cylinders, freestream }
    }
}

impl Plugin for CylinderIbmPlugin {
    fn build(&self, app: &mut App) {
        // Stagnant = freestream at rest (same ρ and internal energy, zero momentum).
        let fs = self.freestream;
        let stagnant = ConsVar::new(fs.rho, 0.0, 0.0, 0.0, fs.rho_e - fs.kinetic_energy());
        app.add_resource(CylindersResource(self.cylinders.clone()));
        app.add_resource(MaskConsts { stagnant, ambient: fs });
        app.add_update_system(refresh_cylinder_mask::<UniformMesh>, MeshScheduleSet::Setup);
        app.add_update_system(compute_cylinder_ghosts::<UniformMesh>, MeshScheduleSet::Setup);
        app.add_update_system(compute_cylinder_markers::<UniformMesh>, MeshScheduleSet::Setup);
    }
    fn provides(&self) -> Vec<&str> {
        vec!["cylinder_ibm"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["ibm_ghost_cell", "mesh"]
    }
}
