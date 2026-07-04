//! Resolved **sphere** immersed-boundary geometry — the 3-D analogue of the
//! parent module's [`Cylinder`](crate::Cylinder) path, reusing the same
//! sharp-interface ghost-cell mirror ([`GhostCellIbmPlugin`](crate::GhostCellIbmPlugin))
//! and the same refinement-independent surface-marker traction integrator
//! ([`MarkerForce`](crate::MarkerForce)).
//!
//! Where the cylinder is z-infinite (a 2-D disc), a [`Sphere`](crate::Sphere) is a
//! fully 3-D body, so this is the geometry the DEM-CFD *resolved* seam runs its
//! 3-D single-sphere drag `Cd(Re)` validation on (`sphere_drag_validation`
//! example) and, with the moving-body support below, the two-particle
//! drafting-kissing-tumbling capstone.
//!
//! Everything here is geometry-only physics over the FIELD mesh — the sphere is
//! *immersed* (`signed_distance`), never a cell owner. Bodies may **move**: each
//! `Setup` phase the mask, ghost list, and markers are rebuilt from the current
//! [`SphereBodies`] state (position + velocity), and the ghost/marker
//! `wall_velocity` carries the rigid-body surface velocity `u + ω×r` so the
//! no-slip mirror and the viscous traction see the moving wall. A fixed sphere is
//! just the `velocity = ω = 0` special case (what the drag validation uses).

use crate::{IbmGhostCells, IbmGhostInfo, IbmMarker, IbmMarkers, MaskConsts};
use cfd_state::{ConsVar, SolidMask};
use field_core::{FieldRegistry, MeshScheduleSet, StructuredMesh, UniformMesh, Vec3};
use grass_app::prelude::*;
use grass_scheduler::{Res, ResMut};

/// One immersed rigid sphere with the kinematic state the moving IBM needs:
/// center, radius, translational velocity, and angular velocity (for the wall
/// velocity `u + ω×(x−center)` and, via the FSI driver, tumbling). A fixed body
/// leaves `velocity` and `omega` at zero.
#[derive(Clone, Copy, Debug)]
pub struct SphereBody {
    pub center: Vec3,
    pub radius: f64,
    pub velocity: Vec3,
    pub omega: Vec3,
}

impl SphereBody {
    /// A stationary sphere (drag-validation case).
    pub fn fixed(center: Vec3, radius: f64) -> Self {
        Self { center, radius, velocity: [0.0; 3], omega: [0.0; 3] }
    }
    /// Signed distance to the surface (negative inside).
    #[inline]
    pub fn signed_distance(&self, p: Vec3) -> f64 {
        let d = [p[0] - self.center[0], p[1] - self.center[1], p[2] - self.center[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - self.radius
    }
    /// Outward unit normal at the surface projection of `p`.
    #[inline]
    pub fn normal(&self, p: Vec3) -> Vec3 {
        let d = [p[0] - self.center[0], p[1] - self.center[1], p[2] - self.center[2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if r < 1e-12 {
            [1.0, 0.0, 0.0]
        } else {
            [d[0] / r, d[1] / r, d[2] / r]
        }
    }
    /// Rigid-body surface velocity `u + ω×(p−center)` at point `p`.
    #[inline]
    pub fn wall_velocity(&self, p: Vec3) -> Vec3 {
        let r = [p[0] - self.center[0], p[1] - self.center[1], p[2] - self.center[2]];
        let w = self.omega;
        [
            self.velocity[0] + w[1] * r[2] - w[2] * r[1],
            self.velocity[1] + w[2] * r[0] - w[0] * r[2],
            self.velocity[2] + w[0] * r[1] - w[1] * r[0],
        ]
    }
}

/// All immersed spheres in the domain. Mutated in place by the FSI driver (the
/// example) between steps; the `Setup` systems below re-read it every step so a
/// moving body's mask/ghosts/markers follow.
#[derive(Clone, Default, Debug)]
pub struct SphereBodies(pub Vec<SphereBody>);

/// `Setup`: mark cells whose center lies inside any sphere as solid (flip back the
/// ones that left as the body moved), seeding freshly-flipped cells with the
/// stagnant (fluid→solid) or ambient (solid→fluid, a "fresh" cell uncovered by the
/// moving body) state so the field stays physical across body motion.
fn refresh_sphere_mask<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    bodies: Res<SphereBodies>,
    consts: Res<MaskConsts>,
) {
    let mut mask = reg.expect_mut::<SolidMask>("SolidMask not registered");
    let mut state = reg.expect_mut::<cfd_state::CfdState>("CfdState not registered");
    let [ni, nj, nk] = mesh.dims();
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let c = mesh.idx(i, j, k);
                let cc = mesh.cell_centroid(c);
                let inside = bodies.0.iter().any(|b| b.signed_distance(cc) < 0.0);
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
/// image point reflected across the nearest sphere surface, `wall_velocity` set to
/// that body's rigid surface velocity at the cell center.
fn compute_sphere_ghosts<M: StructuredMesh>(
    mesh: Res<M>,
    reg: Res<FieldRegistry>,
    bodies: Res<SphereBodies>,
    mut ghosts: ResMut<IbmGhostCells>,
) {
    ghosts.cells.clear();
    if bodies.0.is_empty() {
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
                    continue; // fully buried
                }
                let cc = mesh.cell_centroid(c);
                let mut best_abs = f64::MAX;
                let mut best_signed = 0.0;
                let mut best_normal = [1.0, 0.0, 0.0];
                let mut best_body = 0u32;
                let mut best_wall = [0.0; 3];
                for (bid, b) in bodies.0.iter().enumerate() {
                    let s = b.signed_distance(cc);
                    if s.abs() < best_abs {
                        best_abs = s.abs();
                        best_signed = s;
                        best_normal = b.normal(cc);
                        best_body = bid as u32;
                        // Surface velocity evaluated at the wall footprint of this cell.
                        let foot = [
                            cc[0] - s * best_normal[0],
                            cc[1] - s * best_normal[1],
                            cc[2] - s * best_normal[2],
                        ];
                        best_wall = b.wall_velocity(foot);
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
                        wall_velocity: best_wall,
                    },
                ));
            }
        }
    }
}

/// `Setup`: place `n_markers` equal-area markers on each sphere via the spherical
/// Fibonacci lattice (analytic per-marker area `4πR²/N`, radial normal), tagged
/// with the body's rigid surface velocity so the moving-wall viscous traction sees
/// the correct slip.
fn compute_sphere_markers<M: StructuredMesh>(
    _mesh: Res<M>,
    bodies: Res<SphereBodies>,
    cfg: Res<SphereMarkerCfg>,
    mut markers: ResMut<IbmMarkers>,
) {
    markers.markers.clear();
    if bodies.0.is_empty() {
        return;
    }
    let n = cfg.n_markers.max(12);
    for (bid, b) in bodies.0.iter().enumerate() {
        sphere_markers(b, bid as u32, n, &mut markers.markers);
    }
}

/// Append `n` equal-area surface markers for one sphere via the golden-angle
/// spherical Fibonacci lattice (radial normal, analytic area `4πR²/N`, moving-wall
/// velocity). Factored out so it is unit-testable independent of the ECS.
pub(crate) fn sphere_markers(b: &SphereBody, body_id: u32, n: usize, out: &mut Vec<IbmMarker>) {
    let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
    let area = 4.0 * std::f64::consts::PI * b.radius * b.radius / n as f64;
    for m in 0..n {
        let zr = 1.0 - 2.0 * (m as f64 + 0.5) / n as f64; // (-1,1), cell-centered
        let rr = (1.0 - zr * zr).max(0.0).sqrt();
        let phi = golden * m as f64;
        let dir = [rr * phi.cos(), rr * phi.sin(), zr];
        let pos = [
            b.center[0] + b.radius * dir[0],
            b.center[1] + b.radius * dir[1],
            b.center[2] + b.radius * dir[2],
        ];
        out.push(IbmMarker {
            body_id,
            position: pos,
            normal: dir,
            area,
            wall_velocity: b.wall_velocity(pos),
        });
    }
}

/// Marker resolution for the sphere surface integral (number of Fibonacci points
/// per sphere). ~2000 gives a sub-percent solid-angle quadrature.
#[derive(Clone, Copy, Debug)]
pub struct SphereMarkerCfg {
    pub n_markers: usize,
}

impl Default for SphereMarkerCfg {
    fn default() -> Self {
        Self { n_markers: 2000 }
    }
}

/// Installs one or more immersed spheres and the per-step mask/ghost/marker
/// refresh. Pairs with [`GhostCellIbmPlugin`](crate::GhostCellIbmPlugin) (the
/// geometry-agnostic mirror + force diagnostics), exactly as
/// [`CylinderIbmPlugin`](crate::CylinderIbmPlugin) does for the 2-D disc.
pub struct SphereIbmPlugin {
    bodies: Vec<SphereBody>,
    freestream: ConsVar,
    markers: SphereMarkerCfg,
}

impl SphereIbmPlugin {
    /// Spheres immersed in a `freestream` state (seeds flipped cells).
    pub fn new(bodies: Vec<SphereBody>, freestream: ConsVar) -> Self {
        Self { bodies, freestream, markers: SphereMarkerCfg::default() }
    }
    /// Override the surface-marker count (drag quadrature resolution).
    pub fn with_markers(mut self, n_markers: usize) -> Self {
        self.markers = SphereMarkerCfg { n_markers };
        self
    }
}

impl Plugin for SphereIbmPlugin {
    fn build(&self, app: &mut App) {
        let fs = self.freestream;
        let stagnant = ConsVar::new(fs.rho, 0.0, 0.0, 0.0, fs.rho_e - fs.kinetic_energy());
        app.add_resource(SphereBodies(self.bodies.clone()));
        app.add_resource(MaskConsts { stagnant, ambient: fs });
        app.add_resource(self.markers);
        app.add_update_system(refresh_sphere_mask::<UniformMesh>, MeshScheduleSet::Setup);
        app.add_update_system(compute_sphere_ghosts::<UniformMesh>, MeshScheduleSet::Setup);
        app.add_update_system(compute_sphere_markers::<UniformMesh>, MeshScheduleSet::Setup);
    }
    fn provides(&self) -> Vec<&str> {
        vec!["sphere_ibm"]
    }
    fn requires(&self) -> Vec<&str> {
        vec!["ibm_ghost_cell", "mesh"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_geometry_and_wall_velocity() {
        let b = SphereBody {
            center: [1.0, 2.0, 3.0],
            radius: 0.5,
            velocity: [0.1, 0.0, 0.0],
            omega: [0.0, 0.0, 2.0], // spin about z
        };
        // Signed distance: on-axis point.
        assert!((b.signed_distance([2.0, 2.0, 3.0]) - 0.5).abs() < 1e-12);
        assert!((b.signed_distance([1.0, 2.0, 3.0]) + 0.5).abs() < 1e-12); // center → −R
        // Outward normal at +x surface point.
        let n = b.normal([1.5, 2.0, 3.0]);
        assert!((n[0] - 1.0).abs() < 1e-12 && n[1].abs() < 1e-12);
        // Rigid wall velocity u + ω×r at the +x surface point (r=[0.5,0,0]):
        // ω×r = [0,0,2]×[0.5,0,0] = [0, 1.0, 0]; plus u = [0.1,0,0].
        let wv = b.wall_velocity([1.5, 2.0, 3.0]);
        assert!((wv[0] - 0.1).abs() < 1e-12);
        assert!((wv[1] - 1.0).abs() < 1e-12);
        assert!(wv[2].abs() < 1e-12);
    }

    #[test]
    fn fibonacci_markers_sum_to_sphere_area_and_lie_on_surface() {
        let b = SphereBody::fixed([0.0, 0.0, 0.0], 0.75);
        let n = 4000;
        let mut m = Vec::new();
        sphere_markers(&b, 0, n, &mut m);
        assert_eq!(m.len(), n);
        // Areas sum to 4πR².
        let total: f64 = m.iter().map(|x| x.area).sum();
        let exact = 4.0 * std::f64::consts::PI * b.radius * b.radius;
        assert!((total / exact - 1.0).abs() < 1e-12, "area {total} vs {exact}");
        // Every marker sits on the surface with a unit radial normal.
        for x in &m {
            let r = (x.position[0].powi(2) + x.position[1].powi(2) + x.position[2].powi(2)).sqrt();
            assert!((r - b.radius).abs() < 1e-9, "off-surface marker r={r}");
            let nlen = (x.normal[0].powi(2) + x.normal[1].powi(2) + x.normal[2].powi(2)).sqrt();
            assert!((nlen - 1.0).abs() < 1e-9);
        }
        // Centroid of an equal-area lattice on a centered sphere ≈ origin.
        let mut c = [0.0; 3];
        for x in &m {
            for d in 0..3 {
                c[d] += x.position[d];
            }
        }
        for d in 0..3 {
            assert!((c[d] / n as f64).abs() < 1e-2, "lattice not centered: {c:?}");
        }
    }
}
