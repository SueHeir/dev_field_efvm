//! Cut-cell (embedded-boundary) geometry for the immersed boundary method —
//! the alternative to the sharp-interface ghost-cell scheme in the parent module.
//!
//! Where the ghost-cell method reflects fluid state across the surface, the
//! cut-cell method resolves the body *geometrically*: each cell gets a fluid
//! **volume fraction** `α_v ∈ [0,1]` and six face **area fractions** `α_f`, and
//! the embedded boundary inside the cell carries an outward (into-fluid)
//! `n̂_eb · A_eb`. A finite-volume update then scales by `α_v·V` and adds the EB
//! wall flux — second-order-ish wall resolution without body-fitted meshing.
//!
//! These metrics depend only on **per-cell geometry** (`center`, `size`) and the
//! body's signed-distance function, so they are deliberately decoupled from the
//! mesh backend: feed them `&[(center, size)]` (e.g. from
//! `ForestMesh::interior_geometry`) and any [`Body`]. Small-cell merging (the
//! Aftosmis–Berger stabilization for the explicit time step) is a follow-up layer.

use cfd_eos::Eos;
use cfd_state::ConsVar;
use field_core::Vec3;

/// An implicit solid body: a signed-distance function, negative inside the solid,
/// positive in the fluid. Generalizes the analytic [`super::Cylinder`] to anything
/// with an SDF (spheres, unions, FSI-coupled particles).
pub trait Body: Sync {
    /// Signed distance to the surface at `p` (negative inside the solid).
    fn signed_distance(&self, p: Vec3) -> f64;

    /// Outward (into-fluid) unit surface normal, ∇φ/|∇φ| by default.
    fn surface_normal(&self, p: Vec3) -> Vec3 {
        let h = 1e-6;
        let phi = self.signed_distance(p);
        let g = [
            (self.signed_distance([p[0] + h, p[1], p[2]]) - phi) / h,
            (self.signed_distance([p[0], p[1] + h, p[2]]) - phi) / h,
            (self.signed_distance([p[0], p[1], p[2] + h]) - phi) / h,
        ];
        let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        if len < 1e-15 {
            [0.0, 0.0, 1.0]
        } else {
            [g[0] / len, g[1] / len, g[2] / len]
        }
    }
}

/// Sphere of `radius` centred at `center`.
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

impl Body for Sphere {
    fn signed_distance(&self, p: Vec3) -> f64 {
        let d = [p[0] - self.center[0], p[1] - self.center[1], p[2] - self.center[2]];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - self.radius
    }
}

impl Body for super::Cylinder {
    fn signed_distance(&self, p: Vec3) -> f64 {
        super::Cylinder::signed_distance(self, p)
    }
}

/// Fluid volume fraction of an axis-aligned box by `n³` regular quadrature.
fn sample_volume_fraction<B: Body + ?Sized>(body: &B, center: Vec3, size: Vec3, n: usize) -> f64 {
    let inv_n = 1.0 / n as f64;
    let mut fluid = 0usize;
    for ix in 0..n {
        let x = center[0] - 0.5 * size[0] + (ix as f64 + 0.5) * inv_n * size[0];
        for iy in 0..n {
            let y = center[1] - 0.5 * size[1] + (iy as f64 + 0.5) * inv_n * size[1];
            for iz in 0..n {
                let z = center[2] - 0.5 * size[2] + (iz as f64 + 0.5) * inv_n * size[2];
                if body.signed_distance([x, y, z]) >= 0.0 {
                    fluid += 1;
                }
            }
        }
    }
    fluid as f64 / (n * n * n) as f64
}

/// Fluid (open) area fraction of an axis-aligned face by `n²` regular quadrature.
fn sample_face_fraction<B: Body + ?Sized>(
    body: &B,
    face_center: Vec3,
    ta: usize,
    ext_a: f64,
    tb: usize,
    ext_b: f64,
    n: usize,
) -> f64 {
    let inv_n = 1.0 / n as f64;
    let mut fluid = 0usize;
    for ia in 0..n {
        for ib in 0..n {
            let mut p = face_center;
            p[ta] += (-0.5 + (ia as f64 + 0.5) * inv_n) * ext_a;
            p[tb] += (-0.5 + (ib as f64 + 0.5) * inv_n) * ext_b;
            if body.signed_distance(p) >= 0.0 {
                fluid += 1;
            }
        }
    }
    fluid as f64 / (n * n) as f64
}

/// Per-cell cut-cell metrics for an embedded body. Indexed by interior-cell index
/// (matching the `geom` slice passed to [`compute`](CutCellMetrics::compute)).
pub struct CutCellMetrics {
    /// Fluid volume fraction: 1 = all fluid, 0 = all solid, between = cut.
    pub alpha_v: Vec<f64>,
    /// Open area fraction of each of the 6 axis faces (XLo,XHi,YLo,YHi,ZLo,ZHi).
    pub alpha_f: Vec<[f64; 6]>,
    /// Outward (into-fluid) `n̂_eb · A_eb` of the embedded boundary in each cell,
    /// from the **face-divergence identity** `(α_f[hi]−α_f[lo])·A_face` per axis —
    /// the discretely-consistent choice given `α_f` (so axis-flux-out and EB-flux-out
    /// sum to a true divergence and momentum doesn't leak).
    pub eb_normal_area: Vec<[f64; 3]>,
}

impl CutCellMetrics {
    /// Number of quadrature points per axis for `α` sampling (8 ⇒ 512 volume /
    /// 64 area samples; a few-percent accurate for smooth bodies).
    pub const SAMPLES: usize = 8;

    /// Build metrics for cells given as `(center, size)` against `body`.
    pub fn compute<B: Body + ?Sized>(geom: &[(Vec3, Vec3)], body: &B) -> Self {
        let n = geom.len();
        let mut alpha_v = vec![1.0; n];
        let mut alpha_f = vec![[1.0; 6]; n];

        for (i, &(center, size)) in geom.iter().enumerate() {
            let phi = body.signed_distance(center);
            // Half-diagonal: the farthest a corner can be from the centre. The
            // |phi| >= diag early-outs (all-fluid / all-solid) are exact ONLY for
            // a true Euclidean (1-Lipschitz) SDF — `|∇φ| = 1`, so the surface
            // cannot reach a corner when the centre is farther than `diag`. A
            // non-metric implicit function (scaled fields, smooth-min unions) may
            // under/over-estimate distance and mis-classify a cut cell as uniform;
            // such bodies should widen the band or skip the early-out.
            let diag = 0.5 * (size[0] * size[0] + size[1] * size[1] + size[2] * size[2]).sqrt();
            let av = if phi >= diag {
                1.0
            } else if phi <= -diag {
                0.0
            } else {
                sample_volume_fraction(body, center, size, Self::SAMPLES)
            };
            alpha_v[i] = av;

            alpha_f[i] = if av <= 0.0 {
                [0.0; 6]
            } else if av >= 1.0 {
                [1.0; 6]
            } else {
                let mut af = [1.0f64; 6];
                for (face, slot) in af.iter_mut().enumerate() {
                    let axis = face / 2;
                    let sign = if face % 2 == 1 { 1.0 } else { -1.0 };
                    let mut fc = center;
                    fc[axis] += 0.5 * sign * size[axis];
                    let (ta, tb) = match axis {
                        0 => (1, 2),
                        1 => (0, 2),
                        _ => (0, 1),
                    };
                    *slot = sample_face_fraction(body, fc, ta, size[ta], tb, size[tb], Self::SAMPLES);
                }
                af
            };
        }

        let eb_normal_area: Vec<[f64; 3]> = geom
            .iter()
            .enumerate()
            .map(|(i, &(_, size))| {
                let af = alpha_f[i];
                [
                    (af[1] - af[0]) * size[1] * size[2],
                    (af[3] - af[2]) * size[0] * size[2],
                    (af[5] - af[4]) * size[0] * size[1],
                ]
            })
            .collect();

        CutCellMetrics { alpha_v, alpha_f, eb_normal_area }
    }

    /// Total solid volume implied by the fractions: `Σ (1−α_v)·V`. For a body fully
    /// inside the domain this recovers the body's analytic volume.
    pub fn solid_volume(&self, geom: &[(Vec3, Vec3)]) -> f64 {
        self.alpha_v
            .iter()
            .zip(geom)
            .map(|(&av, &(_, s))| (1.0 - av) * s[0] * s[1] * s[2])
            .sum()
    }

    /// Net pressure force the fluid exerts on the embedded body,
    /// `−Σ p_cell · (n̂_eb·A_eb)` (the EB normal points into the fluid, so the
    /// force on the body is along `−n̂_eb`). For a uniform pressure field over a
    /// closed body this is ≈ 0 — the consistency check on `eb_normal_area`.
    pub fn pressure_force(&self, u: &[ConsVar], eos: &dyn Eos) -> Vec3 {
        let mut f = [0.0; 3];
        for (i, na) in self.eb_normal_area.iter().enumerate() {
            if self.alpha_v[i] <= 0.0 || self.alpha_v[i] >= 1.0 {
                continue;
            }
            let p = eos.pressure(&u[i]);
            for d in 0..3 {
                f[d] -= p * na[d];
            }
        }
        f
    }
}
