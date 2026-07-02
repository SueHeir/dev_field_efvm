//! Cut-cell (embedded-boundary) IBM geometry on a real p4est forest: a sphere is
//! resolved by per-cell fluid volume fractions α_v and embedded-boundary normals.
//! We validate the geometry quantitatively: (1) Σ(1−α_v)·V recovers the sphere's
//! analytic volume, and (2) the EB normals are closed — a uniform pressure field
//! exerts ≈ zero net force.
//! Integration test (own process) for `MPI_Init`.

use cfd_eos::{Eos, IdealGas};
use cfd_ibm::{Body, CutCellMetrics, Sphere};
use cfd_state::PrimVar;
use field_p4est::{ForestGrid, ForestLayout, ForestMesh};

#[test]
fn cutcell_recovers_sphere_volume_and_closed_normals() {
    // Cubic domain [0,1]³, sphere fully inside.
    let layout = ForestLayout {
        trees_x: 4,
        trees_y: 4,
        trees_z: 4,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 1.0,
        min_level: 0,
    };
    let sphere = Sphere { center: [0.5, 0.5, 0.5], radius: 0.2 };

    let mut grid = ForestGrid::new(layout).expect("forest");
    // Refine to level 2 in a shell around the surface so cut cells are well resolved.
    grid.refine(2, |cx, cy, cz, _h| sphere.signed_distance([cx, cy, cz]).abs() < 0.18);
    let mesh = ForestMesh::new(grid);

    let geom = mesh.interior_geometry();
    let metrics = CutCellMetrics::compute(&geom, &sphere);

    // 1) Solid volume recovered from α_v ≈ analytic sphere volume.
    let analytic = 4.0 / 3.0 * std::f64::consts::PI * 0.2_f64.powi(3);
    let recovered = metrics.solid_volume(&geom);
    let rel = (recovered - analytic).abs() / analytic;
    assert!(
        rel < 0.05,
        "cut-cell volume off by {:.1}%: recovered {recovered:.6} vs analytic {analytic:.6}",
        rel * 100.0,
    );

    // There genuinely are cut cells (0 < α_v < 1).
    let n_cut = metrics.alpha_v.iter().filter(|&&a| a > 0.0 && a < 1.0).count();
    assert!(n_cut > 50, "expected a well-resolved cut band, got {n_cut} cut cells");

    // 2) Closed embedded boundary: uniform pressure ⇒ ≈ zero net force.
    let eos = IdealGas::air();
    let p0 = 1.0e5;
    let u0 = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, p0, p0 / 287.058));
    let u = vec![u0; geom.len()];
    let force = metrics.pressure_force(&u, &eos);
    let fmag = (force[0] * force[0] + force[1] * force[1] + force[2] * force[2]).sqrt();
    // Reference scale: pressure × sphere cross-section.
    let scale = p0 * std::f64::consts::PI * 0.2_f64.powi(2);
    assert!(
        fmag < 0.02 * scale,
        "uniform-pressure net force should vanish: |F| = {fmag:.2}, scale = {scale:.2}",
    );
}
