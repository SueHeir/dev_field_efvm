//! Dynamic, solution-adaptive AMR on a real p4est forest: refine where the
//! solution (a pressure pulse) warrants it, then mass-conservingly transfer the
//! field onto the regridded mesh. This is the core AMR loop p4est exists for.
//! Integration test (own process) for `MPI_Init`.

use cfd_eos::{Eos, IdealGas};
use cfd_state::{ConsVar, PrimVar};
use field_core::{AdaptiveMesh, FvMesh};
use field_p4est::{locate_in, transfer_recursive, ForestGrid, ForestLayout, ForestMesh};

/// Component-wise mean of conserved states (equal-volume sub-octants).
fn mean_cons(vals: &[ConsVar]) -> ConsVar {
    let n = vals.len() as f64;
    let mut s = ConsVar::ZERO;
    for v in vals {
        s.rho += v.rho;
        s.rho_u += v.rho_u;
        s.rho_v += v.rho_v;
        s.rho_w += v.rho_w;
        s.rho_e += v.rho_e;
    }
    ConsVar::new(s.rho / n, s.rho_u / n, s.rho_v / n, s.rho_w / n, s.rho_e / n)
}

#[test]
fn regrid_follows_the_solution_and_transfers_conservatively() {
    // 8×8×1 base forest over [0,1]² × thin z; per-tree extent 0.125 in x/y.
    let layout = ForestLayout {
        trees_x: 8,
        trees_y: 8,
        trees_z: 1,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 0.125,
        min_level: 0,
    };
    let mut grid = ForestGrid::new(layout).expect("forest");
    grid.refine(0, |_, _, _, _| false); // build the p4est mesh at the base level
    let mut mesh = ForestMesh::new(grid);

    let eos = IdealGas::air();
    // A pulse at (0.3, 0.5): high pressure AND density inside, ambient outside.
    let center = [0.3, 0.5];
    let pulse = |x: field_core::Vec3| {
        let r2 = (x[0] - center[0]).powi(2) + (x[1] - center[1]).powi(2);
        if r2 < 0.12 * 0.12 {
            eos.prim_to_cons(&PrimVar::new(1.5, 0.0, 0.0, 0.0, 2e5, 2e5 / (1.5 * 287.058)))
        } else {
            eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 1e5, 1e5 / 287.058))
        }
    };

    // Initialize the (uniform, base-level) field.
    let n0 = mesh.n_local_cells();
    let old_geom = mesh.interior_geometry();
    let old_u: Vec<ConsVar> = (0..n0).map(|c| pulse(mesh.cell_centroid(c))).collect();
    let mass_old: f64 = (0..n0).map(|c| old_u[c].rho * old_geom[c].1.iter().product::<f64>()).sum();

    // ── Solution-adaptive refinement: refine where the *old field's* pressure is
    //    high (i.e. the pulse). The criterion samples the snapshotted solution. ──
    {
        let geom = old_geom.clone();
        let ou = old_u.clone();
        mesh.refine(1, move |cx, cy, cz, _h| match locate_in(&geom, [cx, cy, cz]) {
            Some(i) => eos.pressure(&ou[i]) > 1.5e5,
            None => false,
        });
    }

    // Refinement happened and is concentrated at the pulse.
    let (_coarse, fine) = mesh.level_counts();
    assert!(fine > 0, "expected solution-adaptive refinement");
    for c in 0..mesh.n_cells_total() {
        if mesh.is_local_cell(c) && mesh.cell_level(c) > 0 {
            let cc = mesh.cell_centroid(c);
            let r = ((cc[0] - center[0]).powi(2) + (cc[1] - center[1]).powi(2)).sqrt();
            assert!(r < 0.25, "fine cell at {cc:?} is not near the pulse");
        }
    }

    // ── Transfer the field onto the regridded mesh by point sampling. ──
    let n1 = mesh.n_cells_total();
    let new_u: Vec<ConsVar> = (0..n1)
        .map(|c| match locate_in(&old_geom, mesh.cell_centroid(c)) {
            Some(i) => old_u[i],
            None => ConsVar::ZERO, // boundary ghosts
        })
        .collect();

    // The pulse survived the transfer: high p at its centre, ambient far away.
    let probe_at = |target: [f64; 2]| -> f64 {
        let (mut best, mut best_d) = (0usize, f64::MAX);
        for c in 0..n1 {
            if !mesh.is_local_cell(c) {
                continue;
            }
            let cc = mesh.cell_centroid(c);
            let d = (cc[0] - target[0]).powi(2) + (cc[1] - target[1]).powi(2);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        eos.pressure(&new_u[best])
    };
    assert!(probe_at([0.3, 0.5]) > 1.5e5, "pulse lost at its centre after transfer");
    assert!(probe_at([0.85, 0.5]) < 1.1e5, "spurious pressure far from the pulse");

    // Refine-only point sampling is exactly mass-conserving (each child inherits
    // the parent ρ; 8 children tile the parent volume).
    let mass_new: f64 = (0..n1)
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| new_u[c].rho * mesh.cell_volume(c))
        .sum();
    assert!(
        (mass_new - mass_old).abs() / mass_old < 1e-9,
        "regrid transfer not mass-conserving: {mass_old} -> {mass_new}",
    );
}

#[test]
fn conservative_coarsen_transfer_preserves_mass_and_momentum() {
    // Start with a forest refined to level 1 on the left half, so we have fine
    // cells carrying a spatially-varying field to coarsen and average back.
    let layout = ForestLayout {
        trees_x: 4,
        trees_y: 4,
        trees_z: 1,
        xmin: 0.0,
        xmax: 1.0,
        ymin: 0.0,
        ymax: 1.0,
        zmin: 0.0,
        zmax: 0.25,
        min_level: 0,
    };
    let mut grid = ForestGrid::new(layout).expect("forest");
    grid.refine(1, |cx, _cy, _cz, _h| cx < 0.5);
    let mut mesh = ForestMesh::new(grid);

    let eos = IdealGas::air();
    // A field that varies *within* each coarse cell, so a single child's value is
    // NOT representative — only the 8-child average conserves the integral.
    let field = |x: field_core::Vec3| {
        let rho = 1.0 + 0.4 * x[0] + 0.3 * x[1];
        let u = 5.0 + 20.0 * x[0];
        eos.prim_to_cons(&PrimVar::new(rho, u, 0.0, 0.0, 1e5, 1e5 / (rho * 287.058)))
    };

    let n0 = mesh.n_local_cells();
    let old_geom = mesh.interior_geometry();
    let old_u: Vec<ConsVar> = (0..n0).map(|c| field(mesh.cell_centroid(c))).collect();
    let integral = |u: &[ConsVar], get: fn(&ConsVar) -> f64, geom: &[(field_core::Vec3, field_core::Vec3)]| -> f64 {
        (0..geom.len()).map(|c| get(&u[c]) * geom[c].1.iter().product::<f64>()).sum()
    };
    let mass_old = integral(&old_u, |u| u.rho, &old_geom);
    let mom_old = integral(&old_u, |u| u.rho_u, &old_geom);

    // Coarsen the refined (left) half back to the base level.
    mesh.coarsen(0, |cx, _cy, _cz, _h| cx < 0.5);
    let (_coarse, fine) = mesh.level_counts();
    assert_eq!(fine, 0, "expected the refined region to coarsen away");

    // Conservative transfer: each (now coarse) cell = volume-mean of its 8 children.
    let n1 = mesh.n_cells_total();
    let new_u: Vec<ConsVar> = (0..n1)
        .map(|c| {
            let s = mesh.cell_size_at(c);
            transfer_recursive(&old_geom, &old_u, ConsVar::ZERO, mesh.cell_centroid(c), s, &mean_cons)
        })
        .collect();

    let mass_new: f64 = (0..n1)
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| new_u[c].rho * mesh.cell_volume(c))
        .sum();
    let mom_new: f64 = (0..n1)
        .filter(|&c| mesh.is_local_cell(c))
        .map(|c| new_u[c].rho_u * mesh.cell_volume(c))
        .sum();

    assert!(
        (mass_new - mass_old).abs() / mass_old < 1e-12,
        "conservative coarsen lost mass: {mass_old} -> {mass_new}",
    );
    assert!(
        (mom_new - mom_old).abs() / mom_old.abs() < 1e-12,
        "conservative coarsen lost momentum: {mom_old} -> {mom_new}",
    );
}
