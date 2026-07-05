//! **Multi-rank** validation of the cross-rank p4est ghost layer + halo exchange.
//!
//! p4est partitions the forest across MPI ranks; cells along a partition boundary
//! become each other's ghosts. This binary exercises the real exchange and checks:
//!   1. a uniform flow stays *exactly* uniform after many steps — if a single
//!      cross-rank ghost value were wrong, the flux at a partition-boundary face
//!      would be wrong and cells there would drift;
//!   2. a closed (slip-wall) box conserves total mass *globally* (summed across
//!      ranks) as a pressure pulse sloshes through the distributed mesh.
//!
//! Build + run (needs an MPI launcher; the normal `cargo test` path never touches
//! this — it compiles to a stub without the feature):
//!   cargo build -p dev_field_efvm --bin p4est_multirank --features mpi_backend
//!   mpirun -np 2 target/debug/p4est_multirank
//!
//! Exits non-zero if either check fails, so `mpirun` propagates the failure.
#![allow(clippy::needless_range_loop)] // explicit axis/cell indexing reads clearest here

#[cfg(not(feature = "mpi_backend"))]
fn main() {
    eprintln!("p4est_multirank requires the `mpi_backend` feature and an MPI launcher:");
    eprintln!("  cargo build -p dev_field_efvm --bin p4est_multirank --features mpi_backend");
    eprintln!("  mpirun -np 2 target/debug/p4est_multirank");
}

#[cfg(feature = "mpi_backend")]
fn main() {
    // Initialize MPI (via rsmpi) BEFORE the first ForestGrid::new — the p4est
    // shim's amr_init then cooperates with the already-running MPI_Init instead
    // of starting its own.
    let rank = grass_mpi::world_rank();
    let size = grass_mpi::world_size();

    let (uniform_ok, exchanged) = mr::uniform_preserved_across_ranks();
    let mass_ok = mr::mass_conserved_globally();
    let (seam_closed, seam_faces) = mr::coarse_fine_seam_is_closed();

    if rank == 0 {
        println!("─────────────────────────────────────────────");
        println!("p4est multi-rank validation on {size} rank(s)");
        println!("  cross-rank ghost cells exchanged        : {exchanged}");
        println!(
            "  uniform flow preserved across partition : {}",
            pass(uniform_ok)
        );
        println!(
            "  closed-box mass conserved globally      : {}",
            pass(mass_ok)
        );
        println!("  cross-rank coarse/fine faces            : {seam_faces}");
        println!(
            "  coarse/fine seam geometrically closed   : {}",
            pass(seam_closed)
        );
        println!("─────────────────────────────────────────────");
        // A multi-rank run with no exchanged ghosts / no cross-rank C/F seam
        // would make the respective checks vacuous.
        if size > 1 && exchanged == 0 {
            eprintln!("FAIL: multi-rank run exchanged zero ghost cells (vacuous)");
            grass_mpi::finalize_mpi();
            std::process::exit(1);
        }
        if size > 1 && seam_faces == 0 {
            eprintln!("FAIL: refinement seam did not straddle the partition (vacuous)");
            grass_mpi::finalize_mpi();
            std::process::exit(1);
        }
    }

    grass_mpi::finalize_mpi();
    if !(uniform_ok && mass_ok && seam_closed) {
        std::process::exit(1);
    }
}

#[cfg(feature = "mpi_backend")]
fn pass(b: bool) -> &'static str {
    if b {
        "PASS"
    } else {
        "FAIL"
    }
}

#[cfg(feature = "mpi_backend")]
mod mr {
    use cfd_boundary::{boundary_system, BoundaryRegistry, SlipWall};
    use cfd_eos::{Eos, EosResource, IdealGas};
    use cfd_riemann::Rusanov;
    use cfd_solver::{
        cfl_generic_system, flux_system, halo_system, update_system, zero_rhs_system,
        EulerIntegrator, RiemannFluxResource, SolverConfig, SolverState, TimeIntegratorResource,
    };
    use cfd_state::{CfdState, PrimVar};
    use field_core::{
        register_field_data, BoundarySide, CommPlugin, FieldRegistry, FieldRegistryPlugin, FvMesh,
        MeshScheduleSet,
    };
    use field_p4est::{ForestGrid, ForestLayout, ForestMesh};
    use grass_app::prelude::*;
    use grass_app::ScheduleSetupSet;
    use grass_mpi::CommResource;
    use grass_scheduler::Res;
    use std::sync::Arc;

    fn resize_forest(mesh: Res<ForestMesh>, reg: Res<FieldRegistry>) {
        reg.resize_all(mesh.n_cells_total());
    }

    fn base_layout() -> ForestLayout {
        ForestLayout {
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
        }
    }

    /// Common solver wiring (inviscid Euler on the forest, with halo exchange).
    fn build_app(mesh: ForestMesh, bcs: BoundaryRegistry) -> App {
        let eos = IdealGas::air();
        let mut app = App::new();
        app.add_plugins(CommPlugin); // ← MPI backend under the feature
        app.add_plugins(FieldRegistryPlugin);
        app.add_resource(mesh);
        app.add_resource(EosResource(Arc::new(eos)));
        app.add_resource(RiemannFluxResource(Arc::new(Rusanov)));
        app.add_resource(TimeIntegratorResource(Arc::new(EulerIntegrator)));
        app.add_resource(SolverState::default());
        app.add_resource(SolverConfig {
            cfl: 0.2,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        });
        app.add_resource(bcs);
        register_field_data!(app, CfdState::default());

        app.add_setup_system(resize_forest, ScheduleSetupSet::Setup);
        app.add_update_system(cfl_generic_system::<ForestMesh>, MeshScheduleSet::Setup);
        app.add_update_system(halo_system::<ForestMesh>, MeshScheduleSet::Halo);
        app.add_update_system(boundary_system::<ForestMesh>, MeshScheduleSet::Boundary);
        app.add_update_system(zero_rhs_system::<ForestMesh>, MeshScheduleSet::PreFlux);
        app.add_update_system(flux_system::<ForestMesh>, MeshScheduleSet::Flux);
        app.add_update_system(update_system, MeshScheduleSet::Update);
        app.prepare();
        app
    }

    /// Sum `local` across all ranks via the app's comm backend.
    fn all_reduce_sum(app: &App, local: f64) -> f64 {
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        comm.0.all_reduce_sum_f64(local)
    }

    /// Build a partitioned forest, run uniform inviscid flow, and check that no
    /// cell drifted from the uniform state on ANY rank (reduced globally).
    /// Returns `(preserved, total_ghost_cells_exchanged)` — the second value is
    /// the global count of received halo cells, proving the test is non-vacuous.
    pub fn uniform_preserved_across_ranks() -> (bool, u64) {
        let mut grid = ForestGrid::new(base_layout()).expect("forest");
        // Refinement triggers p8est_partition → leaves split across ranks, and
        // creates coarse/fine faces (some straddling the partition boundary).
        grid.refine(1, |cx, cy, _cz, _h| {
            ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt() < 0.3
        });
        let mesh = ForestMesh::new(grid);

        let eos = IdealGas::air();
        let u0 = eos.prim_to_cons(&PrimVar::new(1.0, 18.0, -7.0, 0.0, 100_000.0, 348.4));

        let mut app = build_app(mesh, BoundaryRegistry::default()); // outflow
                                                                    // Count the halo cells this rank receives (proves cross-rank coupling).
        let local_recv = {
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            mesh.halo_plan()
                .links
                .iter()
                .map(|l| l.recv_cells.len())
                .sum::<usize>() as f64
        };
        {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mut s = reg.get_mut::<CfdState>().unwrap();
            for u in s.u.iter_mut() {
                *u = u0;
            }
        }
        for _ in 0..30 {
            app.run();
        }

        // Local max drift over owned cells.
        let local_violations = {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let s = reg.get::<CfdState>().unwrap();
            let mut bad = 0.0f64;
            for c in 0..mesh.n_cells_total() {
                if !mesh.is_local_cell(c) {
                    continue;
                }
                let u = s.u[c];
                if (u.rho - u0.rho).abs() > 1e-9
                    || (u.rho_u - u0.rho_u).abs() > 1e-6
                    || (u.rho_e - u0.rho_e).abs() > 1e-3
                {
                    bad += 1.0;
                }
            }
            bad
        };
        // Zero iff every cell on every rank held uniform.
        let preserved = all_reduce_sum(&app, local_violations) == 0.0;
        let exchanged = all_reduce_sum(&app, local_recv) as u64;
        (preserved, exchanged)
    }

    /// Force coarse/fine interfaces to straddle the partition by refining a
    /// **checkerboard** of base cells to level 1 — every refined cell is ringed
    /// by coarse face-neighbours, so C/F interfaces blanket the domain and any
    /// SFC partition cut slices through some of them (true even at np=2). Then
    /// check every *local* cell is geometrically closed (Σ area_normal ≈ 0).
    ///
    /// This is the decisive cross-rank coarse/fine test: a coarse cell on the
    /// rank that owns it, whose fine neighbours live on another rank, is missing
    /// its fine sub-faces unless cross-rank C/F is handled — and an unclosed cell
    /// shows up *immediately*, no time-stepping needed.
    /// Returns `(closed, cross_rank_cf_faces)`.
    pub fn coarse_fine_seam_is_closed() -> (bool, u64) {
        let mut grid = ForestGrid::new(base_layout()).expect("forest");
        grid.refine(1, |cx, cy, _cz, h| {
            // Only level-0 cells (size > 0.1), checkerboard by base (i+j) parity.
            h > 0.1 && (((cx * 8.0).floor() as i64 + (cy * 8.0).floor() as i64) % 2 == 0)
        });
        let mesh = ForestMesh::new(grid);

        let eos = IdealGas::air();
        let u0 = eos.prim_to_cons(&PrimVar::new(1.0, 18.0, -7.0, 0.0, 100_000.0, 348.4));
        let mut app = build_app(mesh, BoundaryRegistry::default());

        // (1) Static geometric closure: worst per-cell |Σ area_normal|, normalized
        //     by a face area so the tolerance is dimensionless. Non-zero at a
        //     coarse cell missing its remote fine sub-faces.
        let (worst, local_seam) = {
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let mut comp = vec![[0.0f64; 3]; mesh.n_cells_total()];
            mesh.for_each_face(&mut |f| {
                for d in 0..3 {
                    comp[f.owner][d] += f.area_normal[d];
                    if f.patch.is_none() && mesh.is_local_cell(f.other) {
                        comp[f.other][d] -= f.area_normal[d];
                    }
                }
            });
            let mut worst = 0.0f64;
            for c in 0..mesh.n_cells_total() {
                if !mesh.is_local_cell(c) {
                    continue;
                }
                let s = mesh.cell_size_at(c);
                let ref_area = s[1] * s[2] + s[0] * s[2] + s[0] * s[1];
                let r = (comp[c][0].powi(2) + comp[c][1].powi(2) + comp[c][2].powi(2)).sqrt();
                worst = worst.max(r / ref_area);
            }
            (worst, mesh.cross_rank_refinement_faces() as f64)
        };

        // (2) Dynamic: uniform flow on the checkerboard must stay exactly uniform —
        //     the cross-rank C/F sub-fluxes must agree on both ranks and telescope.
        {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mut s = reg.get_mut::<CfdState>().unwrap();
            for u in s.u.iter_mut() {
                *u = u0;
            }
        }
        for _ in 0..20 {
            app.run();
        }
        let local_drift = {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let s = reg.get::<CfdState>().unwrap();
            let mut bad = 0.0f64;
            for c in 0..mesh.n_cells_total() {
                if mesh.is_local_cell(c)
                    && ((s.u[c].rho - u0.rho).abs() > 1e-9
                        || (s.u[c].rho_u - u0.rho_u).abs() > 1e-6
                        || (s.u[c].rho_e - u0.rho_e).abs() > 1e-3)
                {
                    bad += 1.0;
                }
            }
            bad
        };

        let global_worst = {
            let comm = app.get_resource_ref::<CommResource>().unwrap();
            -comm.0.all_reduce_min_f64(-worst) // max via -min(-x)
        };
        let seam = all_reduce_sum(&app, local_seam) as u64;
        let drift = all_reduce_sum(&app, local_drift);
        (global_worst < 1e-9 && drift == 0.0, seam)
    }

    /// Build a partitioned closed (slip-wall) box, slosh a pressure pulse, and
    /// check the GLOBAL mass (summed across ranks) is conserved.
    pub fn mass_conserved_globally() -> bool {
        let mut grid = ForestGrid::new(base_layout()).expect("forest");
        grid.refine(1, |cx, cy, _cz, _h| {
            (cx - 0.5).abs() < 0.25 && (cy - 0.5).abs() < 0.25
        });
        let mesh = ForestMesh::new(grid);

        let eos = IdealGas::air();
        let pulse = |x: field_core::Vec3| {
            let r2 = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
            let p = if r2 < 0.1 * 0.1 { 2.0e5 } else { 1.0e5 };
            eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, p, p / 287.058))
        };

        let mut bcs = BoundaryRegistry::default();
        for side in [
            BoundarySide::XLo,
            BoundarySide::XHi,
            BoundarySide::YLo,
            BoundarySide::YHi,
            BoundarySide::ZLo,
            BoundarySide::ZHi,
        ] {
            bcs.set(side, SlipWall);
        }
        let mut app = build_app(mesh, bcs);

        let local_mass = |app: &App| -> f64 {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let s = reg.get::<CfdState>().unwrap();
            (0..mesh.n_cells_total())
                .filter(|&c| mesh.is_local_cell(c))
                .map(|c| s.u[c].rho * mesh.cell_volume(c))
                .sum()
        };

        {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let mut s = reg.get_mut::<CfdState>().unwrap();
            for c in 0..mesh.n_cells_total() {
                s.u[c] = pulse(mesh.cell_centroid(c));
            }
        }
        let mass0 = all_reduce_sum(&app, local_mass(&app));
        for _ in 0..40 {
            app.run();
        }
        let mass1 = all_reduce_sum(&app, local_mass(&app));

        (mass1 - mass0).abs() / mass0 < 1e-10
    }
}
