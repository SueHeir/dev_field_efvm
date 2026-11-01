//! A periodic advected pulse gives the same answer on 1, 2, 3 and 4 ranks.
//!
//! Periodic axes are FIELD halo links (`MeshPeriodicity`, added by
//! `StructuredBoundaryPlugin`), so the wrap crosses ranks like any other halo.
//! Each run reduces mass and two moments of density across ranks and writes
//! them to a file; the coordinator checks every rank count against one rank.
//!
//! `cargo test` runs a single process, so the coordinator re-launches this
//! binary under `mpirun` (the grass_mpi pattern). Needs `--features
//! mpi_backend` and `mpirun` on PATH; otherwise it skips with a note.
#![cfg(feature = "mpi_backend")]

use cfd_boundary::{BoundaryRegistry, Periodic, SlipWall, StructuredBoundaryPlugin};
use cfd_eos::{Eos, IdealGas};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
};
use cfd_state::{CfdState, PrimVar};
use field_core::{FieldDefaultPlugins, FieldRegistry, FvMesh, UniformMesh, UniformMeshConfig, Vec3};
use grass_app::prelude::*;
use grass_mpi::CommResource;
use std::process::Command;

const CHILD_ENV: &str = "EFVM_PERIODIC_MPI_OUT";
const RANKS: [usize; 4] = [1, 2, 3, 4];
const STEPS: usize = 30; // RK3 stages; 10 full steps

fn pulse(x: Vec3) -> cfd_state::ConsVar {
    let eos = IdealGas { gamma: 1.4, r_gas: 1.0 };
    // Starts near the high-x wall so it crosses the periodic seam.
    let d = x[0] - 0.9;
    let d = d - d.round();
    let rho = 1.0 + 0.2 * (-0.5 * (d / 0.08).powi(2)).exp();
    eos.prim_to_cons(&PrimVar::new(rho, 1.0, 0.0, 0.0, 1.0, 1.0 / rho))
}

/// Run the pulse on whatever communicator this process has and return the
/// global `[Σ ρV, Σ ρ²V, Σ ρxV]`.
fn run() -> [f64; 3] {
    let mesh = UniformMeshConfig {
        nx: 48,
        ny: 12,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [1.0, 0.25, 1.0 / 48.0],
        y_edges: None,
        z_edges: None,
    };
    let bcs = BoundaryRegistry::default()
        .with_axis(0, Periodic)
        .with_axis(1, Periodic)
        .with_axis(2, SlipWall);
    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh })
        .add_plugins(CfdStatePlugin::new(pulse))
        .add_plugins(IdealGasPlugin)
        .add_plugins(StructuredBoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.4,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();
    for _ in 0..STEPS {
        app.run();
    }

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let comm = app.get_resource_ref::<CommResource>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let mut sums = [0.0; 3];
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let rho = state.u[c].rho;
        assert!(rho.is_finite() && rho > 0.0, "non-physical density");
        let v = mesh.cell_volume(c);
        let x = mesh.cell_centroid(c)[0];
        sums[0] += rho * v;
        sums[1] += rho * rho * v;
        sums[2] += rho * x * v;
    }
    sums.map(|s| comm.all_reduce_sum_f64(s))
}

#[test]
fn periodic_pulse_matches_across_rank_counts() {
    if let Some(out) = std::env::var_os(CHILD_ENV) {
        let sums = run();
        if grass_mpi::world_rank() == 0 {
            let line: Vec<String> = sums.iter().map(|s| format!("{s:.17e}")).collect();
            std::fs::write(out, line.join(" ")).expect("write result");
        }
        // The App (and its communicator) is dropped inside `run`.
        grass_mpi::finalize_mpi();
        return;
    }
    if Command::new("mpirun").arg("--version").output().is_err() {
        eprintln!("SKIP periodic_pulse_matches_across_rank_counts: `mpirun` not on PATH");
        return;
    }
    let exe = std::env::current_exe().expect("locate test binary");
    let dir = std::env::temp_dir().join(format!("efvm_periodic_mpi_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut results = Vec::new();
    for np in RANKS {
        let out = dir.join(format!("np{np}.txt"));
        let status = Command::new("mpirun")
            .args(["--oversubscribe", "-np", &np.to_string()])
            .arg(&exe)
            .args(["--exact", "periodic_pulse_matches_across_rank_counts", "--nocapture"])
            .env(CHILD_ENV, &out)
            .status()
            .expect("spawn mpirun");
        assert!(status.success(), "periodic pulse failed on {np} ranks: {status}");
        let text = std::fs::read_to_string(&out).expect("read result");
        let v: Vec<f64> = text.split_whitespace().map(|s| s.parse().unwrap()).collect();
        results.push((np, v));
    }
    let _ = std::fs::remove_dir_all(&dir);
    let (_, base) = &results[0];
    for (np, v) in &results {
        for k in 0..3 {
            let rel = (v[k] - base[k]).abs() / base[k].abs();
            assert!(
                rel < 1e-12,
                "moment {k} on {np} ranks = {} vs 1 rank = {} (rel {rel:e})",
                v[k],
                base[k]
            );
        }
        eprintln!("np={np}: {:?}", v);
    }
}
