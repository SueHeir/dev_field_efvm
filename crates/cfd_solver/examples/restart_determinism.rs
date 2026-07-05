//! Restart determinism and rank-count invariance checks for the CFD tier.
//!
//! Build + run:
//!   cargo build -p cfd_solver --example restart_determinism --features mpi_backend
//!   mpirun -np 1 target/debug/examples/restart_determinism
//!   mpirun -np 2 target/debug/examples/restart_determinism
//!
//! The restart check writes a rank-local checkpoint after a completed step,
//! rebuilds the app, reads the checkpoint, continues, and requires the final
//! state to be bitwise-identical to an uninterrupted run. The rank-count check
//! advances a fixed smooth Euler problem and compares a coordinate-keyed digest
//! to the single-rank reference, so the same binary gates both np=1 and np=2.

#[cfg(not(feature = "mpi_backend"))]
fn main() {
    eprintln!("restart_determinism requires the `mpi_backend` feature:");
    eprintln!("  cargo build -p cfd_solver --example restart_determinism --features mpi_backend");
    eprintln!("  mpirun -np 1 target/debug/examples/restart_determinism");
    eprintln!("  mpirun -np 2 target/debug/examples/restart_determinism");
}

#[cfg(feature = "mpi_backend")]
fn main() {
    use grass_mpi::{world_rank, world_size};

    let rank = world_rank();
    let size = world_size();

    let restart_ok = rd::restart_checkpoint_is_bitwise();
    let (rank_ok, digest) = rd::fixed_problem_digest_matches_reference();

    if rank == 0 {
        println!("------------------------------------------------");
        println!("CFD restart/determinism validation on {size} rank(s)");
        println!("  restart checkpoint bitwise replay : {}", pass(restart_ok));
        println!(
            "  fixed-problem rank-count digest   : ({:.0}, {:.0})",
            digest.0, digest.1
        );
        println!("  rank-count invariant digest       : {}", pass(rank_ok));
        println!("------------------------------------------------");
    }

    grass_mpi::finalize_mpi();
    if !(restart_ok && rank_ok) {
        std::process::exit(1);
    }
}

#[cfg(feature = "mpi_backend")]
fn pass(ok: bool) -> &'static str {
    if ok {
        "PASS"
    } else {
        "FAIL"
    }
}

#[cfg(feature = "mpi_backend")]
mod rd {
    use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, SlipWall};
    use cfd_eos::{Eos, IdealGas};
    use cfd_solver::{
        CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
        SolverState,
    };
    use cfd_state::{CfdState, ConsVar, PrimVar};
    use field_core::{
        BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, UniformMesh, UniformMeshConfig,
        Vec3,
    };
    use grass_app::prelude::*;
    use grass_mpi::CommResource;
    use std::fs;
    use std::path::PathBuf;

    const STEPS_TOTAL: usize = 10;
    const STEPS_CHECKPOINT: usize = 5;
    const FIXED_DIGEST_REF: (f64, f64) = (21_398_001_286.0, 21_099_935_807.0);

    fn mesh_config() -> UniformMeshConfig {
        UniformMeshConfig {
            nx: 16,
            ny: 4,
            nz: 2,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [1.0, 0.25, 0.125],
            y_edges: None,
            z_edges: None,
        }
    }

    fn boundaries() -> BoundaryRegistry {
        let mut bcs = BoundaryRegistry::default();
        for side in [
            BoundarySide::YLo,
            BoundarySide::YHi,
            BoundarySide::ZLo,
            BoundarySide::ZHi,
        ] {
            bcs.set(side, SlipWall);
        }
        bcs
    }

    fn initial_state(x: Vec3) -> ConsVar {
        let eos = IdealGas::air();
        let r2 = (x[0] - 0.5).powi(2) + 16.0 * (x[1] - 0.125).powi(2);
        let bump = (-80.0 * r2).exp();
        let rho = 1.0 + 0.04 * bump;
        let u = 9.0 + 0.7 * (2.0 * std::f64::consts::PI * x[0]).sin();
        let v = 0.15 * (4.0 * std::f64::consts::PI * x[1]).cos();
        let p = 100_000.0 * (1.0 + 0.03 * bump);
        eos.prim_to_cons(&PrimVar::new(rho, u, v, 0.0, p, p / (rho * 287.058)))
    }

    fn build_app() -> App {
        let mut app = App::new();
        app.add_plugins(FieldDefaultPlugins {
            mesh: mesh_config(),
        })
        .add_plugins(CfdStatePlugin::new(initial_state))
        .add_plugins(IdealGasPlugin)
        .add_plugins(BoundaryPlugin::<UniformMesh>::new(boundaries()))
        .add_plugins(FluxPlugin::<UniformMesh>::rusanov())
        .add_plugins(IntegratorPlugin::euler())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.2,
            fixed_dt: Some(2.0e-5),
            viscous: false,
            muscl: false,
        }));
        app.prepare();
        app
    }

    fn run_steps(app: &mut App, n: usize) {
        for _ in 0..n {
            app.run();
        }
    }

    pub fn restart_checkpoint_is_bitwise() -> bool {
        let mut continuous = build_app();
        run_steps(&mut continuous, STEPS_TOTAL);

        let mut restarted = build_app();
        run_steps(&mut restarted, STEPS_CHECKPOINT);
        let path = checkpoint_path(&restarted);
        write_checkpoint(&restarted, &path).expect("write checkpoint");

        let mut resumed = build_app();
        read_checkpoint(&mut resumed, &path).expect("read checkpoint");
        let _ = fs::remove_file(&path);
        run_steps(&mut resumed, STEPS_TOTAL - STEPS_CHECKPOINT);

        same_local_state(&continuous, &resumed)
    }

    pub fn fixed_problem_digest_matches_reference() -> (bool, (f64, f64)) {
        let mut app = build_app();
        run_steps(&mut app, STEPS_TOTAL);
        let digest = state_digest(&app);
        (digest == FIXED_DIGEST_REF, digest)
    }

    fn checkpoint_path(app: &App) -> PathBuf {
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        std::env::temp_dir().join(format!(
            "test_cfd_restart_{}_rank{}.bin",
            std::process::id(),
            comm.rank()
        ))
    }

    fn write_checkpoint(app: &App, path: &PathBuf) -> std::io::Result<()> {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        let solver = app.get_resource_ref::<SolverState>().unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"CFDRST01");
        bytes.extend_from_slice(&solver.dt.to_le_bytes());
        bytes.extend_from_slice(&solver.time.to_le_bytes());
        bytes.extend_from_slice(&solver.step.to_le_bytes());
        bytes.push(match solver.rk_stage {
            cfd_solver::RkStage::First => 0,
            cfd_solver::RkStage::Second => 1,
            cfd_solver::RkStage::Third => 2,
        });
        write_cons_vec(&mut bytes, &state.u);
        write_cons_vec(&mut bytes, &state.rhs);
        write_cons_vec(&mut bytes, &state.u0);
        fs::write(path, bytes)
    }

    fn read_checkpoint(app: &mut App, path: &PathBuf) -> std::io::Result<()> {
        let bytes = fs::read(path)?;
        let mut cursor = Cursor {
            bytes: &bytes,
            pos: 0,
        };
        assert_eq!(cursor.take(8), b"CFDRST01");

        let solver = SolverState {
            dt: cursor.f64(),
            time: cursor.f64(),
            step: cursor.u64(),
            rk_stage: match cursor.u8() {
                0 => cfd_solver::RkStage::First,
                1 => cfd_solver::RkStage::Second,
                2 => cfd_solver::RkStage::Third,
                other => panic!("bad checkpoint rk stage {other}"),
            },
        };
        app.add_resource(solver);

        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let mut state = reg.get_mut::<CfdState>().unwrap();
        read_cons_vec(&mut cursor, &mut state.u);
        read_cons_vec(&mut cursor, &mut state.rhs);
        read_cons_vec(&mut cursor, &mut state.u0);
        assert_eq!(cursor.pos, bytes.len());
        Ok(())
    }

    fn write_cons_vec(bytes: &mut Vec<u8>, v: &[ConsVar]) {
        bytes.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for c in v {
            for x in [c.rho, c.rho_u, c.rho_v, c.rho_w, c.rho_e] {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }

    fn read_cons_vec(cursor: &mut Cursor<'_>, out: &mut [ConsVar]) {
        let len = cursor.u64() as usize;
        assert_eq!(len, out.len(), "checkpoint vector length changed");
        for c in out {
            *c = ConsVar::new(
                cursor.f64(),
                cursor.f64(),
                cursor.f64(),
                cursor.f64(),
                cursor.f64(),
            );
        }
    }

    struct Cursor<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn take(&mut self, n: usize) -> &'a [u8] {
            let start = self.pos;
            self.pos += n;
            &self.bytes[start..self.pos]
        }
        fn u8(&mut self) -> u8 {
            self.take(1)[0]
        }
        fn u64(&mut self) -> u64 {
            let mut b = [0u8; 8];
            b.copy_from_slice(self.take(8));
            u64::from_le_bytes(b)
        }
        fn f64(&mut self) -> f64 {
            let mut b = [0u8; 8];
            b.copy_from_slice(self.take(8));
            f64::from_le_bytes(b)
        }
    }

    fn same_local_state(a: &App, b: &App) -> bool {
        let mesh_a = a.get_resource_ref::<UniformMesh>().unwrap();
        let mesh_b = b.get_resource_ref::<UniformMesh>().unwrap();
        let reg_a = a.get_resource_ref::<FieldRegistry>().unwrap();
        let reg_b = b.get_resource_ref::<FieldRegistry>().unwrap();
        let state_a = reg_a.get::<CfdState>().unwrap();
        let state_b = reg_b.get::<CfdState>().unwrap();
        let solver_a = a.get_resource_ref::<SolverState>().unwrap();
        let solver_b = b.get_resource_ref::<SolverState>().unwrap();

        if solver_a.step != solver_b.step
            || solver_a.time.to_bits() != solver_b.time.to_bits()
            || solver_a.dt.to_bits() != solver_b.dt.to_bits()
            || solver_a.rk_stage != solver_b.rk_stage
            || mesh_a.n_cells_total() != mesh_b.n_cells_total()
        {
            return false;
        }

        for c in 0..mesh_a.n_cells_total() {
            if mesh_a.is_local_cell(c)
                && (!mesh_b.is_local_cell(c) || !same_cons(state_a.u[c], state_b.u[c]))
            {
                return false;
            }
        }
        true
    }

    fn state_digest(app: &App) -> (f64, f64) {
        let comm = app.get_resource_ref::<CommResource>().unwrap();
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        let cfg = mesh_config();
        let dx = (cfg.bounds_hi[0] - cfg.bounds_lo[0]) / cfg.nx as f64;
        let dy = (cfg.bounds_hi[1] - cfg.bounds_lo[1]) / cfg.ny as f64;
        let dz = (cfg.bounds_hi[2] - cfg.bounds_lo[2]) / cfg.nz as f64;

        let mut lo = 0.0f64;
        let mut hi = 0.0f64;
        for c in 0..mesh.n_cells_total() {
            if !mesh.is_local_cell(c) {
                continue;
            }
            let x = mesh.cell_centroid(c);
            let ix = ((x[0] - cfg.bounds_lo[0]) / dx).floor() as u64;
            let iy = ((x[1] - cfg.bounds_lo[1]) / dy).floor() as u64;
            let iz = ((x[2] - cfg.bounds_lo[2]) / dz).floor() as u64;
            let key = (ix << 32) ^ (iy << 16) ^ iz;
            for (component, bits) in [
                state.u[c].rho.to_bits(),
                state.u[c].rho_u.to_bits(),
                state.u[c].rho_v.to_bits(),
                state.u[c].rho_w.to_bits(),
                state.u[c].rho_e.to_bits(),
            ]
            .into_iter()
            .enumerate()
            {
                let h = splitmix64(key ^ ((component as u64) << 48) ^ bits);
                lo += (h & ((1u64 << 26) - 1)) as f64;
                hi += ((h >> 26) & ((1u64 << 26) - 1)) as f64;
            }
        }
        (comm.all_reduce_sum_f64(lo), comm.all_reduce_sum_f64(hi))
    }

    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }

    fn same_cons(a: ConsVar, b: ConsVar) -> bool {
        a.rho.to_bits() == b.rho.to_bits()
            && a.rho_u.to_bits() == b.rho_u.to_bits()
            && a.rho_v.to_bits() == b.rho_v.to_bits()
            && a.rho_w.to_bits() == b.rho_w.to_bits()
            && a.rho_e.to_bits() == b.rho_e.to_bits()
    }
}
