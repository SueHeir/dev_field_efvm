//! DEMO (not a validation): the classic Sod shock tube, run end-to-end on the
//! FIELD substrate and printed as a 1-D density/velocity/pressure profile.
//!
//! This example is intentionally a visual/smoke demo: it writes `sod_shock_tube.vtk`
//! and prints sampled states, but it does not compare against an exact Riemann
//! solution or exit with a PASS/FAIL verdict.
//!
//! `cargo run -p test_cfd --example sod_shock_tube`

use cfd_solver::SolverState;
use cfd_state::CfdState;
use test_cfd::prelude::*;

fn main() {
    println!("DEMO (not a validation): Sod shock tube profile; no PASS/FAIL gate");

    let eos = IdealGas::air();
    // Sod's classic left/right states (quiescent, p & rho jump at x = 0.5).
    let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100_000.0, 348.4));
    let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10_000.0, 278.7));
    let init = move |x: Vec3| if x[0] < 0.5 { left } else { right };

    // A thin 1-D tube: slip walls on the transverse faces, transmissive x-ends.
    let mesh = UniformMeshConfig {
        nx: 100,
        ny: 1,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [1.0, 0.01, 0.01],
        y_edges: None,
        z_edges: None,
    };
    let bcs = BoundaryRegistry::default()
        .with_axis(1, SlipWall)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh }).add_plugins(
        CfdDefaultPlugins::new(init)
            .hllc()
            .rk3()
            .boundary(bcs)
            .config(SolverConfig {
                cfl: 0.4,
                fixed_dt: None,
                viscous: false,
                muscl: false,
            }),
    );
    app.prepare();

    let n_steps = 60usize;
    for _ in 0..(3 * n_steps) {
        app.run(); // one substep per iteration; 3 substeps = one RK3 step
    }

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let ss = app.get_resource_ref::<SolverState>().unwrap();

    println!(
        "Sod shock tube — {} RK3 steps, t = {:.5e} s",
        ss.step, ss.time
    );
    println!("   x      rho       u        p");
    let [ni, _, _] = mesh.dims();
    for i in (0..ni).step_by(4) {
        let c = mesh.idx(i, 0, 0);
        let p = eos.cons_to_prim(&state.u[c]);
        let x = mesh.cell_centroid(c)[0];
        println!("{:5.3}  {:7.4}  {:8.2}  {:9.0}", x, p.rho, p.u, p.p);
    }

    match write_vtk("sod_shock_tube.vtk", &*mesh, &state, &eos) {
        Ok(()) => println!("\nwrote sod_shock_tube.vtk"),
        Err(e) => eprintln!("\nfailed to write VTK: {e}"),
    }
}
