//! DEMO (not a validation): config-driven Sod shock tube. The grid, solver
//! settings, and run length all come from a TOML file (toy-cfd's `grass_io`
//! config workflow).
//!
//! This demonstrates the declarative input path and writes VTK, but it does not
//! compare against an exact Riemann solution or exit with a PASS/FAIL verdict.
//!
//! `cargo run -p dev_field_efvm --bin config_driven -- examples/config_driven/config.toml`

use cfd_solver::SolverConfig;
use grass_io::{Config, InputPlugin, RunPlugin};
use test_cfd::prelude::*;

fn main() {
    println!("DEMO (not a validation): config-driven Sod shock tube; no PASS/FAIL gate");

    let mut app = App::new();
    // InputPlugin parses argv[1] as a TOML file into the `Config` resource.
    app.add_plugins(InputPlugin);

    // Read the `[grid]` and `[solver]` sections (defaults fill anything omitted).
    let (mesh_cfg, solver_cfg) = {
        let cfg = app.get_resource_ref::<Config>().expect("config");
        (
            cfg.section::<UniformMeshConfig>("grid"),
            cfg.section::<SolverConfig>("solver"),
        )
    };

    // Sod initial state (the IC is code, not config).
    let eos = IdealGas::air();
    let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100_000.0, 348.4));
    let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10_000.0, 278.7));
    let split = 0.5 * (mesh_cfg.bounds_lo[0] + mesh_cfg.bounds_hi[0]);
    let init = move |x: Vec3| if x[0] < split { left } else { right };

    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(
            CfdDefaultPlugins::new(init).config(solver_cfg).boundary(
                BoundaryRegistry::default()
                    .with_axis(1, SlipWall)
                    .with_axis(2, SlipWall),
            ),
        )
        // RunPlugin drives `[run].steps` iterations, then ends the App.
        .add_plugins(RunPlugin)
        .add_plugins(OutputPlugin {
            path: "config_driven.vtk".into(),
        });

    app.start();
    println!("config-driven run complete");
}
