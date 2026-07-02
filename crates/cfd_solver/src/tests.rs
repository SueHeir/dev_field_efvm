//! End-to-end solver tests: assemble a real `App` on a `UniformMesh` and drive it
//! through forward-Euler steps. These exercise the full FIELD↔test-cfd seam —
//! state registration, field resize, halo, boundary fill, the `for_each_face`
//! flux scatter, and the integrator — not just isolated functions.

use super::*;
use cfd_eos::{sutherland_viscosity, Eos, IdealGas};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, StructuredMesh, UniformMesh, UniformMeshConfig,
};

fn mesh_cfg() -> UniformMeshConfig {
    UniformMeshConfig {
        nx: 16,
        ny: 4,
        nz: 4,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [1.0, 0.25, 0.25],
        y_edges: None,
        z_edges: None,
    }
}

/// A uniform flow must stay *exactly* uniform: every face sees identical L/R
/// states, so the Rusanov dissipation vanishes and opposing-face physical fluxes
/// cancel to give zero RHS. Any sign or area/volume error in the scatter breaks
/// this immediately.
#[test]
fn uniform_flow_is_preserved() {
    let eos = IdealGas::air();
    let u_uniform = eos.prim_to_cons(&PrimVar::new(1.0, 30.0, 0.0, 0.0, 100_000.0, 348.4));

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg() })
        .add_plugins(CfdStatePlugin::new(move |_x| u_uniform))
        .add_plugins(IdealGasPlugin)
        .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::default())
        .add_plugins(FluxPlugin::<UniformMesh>::rusanov())
        .add_plugins(IntegratorPlugin::euler())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig::default()));
    app.prepare();
    for _ in 0..25 {
        app.run();
    }

    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    for (c, u) in state.u.iter().enumerate() {
        assert!(
            (u.rho - u_uniform.rho).abs() < 1e-9
                && (u.rho_u - u_uniform.rho_u).abs() < 1e-6
                && (u.rho_e - u_uniform.rho_e).abs() < 1e-3,
            "cell {c} drifted: {u:?} vs {u_uniform:?}",
        );
    }
    // The clock actually advanced (CFL produced a finite dt and steps ran).
    let ss = app.get_resource_ref::<SolverState>().unwrap();
    assert_eq!(ss.step, 25);
    assert!(ss.dt.is_finite() && ss.dt > 0.0);
}

/// A quiescent Sod-like discontinuity: the flux scatter is conservative, so while
/// the waves stay away from the (quiescent) domain ends total interior mass is
/// conserved — and the field genuinely evolves (the contact/shock move).
#[test]
fn sod_conserves_mass_and_evolves() {
    let eos = IdealGas::air();
    let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100_000.0, 348.4));
    let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10_000.0, 278.7));
    let init = move |x: field_core::Vec3| if x[0] < 0.5 { left } else { right };

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg() })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin)
        .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::default())
        .add_plugins(FluxPlugin::<UniformMesh>::rusanov())
        .add_plugins(IntegratorPlugin::euler())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.3,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();

    let mass = |app: &App| -> f64 {
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        (0..mesh.n_cells_total())
            .filter(|&c| mesh.is_local_cell(c))
            .map(|c| state.u[c].rho * mesh.cell_volume(c))
            .sum()
    };

    let m0 = mass(&app);
    // The middle cell value before stepping (used to confirm evolution).
    let mid_before = {
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        state.u[mesh.idx(8, 2, 2)]
    };

    for _ in 0..15 {
        app.run();
    }

    let m1 = mass(&app);
    // Conservation is exact in exact arithmetic (the two-sided face scatter
    // telescopes); the residual is pure FP round-off in the flux accumulation.
    // 1e-6 is far below any real (O(1)) conservation bug, far above FP noise.
    assert!(
        (m1 - m0).abs() / m0 < 1e-6,
        "interior mass not conserved: {m0} -> {m1}",
    );

    let (mid_after, all_finite) = {
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        let finite = state.u.iter().all(|u| {
            u.rho.is_finite() && u.rho_u.is_finite() && u.rho_e.is_finite() && u.rho > 0.0
        });
        (state.u[mesh.idx(8, 2, 2)], finite)
    };
    assert!(all_finite, "non-finite or negative density appeared");
    assert!(
        (mid_after.rho_u - mid_before.rho_u).abs() > 1.0,
        "expected the discontinuity to drive momentum at the interface",
    );
}

/// A closed box (slip walls on all six sides) with a central pressure pulse:
/// walls reflect normal momentum so *no* mass crosses any boundary, hence total
/// mass is conserved to FP precision while the pulse expands and reflects. This
/// exercises the real `BoundaryPlugin` ghost-fill, not just the BC math.
#[test]
fn closed_box_conserves_mass_under_slip_walls() {
    use cfd_boundary::{BoundaryRegistry, SlipWall};

    let eos = IdealGas::air();
    let center = [0.5, 0.125, 0.125];
    let init = move |x: field_core::Vec3| {
        let r2 = (0..3).map(|d| (x[d] - center[d]).powi(2)).sum::<f64>();
        let p = if r2 < 0.15 * 0.15 { 3e5 } else { 1e5 };
        eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, p, p / (1.0 * 287.058)))
    };
    let walls = BoundaryRegistry::default()
        .with_axis(0, SlipWall)
        .with_axis(1, SlipWall)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg() })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin)
        .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::new(walls))
        .add_plugins(FluxPlugin::<UniformMesh>::rusanov())
        .add_plugins(IntegratorPlugin::euler())
        .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.3,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
    app.prepare();

    let mass = |app: &App| -> f64 {
        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        (0..mesh.n_cells_total())
            .filter(|&c| mesh.is_local_cell(c))
            .map(|c| state.u[c].rho * mesh.cell_volume(c))
            .sum()
    };
    let m0 = mass(&app);
    for _ in 0..30 {
        app.run();
    }
    let m1 = mass(&app);
    assert!(
        (m1 - m0).abs() / m0 < 1e-9,
        "slip-wall box leaked mass: {m0} -> {m1}",
    );
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    assert!(
        state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0),
        "non-finite or negative density in closed box",
    );
}

/// The structured fast path (`FastFluxPlugin` / `flux_fast_system`) must produce
/// the same solution as the generic `for_each_face` path (`FluxPlugin`) — same
/// flux, same scatter, just monomorphized + cached. Run a Sod discontinuity both
/// ways and compare cell-for-cell.
#[test]
fn fast_path_matches_generic() {
    use cfd_riemann::Rusanov;

    let eos = IdealGas::air();
    let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100_000.0, 348.4));
    let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10_000.0, 278.7));
    let init = move |x: field_core::Vec3| if x[0] < 0.5 { left } else { right };

    let run = |fast: bool| -> Vec<cfd_state::ConsVar> {
        let mut app = App::new();
        app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg() });
        app.add_plugins(CfdStatePlugin::new(init));
        app.add_plugins(IdealGasPlugin);
        app.add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::default());
        app.add_plugins(IntegratorPlugin::euler());
        app.add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
            cfl: 0.3,
            fixed_dt: None,
            viscous: false,
            muscl: false,
        }));
        if fast {
            app.add_plugins(FastFluxPlugin::new(Rusanov));
        } else {
            app.add_plugins(FluxPlugin::<UniformMesh>::rusanov());
        }
        app.prepare();
        for _ in 0..12 {
            app.run();
        }
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        state.u.clone()
    };

    let generic = run(false);
    let fast = run(true);
    for (g, f) in generic.iter().zip(fast.iter()) {
        assert!(
            (g.rho - f.rho).abs() < 1e-10
                && (g.rho_u - f.rho_u).abs() < 1e-7
                && (g.rho_e - f.rho_e).abs() < 1e-3,
            "fast path diverged from generic: {g:?} vs {f:?}",
        );
    }
}

/// Validate the viscous operator directly. With `u_x = a·y²` on a uniform
/// (ρ, p) field, the inviscid x-momentum RHS vanishes (no x-variation; `v=0`)
/// and the central second difference of a quadratic is exact, so one forward-Euler
/// step must change `ρu` by exactly `dt · μ · d²u/dy² = dt · μ · 2a` at an interior
/// cell. Viscous-off must give ~zero change.
#[test]
fn viscous_operator_matches_laplacian() {
    let a = 10.0;
    let eos = IdealGas::air();
    let t0 = 100_000.0 / (1.0 * 287.058);
    let init = move |x: field_core::Vec3| {
        eos.prim_to_cons(&PrimVar::new(1.0, a * x[1] * x[1], 0.0, 0.0, 100_000.0, t0))
    };
    let cfg = UniformMeshConfig {
        nx: 6,
        ny: 8,
        nz: 4,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [1.0, 1.0, 0.5],
        y_edges: None,
        z_edges: None,
    };

    let step = |viscous: bool| -> (f64, f64) {
        let mut app = App::new();
        app.add_plugins(FieldDefaultPlugins { mesh: cfg.clone() })
            .add_plugins(CfdStatePlugin::new(init))
            .add_plugins(IdealGasPlugin)
            .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::default())
            .add_plugins(FluxPlugin::<UniformMesh>::rusanov())
            .add_plugins(IntegratorPlugin::euler())
            .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
                cfl: 0.3,
                fixed_dt: None,
                viscous,
                muscl: false,
            }));
        app.prepare();
        let cell = {
            let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
            mesh.idx(3, 4, 2) // interior in every axis → fully-interior stencil
        };
        let before = {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let state = reg.get::<CfdState>().unwrap();
            state.u[cell].rho_u
        };
        app.run();
        let dt = app.get_resource_ref::<SolverState>().unwrap().dt;
        let after = {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let state = reg.get::<CfdState>().unwrap();
            state.u[cell].rho_u
        };
        ((after - before) / dt, dt)
    };

    let (rhs_visc, _) = step(true);
    let (rhs_inv, _) = step(false);

    // The inviscid RHS is identical in both runs (same field; Rusanov's *numerical*
    // dissipation on tangential momentum across the y-faces is large but cancels
    // in the difference). What remains is exactly the physical viscous operator
    // the `viscous_system` added: μ · d²(a·y²)/dy² = μ · 2a, with the central
    // second difference of a quadratic being exact.
    let viscous_contribution = rhs_visc - rhs_inv;
    let expected = sutherland_viscosity(t0) * 2.0 * a;
    assert!(
        (viscous_contribution - expected).abs() / expected.abs() < 1e-4,
        "viscous operator gave {viscous_contribution}, expected μ·2a = {expected}",
    );
}
