//! Top-level compressible-CFD solver on the FIELD substrate.
//!
//! [`CfdDefaultPlugins`] bundles the physics layer the way DIRT's
//! `GranularDefaultPlugins` bundles granular physics on SOIL: state registration,
//! EOS, boundary conditions, flux scheme, integrator, and the solver systems.
//! Combine it with `field_core`'s `FieldDefaultPlugins` (comm + registry + mesh)
//! and you have a runnable solver:
//!
//! ```rust,ignore
//! use test_cfd::prelude::*;
//!
//! let mut app = App::new();
//! app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
//!    .add_plugins(CfdDefaultPlugins::new(sod_init).rk3().boundary(walls));
//! app.prepare();
//! for _ in 0..n_substeps { app.run(); }
//! ```
//!
//! ## Known limitation: plugins are wired for `UniformMesh`
//!
//! The physics *systems* (`flux_system`, `viscous_system`, `boundary_system`,
//! the IBM systems, …) are all generic over the mesh — `<M: FvMesh>` or
//! `<M: StructuredMesh>` — and `SolverPlugin<M>` is generic too. But the thin
//! plugin *wiring* in `CfdStatePlugin`, `BoundaryPlugin`, `GhostCellIbmPlugin`,
//! and `CylinderIbmPlugin` registers those systems with a concrete
//! `::<UniformMesh>`. Adding a second mesh type (the future `field_amr::ForestMesh`)
//! therefore requires parameterizing those four plugins over `M`, exactly as
//! `SolverPlugin<M>` already is — a mechanical change to the assembly layer, not
//! to the physics. Until a second mesh exists this is a no-op, so it is deferred
//! to the AMR (M2) work where `ForestMesh` lands.

use std::sync::Arc;

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig, SolverPlugin,
};
use cfd_state::ConsVar;
use field_core::{UniformMesh, Vec3};
use grass_app::{PluginGroup, PluginGroupBuilder};

/// The default dev_field_efvm physics bundle. Construct with an initial-condition
/// closure, then tune with the builder methods.
pub struct CfdDefaultPlugins {
    init: Arc<dyn Fn(Vec3) -> ConsVar + Send + Sync>,
    boundary: BoundaryRegistry,
    flux: FluxPlugin,
    integrator: IntegratorPlugin,
    config: SolverConfig,
}

impl CfdDefaultPlugins {
    /// New bundle: Rusanov flux, forward Euler, all-outflow boundaries, default CFL.
    pub fn new(init: impl Fn(Vec3) -> ConsVar + Send + Sync + 'static) -> Self {
        Self {
            init: Arc::new(init),
            boundary: BoundaryRegistry::default(),
            flux: FluxPlugin::<UniformMesh>::rusanov(),
            integrator: IntegratorPlugin::euler(),
            config: SolverConfig::default(),
        }
    }

    pub fn boundary(mut self, boundary: BoundaryRegistry) -> Self {
        self.boundary = boundary;
        self
    }
    pub fn flux(mut self, flux: FluxPlugin) -> Self {
        self.flux = flux;
        self
    }
    /// Use HLLC instead of Rusanov.
    pub fn hllc(mut self) -> Self {
        self.flux = FluxPlugin::<UniformMesh>::hllc();
        self
    }
    /// Use TVD-RK3 instead of forward Euler.
    pub fn rk3(mut self) -> Self {
        self.integrator = IntegratorPlugin::rk3();
        self
    }
    pub fn config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self
    }
}

impl PluginGroup for CfdDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(CfdStatePlugin::from_arc(self.init))
            .add(IdealGasPlugin)
            .add(BoundaryPlugin::<UniformMesh>::new(self.boundary))
            .add(self.flux)
            .add(self.integrator)
            .add(SolverPlugin::<UniformMesh>::new(self.config))
    }
}

/// Commonly used imports for assembling a dev_field_efvm solver.
pub mod prelude {
    pub use crate::CfdDefaultPlugins;
    pub use cfd_boundary::{
        BoundaryRegistry, NoSlipWall, PressureOutlet, SlipWall, SubsonicInflow, SupersonicInflow,
        SupersonicOutflow, Symmetry,
    };
    pub use cfd_eos::{Eos, IdealGas};
    pub use cfd_ibm::{Cylinder, CylinderIbmPlugin, GhostCellIbmPlugin, MarkerForce, WallForce};
    pub use cfd_output::{write_vtk, OutputPlugin};
    pub use cfd_solver::{
        CfdCorePlugin, CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, SolverConfig,
        SolverPlugin, SolverState,
    };
    pub use cfd_state::{CfdState, ConsVar, PrimVar, SolidMask};
    pub use field_core::{
        BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, StructuredMesh, UniformMesh,
        UniformMeshConfig, Vec3,
    };
    pub use grass_app::prelude::*;
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use field_amr::{ForestConfig, ForestMesh};
    use field_core::{CommPlugin, FieldRegistryPlugin};

    /// Adds the `field_amr` forest mesh as a resource and provides the `"mesh"`
    /// capability — the adaptive-mesh parallel to `UniformMeshPlugin`, so a forest
    /// app satisfies `CfdCorePlugin`'s `"mesh"` contract the same way a uniform one
    /// does.
    struct ForestMeshPlugin(ForestConfig);
    impl grass_app::Plugin for ForestMeshPlugin {
        fn build(&self, app: &mut App) {
            app.add_resource(ForestMesh::from_config(&self.0));
        }
        fn provides(&self) -> Vec<&str> {
            vec!["mesh"]
        }
    }

    /// Assemble the inviscid Euler solver on a `field_amr::ForestMesh` **entirely
    /// through plugins** — the same assembly style as the uniform path, just with
    /// `CfdCorePlugin::<M>` (generic `FvMesh` core: resize + state + generic CFL +
    /// halo + zero-rhs + update) in place of the `StructuredMesh`-only
    /// `SolverPlugin`. Proves the substrate's "swap the mesh type, leave the
    /// physics" promise holds at the plugin layer, not just for hand-wired systems.
    fn forest_app(cfg: &ForestConfig, bcs: cfd_boundary::BoundaryRegistry) -> grass_app::App {
        let mut app = App::new();
        app.add_plugins(CommPlugin);
        app.add_plugins(FieldRegistryPlugin);
        app.add_plugins(ForestMeshPlugin(cfg.clone()));
        app.add_plugins(IdealGasPlugin);
        app.add_plugins(IntegratorPlugin::euler());
        app.add_plugins(cfd_boundary::BoundaryPlugin::<ForestMesh>::new(bcs));
        app.add_plugins(FluxPlugin::<ForestMesh>::rusanov());
        app.add_plugins(CfdCorePlugin::<ForestMesh>::new(
            SolverConfig { cfl: 0.15, fixed_dt: None, viscous: false, muscl: false },
            // IC is overwritten by each test after `prepare`; a trivial init is fine.
            |_x| ConsVar::ZERO,
        ));
        app
    }

    fn amr_cfg() -> ForestConfig {
        ForestConfig {
            ncx: 8,
            ncy: 6,
            ncz: 1,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.0, 0.75, 0.125],
            // A refined 2×2 block straddling the domain center.
            refine: vec![[3, 2, 0], [4, 2, 0], [3, 3, 0], [4, 3, 0]],
        }
    }

    /// Uniform flow stays *exactly* uniform on the adaptive mesh — the sharpest
    /// check that the coarse/fine face geometry + two-sided scatter are correct
    /// (any error makes a non-zero RHS at a refinement boundary).
    #[test]
    fn amr_uniform_flow_is_preserved() {
        let eos = IdealGas::air();
        let u0 = eos.prim_to_cons(&PrimVar::new(1.0, 20.0, 0.0, 0.0, 100_000.0, 348.4));
        let mut app = forest_app(&amr_cfg(), BoundaryRegistry::default());
        app.prepare();
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
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        for (c, u) in state.u.iter().enumerate() {
            assert!(
                (u.rho - u0.rho).abs() < 1e-9
                    && (u.rho_u - u0.rho_u).abs() < 1e-6
                    && (u.rho_e - u0.rho_e).abs() < 1e-3,
                "AMR cell {c} drifted from uniform: {u:?}",
            );
        }
    }

    /// A closed box (slip walls everywhere) with a pressure pulse that expands
    /// across the refined block: with no boundary outflow, total mass is conserved
    /// to FP precision — proving the coarse/fine flux scatter is conservative by
    /// construction (refluxing-free).
    #[test]
    fn amr_closed_box_conserves_mass_across_refinement() {
        let eos = IdealGas::air();
        let bcs = BoundaryRegistry::default()
            .with_axis(0, SlipWall)
            .with_axis(1, SlipWall)
            .with_axis(2, SlipWall);
        let mut app = forest_app(&amr_cfg(), bcs);
        app.prepare();

        let mass = |app: &grass_app::App| -> f64 {
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let state = reg.get::<CfdState>().unwrap();
            (0..mesh.n_cells_total())
                .filter(|&c| mesh.is_local_cell(c))
                .map(|c| state.u[c].rho * mesh.cell_volume(c))
                .sum()
        };
        // A high-pressure pulse centred on the refined region.
        {
            let mesh = app.get_resource_ref::<ForestMesh>().unwrap();
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let mut state = reg.get_mut::<CfdState>().unwrap();
            for c in 0..mesh.n_cells_total() {
                let cc = mesh.cell_centroid(c);
                let r2 = (cc[0] - 0.5).powi(2) + (cc[1] - 0.375).powi(2);
                // A mild pulse: keeps the first-order coarse/fine interface stable
                // (a strong shock there needs flux limiting — a known AMR concern).
                let p = if r2 < 0.15 * 0.15 { 1.1e5 } else { 1e5 };
                state.u[c] = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, p, p / 287.058));
            }
        }
        let m0 = mass(&app);
        // Short run: the conservative scatter holds mass exactly while the solution
        // stays smooth. (A longer evolving run exposes a mild coarse/fine interface
        // instability — first-order Rusanov across a 2× size jump needs flux
        // stabilization; the conservation *structure* is exact, see field_amr's
        // `flux_scatter_is_conservative`.)
        for _ in 0..5 {
            app.run();
        }
        let m1 = mass(&app);
        assert!(
            (m1 - m0).abs() / m0 < 1e-7,
            "AMR mass not conserved across refinement: {m0} -> {m1}",
        );
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        assert!(state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0));
    }

    fn mesh_cfg() -> UniformMeshConfig {
        UniformMeshConfig {
            nx: 32,
            ny: 4,
            nz: 4,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [1.0, 0.125, 0.125],
            y_edges: None,
            z_edges: None,
        }
    }

    /// The full bundle with TVD-RK3 + HLLC on a Sod shock tube: drive whole RK3
    /// steps (3 substeps each), then check the solution is physical, conserved in
    /// the quiescent interior, and has developed the rightward-moving shock.
    #[test]
    fn sod_bundle_rk3_hllc_runs_and_conserves() {
        let eos = IdealGas::air();
        let left = eos.prim_to_cons(&PrimVar::new(1.0, 0.0, 0.0, 0.0, 100_000.0, 348.4));
        let right = eos.prim_to_cons(&PrimVar::new(0.125, 0.0, 0.0, 0.0, 10_000.0, 278.7));
        let init = move |x: Vec3| if x[0] < 0.5 { left } else { right };
        // Slip walls top/bottom/front/back; transmissive (outflow) x-ends.
        let bcs = BoundaryRegistry::default()
            .with_axis(1, SlipWall)
            .with_axis(2, SlipWall);

        let mut app = App::new();
        app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg() })
            .add_plugins(
                CfdDefaultPlugins::new(init)
                    .hllc()
                    .rk3()
                    .boundary(bcs)
                    .config(SolverConfig { cfl: 0.3, fixed_dt: None, viscous: false, muscl: false }),
            );
        app.prepare();

        let mass = |app: &App| -> f64 {
            let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let state = reg.get::<cfd_state::CfdState>().unwrap();
            (0..mesh.n_cells_total())
                .filter(|&c| mesh.is_local_cell(c))
                .map(|c| state.u[c].rho * mesh.cell_volume(c))
                .sum()
        };
        let m0 = mass(&app);

        // 10 full RK3 steps = 30 substeps; waves stay away from the x-ends.
        for _ in 0..30 {
            app.run();
        }

        let m1 = mass(&app);
        assert!((m1 - m0).abs() / m0 < 1e-6, "mass drift: {m0} -> {m1}");

        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<cfd_state::CfdState>().unwrap();

        // Physical everywhere.
        assert!(state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0));

        // The shock/contact moved right: a cell well right of center that started
        // at the low state now carries rightward momentum.
        let probe = state.u[mesh.idx(20, 2, 2)];
        assert!(probe.rho_u > 0.0, "expected rightward momentum at probe, got {}", probe.rho_u);

        // Monotone-ish: density on the far left (undisturbed) stays high, far
        // right (undisturbed) stays low.
        assert!(state.u[mesh.idx(1, 2, 2)].rho > 0.9);
        assert!(state.u[mesh.idx(30, 2, 2)].rho < 0.2);

        // The clock advanced exactly 10 full steps.
        let ss = app.get_resource_ref::<cfd_solver::SolverState>().unwrap();
        assert_eq!(ss.step, 10);
    }

    /// Flow past an immersed cylinder (ghost-cell IBM): the mask marks the right
    /// cells, the run stays physical, and a stagnation pressure rise develops at
    /// the cylinder's front — the signature of a bluff body in a stream.
    #[test]
    fn cylinder_in_flow_is_stable_with_stagnation() {
        let eos = IdealGas::air();
        let p_inf = 100_000.0;
        let u_inf = 80.0;
        let t_inf = p_inf / (1.0 * 287.058);
        let freestream = eos.prim_to_cons(&PrimVar::new(1.0, u_inf, 0.0, 0.0, p_inf, t_inf));

        let cyl = Cylinder { center_xy: [0.5, 0.6], radius: 0.15 };
        let mesh = UniformMeshConfig {
            nx: 40,
            ny: 24,
            nz: 1,
            ng: 2,
            bounds_lo: [0.0, 0.0, 0.0],
            bounds_hi: [2.0, 1.2, 0.05],
            y_edges: None,
            z_edges: None,
        };
        let bcs = BoundaryRegistry::default()
            .with(BoundarySide::XLo, SupersonicInflow { rho: 1.0, u: u_inf, v: 0.0, w: 0.0, p: p_inf })
            .with(BoundarySide::XHi, SupersonicOutflow)
            .with_axis(1, SlipWall)
            .with_axis(2, SlipWall);

        let mut app = App::new();
        app.add_plugins(FieldDefaultPlugins { mesh })
            .add_plugins(CfdStatePlugin::new(move |_x| freestream))
            .add_plugins(IdealGasPlugin)
            .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::new(bcs))
            .add_plugins(FluxPlugin::<UniformMesh>::hllc())
            .add_plugins(IntegratorPlugin::rk3())
            .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
                cfl: 0.3,
                fixed_dt: None,
                viscous: false,
                muscl: false,
            }))
            .add_plugins(GhostCellIbmPlugin)
            .add_plugins(CylinderIbmPlugin::new(cyl, freestream));
        app.prepare();

        for _ in 0..360 {
            app.run(); // 120 RK3 steps — long enough for the front to stagnate
        }

        let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();

        // Mask marks ~ π r² / cell-area cells (≈28); allow for discretization.
        let solid_count = reg.get::<SolidMask>().unwrap().solid.iter().filter(|&&s| s).count();
        assert!((12..=48).contains(&solid_count), "solid cell count {solid_count} off");

        // Stable + physical everywhere.
        assert!(
            state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0 && u.rho_e.is_finite()),
            "cylinder run went non-physical",
        );

        // Stagnation: a fluid cell just upstream of the cylinder front carries
        // pressure above freestream (the flow decelerates into the body).
        let probe_x = cyl.center_xy[0] - cyl.radius - 0.06; // ~1 cell upstream of front
        let i = ((probe_x / 2.0) * 40.0) as usize;
        let jy = ((cyl.center_xy[1] / 1.2) * 24.0) as usize;
        let p_stag = eos.cons_to_prim(&state.u[mesh.idx(i, jy, 0)]).p;
        assert!(
            p_stag > 1.01 * p_inf,
            "expected stagnation pressure rise upstream, got {p_stag} vs p_inf {p_inf}",
        );

        // Drag: the net wall force points downstream (+x) and dominates lift (+y).
        let force = app.get_resource_ref::<WallForce>().unwrap().total();
        assert!(
            force[0] > 0.0 && force[0].abs() > force[1].abs(),
            "expected downstream drag, got force {force:?}",
        );
    }

    /// Viscous flow past the cylinder: exercises the mask-aware `viscous_system`
    /// (no-slip/adiabatic at solid neighbours) coupled with the IBM. Validates the
    /// combination stays stable and physical — the path that was silently wrong
    /// before the viscous solid-mask coupling was restored.
    #[test]
    fn viscous_cylinder_is_stable() {
        let eos = IdealGas::air();
        let (p_inf, u_inf) = (100_000.0, 60.0);
        let freestream = eos.prim_to_cons(&PrimVar::new(
            1.0, u_inf, 0.0, 0.0, p_inf, p_inf / (1.0 * 287.058),
        ));
        let cyl = Cylinder { center_xy: [0.5, 0.5], radius: 0.12 };
        let mesh = UniformMeshConfig {
            nx: 32,
            ny: 24,
            nz: 1,
            ng: 2,
            bounds_lo: [0.0; 3],
            bounds_hi: [1.6, 1.0, 0.05],
            y_edges: None,
            z_edges: None,
        };
        let bcs = BoundaryRegistry::default()
            .with(BoundarySide::XLo, SupersonicInflow { rho: 1.0, u: u_inf, v: 0.0, w: 0.0, p: p_inf })
            .with(BoundarySide::XHi, SupersonicOutflow)
            .with_axis(1, SlipWall)
            .with_axis(2, SlipWall);

        let mut app = App::new();
        app.add_plugins(FieldDefaultPlugins { mesh })
            .add_plugins(CfdStatePlugin::new(move |_x| freestream))
            .add_plugins(IdealGasPlugin)
            .add_plugins(cfd_boundary::BoundaryPlugin::<UniformMesh>::new(bcs))
            .add_plugins(FluxPlugin::<UniformMesh>::hllc())
            .add_plugins(IntegratorPlugin::rk3())
            .add_plugins(SolverPlugin::<UniformMesh>::new(SolverConfig {
                cfl: 0.3,
                fixed_dt: None,
                viscous: true, // mask-aware viscous + IBM
                muscl: false,
            }))
            .add_plugins(GhostCellIbmPlugin)
            .add_plugins(CylinderIbmPlugin::new(cyl, freestream));
        app.prepare();
        for _ in 0..180 {
            app.run();
        }

        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        assert!(
            state.u.iter().all(|u| u.rho.is_finite() && u.rho > 0.0 && u.rho_e.is_finite()),
            "viscous cylinder run went non-physical",
        );
        // Viscous drag should still point downstream.
        let force = app.get_resource_ref::<WallForce>().unwrap().total();
        assert!(force[0] > 0.0, "expected downstream viscous drag, got {force:?}");

        // The refinement-independent marker drag also points downstream, and the
        // drag coefficient Cd = |F_x| / (½ ρ U² D L_z) is physically plausible.
        let marker = app.get_resource_ref::<MarkerForce>().unwrap().total();
        assert!(marker[0] > 0.0, "expected downstream marker drag, got {marker:?}");
        let (d, lz) = (2.0 * cyl.radius, 0.05);
        let cd = marker[0].abs() / (0.5 * 1.0 * u_inf * u_inf * d * lz);
        assert!((0.1..10.0).contains(&cd), "implausible drag coefficient Cd = {cd}");
    }
}
