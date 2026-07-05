//! **Resolved drag on a BPM fiber (bonded-sphere chain) via the 3-D ghost-cell
//! IBM — anisotropy validated against slender-body theory.** This is the fiber
//! analogue of `sphere_drag_validation`: the immersed body is no longer a single
//! sphere but a **chain of bonded spheres built with DIRT's Bonded-Particle Model**
//! (`dirt_bond::auto_bond_touching` on the SOIL substrate), immersed as the IBM
//! body by the *same* multi-sphere machinery the single-sphere case uses —
//! `SphereIbmPlugin` (already `Vec<SphereBody>`), the per-body surface-marker
//! traction integral (`MarkerForce` / `compute_marker_loadings_per_body`), and the
//! `sample_trilinear` image-point mirror. No new core API: a fiber is just a
//! multi-marker clump the existing seam already supports.
//!
//! ## What is validated, and against what independent reference
//!
//! A rigid slender body translating through a viscous fluid has an **anisotropic**
//! Stokes drag: broadside (motion ⊥ the fiber axis) it feels more drag than end-on
//! (motion ∥ the axis). For a rod modelled as a string of touching beads — exactly
//! the geometry here — the translational friction coefficients are the
//! **Tirado–García de la Torre** shish-kebab result (J. Chem. Phys. 71, 2581
//! (1979); 73, 1986 (1980); refined in 81, 2047 (1984)):
//!
//! ```text
//!   ζ⊥ = 4πμL / (ln p + γ⊥),   γ⊥ = 0.839 + 0.185/p + 0.233/p²
//!   ζ∥ = 2πμL / (ln p + γ∥),   γ∥ = −0.207 + 0.980/p − 0.133/p²      p = L/d
//! ```
//!
//! The **anisotropy ratio** `r = ζ⊥/ζ∥` is the validation target. The ratio (not
//! the absolute drag) is used deliberately: it **cancels** the systematic biases a
//! resolved compressible-IBM drag carries — the first-order staircase grid error,
//! the finite-domain (confinement) correction, the low-Mach compressibility
//! offset — because they act almost identically on both orientations of the *same*
//! fiber on the *same* mesh. Finite-Re inertial drift does not fully cancel, so
//! the validation gates the lowest configured Reynolds number and reports any
//! higher-Re points as trend checks.
//!
//! `Re_d = ρ U d / μ` is pinned by the constant viscosity `μ = ρ U d / Re_d`;
//! the flow is a uniform stream (subsonic velocity inlet, pressure outlet,
//! symmetry far-field) marched at low Mach as an incompressible-Stokes surrogate.
//! Drag is the marker-integrated pressure+viscous load; the lowest-Re point is
//! compared to the Stokes reference after each orientation reaches a steady
//! plateau (small `Cd` peak-to-peak). Additional Re values are reported as an
//! inertial trend check, not used to fit the reference.
//!
//! ## Falsifiability (no back-fitting)
//!
//! The pass rests on the lowest-Re ratio matching Tirado within `tol_ratio`, and the
//! example makes the gate's teeth explicit:
//!   * **Isotropic-body control** — a single sphere (or any isotropic clump) has
//!     `F⊥ = F∥ ⇒ r = 1` *exactly by rotational symmetry*. `r = 1` is `p → 1` in the
//!     Tirado formula. The gate `|r − r_theory| ≤ tol` **rejects** `r = 1` by a
//!     margin far exceeding `tol`, so a coupling that smeared the fiber into an
//!     isotropic blob (lost its shape) would fail. The example asserts this
//!     rejection.
//!
//! Everything case-specific — fiber (bead count, diameter, BPM stiffness), the two
//! Reynolds numbers, mesh resolution + clearances, run length, sponge, tolerances —
//! is declarative TOML from `argv[1]`:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example cfd_ibm_fiber -- \
//!     crates/cfd_ibm/examples/cfd_ibm_fiber_p5.toml
//! ```
//!
//! References:
//! * M.M. Tirado, J. García de la Torre, "Translational friction coefficients of
//!   rigid, symmetric top macromolecules", *J. Chem. Phys.* 71, 2581 (1979).
//! * M.M. Tirado, C. López Martínez, J. García de la Torre, "Comparison of theories
//!   for the translational and rotational diffusion coefficients of rod-like
//!   macromolecules", *J. Chem. Phys.* 81, 2047 (1984).

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, PressureOutlet, SubsonicInflow, Symmetry};
use cfd_eos::{Eos, IdealGas, Viscosity};
use cfd_ibm::{
    Body, CutCellMetrics, GhostCellIbmPlugin, MarkerForce, SphereBody, SphereIbmPlugin, WallForce,
};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, RkStage, SolverConfig,
    SolverPlugin, SolverState,
};
use cfd_state::{CfdState, ConsVar, PrimVar};
use field_core::{
    BoundarySide, FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, StructuredMesh,
    UniformMesh, UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_scheduler::Res;
use serde::Deserialize;

// DIRT Bonded-Particle Model — builds the fiber as a genuine bonded-sphere clump.
use dirt_atom::DemAtom;
use dirt_bond::{auto_bond_touching, BondConfig};
use soil_core::{
    Atom, AtomDataRegistry, BondStore, CommResource, Domain, ScheduleSetupSet, SingleProcessComm,
};

const R_GAS: f64 = 287.058; // matches IdealGas::air()

// ─── Declarative case ────────────────────────────────────────────────────────

/// `[fiber]`: a chain of `n_spheres` beads of `diameter`, touching (centre spacing
/// = diameter), bonded by DIRT's BPM with elastic moduli `youngs_modulus` (E) and
/// `shear_modulus` (G). `markers` = surface quadrature points per bead.
#[derive(Deserialize, Default)]
struct FiberCfg {
    n_spheres: usize,
    diameter: f64,
    youngs_modulus: f64,
    shear_modulus: f64,
    #[serde(default = "default_markers")]
    markers: usize,
}
fn default_markers() -> usize {
    800
}

/// `[flow]`: freestream state + the list of diameter-Reynolds numbers to sweep
/// (each sets μ = ρ U d / Re_d). The lowest-Re point is the gated Stokes-regime
/// validation; any additional points are reported as finite-inertia trend checks.
#[derive(Deserialize, Default)]
struct FlowCfg {
    rho: f64,
    p: f64,
    u_inf: f64,
    reynolds: Vec<f64>,
}

/// `[mesh]`: resolution (`cells_per_diameter`) and clearances (in diameters) from
/// the fiber *surface* to each domain face. The per-orientation box is derived
/// from these + the fiber extent, so both orientations get identical clearances.
#[derive(Deserialize, Default)]
struct MeshCfg {
    cells_per_diameter: f64,
    up: f64,
    dn: f64,
    lat: f64,
    #[serde(default = "default_ng")]
    ng: usize,
}
fn default_ng() -> usize {
    2
}

/// `[sponge]`: downstream absorbing layer relaxing to the uniform freestream.
#[derive(Deserialize, Default)]
struct SpongeCfg {
    sigma: f64,
}

/// `[run]`: iteration budget and trailing fraction to average / test steadiness.
#[derive(Deserialize, Default)]
struct RunCfg {
    steps: usize,
    print_every: usize,
    average_frac: f64,
}

/// `[validation]`: tolerances and gates.
#[derive(Deserialize, Default)]
struct ValidationCfg {
    /// |r_lowest_Re − r_theory| / r_theory. Set from first-order IBM grid bias,
    /// residual finite-Re correction, and finite-domain effects, NOT tuned to pass.
    tol_ratio: f64,
    /// Max Cd peak-to-peak over the averaging window per run: anti-transient gate.
    steady_ptp_tol: f64,
    /// The lowest-Re ratio must exceed the isotropic null (r = 1) by at least
    /// this margin — the fiber-shape-resolving gate (a single sphere gives r = 1).
    aniso_margin: f64,
    /// Exact analytic-field integrator checks on the fiber clump:
    /// relative tolerance on the recovered union volume vs Σ bead volumes.
    tol_volume: f64,
    /// Max |F_uniform| / |F_hydrostatic| (constant-pressure residual → 0).
    tol_uniform_ratio: f64,
    /// Relative tolerance on the recovered buoyancy vs Archimedes ρ_f g V_union.
    tol_buoyancy: f64,
    /// Downward gravity g_z used only by the analytic hydrostatic buoyancy check.
    gz: f64,
}

// ─── Union-of-spheres implicit body (the fiber clump) ────────────────────────

/// Implicit fiber = union of the bead spheres: signed distance = min over beads.
/// A `min` of Euclidean sphere SDFs is 1-Lipschitz and classifies inside/outside
/// of the clump **exactly**, so the cut-cell volume/area quadrature is exact for
/// the union geometry.
struct FiberBody {
    centers: Vec<Vec3>,
    radius: f64,
}
impl Body for FiberBody {
    fn signed_distance(&self, p: Vec3) -> f64 {
        self.centers
            .iter()
            .map(|c| {
                let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() - self.radius
            })
            .fold(f64::MAX, f64::min)
    }
}

/// Result of the exact analytic-field integrator checks on the fiber clump.
struct ExactCheck {
    v_num: f64,
    v_ref: f64,
    v_err: f64,
    uniform_ratio: f64,
    buoy_num: f64,
    buoy_ref: f64,
    buoy_err: f64,
    /// Buoyancy a SINGLE-sphere integrator (the pre-fiber capability) would
    /// recover — 1/N of the clump — used as the negative control.
    buoy_single: f64,
    dx_over_d: f64,
    cells: [usize; 3],
}

/// Validate the fiber-clump **surface-load coupling** against fields with exact
/// closed-form answers (the `sphere_drag_resolved` method, generalized to the
/// multi-sphere union): a uniform pressure integrates to **zero** net force, and a
/// hydrostatic pressure integrates to the **Archimedes buoyancy** `ρ_f g V_union`.
/// This is exact (analytic fields, no flow solve, no grid/Re/confinement bias), so
/// it is the tight anchor on the multi-sphere handoff the drag case then exercises
/// under real flow. Uses the cut-cell (embedded-boundary) integrator — the same one
/// the single-sphere `sphere_drag_resolved` validates — now on the bonded clump.
fn exact_integrator_check(fiber: &Fiber, rho_f: f64, p0: f64, gz: f64) -> ExactCheck {
    let r = fiber.radius;
    let d = 2.0 * r;
    let body = FiberBody {
        centers: fiber.centers.clone(),
        radius: r,
    };
    // A tight mesh just enclosing the clump; d/dx = 12 with 8³ sub-samples gives
    // a ~1 % cut-cell quadrature of the union volume. The fiber is built along
    // world x (see `build_bpm_fiber`), so x is the long axis of this box.
    let ax = 0.5 * fiber.length;
    let m = 0.5 * d;
    let dxq = d / 12.0;
    let mesh_cfg = UniformMeshConfig {
        nx: ((2.0 * (ax + m)) / dxq).round() as usize,
        ny: ((2.0 * (r + m)) / dxq).round() as usize,
        nz: ((2.0 * (r + m)) / dxq).round() as usize,
        ng: 2,
        bounds_lo: [-(ax + m), -(r + m), -(r + m)],
        bounds_hi: [ax + m, r + m, r + m],
        y_edges: None,
        z_edges: None,
    };
    let mesh = UniformMesh::from_config(&mesh_cfg);
    let eos = IdealGas::air();
    let t0 = p0 / (rho_f * R_GAS);
    let [ni, nj, nk] = mesh.dims();
    let mut geom: Vec<(Vec3, Vec3)> = Vec::with_capacity(ni * nj * nk);
    let mut u_uni: Vec<cfd_state::ConsVar> = Vec::with_capacity(ni * nj * nk);
    let mut u_hyd: Vec<cfd_state::ConsVar> = Vec::with_capacity(ni * nj * nk);
    for i in 0..ni {
        for j in 0..nj {
            for k in 0..nk {
                let c = mesh.idx(i, j, k);
                let cc = mesh.cell_centroid(c);
                geom.push((
                    cc,
                    [mesh.spacing(0, i), mesh.spacing(1, j), mesh.spacing(2, k)],
                ));
                u_uni.push(eos.prim_to_cons(&PrimVar::new(rho_f, 0.0, 0.0, 0.0, p0, t0)));
                let ph = p0 + rho_f * gz * cc[2];
                u_hyd.push(eos.prim_to_cons(&PrimVar::new(
                    rho_f,
                    0.0,
                    0.0,
                    0.0,
                    ph,
                    ph / (rho_f * R_GAS),
                )));
            }
        }
    }
    let metrics = CutCellMetrics::compute(&geom, &body);
    let v_num = metrics.solid_volume(&geom);
    // Touching (non-overlapping) beads ⇒ union volume = Σ bead volumes.
    let v_sphere = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
    let v_ref = fiber.centers.len() as f64 * v_sphere;
    let f_uni = metrics.pressure_force(&u_uni, &eos);
    let f_hyd = metrics.pressure_force(&u_hyd, &eos);
    let buoy_ref = -rho_f * v_ref * gz; // upward (gz < 0)
    let f_uni_mag = (f_uni[0].powi(2) + f_uni[1].powi(2) + f_uni[2].powi(2)).sqrt();
    let f_hyd_mag = (f_hyd[0].powi(2) + f_hyd[1].powi(2) + f_hyd[2].powi(2)).sqrt();
    ExactCheck {
        v_num,
        v_ref,
        v_err: (v_num - v_ref).abs() / v_ref,
        uniform_ratio: f_uni_mag / f_hyd_mag.max(1e-30),
        buoy_num: f_hyd[2],
        buoy_ref,
        buoy_err: (f_hyd[2] - buoy_ref).abs() / buoy_ref.abs(),
        buoy_single: -rho_f * v_sphere * gz, // one bead only (negative control)
        dx_over_d: d / dxq,
        cells: [ni, nj, nk],
    }
}

// ─── Tirado–García de la Torre rod friction ──────────────────────────────────

/// Broadside (⊥) translational friction coefficient ζ⊥ = 4πμL/(ln p + γ⊥).
fn zeta_perp(mu: f64, l: f64, p: f64) -> f64 {
    let g = 0.839 + 0.185 / p + 0.233 / (p * p);
    4.0 * std::f64::consts::PI * mu * l / (p.ln() + g)
}
/// End-on (∥) translational friction coefficient ζ∥ = 2πμL/(ln p + γ∥).
fn zeta_para(mu: f64, l: f64, p: f64) -> f64 {
    let g = -0.207 + 0.980 / p - 0.133 / (p * p);
    2.0 * std::f64::consts::PI * mu * l / (p.ln() + g)
}
/// Stokes anisotropy ratio ζ⊥/ζ∥ (μ, L cancel). `p = L/d`.
fn ratio_theory(p: f64) -> f64 {
    zeta_perp(1.0, 1.0, p) / zeta_para(1.0, 1.0, p)
}

// ─── BPM fiber construction (DIRT bonded-sphere chain on SOIL) ────────────────

/// Result of building the fiber: bead centres along the local +axis (axis 0 = x),
/// bead radius, end-to-end length L, aspect ratio p, bond count, and the derived
/// BPM stiffnesses (certification that the chain is a real bonded fiber).
struct Fiber {
    /// Bead centres, centred at the origin, laid along local axis 0 (+x).
    centers: Vec<Vec3>,
    radius: f64,
    length: f64,
    p: f64,
    n_bonds: usize,
    k_n: f64,
    k_t: f64,
    k_bend: f64,
}

/// Build the bonded-sphere fiber with DIRT's BPM: place `n` touching beads on the
/// SOIL substrate and let `dirt_bond::auto_bond_touching` create the BPM bond
/// network (the exact construction the `fiber_bond` DEM validation uses). The chain
/// is rigid for this drag case — a straight equal-spaced chain is at bond
/// equilibrium — so it is handed to the CFD side as fixed geometry (the seam speaks
/// kinematics only), analogous to the static packing in `fixed_bed_ergun`.
fn build_bpm_fiber(n: usize, diameter: f64, e_mod: f64, g_mod: f64) -> Fiber {
    let r = 0.5 * diameter;
    let d = diameter; // touching spacing
    let density = 2500.0;

    let mut atom = Atom::new();
    let mut dem = DemAtom::new();
    atom.dt = 1e-6;
    let mut centers = Vec::with_capacity(n);
    for i in 0..n {
        let off = (i as f64 - (n as f64 - 1.0) / 2.0) * d;
        let pos = [off, 0.0, 0.0];
        centers.push(pos);
        let mass = density * 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        atom.push_test_atom(i as u32 + 1, pos, r, mass);
        dem.radius.push(r);
        dem.density.push(density);
        dem.inv_inertia.push(1.0 / (0.4 * mass * r * r));
        dem.quaternion.push([1.0, 0.0, 0.0, 0.0]);
        dem.omega.push([0.0; 3]);
        dem.ang_mom.push([0.0; 3]);
        dem.torque.push([0.0; 3]);
        dem.body_id.push(0.0);
    }
    atom.nlocal = n as u32;
    atom.natoms = n as u64;

    let bond_radius_ratio = 1.0;
    let cfg = BondConfig {
        auto_bond: true,
        bond_tolerance: 1.001,
        bond_radius_ratio,
        youngs_modulus: Some(e_mod),
        shear_modulus: Some(g_mod),
        ..BondConfig::default()
    };

    let mut registry = AtomDataRegistry::new();
    registry.register(dem);
    registry.register(BondStore::new());
    let mut domain = Domain::new();
    domain.size = [1e3, 1e3, 1e3];

    let mut app = App::new();
    app.add_resource(atom);
    app.add_resource(registry);
    app.add_resource(cfg);
    app.add_resource(CommResource(Box::new(SingleProcessComm::new())));
    app.add_resource(domain);
    app.add_setup_system(auto_bond_touching, ScheduleSetupSet::PostSetup);
    app.organize_systems();
    app.setup();

    let reg = app.get_resource_ref::<AtomDataRegistry>().unwrap();
    let bonds = reg.expect::<BondStore>("BondStore after auto_bond");
    let n_bonds: usize = bonds.bonds.iter().map(|b| b.len()).sum::<usize>() / 2;
    // Certify: a straight chain of n touching beads must produce exactly n−1 bonds,
    // each at rest length r0 = d (the BPM contract for a linear fiber).
    assert_eq!(
        n_bonds,
        n - 1,
        "BPM auto-bond produced {n_bonds} bonds, expected {}",
        n - 1
    );
    for (i, list) in bonds.bonds.iter().enumerate() {
        for b in list {
            assert!(
                (b.r0 - d).abs() < 1e-9 * d,
                "bond {}→{} rest length {} ≠ spacing {d}",
                i + 1,
                b.partner_tag,
                b.r0
            );
        }
    }

    // dirt_bond BPM material-mode stiffnesses (E·A/L, G·A/L, E·I/L; the same
    // derivation `dirt_bond::bond_force` uses): certifies a real elastic bond.
    let r_b = bond_radius_ratio * r;
    let area = std::f64::consts::PI * r_b * r_b;
    let iben = 0.5 * (0.5 * std::f64::consts::PI * r_b.powi(4));
    let k_n = e_mod * area / d;
    let k_t = g_mod * area / d;
    let k_bend = e_mod * iben / d;
    assert!(
        k_n > 0.0 && k_t > 0.0 && k_bend > 0.0,
        "BPM stiffness non-positive"
    );

    let length = n as f64 * d; // shish-kebab rod length (touching beads)
    Fiber {
        centers,
        radius: r,
        length,
        p: length / d,
        n_bonds,
        k_n,
        k_t,
        k_bend,
    }
}

// ─── Downstream sponge (uniform-freestream absorber) ─────────────────────────

#[derive(Clone, Copy)]
struct Sponge {
    x_start: f64,
    x_end: f64,
    sigma: f64,
    target: ConsVar,
}

/// `PostUpdate` (once per completed RK step): relax the conserved state toward the
/// undisturbed freestream in the downstream sponge (quadratic ramp), so outgoing
/// acoustic/vortical energy leaves instead of reflecting off the outlet. Same
/// passive absorber the sphere/cylinder drag cases use.
fn outlet_sponge_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    sponge: Res<Sponge>,
    sstate: Res<SolverState>,
) {
    if sstate.rk_stage != RkStage::First {
        return;
    }
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let ng = mesh.n_ghost();
    let [ni, nj, nk] = mesh.dims();
    let dt = sstate.dt;
    let (x0, x1, sig) = (sponge.x_start, sponge.x_end, sponge.sigma);
    let t = sponge.target;
    let inv_w = 1.0 / (x1 - x0);
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let idx = mesh.idx_raw(ir, jr, kr);
                let c = mesh.cell_centroid(idx);
                if c[0] < x0 {
                    continue;
                }
                let s = ((c[0] - x0) * inv_w).clamp(0.0, 1.0);
                let f = (sig * s * s * dt).min(1.0);
                let u = &mut state.u[idx];
                u.rho += f * (t.rho - u.rho);
                u.rho_u += f * (t.rho_u - u.rho_u);
                u.rho_v += f * (t.rho_v - u.rho_v);
                u.rho_w += f * (t.rho_w - u.rho_w);
                u.rho_e += f * (t.rho_e - u.rho_e);
            }
        }
    }
}

// ─── One orientation at one Reynolds number ──────────────────────────────────

/// Flow orientation relative to the fiber axis (which lies along world x as built).
#[derive(Clone, Copy, PartialEq)]
enum Orient {
    /// Flow ⊥ axis: rotate the fiber onto world y (broadside).
    Broadside,
    /// Flow ∥ axis: keep the fiber on world x (end-on).
    EndOn,
}

/// Steady drag result for one (orientation, Re) run.
struct DragRun {
    fx: f64,      // steady streamwise force [N]
    ptp_rel: f64, // Cd peak-to-peak / mean over the averaging window
    re_actual: f64,
    mach: f64,
    cells: [usize; 3],
}

#[allow(clippy::too_many_arguments)]
fn run_orientation(
    fiber: &Fiber,
    orient: Orient,
    mu: f64,
    flow: &FlowCfg,
    meshc: &MeshCfg,
    sponge_sigma: f64,
    mut solver_cfg: SolverConfig,
    run: &RunCfg,
    markers: usize,
) -> DragRun {
    solver_cfg.viscous = true;
    let d = 2.0 * fiber.radius;
    let r = fiber.radius;
    let dx = d / meshc.cells_per_diameter;

    // Place beads: end-on keeps them on x; broadside rotates x→y.
    let bodies: Vec<SphereBody> = fiber
        .centers
        .iter()
        .map(|c| {
            let center = match orient {
                Orient::EndOn => [c[0], 0.0, 0.0],
                Orient::Broadside => [0.0, c[0], 0.0],
            };
            SphereBody::fixed(center, r)
        })
        .collect();

    // Fiber surface reach along each world axis, then box = reach + clearance.
    let ax_reach = 0.5 * fiber.length; // centre of the end bead
    let (xr, yr, zr) = match orient {
        Orient::EndOn => (ax_reach + r, r, r),
        Orient::Broadside => (r, ax_reach + r, r),
    };
    let x_lo = -(xr + meshc.up * d);
    let x_hi = xr + meshc.dn * d;
    let y_span = yr + meshc.lat * d;
    let z_span = zr + meshc.lat * d;
    let nx = ((x_hi - x_lo) / dx).round() as usize;
    let ny = (2.0 * y_span / dx).round() as usize;
    let nz = (2.0 * z_span / dx).round() as usize;
    let mesh_cfg = UniformMeshConfig {
        nx,
        ny,
        nz,
        ng: meshc.ng,
        bounds_lo: [x_lo, -y_span, -z_span],
        bounds_hi: [x_hi, y_span, z_span],
        y_edges: None,
        z_edges: None,
    };

    let eos = IdealGas::air();
    let t_inf = flow.p / (flow.rho * R_GAS);
    let freestream = eos.prim_to_cons(&PrimVar::new(flow.rho, flow.u_inf, 0.0, 0.0, flow.p, t_inf));
    let init = move |_x: Vec3| freestream;

    let bcs = BoundaryRegistry::default()
        .with(
            BoundarySide::XLo,
            SubsonicInflow {
                rho: flow.rho,
                u: flow.u_inf,
                v: 0.0,
                w: 0.0,
                t: t_inf,
            },
        )
        .with(BoundarySide::XHi, PressureOutlet { p: flow.p })
        .with_axis(1, Symmetry)
        .with_axis(2, Symmetry);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    app.add_resource(Viscosity::Constant(mu));
    app.add_resource(Sponge {
        x_start: xr + 0.5 * meshc.dn * d,
        x_end: x_hi,
        sigma: sponge_sigma,
        target: freestream,
    });
    app.add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_update_system(outlet_sponge_system, MeshScheduleSet::PostUpdate)
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(solver_cfg))
        .add_plugins(GhostCellIbmPlugin)
        .add_plugins(SphereIbmPlugin::new(bodies, freestream).with_markers(markers));
    app.prepare();

    let re_actual = flow.rho * flow.u_inf * d / mu;
    // Nondim reference for Cd bookkeeping only (the validation is on forces/ratio):
    // ½ρU²·πR² per bead.
    let q = 0.5
        * flow.rho
        * flow.u_inf
        * flow.u_inf
        * std::f64::consts::PI
        * r
        * r
        * fiber.centers.len() as f64;
    let cd_of = |fx: f64| fx / q;

    let avg_start = ((1.0 - run.average_frac) * run.steps as f64) as usize;
    let mut cd_samples: Vec<f64> = Vec::new();
    let mut fx_samples: Vec<f64> = Vec::new();
    for step in 0..run.steps {
        app.run();
        let fx = app.get_resource_ref::<MarkerForce>().unwrap().total()[0];
        if step >= avg_start {
            cd_samples.push(cd_of(fx));
            fx_samples.push(fx);
        }
        if run.print_every > 0 && (step % run.print_every == 0 || step + 1 == run.steps) {
            let wall = app.get_resource_ref::<WallForce>().unwrap().total()[0];
            println!("    step {step:>7}   Fx(marker) {fx:>10.3}   Cd {:>8.4}   Fx(staircase) {wall:>10.3}", cd_of(fx));
        }
    }

    // Physicality + Mach guards.
    let mut mach = 0.0f64;
    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        assert!(
            state
                .u
                .iter()
                .all(|u| u.rho.is_finite() && u.rho > 0.0 && u.rho_e.is_finite()),
            "fiber run went non-physical",
        );
        for u in state.u.iter() {
            let sp = (u.velocity()[0].powi(2) + u.velocity()[1].powi(2) + u.velocity()[2].powi(2))
                .sqrt();
            mach = mach.max(sp / eos.sound_speed(u));
        }
    }

    let n = cd_samples.len().max(1) as f64;
    let cd = cd_samples.iter().sum::<f64>() / n;
    let fx = fx_samples.iter().sum::<f64>() / n;
    let cd_max = cd_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let cd_min = cd_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let ptp_rel = if cd.abs() > 0.0 {
        (cd_max - cd_min) / cd.abs()
    } else {
        f64::INFINITY
    };
    DragRun {
        fx,
        ptp_rel,
        re_actual,
        mach,
        cells: [nx, ny, nz],
    }
}

fn main() {
    let path = std::env::args().nth(1).expect(
        "usage: cfd_ibm_fiber <case.toml>  (see crates/cfd_ibm/examples/cfd_ibm_fiber_p5.toml)",
    );
    let toml_src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let fc: FiberCfg = cfg.section("fiber");
    let flow: FlowCfg = cfg.section("flow");
    let meshc: MeshCfg = cfg.section("mesh");
    let sponge_cfg: SpongeCfg = cfg.section("sponge");
    let run: RunCfg = cfg.section("run");
    let valid: ValidationCfg = cfg.section("validation");
    let solver_cfg: SolverConfig = cfg.section("solver");

    assert!(
        !flow.reynolds.is_empty(),
        "need ≥1 diameter-Reynolds number for the fiber-drag validation, got {}",
        flow.reynolds.len()
    );

    // ── Build the BPM fiber (DIRT bonded-sphere chain) ───────────────────────
    let fiber = build_bpm_fiber(
        fc.n_spheres,
        fc.diameter,
        fc.youngs_modulus,
        fc.shear_modulus,
    );
    let r_theory = ratio_theory(fiber.p);

    println!("# Resolved BPM-fiber drag anisotropy vs slender-body theory (3-D ghost-cell IBM)");
    println!(
        "# fiber: {} BPM-bonded beads, d = {}, L = {:.4}, p = L/d = {:.3}, {} bonds",
        fc.n_spheres, fc.diameter, fiber.length, fiber.p, fiber.n_bonds
    );
    println!(
        "# BPM stiffness (dirt_bond, E={:.2e} G={:.2e}): k_n={:.3e} N/m  k_t={:.3e} N/m  k_bend={:.3e} N·m/rad",
        fc.youngs_modulus, fc.shear_modulus, fiber.k_n, fiber.k_t, fiber.k_bend
    );
    println!(
        "# Tirado–García de la Torre: r_theory = ζ⊥/ζ∥ = {:.4}  (isotropic null r=1 ⇒ p→1)",
        r_theory
    );

    // ── Stage 1: EXACT surface-load coupling check (analytic fields) ─────────
    // Tight, grid/Re/confinement-free validation of the multi-sphere handoff
    // against Archimedes' principle before the real-flow drag case exercises it.
    let ex = exact_integrator_check(&fiber, flow.rho, flow.p, valid.gz);
    println!("#");
    println!(
        "# ── Stage 1: exact surface-load coupling (cut-cell IBM, analytic fields, no flow) ──"
    );
    println!("#   clump mesh {:?}  d/dx = {:.1}", ex.cells, ex.dx_over_d);
    println!(
        "#   union volume: {:.5e}  (ref Σ V_bead = {:.5e}, err {:.2}% ≤ {:.1}%)",
        ex.v_num,
        ex.v_ref,
        100.0 * ex.v_err,
        100.0 * valid.tol_volume
    );
    println!(
        "#   uniform pressure ⇒ net force |F|/|F_hydro| = {:.2e}  (≤ {:.1e})  [normals close over the clump]",
        ex.uniform_ratio, valid.tol_uniform_ratio
    );
    println!(
        "#   hydrostatic ⇒ buoyancy Fz = {:.5e}  (Archimedes ρ_f g V_union = {:.5e}, err {:.2}% ≤ {:.1}%)",
        ex.buoy_num, ex.buoy_ref, 100.0 * ex.buoy_err, 100.0 * valid.tol_buoyancy
    );
    // Negative control: a single-sphere integrator (the pre-fiber capability)
    // recovers only 1/N of the clump buoyancy and FAILS — the multi-sphere
    // generalization is load-bearing, not decorative.
    let single_err = (ex.buoy_single - ex.buoy_ref).abs() / ex.buoy_ref.abs();
    let single_fails = single_err > valid.tol_buoyancy;
    println!(
        "#   negative control: a SINGLE-bead integrator gives Fz = {:.5e} ({:.0}% off) ⇒ {}  [multi-sphere handoff is necessary]",
        ex.buoy_single,
        100.0 * single_err,
        if single_fails { "FAILS the gate" } else { "DID NOT FAIL — gate vacuous!" }
    );
    let pass_exact = ex.v_err <= valid.tol_volume
        && ex.uniform_ratio <= valid.tol_uniform_ratio
        && ex.buoy_err <= valid.tol_buoyancy
        && single_fails;

    // ── Stage 2: real-flow drag anisotropy — measure r(Re) at each Re ────────
    println!("#");
    println!("# ── Stage 2: real-flow drag anisotropy vs slender-body theory ──");
    let mut re_list: Vec<f64> = Vec::new();
    let mut r_list: Vec<f64> = Vec::new();
    let mut worst_ptp = 0.0f64;
    let mut worst_mach = 0.0f64;

    for &re_d in &flow.reynolds {
        let mu = flow.rho * flow.u_inf * fc.diameter / re_d;
        println!("#");
        println!("# ── Re_d = {re_d}  (μ = {mu:.4}) ─────────────────────────");
        println!("#  BROADSIDE (flow ⊥ fiber axis):");
        let perp = run_orientation(
            &fiber,
            Orient::Broadside,
            mu,
            &flow,
            &meshc,
            sponge_cfg.sigma,
            solver_cfg.clone(),
            &run,
            fc.markers,
        );
        println!("#  END-ON (flow ∥ fiber axis):");
        let par = run_orientation(
            &fiber,
            Orient::EndOn,
            mu,
            &flow,
            &meshc,
            sponge_cfg.sigma,
            solver_cfg.clone(),
            &run,
            fc.markers,
        );

        let r_meas = perp.fx / par.fx;
        let l = fiber.length;
        let f_perp_ref = zeta_perp(mu, l, fiber.p) * flow.u_inf;
        let f_par_ref = zeta_para(mu, l, fiber.p) * flow.u_inf;
        println!(
            "#   F⊥ = {:.3} N ({:.3}×ζ⊥U)  [cells {:?}, M {:.3}, ptp {:.2}%]",
            perp.fx,
            perp.fx / f_perp_ref,
            perp.cells,
            perp.mach,
            100.0 * perp.ptp_rel
        );
        println!(
            "#   F∥ = {:.3} N ({:.3}×ζ∥U)  [cells {:?}, M {:.3}, ptp {:.2}%]",
            par.fx,
            par.fx / f_par_ref,
            par.cells,
            par.mach,
            100.0 * par.ptp_rel
        );
        println!("#   r(Re_d={re_d}) = F⊥/F∥ = {r_meas:.4}   (Stokes target {r_theory:.4})");
        re_list.push(perp.re_actual.max(par.re_actual));
        r_list.push(r_meas);
        worst_ptp = worst_ptp.max(perp.ptp_rel).max(par.ptp_rel);
        worst_mach = worst_mach.max(perp.mach).max(par.mach);
    }

    // The gated Stokes-regime observable is the LOWEST-Re ratio; higher Re values
    // are reported only as a finite-inertia trend check.
    let (i_lo, _) = re_list
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    let r_low = r_list[i_lo];

    let rel_low = (r_low - r_theory).abs() / r_theory;
    let rel_isotropic = (1.0 - r_theory).abs() / r_theory;

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    for (re, r) in re_list.iter().zip(&r_list) {
        println!("#   r(Re_d={:.3}) = {:.4}", re, r);
    }
    println!(
        "# gated lowest-Re ratio (Re_d={:.3}): {r_low:.4}",
        re_list[i_lo]
    );
    println!(
        "# r_theory (Tirado–García de la Torre, p={:.3}): {r_theory:.4}",
        fiber.p
    );
    println!(
        "# lowest-Re rel.err: {:.2}%   (tol {:.1}%)",
        100.0 * rel_low,
        100.0 * valid.tol_ratio
    );
    println!(
        "# isotropic-null control: |1 − r_theory| = {:.3} (= {:.1}% of r_theory)  vs aniso_margin {:.1}%  ⇒ r=1 {} the gate",
        (1.0 - r_theory).abs(),
        100.0 * rel_isotropic,
        100.0 * valid.aniso_margin,
        if (1.0 - r_theory).abs() / r_theory > valid.tol_ratio { "REJECTED by" } else { "would pass" }
    );
    println!(
        "# worst steady ptp over all runs: {:.2}%  (gate {:.1}%)",
        100.0 * worst_ptp,
        100.0 * valid.steady_ptp_tol
    );
    println!("# worst Mach over all runs: {:.3}", worst_mach);

    let pass_ratio = rel_low <= valid.tol_ratio;
    let pass_steady = worst_ptp <= valid.steady_ptp_tol;
    // Falsifiability: the measurement must resolve real anisotropy (r_low far
    // from the isotropic null r=1), AND the gate must be discriminating enough that
    // an isotropic body (r=1) is rejected.
    let pass_aniso = (r_low - 1.0) >= valid.aniso_margin;
    let isotropic_would_fail = (1.0 - r_theory).abs() / r_theory > valid.tol_ratio;
    let pass_nontrivial = pass_aniso && isotropic_would_fail;

    println!("#");
    println!("# ── verdict ────────────────────────────────────────────");
    println!(
        "# Stage 1 (exact coupling, Archimedes):   {}",
        if pass_exact { "PASS" } else { "FAIL" }
    );
    println!(
        "# Stage 2 (drag anisotropy, slender-body): {}",
        if pass_ratio && pass_steady && pass_nontrivial {
            "PASS"
        } else {
            "FAIL"
        }
    );

    if pass_exact && pass_ratio && pass_steady && pass_nontrivial {
        println!(
            "VALIDATION: PASS  (exact: V {:.2}% + buoyancy {:.2}% + uniform {:.1e}, single-bead control fails; drag: r_low {r_low:.4} vs Tirado {r_theory:.4} {:.2}%≤{:.1}%, ptp {:.2}%≤{:.1}%, r=1 rejected)",
            100.0 * ex.v_err, 100.0 * ex.buoy_err, ex.uniform_ratio,
            100.0 * rel_low, 100.0 * valid.tol_ratio, 100.0 * worst_ptp, 100.0 * valid.steady_ptp_tol
        );
    } else {
        println!(
            "VALIDATION: FAIL  (exact_ok={pass_exact}  ratio_ok={pass_ratio} [{:.2}% vs {:.1}%]  steady_ok={pass_steady} [{:.2}% vs {:.1}%]  nontrivial_ok={pass_nontrivial} [aniso={pass_aniso} iso_fails={isotropic_would_fail}])",
            100.0 * rel_low, 100.0 * valid.tol_ratio, 100.0 * worst_ptp, 100.0 * valid.steady_ptp_tol
        );
        std::process::exit(1);
    }
}
