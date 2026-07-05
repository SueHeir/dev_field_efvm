//! **Drafting–Kissing–Tumbling (DKT): resolved two-body IBM sedimentation**. Two
//! heavy circular bodies settle in a
//! viscous fluid and reproduce the canonical particle–fluid–particle sequence:
//! the trailing particle is drawn into the leading particle's low-pressure wake
//! (**drafting**), catches up until the two touch (**kissing**), and the tandem
//! is then unstable and rotates apart (**tumbling**), the former trailing particle
//! overtaking. Reference: Fortes, Joseph & Lundgren 1987 (experiment);
//! Glowinski et al. 2001 and Feng & Michaelides 2004 (resolved simulation) — both
//! of which present this as the **2-D two-particle** benchmark, which is what this
//! example runs (a full 3-D two-sphere DKT reuses the same driver on
//! `cfd_ibm::Sphere` + `SphereIbmPlugin`, but is compute-bound; see the PR notes).
//!
//! ## What is resolved
//!
//! The particles are **geometrically resolved** on the FIELD mesh by the same
//! sharp-interface ghost-cell IBM the single-sphere drag case validates
//! (`sphere_drag_validation`): each step the mask/ghosts/markers are rebuilt at
//! the particles' current pose, the no-slip mirror imposes the moving-wall
//! velocity `u + ω×r`, and the **surface-marker traction integral**
//! ([`compute_marker_loadings_per_body`]) returns each particle's own net fluid
//! load + torque. Rigid-body motion under the fluid load, the reduced-gravity
//! buoyant weight `(ρ_p−ρ_f)Vg`, and a **soft-sphere (linear-spring) contact** for
//! the kiss are integrated here in the example (the FSI driver is case logic, not
//! core API). A quiescent gas
//! carries no hydrostatic field, so buoyancy is added analytically on the
//! particle (reduced-gravity formulation) and the marker load is the pure dynamic
//! (drag + pressure + added-mass) reaction.
//!
//! ## Low-Mach incompressible surrogate + matched dimensionless groups
//!
//! As with the cylinder/sphere drag cases, this compressible solver is marched at
//! low Mach as an incompressible surrogate. DKT is governed by the **density
//! ratio** `ρ_p/ρ_f` and the **Galileo number** `Ga = u_g d/ν`,
//! `u_g=√((ρ_p/ρ_f−1) g d)` — matched here — not by the absolute fluid; the
//! gravitational velocity `u_g` is held to a small Mach so compressibility stays
//! sub-percent. Everything is declarative (`argv[1]`):
//!
//! ```text
//! cargo run --release -p dev_field_efvm --bin dkt_sedimentation -- \
//!     examples/dkt_sedimentation/config.toml
//! ```

use cfd_boundary::{BoundaryPlugin, BoundaryRegistry, NoSlipWall, SlipWall};
use cfd_eos::{Eos, EosResource, IdealGas, Viscosity};
use cfd_ibm::{
    compute_marker_loadings_per_body, GhostCellIbmPlugin, IbmGhostCells, IbmGhostInfo, IbmMarker,
    IbmMarkers, MaskConsts,
};
use cfd_solver::{
    CfdStatePlugin, FluxPlugin, IdealGasPlugin, IntegratorPlugin, RkStage, SolverConfig,
    SolverPlugin, SolverState,
};
use cfd_state::{CfdState, ConsVar, PrimVar, SolidMask};
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, StructuredMesh, UniformMesh,
    UniformMeshConfig, Vec3,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_scheduler::{Res, ResMut};
use serde::Deserialize;

const R_GAS: f64 = 287.058;

// ─── One settling disc (rigid body, per-unit-depth 2-D) ──────────────────────

#[derive(Clone, Copy, Debug)]
struct Disc {
    center: [f64; 2],
    vel: [f64; 2],
    omega: f64,
    radius: f64,
    mass: f64,    // per depth L_z
    inertia: f64, // per depth L_z
}

impl Disc {
    fn signed_distance(&self, p: Vec3) -> f64 {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        (dx * dx + dy * dy).sqrt() - self.radius
    }
    fn normal(&self, p: Vec3) -> Vec3 {
        let dx = p[0] - self.center[0];
        let dy = p[1] - self.center[1];
        let r = (dx * dx + dy * dy).sqrt();
        if r < 1e-12 {
            [1.0, 0.0, 0.0]
        } else {
            [dx / r, dy / r, 0.0]
        }
    }
    /// Rigid surface velocity `u + ω ẑ × (p−c)` (2-D, z-out-of-plane spin).
    fn wall_velocity(&self, p: Vec3) -> Vec3 {
        let rx = p[0] - self.center[0];
        let ry = p[1] - self.center[1];
        [
            self.vel[0] - self.omega * ry,
            self.vel[1] + self.omega * rx,
            0.0,
        ]
    }
}

/// The immersed discs (mutated in place by the FSI integrator each step).
#[derive(Clone, Default, Debug)]
struct Discs(Vec<Disc>);

/// Static physics + numerics the FSI integrator needs.
#[derive(Clone, Copy, Debug)]
struct FsiParams {
    rho_f: f64,
    g: f64,  // gravitational acceleration magnitude (−y)
    lz: f64, // cell thickness (per-depth normalization)
    n_theta: usize,
    // Contact (soft-sphere linear spring): range δ and stiffness k_n; wall_x are
    // the two no-slip side walls; wall_y the top/bottom.
    contact_range: f64,
    k_n: f64,
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
}

/// DKT diagnostics accumulated over the run (phase detection + timing).
#[derive(Clone, Default)]
struct DktTrace {
    t: Vec<f64>,
    // per-disc y (fall) and x (lateral); disc 0 = initially leading (lower).
    y0: Vec<f64>,
    y1: Vec<f64>,
    x0: Vec<f64>,
    x1: Vec<f64>,
    gap: Vec<f64>, // surface-to-surface gap
    started: bool,
}

// ─── Moving-disc IBM geometry (example-local Setup systems) ───────────────────

/// `Setup`: mark cells inside any disc solid (flip back the ones the moving discs
/// uncovered), seeding flips with the stagnant/ambient state.
fn refresh_disc_mask(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    discs: Res<Discs>,
    consts: Res<MaskConsts>,
) {
    let mut mask = reg.expect_mut::<SolidMask>("SolidMask not registered");
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let [ni, nj, nk] = mesh.dims();
    for k in 0..nk {
        for j in 0..nj {
            for i in 0..ni {
                let c = mesh.idx(i, j, k);
                let cc = mesh.cell_centroid(c);
                let inside = discs.0.iter().any(|d| d.signed_distance(cc) < 0.0);
                let was = mask.solid[c];
                if inside && !was {
                    mask.solid[c] = true;
                    state.u[c] = consts.stagnant;
                    state.rhs[c] = ConsVar::ZERO;
                } else if !inside && was {
                    mask.solid[c] = false;
                    state.u[c] = consts.ambient;
                    state.rhs[c] = ConsVar::ZERO;
                }
            }
        }
    }
}

/// `Setup`: rebuild the ghost-cell list, image points reflected across the nearest
/// disc surface, `wall_velocity` = that disc's moving surface velocity.
fn compute_disc_ghosts(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    discs: Res<Discs>,
    mut ghosts: ResMut<IbmGhostCells>,
) {
    ghosts.cells.clear();
    if discs.0.is_empty() {
        return;
    }
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let ng = mesh.n_ghost();
    let [ni, nj, nk] = mesh.dims();
    for k in 0..nk {
        let kr = k + ng;
        for j in 0..nj {
            let jr = j + ng;
            for i in 0..ni {
                let ir = i + ng;
                let c = mesh.idx_raw(ir, jr, kr);
                if !mask.solid[c] {
                    continue;
                }
                let nbrs = [
                    mesh.idx_raw(ir + 1, jr, kr),
                    mesh.idx_raw(ir - 1, jr, kr),
                    mesh.idx_raw(ir, jr + 1, kr),
                    mesh.idx_raw(ir, jr - 1, kr),
                ];
                if !nbrs.iter().any(|&n| !mask.solid[n]) {
                    continue;
                }
                let cc = mesh.cell_centroid(c);
                let mut best_abs = f64::MAX;
                let mut best_signed = 0.0;
                let mut best_normal = [1.0, 0.0, 0.0];
                let mut best_body = 0u32;
                let mut best_wall = [0.0; 3];
                for (bid, d) in discs.0.iter().enumerate() {
                    let s = d.signed_distance(cc);
                    if s.abs() < best_abs {
                        best_abs = s.abs();
                        best_signed = s;
                        best_normal = d.normal(cc);
                        best_body = bid as u32;
                        let foot = [
                            cc[0] - s * best_normal[0],
                            cc[1] - s * best_normal[1],
                            cc[2],
                        ];
                        best_wall = d.wall_velocity(foot);
                    }
                }
                let push = 2.0 * (-best_signed);
                let image_point = [
                    cc[0] + push * best_normal[0],
                    cc[1] + push * best_normal[1],
                    cc[2],
                ];
                ghosts.cells.push((
                    c,
                    IbmGhostInfo {
                        cell_center: cc,
                        image_point,
                        normal: best_normal,
                        body_id: best_body,
                        wall_velocity: best_wall,
                    },
                ));
            }
        }
    }
}

/// `Setup`: `n_theta` equal-arc markers on each disc's mid-plane circle, analytic
/// area `2π r L_z / n_theta`, tagged with the moving-wall velocity.
fn compute_disc_markers(
    mesh: Res<UniformMesh>,
    discs: Res<Discs>,
    params: Res<FsiParams>,
    mut markers: ResMut<IbmMarkers>,
) {
    markers.markers.clear();
    if discs.0.is_empty() {
        return;
    }
    // Mid-plane z of the (one-cell-thick) domain.
    let ng = mesh.n_ghost();
    let [_, _, nk] = mesh.dims();
    let z_mid = mesh.cell_centroid(mesh.idx_raw(ng, ng, ng + nk / 2))[2];
    let n_theta = params.n_theta;
    for (bid, d) in discs.0.iter().enumerate() {
        let area = 2.0 * std::f64::consts::PI * d.radius * params.lz / n_theta as f64;
        for it in 0..n_theta {
            let theta = 2.0 * std::f64::consts::PI * (it as f64 + 0.5) / n_theta as f64;
            let (nx, ny) = (theta.cos(), theta.sin());
            let pos = [
                d.center[0] + d.radius * nx,
                d.center[1] + d.radius * ny,
                z_mid,
            ];
            markers.markers.push(IbmMarker {
                body_id: bid as u32,
                position: pos,
                normal: [nx, ny, 0.0],
                area,
                wall_velocity: d.wall_velocity(pos),
            });
        }
    }
}

// ─── FSI integrator + contact (once per completed RK step) ────────────────────

/// Linear-spring (Hooke) contact repulsion between the two discs and against the
/// four no-slip walls, active only inside the short `contact_range` band — the
/// soft-sphere collision that resolves the kiss the sub-grid fluid film cannot.
/// Returns `(force0, force1)` [N per depth].
fn contact_forces(discs: &[Disc], p: &FsiParams) -> ([f64; 2], [f64; 2]) {
    let mut f = [[0.0f64; 2]; 2];
    // Disc–disc.
    if discs.len() == 2 {
        let (a, b) = (&discs[0], &discs[1]);
        let dx = b.center[0] - a.center[0];
        let dy = b.center[1] - a.center[1];
        let dist = (dx * dx + dy * dy).sqrt().max(1e-12);
        let overlap = (a.radius + b.radius + p.contact_range) - dist;
        if overlap > 0.0 {
            let (nx, ny) = (dx / dist, dy / dist); // a→b
            let fmag = p.k_n * overlap;
            f[0][0] -= fmag * nx;
            f[0][1] -= fmag * ny;
            f[1][0] += fmag * nx;
            f[1][1] += fmag * ny;
        }
    }
    // Disc–wall (repel from each of the four walls when within the band).
    for (i, d) in discs.iter().enumerate() {
        let gaps = [
            (d.center[0] - p.x_lo - d.radius, [1.0, 0.0]), // left wall pushes +x
            (p.x_hi - d.center[0] - d.radius, [-1.0, 0.0]), // right wall pushes −x
            (d.center[1] - p.y_lo - d.radius, [0.0, 1.0]), // bottom pushes +y
            (p.y_hi - d.center[1] - d.radius, [0.0, -1.0]), // top pushes −y
        ];
        for (gap, dir) in gaps {
            let overlap = p.contact_range - gap;
            if overlap > 0.0 {
                let fmag = p.k_n * overlap;
                f[i][0] += fmag * dir[0];
                f[i][1] += fmag * dir[1];
            }
        }
    }
    (f[0], f[1])
}

/// `Output` (gated to a completed RK step): the resolved-FSI hand-back. Integrate
/// each disc under (fluid marker load + torque) + reduced-gravity buoyant weight
/// + soft-sphere contact, using the just-completed field. Advances by the full
/// step `dt` and records the DKT trace.
fn fsi_step_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    visc: Res<Viscosity>,
    markers: Res<IbmMarkers>,
    params: Res<FsiParams>,
    sstate: Res<SolverState>,
    mut discs: ResMut<Discs>,
    mut trace: ResMut<DktTrace>,
) {
    if sstate.rk_stage != RkStage::First {
        return; // act once per completed full step
    }
    let dt = sstate.dt;
    if dt <= 0.0 {
        return;
    }
    let eos: &dyn Eos = &*eos.0;
    let mask = reg.expect::<SolidMask>("SolidMask not registered");
    let state = reg.expect::<CfdState>("CfdState not registered");
    let dx = mesh.spacing(0, 0);

    let centers: Vec<Vec3> = discs
        .0
        .iter()
        .map(|d| [d.center[0], d.center[1], 0.0])
        .collect();
    let loads =
        compute_marker_loadings_per_body(&*mesh, &state, &mask, eos, &markers, dx, *visc, &centers);
    let (fc0, fc1) = contact_forces(&discs.0, &params);
    let contact = [fc0, fc1];

    let vol = |r: f64| std::f64::consts::PI * r * r * params.lz; // per-depth disc "volume"
    for (i, d) in discs.0.iter_mut().enumerate() {
        let ld = loads.get(i).copied().unwrap_or_default();
        let f_fluid = ld.total(); // dynamic pressure + viscous
        let v = vol(d.radius);
        // Reduced-gravity buoyant weight (ρ_p−ρ_f)·V·g downward. m = ρ_p V.
        let rho_p = d.mass / v;
        let w_net = (rho_p - params.rho_f) * v * params.g; // magnitude, −y
        let fx = f_fluid[0] + contact[i][0];
        let fy = f_fluid[1] - w_net + contact[i][1];
        // Semi-implicit: velocity then position.
        d.vel[0] += fx / d.mass * dt;
        d.vel[1] += fy / d.mass * dt;
        d.center[0] += d.vel[0] * dt;
        d.center[1] += d.vel[1] * dt;
        // Spin from fluid torque (z-component).
        d.omega += ld.torque[2] / d.inertia * dt;
    }

    // ── DKT trace ────────────────────────────────────────────────────────────
    if discs.0.len() == 2 {
        let a = discs.0[0];
        let b = discs.0[1];
        let dxc = b.center[0] - a.center[0];
        let dyc = b.center[1] - a.center[1];
        let gap = (dxc * dxc + dyc * dyc).sqrt() - a.radius - b.radius;
        trace.started = true;
        trace.t.push(sstate.time);
        trace.y0.push(a.center[1]);
        trace.y1.push(b.center[1]);
        trace.x0.push(a.center[0]);
        trace.x1.push(b.center[0]);
        trace.gap.push(gap);
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct FlowCfg {
    rho: f64,
    p: f64,
}

#[derive(Deserialize, Default)]
struct ParticleCfg {
    diameter: f64,
    density_ratio: f64, // ρ_p/ρ_f
    u_grav: f64,        // gravitational velocity scale √((s−1) g d) → sets g
    re_settle: f64,     // Ga = u_grav d / ν → sets μ
    n_theta: usize,
}

#[derive(Deserialize, Default)]
struct InitCfg {
    // Disc 0 (leading, lower) and disc 1 (trailing, upper) centers.
    lead_x: f64,
    lead_y: f64,
    trail_x: f64,
    trail_y: f64,
}

#[derive(Deserialize, Default)]
struct ContactCfg {
    range_cells: f64,     // contact band in cell widths
    stiffness_scale: f64, // k_n = scale · (net weight)/(cell width)
}

#[derive(Deserialize, Default)]
struct RunCfg {
    steps: usize,
    print_every: usize,
}

#[derive(Deserialize, Default)]
struct ValidationCfg {
    draft_close_frac: f64, // drafting: gap must close by ≥ this fraction of its initial value
    kiss_gap_cells: f64,   // kissing: min surface gap ≤ this many cell widths
    tumble_lateral_d: f64, // tumbling: post-kiss |Δx| must exceed this many diameters
    max_mach: f64,         // incompressible-surrogate guard
}

fn main() {
    let path = std::env::args().nth(1).expect(
        "usage: dkt_sedimentation <case.toml>  (see examples/dkt_sedimentation/config.toml)",
    );
    let toml_src =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let mesh_cfg: UniformMeshConfig = cfg.section("grid");
    let flow: FlowCfg = cfg.section("flow");
    let pc: ParticleCfg = cfg.section("particle");
    let ic: InitCfg = cfg.section("init");
    let cc: ContactCfg = cfg.section("contact");
    let run: RunCfg = cfg.section("run");
    let valid: ValidationCfg = cfg.section("validation");
    let mut solver_cfg: SolverConfig = cfg.section("solver");
    solver_cfg.viscous = true;

    let d = pc.diameter;
    let r = 0.5 * d;
    let s = pc.density_ratio;
    // Match the dimensionless groups: u_g = √((s−1) g d) ⇒ g = u_g²/((s−1) d);
    // Ga = u_g d / ν ⇒ ν = u_g d / Ga, μ = ρ_f ν.
    let g = pc.u_grav * pc.u_grav / ((s - 1.0) * d);
    let nu = pc.u_grav * d / pc.re_settle;
    let mu = flow.rho * nu;
    let rho_p = s * flow.rho;

    let (nx_cells, ny_cells) = (mesh_cfg.nx, mesh_cfg.ny);
    let dx = (mesh_cfg.bounds_hi[0] - mesh_cfg.bounds_lo[0]) / mesh_cfg.nx as f64;
    let lz = mesh_cfg.bounds_hi[2] - mesh_cfg.bounds_lo[2];
    let (x_lo, x_hi) = (mesh_cfg.bounds_lo[0], mesh_cfg.bounds_hi[0]);
    let (y_lo, y_hi) = (mesh_cfg.bounds_lo[1], mesh_cfg.bounds_hi[1]);

    let eos = IdealGas::air();
    let t_inf = flow.p / (flow.rho * R_GAS);
    // Quiescent gas (reduced-gravity formulation: no fluid body force).
    let freestream = eos.prim_to_cons(&PrimVar::new(flow.rho, 0.0, 0.0, 0.0, flow.p, t_inf));
    let init = move |_x: Vec3| freestream;

    let vol = std::f64::consts::PI * r * r * lz;
    let mass = rho_p * vol;
    let inertia = 0.5 * mass * r * r;
    let mk_disc = |cx: f64, cy: f64| Disc {
        center: [cx, cy],
        vel: [0.0, 0.0],
        omega: 0.0,
        radius: r,
        mass,
        inertia,
    };
    // Disc 0 = leading (lower), disc 1 = trailing (upper).
    let discs = Discs(vec![
        mk_disc(ic.lead_x, ic.lead_y),
        mk_disc(ic.trail_x, ic.trail_y),
    ]);

    // Contact stiffness from the net weight scale so overlaps stay ~sub-cell.
    let w_net = (rho_p - flow.rho) * vol * g;
    let k_n = cc.stiffness_scale * w_net / dx;
    let params = FsiParams {
        rho_f: flow.rho,
        g,
        lz,
        n_theta: pc.n_theta.max(16),
        contact_range: cc.range_cells * dx,
        k_n,
        x_lo,
        x_hi,
        y_lo,
        y_hi,
    };

    let bcs = BoundaryRegistry::default()
        .with_axis(0, NoSlipWall) // channel side walls
        .with_axis(1, NoSlipWall) // closed top/bottom (sedimentation box)
        .with_axis(2, SlipWall); // 2-D span

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    app.add_resource(Viscosity::Constant(mu));
    // Stagnant seed = ambient at rest (already at rest here).
    let stagnant = ConsVar::new(
        freestream.rho,
        0.0,
        0.0,
        0.0,
        freestream.rho_e - freestream.kinetic_energy(),
    );
    app.add_resource(MaskConsts {
        stagnant,
        ambient: freestream,
    });
    app.add_resource(discs.clone());
    app.add_resource(params);
    app.add_resource(DktTrace::default());
    app.add_plugins(BoundaryPlugin::<UniformMesh>::new(bcs))
        .add_plugins(FluxPlugin::<UniformMesh>::hllc())
        .add_plugins(IntegratorPlugin::rk3())
        .add_plugins(SolverPlugin::<UniformMesh>::new(solver_cfg))
        .add_plugins(GhostCellIbmPlugin)
        // Moving-disc geometry refresh (Setup, before the ghost mirror in PreFlux).
        .add_update_system(refresh_disc_mask, MeshScheduleSet::Setup)
        .add_update_system(compute_disc_ghosts, MeshScheduleSet::Setup)
        .add_update_system(compute_disc_markers, MeshScheduleSet::Setup)
        // Resolved-FSI hand-back (Output, once per completed step).
        .add_update_system(fsi_step_system, MeshScheduleSet::Output);
    app.prepare();

    let c_snd = eos.sound_speed(&freestream);
    let ga = pc.u_grav * d / nu;
    println!("# Drafting–Kissing–Tumbling — resolved two-particle FSI (2-D ghost-cell IBM)");
    println!(
        "# d = {d}   d/dx = {:.1}   cells = {}x{}   rho_p/rho_f = {s}",
        d / dx,
        nx_cells,
        ny_cells
    );
    println!(
        "# u_g = {:.3}   Ga = {ga:.2}   g = {g:.3e}   mu = {mu:.4e}   M_g = {:.3}",
        pc.u_grav,
        pc.u_grav / c_snd
    );
    println!(
        "# k_n = {k_n:.3e}   contact_range = {:.3e} ({} cells)",
        params.contact_range, cc.range_cells
    );
    println!("# step        time        y_lead      y_trail     gap/d       x_lead     x_trail");

    let mut mach_max = 0.0f64;
    for step in 0..run.steps {
        app.run();
        // Mach guard from the current field.
        if step % 200 == 0 {
            let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
            let state = reg.get::<CfdState>().unwrap();
            let mut mmax = 0.0f64;
            for u in state.u.iter() {
                if u.rho.is_finite() && u.rho > 0.0 {
                    let sp = (u.velocity()[0].powi(2)
                        + u.velocity()[1].powi(2)
                        + u.velocity()[2].powi(2))
                    .sqrt();
                    mmax = mmax.max(sp / eos.sound_speed(u));
                }
            }
            mach_max = mach_max.max(mmax);
        }
        if run.print_every > 0 && (step % run.print_every == 0 || step + 1 == run.steps) {
            let dref = app.get_resource_ref::<Discs>().unwrap();
            let tr = app.get_resource_ref::<DktTrace>().unwrap();
            if dref.0.len() == 2 && tr.started {
                let a = dref.0[0];
                let b = dref.0[1];
                let gap = (*tr.gap.last().unwrap()) / d;
                let t = *tr.t.last().unwrap();
                println!(
                    "{step:>8}  {t:>10.5}  {:>10.5}  {:>10.5}  {:>9.4}  {:>9.5}  {:>9.5}",
                    a.center[1], b.center[1], gap, a.center[0], b.center[0]
                );
            }
        }
    }

    // Physicality guard.
    {
        let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
        let state = reg.get::<CfdState>().unwrap();
        assert!(
            state
                .u
                .iter()
                .all(|u| u.rho.is_finite() && u.rho > 0.0 && u.rho_e.is_finite()),
            "DKT run went non-physical",
        );
    }

    // ── DKT phase analysis ───────────────────────────────────────────────────
    let tr = app.get_resource_ref::<DktTrace>().unwrap().clone();
    let n = tr.t.len();
    assert!(n > 10, "no trace recorded");
    let diam = pc.diameter;

    // Kissing = the closest approach: global minimum surface gap and when it occurs.
    let (mut kiss_i, mut kiss_gap) = (0usize, f64::MAX);
    for i in 0..n {
        if tr.gap[i] < kiss_gap {
            kiss_gap = tr.gap[i];
            kiss_i = i;
        }
    }
    let t_kiss = tr.t[kiss_i];

    // Drafting is measured from the MAXIMUM gap reached on the *approach* (up to the
    // kiss): the leading particle's forming wake first nudges the pair slightly
    // apart, then the wake draws the trailing particle in and the gap closes. The
    // physically meaningful closure is (gap_max − gap_min)/gap_max over that window.
    let mut gap_max_pre = tr.gap[0];
    for i in 0..=kiss_i {
        gap_max_pre = gap_max_pre.max(tr.gap[i]);
    }
    let draft_closed = (gap_max_pre - kiss_gap) / gap_max_pre;

    // Independent drafting signature: at some instant before the kiss the trailing
    // (disc 1, upper) falls FASTER than the leading (disc 0) — the wake-shielding
    // catch-up. (dy<0 is downward; "faster" ⇒ more negative dy.)
    let mut trailing_caught_up = false;
    for i in 1..=kiss_i {
        let v0 = tr.y0[i] - tr.y0[i - 1];
        let v1 = tr.y1[i] - tr.y1[i - 1];
        if v1 < v0 - 1e-9 {
            trailing_caught_up = true;
            break;
        }
    }

    // Tumbling: post-kiss lateral separation |x1−x0|, in diameters.
    let mut tumble_dx = 0.0f64;
    for i in kiss_i..n {
        tumble_dx = tumble_dx.max((tr.x1[i] - tr.x0[i]).abs());
    }
    let tumble_diam = tumble_dx / diam;

    // Did the trailing particle overtake (lead/trail swap in y)? Post-kiss the
    // former trailing (disc 1) should fall below the former leading (disc 0).
    let mut overtook = false;
    for i in kiss_i..n {
        if tr.y1[i] < tr.y0[i] {
            overtook = true;
            break;
        }
    }

    let dt_kiss_scale = t_kiss * pc.u_grav / diam; // nondimensional kiss time (t·u_g/d)

    println!("#");
    println!("# ── DKT phase analysis ─────────────────────────────────");
    println!(
        "# max approach gap:           {:.4} d  (initial {:.4} d)",
        gap_max_pre / diam,
        tr.gap[0] / diam
    );
    println!("# DRAFTING: gap closed by      {:.1}% of the max approach gap  (gate ≥ {:.0}%);  trailing-caught-up: {trailing_caught_up}", 100.0 * draft_closed, 100.0 * valid.draft_close_frac);
    println!("# KISSING:  min gap            {:.3} d = {:.2} cells at t = {:.5}s  (t·u_g/d = {dt_kiss_scale:.2})  (gate ≤ {:.1} cells)", kiss_gap / diam, kiss_gap / dx, t_kiss, valid.kiss_gap_cells);
    println!(
        "# TUMBLING: max post-kiss |Δx| {:.3} d   (gate ≥ {:.2} d);  overtake: {overtook}",
        tumble_diam, valid.tumble_lateral_d
    );
    println!(
        "# max Mach over run:           {:.3}   (gate ≤ {:.2})",
        mach_max, valid.max_mach
    );

    let pass_draft = draft_closed >= valid.draft_close_frac && trailing_caught_up;
    let pass_kiss = (kiss_gap / dx) <= valid.kiss_gap_cells;
    let pass_tumble = tumble_diam >= valid.tumble_lateral_d && overtook;
    let pass_mach = mach_max <= valid.max_mach;

    println!("#");
    if pass_draft && pass_kiss && pass_tumble && pass_mach {
        println!(
            "VALIDATION: PASS  (draft {:.0}%≥{:.0}%; kiss {:.2}≤{:.1} cells; tumble {:.2}d≥{:.2}d & overtake; M {:.3}≤{:.2})",
            100.0 * draft_closed, 100.0 * valid.draft_close_frac, kiss_gap / dx, valid.kiss_gap_cells,
            tumble_diam, valid.tumble_lateral_d, mach_max, valid.max_mach
        );
    } else {
        println!(
            "VALIDATION: FAIL  (draft_ok={pass_draft}  kiss_ok={pass_kiss}  tumble_ok={pass_tumble}  mach_ok={pass_mach})"
        );
        std::process::exit(1);
    }
}
