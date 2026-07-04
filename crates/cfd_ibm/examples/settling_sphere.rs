//! **Single-sphere terminal velocity via the unresolved DEM–CFD seam** — the
//! point-particle regime of the cross-substrate coupling.
//!
//! A single dense sphere is released from rest in quiescent gas and settles under
//! gravity. The sphere lives on the SOIL (Lagrangian) substrate — a `soil::Atom`
//! integrated by `soil_verlet`; the gas lives on the FIELD (Eulerian) substrate —
//! a `CfdState` on a `UniformMesh`. They are two independent `grass_app::App`s run
//! as sub-Apps of one `grass_multi` parent, coupled *only* through the seam data
//! contract in [`cfd_ibm::coupling`]:
//!
//! ```text
//!   Export : soil Atom (pos, vel) ──▶ cfd  ParticleSet          (kinematics)
//!   TickCfd: cfd computes drag+buoyancy from the local gas, deposits void
//!            fraction ε and the momentum sink −F_drag back into the gas
//!   Import : cfd InterphaseForces ──▶ soil FluidForce           (fluid load)
//!   TickSoil: soil integrates the particle under gravity + the fluid load
//! ```
//!
//! FIELD is the sole mesh owner; the particle is immersed and never owns a cell.
//!
//! ## What is validated (independent references)
//!
//! 1. **Terminal velocity vs Stokes (1851).** Parameters put the settling Reynolds
//!    number at `Re ≪ 1`, where the closed form
//!    `v_t = (2/9)(ρ_p − ρ_f) g R²/μ` holds. The measured plateau velocity must
//!    match it within `tol_rel_stokes` (the Schiller–Naumann correction at this Re
//!    is a known few-percent, so the tolerance is set from that, not tuned).
//! 2. **Force balance at steady state.** At terminal the total fluid load equals
//!    the weight: `|F_fluid − m g| / (m g)` must vanish — the coupled ODE has
//!    reached the drag law's fixed point (validates the force *handoff* and the
//!    integrator independently of the drag physics).
//! 3. **Momentum conservation of the two-way exchange.** The momentum the sink
//!    deposits in the gas must equal `−Σ F_drag Δt` — the reaction the particle
//!    received — to round-off.
//!
//! The drag closure ([`beta_gidaspow`], Wen & Yu 1966 in the dilute limit) and its
//! `Cd(Re)` against Schiller–Naumann (1935) / Clift–Grace–Weber (1978) are
//! unit-tested in `cfd_ibm::coupling`.
//!
//! Everything case-specific is declarative TOML read from `argv[1]`:
//!
//! ```text
//! cargo run --release -p cfd_ibm --example settling_sphere -- \
//!     crates/cfd_ibm/examples/settling_sphere_stokes.toml
//! ```

use std::any::TypeId;

use cfd_eos::{Eos, EosResource, IdealGas, Viscosity};
use cfd_ibm::coupling::{
    self, beta_gidaspow, cd_schiller_naumann, deposit_void_fraction, particle_reynolds,
    terminal_velocity, terminal_velocity_stokes, InterphaseForces, ParticleKinematics, ParticleSet,
};
use cfd_solver::{CfdStatePlugin, IdealGasPlugin};
use cfd_state::{CfdState, PrimVar};
use field_core::{
    FieldDefaultPlugins, FieldRegistry, FvMesh, MeshScheduleSet, UniformMesh, UniformMeshConfig,
    Vec3,
};
use grass_app::prelude::*;
use grass_io::Config;
use grass_multi::{tick_subapp, Multi, MultiAppExt, SubApps};
use grass_scheduler::prelude::*;
use grass_scheduler::{Res, ResMut};
use serde::Deserialize;
use soil_core::{Atom, ParticleSimScheduleSet};
use soil_verlet::VelocityVerletPlugin;

const R_GAS: f64 = 287.058; // matches IdealGas::air()

// ─── Declarative case ────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
struct GasCfg {
    rho: f64,
    p: f64,
    /// Dynamic viscosity μ [Pa·s].
    mu: f64,
}

#[derive(Deserialize, Default)]
struct ParticleCfg {
    radius: f64,
    density: f64,
    x0: f64,
    y0: f64,
    z0: f64,
}

#[derive(Deserialize, Default)]
struct RunCfg {
    dt: f64,
    steps: usize,
    print_every: usize,
    average_frac: f64,
}

#[derive(Deserialize, Default)]
struct GravityCfg {
    gz: f64,
}

#[derive(Deserialize, Default)]
struct ValidationCfg {
    tol_rel_stokes: f64,
    tol_force_balance: f64,
    re_max: f64,
    tol_momentum: f64,
}

// ─── Seam resources shared across the namespace boundary ─────────────────────

/// Total fluid force on the particle for this step (buoyancy + drag), written by
/// the parent from the CFD side and read by the SOIL force system.
#[derive(Clone, Copy, Default)]
struct FluidForce {
    f: Vec3,
}

/// Gravity acceleration applied to the particle (SOIL side).
#[derive(Clone, Copy)]
struct BodyAccel {
    g: Vec3,
}

/// Gravity vector the CFD side reads to assemble the buoyancy (a fluid force).
#[derive(Clone, Copy)]
struct GravityVec {
    g: Vec3,
}

/// Particle radius, on the parent, so `export` can build [`ParticleKinematics`]
/// from the bare `Atom` (which carries no DEM radius).
#[derive(Clone, Copy)]
struct ParticleSpec {
    radius: f64,
}

/// Fixed coupling timestep handed to the CFD side for the momentum-sink integral.
#[derive(Clone, Copy)]
struct CouplingDt(f64);

/// Gas transport the CFD drag system reads.
#[derive(Clone, Copy)]
struct GasProps {
    mu: f64,
}

/// Cumulative drag impulse delivered to the particle, `Σ F_drag Δt`. Its negative
/// must equal the gas momentum the sink accumulated (the conservation check).
#[derive(Clone, Copy, Default)]
struct DragImpulse {
    total: Vec3,
}

// ─── SOIL sub-App: one settling particle ─────────────────────────────────────

/// `Force` phase: force = gravity + the fluid load handed in through the seam. No
/// `AtomPlugin` (no force-zeroing), so this *assigns* the force (idempotent).
fn particle_force(mut atoms: ResMut<Atom>, ff: Res<FluidForce>, body: Res<BodyAccel>) {
    let n = atoms.nlocal as usize;
    for i in 0..n {
        let m = atoms.mass[i] as f64;
        atoms.force[i] = [
            (m * body.g[0] + ff.f[0]) as _,
            (m * body.g[1] + ff.f[1]) as _,
            (m * body.g[2] + ff.f[2]) as _,
        ];
    }
}

fn build_soil(pc: &ParticleCfg, gz: f64, dt: f64) -> App {
    let mut atoms = Atom::new();
    atoms.dt = dt;
    let mass = pc.density * 4.0 / 3.0 * std::f64::consts::PI * pc.radius.powi(3);
    atoms.push_test_atom(0, [pc.x0, pc.y0, pc.z0], pc.radius, mass);
    atoms.nlocal = 1;
    atoms.natoms = 1;

    let mut app = App::new();
    app.add_resource(atoms);
    app.add_resource(FluidForce::default());
    app.add_resource(BodyAccel { g: [0.0, 0.0, gz] });
    app.add_update_system(particle_force, ParticleSimScheduleSet::Force);
    app.add_plugins(VelocityVerletPlugin::new());
    app
}

// ─── FIELD sub-App: the quiescent gas + drag/void/sink coupling system ───────

/// `Output` phase on the CFD sub-App: for each immersed particle sample the local
/// gas, evaluate the Wen–Yu/Gidaspow drag + buoyancy, write it to
/// [`InterphaseForces`] (read back by the parent), and deposit the equal-and-
/// opposite momentum sink into the gas. This is the FIELD half of the seam.
#[allow(clippy::too_many_arguments)]
fn cfd_interphase_system(
    mesh: Res<UniformMesh>,
    reg: Res<FieldRegistry>,
    eos: Res<EosResource>,
    gas: Res<GasProps>,
    grav: Res<GravityVec>,
    dt: Res<CouplingDt>,
    pset: Res<ParticleSet>,
    mut forces: ResMut<InterphaseForces>,
    mut impulse: ResMut<DragImpulse>,
) {
    let eos: &dyn Eos = &*eos.0;
    let mut state = reg.expect_mut::<CfdState>("CfdState not registered");
    let parts = &pset.particles;
    forces.reset(parts.len());
    if parts.is_empty() {
        return;
    }

    let eps_field = deposit_void_fraction(&*mesh, parts, 1e-3);
    let mut drag_on_particle = vec![[0.0f64; 3]; parts.len()];

    for (i, p) in parts.iter().enumerate() {
        let u_gas = coupling::sample_gas_velocity(&*mesh, &state, eos, p.center).unwrap_or([0.0; 3]);
        let rho_f = coupling::sample_gas_density(&*mesh, &state, p.center).unwrap_or(0.0);
        let eps = coupling::sample_void_fraction(&*mesh, &eps_field, p.center).unwrap_or(1.0);

        let rel = [
            u_gas[0] - p.velocity[0],
            u_gas[1] - p.velocity[1],
            u_gas[2] - p.velocity[2],
        ];
        let rel_speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let d = p.diameter();
        // Wen–Yu uses the superficial (ε-weighted) slip Reynolds number.
        let re = particle_reynolds(rho_f, rel_speed, d, gas.mu) * eps;
        let cd = cd_schiller_naumann(re.max(1e-12));

        let beta = beta_gidaspow(eps, rho_f, gas.mu, d, rel_speed, cd);
        let drag = coupling::drag_force_from_beta(beta, p.volume(), eps, rel);

        // Generalized buoyancy: undisturbed pressure-gradient force −ρ_f V g
        // (opposes gravity). Negligible in gas; included for regime-independence.
        let buoy = [
            -rho_f * p.volume() * grav.g[0],
            -rho_f * p.volume() * grav.g[1],
            -rho_f * p.volume() * grav.g[2],
        ];

        forces.force[i] = [drag[0] + buoy[0], drag[1] + buoy[1], drag[2] + buoy[2]];
        drag_on_particle[i] = drag;
        impulse.total[0] += drag[0] * dt.0;
        impulse.total[1] += drag[1] * dt.0;
        impulse.total[2] += drag[2] * dt.0;
    }

    coupling::apply_momentum_sink(&*mesh, &mut state, parts, &drag_on_particle, dt.0);
}

fn build_cfd(gc: &GasCfg, mesh_cfg: UniformMeshConfig, gz: f64, dt: f64) -> App {
    let (rho, p) = (gc.rho, gc.p);
    let t = p / (rho * R_GAS);
    let init = move |_x: Vec3| {
        let eos = IdealGas::air();
        eos.prim_to_cons(&PrimVar::new(rho, 0.0, 0.0, 0.0, p, t))
    };

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh: mesh_cfg })
        .add_plugins(CfdStatePlugin::new(init))
        .add_plugins(IdealGasPlugin);
    app.add_resource(Viscosity::Constant(gc.mu));
    app.add_resource(GasProps { mu: gc.mu });
    app.add_resource(GravityVec { g: [0.0, 0.0, gz] });
    app.add_resource(CouplingDt(dt));
    app.add_resource(ParticleSet::default());
    app.add_resource(InterphaseForces::default());
    app.add_resource(DragImpulse::default());
    app.add_update_system(cfd_interphase_system, MeshScheduleSet::Output);
    app
}

// ─── Parent coupling schedule ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum Phase {
    Export,
    TickCfd,
    Import,
    TickSoil,
}
impl ScheduleSet for Phase {
    fn to_index(&self) -> u32 {
        match self {
            Self::Export => 0,
            Self::TickCfd => 1,
            Self::Import => 2,
            Self::TickSoil => 3,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Export => "Export",
            Self::TickCfd => "TickCfd",
            Self::Import => "Import",
            Self::TickSoil => "TickSoil",
        }
    }
}

fn export_kinematics(world: Multi, spec: Res<ParticleSpec>) {
    let atoms = world.expect_read::<Atom>("soil");
    let n = atoms.nlocal as usize;
    let mut set = world.expect_write::<ParticleSet>("cfd");
    set.particles.clear();
    for i in 0..n {
        set.particles.push(ParticleKinematics {
            center: [atoms.pos[i][0] as f64, atoms.pos[i][1] as f64, atoms.pos[i][2] as f64],
            velocity: [atoms.vel[i][0] as f64, atoms.vel[i][1] as f64, atoms.vel[i][2] as f64],
            radius: spec.radius,
        });
    }
}

fn import_force(world: Multi) {
    let forces = world.expect_read::<InterphaseForces>("cfd");
    let f = forces.force.first().copied().unwrap_or([0.0; 3]);
    drop(forces);
    world.expect_write::<FluidForce>("soil").f = f;
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: settling_sphere <case.toml>");
    let toml_src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let cfg = Config::from_str(&toml_src);

    let gas: GasCfg = cfg.section("gas");
    let pc: ParticleCfg = cfg.section("particle");
    let run: RunCfg = cfg.section("run");
    let grav: GravityCfg = cfg.section("gravity");
    let valid: ValidationCfg = cfg.section("validation");
    let mesh_cfg: UniformMeshConfig = cfg.section("grid");

    let mass = pc.density * 4.0 / 3.0 * std::f64::consts::PI * pc.radius.powi(3);
    let g_eff = (pc.density - gas.rho) / pc.density * grav.gz.abs();
    let weight_full = mass * grav.gz.abs();

    // Analytic references.
    let v_stokes = terminal_velocity_stokes(pc.density, gas.rho, pc.radius, grav.gz.abs(), gas.mu);
    let v_balance = terminal_velocity(pc.density, gas.rho, pc.radius, grav.gz.abs(), gas.mu, cd_schiller_naumann);
    let re_balance = particle_reynolds(gas.rho, v_balance, 2.0 * pc.radius, gas.mu);

    let soil = build_soil(&pc, grav.gz, run.dt);
    let cfd = build_cfd(&gas, mesh_cfg, grav.gz, run.dt);

    let mut parent = App::new();
    parent.add_subapp("soil", soil);
    parent.add_subapp("cfd", cfd);
    parent.add_resource(ParticleSpec { radius: pc.radius });
    parent.add_update_system(export_kinematics, Phase::Export);
    parent.add_update_system(tick_subapp("cfd", 1), Phase::TickCfd);
    parent.add_update_system(import_force, Phase::Import);
    parent.add_update_system(tick_subapp("soil", 1), Phase::TickSoil);
    parent.prepare();

    println!("# Single-sphere settling — unresolved DEM-CFD seam (Wen-Yu/Gidaspow drag)");
    println!(
        "# R = {:.3e} m   d = {:.3e} m   rho_p = {}   rho_f = {}   mu = {:.3e}",
        pc.radius, 2.0 * pc.radius, pc.density, gas.rho, gas.mu
    );
    println!(
        "# m = {:.3e} kg   gz = {}   g_eff = {:.4}   tau = {:.3e} s",
        mass, grav.gz, g_eff, mass / (6.0 * std::f64::consts::PI * gas.mu * pc.radius)
    );
    println!("# v_stokes = {v_stokes:.6}   v_balance(SN) = {v_balance:.6}   Re_balance = {re_balance:.4}");
    println!("# step        v_z [m/s]      |u_gas| [m/s]      Re");

    let avg_start = ((1.0 - run.average_frac) * run.steps as f64) as usize;
    let mut vz_samples: Vec<f64> = Vec::new();
    let (mut last_ugas, mut last_re, mut last_ffluid) = (0.0f64, 0.0f64, [0.0f64; 3]);

    for step in 0..run.steps {
        parent.run();
        let (vz, speed, ffluid) = read_particle(&parent);
        let ugas = gas_speed_at_particle(&parent);
        let re = particle_reynolds(gas.rho, speed, 2.0 * pc.radius, gas.mu);
        last_ugas = ugas;
        last_re = re;
        last_ffluid = ffluid;
        if step >= avg_start {
            vz_samples.push(vz);
        }
        if run.print_every > 0 && (step % run.print_every == 0 || step + 1 == run.steps) {
            println!("{step:>8}   {vz:>12.6}   {ugas:>14.3e}   {re:>8.4}");
        }
    }

    // Cleanup sub-Apps (no-op for these serial apps, but keep the contract).
    if let Some(cell) = parent.get_mut_resource(TypeId::of::<SubApps>()) {
        cell.borrow_mut().downcast_mut::<SubApps>().unwrap().cleanup_all();
    }

    let n = vz_samples.len().max(1) as f64;
    let v_term = (vz_samples.iter().sum::<f64>() / n).abs();

    let rel_stokes = (v_term - v_stokes).abs() / v_stokes;
    let f_fluid_mag =
        (last_ffluid[0].powi(2) + last_ffluid[1].powi(2) + last_ffluid[2].powi(2)).sqrt();
    let force_bal = (f_fluid_mag - weight_full).abs() / weight_full;

    let (gas_mom, drag_impulse) = read_momentum(&parent);
    let mom_err = {
        let diff = [
            gas_mom[0] + drag_impulse[0],
            gas_mom[1] + drag_impulse[1],
            gas_mom[2] + drag_impulse[2],
        ];
        let dn = (diff[0].powi(2) + diff[1].powi(2) + diff[2].powi(2)).sqrt();
        let sc = (drag_impulse[0].powi(2) + drag_impulse[1].powi(2) + drag_impulse[2].powi(2)).sqrt();
        dn / sc.max(1e-30)
    };

    println!("#");
    println!("# ── result ─────────────────────────────────────────────");
    println!("# v_terminal (last {:.0}%):  {v_term:.6} m/s", 100.0 * run.average_frac);
    println!(
        "# v_stokes (1851):          {v_stokes:.6} m/s   rel err {:.2}%  (tol {:.1}%)",
        100.0 * rel_stokes, 100.0 * valid.tol_rel_stokes
    );
    println!("# terminal Re:              {last_re:.4}  (regime gate Re < {})", valid.re_max);
    println!(
        "# force balance |F_f-mg|/mg: {:.2}%  (tol {:.1}%)",
        100.0 * force_bal, 100.0 * valid.tol_force_balance
    );
    println!(
        "# |u_gas| at particle:      {last_ugas:.3e} m/s  ({:.2}% of v_t)",
        100.0 * last_ugas / v_term.max(1e-30)
    );
    println!("# momentum conservation err: {mom_err:.2e}  (tol {:.0e})", valid.tol_momentum);

    let pass_stokes = rel_stokes <= valid.tol_rel_stokes;
    let pass_re = last_re < valid.re_max;
    let pass_balance = force_bal <= valid.tol_force_balance;
    let pass_mom = mom_err <= valid.tol_momentum;

    if pass_stokes && pass_re && pass_balance && pass_mom {
        println!(
            "VALIDATION: PASS  (v_t={v_term:.6} vs Stokes {v_stokes:.6}, {:.2}%<={:.1}%; Re {last_re:.3}<{}; force-bal {:.2}%; mom {mom_err:.1e})",
            100.0 * rel_stokes, 100.0 * valid.tol_rel_stokes, valid.re_max, 100.0 * force_bal
        );
    } else {
        println!("VALIDATION: FAIL  (stokes_ok={pass_stokes} re_ok={pass_re} balance_ok={pass_balance} mom_ok={pass_mom})");
        std::process::exit(1);
    }
}

// ─── Post-run reads via the parent's SubApps (outside any system) ────────────

/// Particle vertical velocity, slip speed |v_p|, and the current fluid force.
fn read_particle(parent: &App) -> (f64, f64, Vec3) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let atom_cell = subs
        .find("soil")
        .unwrap()
        .resource_cell(TypeId::of::<Atom>())
        .unwrap()
        .borrow();
    let atoms = atom_cell.downcast_ref::<Atom>().unwrap();
    let v = [atoms.vel[0][0] as f64, atoms.vel[0][1] as f64, atoms.vel[0][2] as f64];
    let vz = v[2];
    let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    drop(atom_cell);

    let f_cell = subs
        .find("cfd")
        .unwrap()
        .resource_cell(TypeId::of::<InterphaseForces>())
        .unwrap()
        .borrow();
    let f = f_cell
        .downcast_ref::<InterphaseForces>()
        .unwrap()
        .force
        .first()
        .copied()
        .unwrap_or([0.0; 3]);
    (vz, speed, f)
}

/// |u_gas| sampled at the particle's cell in the CFD sub-App.
fn gas_speed_at_particle(parent: &App) -> f64 {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cfd = subs.find("cfd").unwrap();

    let set_cell = cfd.resource_cell(TypeId::of::<ParticleSet>()).unwrap().borrow();
    let Some(p) = set_cell.downcast_ref::<ParticleSet>().unwrap().particles.first().copied() else {
        return 0.0;
    };
    drop(set_cell);

    let reg_cell = cfd.resource_cell(TypeId::of::<FieldRegistry>()).unwrap().borrow();
    let mesh_cell = cfd.resource_cell(TypeId::of::<UniformMesh>()).unwrap().borrow();
    let eos_cell = cfd.resource_cell(TypeId::of::<EosResource>()).unwrap().borrow();
    let reg = reg_cell.downcast_ref::<FieldRegistry>().unwrap();
    let mesh = mesh_cell.downcast_ref::<UniformMesh>().unwrap();
    let eos = &*eos_cell.downcast_ref::<EosResource>().unwrap().0;
    let state = reg.expect::<CfdState>("CfdState");
    coupling::sample_gas_velocity(mesh, &state, eos, p.center)
        .map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt())
        .unwrap_or(0.0)
}

/// Total gas momentum (Σ ρu·V over owned cells) and the accumulated drag impulse.
fn read_momentum(parent: &App) -> (Vec3, Vec3) {
    let subs = parent.get_resource_ref::<SubApps>().unwrap();
    let cfd = subs.find("cfd").unwrap();

    let reg_cell = cfd.resource_cell(TypeId::of::<FieldRegistry>()).unwrap().borrow();
    let mesh_cell = cfd.resource_cell(TypeId::of::<UniformMesh>()).unwrap().borrow();
    let reg = reg_cell.downcast_ref::<FieldRegistry>().unwrap();
    let mesh = mesh_cell.downcast_ref::<UniformMesh>().unwrap();
    let state = reg.expect::<CfdState>("CfdState");
    let mut mom = [0.0f64; 3];
    for c in 0..mesh.n_cells_total() {
        if !mesh.is_local_cell(c) {
            continue;
        }
        let v = mesh.cell_volume(c);
        let u = &state.u[c];
        mom[0] += u.rho_u * v;
        mom[1] += u.rho_v * v;
        mom[2] += u.rho_w * v;
    }
    drop(state);
    drop(reg_cell);
    drop(mesh_cell);

    let imp_cell = cfd.resource_cell(TypeId::of::<DragImpulse>()).unwrap().borrow();
    let imp = imp_cell.downcast_ref::<DragImpulse>().unwrap().total;
    (mom, imp)
}
