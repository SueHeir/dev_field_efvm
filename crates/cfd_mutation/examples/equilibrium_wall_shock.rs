//! VALIDATION: a reflected shock in equilibrium air, through the unmodified
//! efvm solver with Mutation++ as its equation of state.
//!
//! Air at 100 Pa and 300 K flows at 5 km/s into a reflecting wall at x = 0. A
//! shock stands off the wall and moves upstream, leaving hot gas at rest
//! behind it. The exact answer is the equilibrium Rankine–Hugoniot jump whose
//! post-shock gas is at rest in the wall frame, computed independently of the
//! solver by `cfd_mutation::normal_shock` (itself checked against Mutation++'s
//! `mppshock` in the tests). The run passes if the solver's plateau and shock
//! speed match it.
//!
//! The perfect-gas answer is printed for contrast: that gas cannot
//! dissociate, so it puts the post-shock temperature far too high.
//!
//! ```sh
//! MUTATIONPP_DIR=... MPP_DATA_DIRECTORY=.../Mutationpp/data \
//!   cargo run --release --example equilibrium_wall_shock
//! ```

use std::sync::Arc;
use std::time::Instant;

use cfd_boundary::FnBoundary;
use cfd_eos::EosResource;
use cfd_mutation::{normal_shock, EquilibriumEos, NormalShock};
use test_cfd::prelude::*;

const P1: f64 = 100.0;
const T1: f64 = 300.0;
const U_WALL: f64 = 5000.0; // speed of the incoming gas toward the wall
const LENGTH: f64 = 0.06;
const CELLS: usize = 60;
const T_END: f64 = 9.0e-5;

/// Shock speed (wall frame) and post-shock state for gas brought to rest by a
/// reflected shock: in the shock frame the gas enters at U_WALL + Vs and must
/// leave at Vs.
fn exact(eos: &EquilibriumEos) -> (f64, NormalShock) {
    let resid = |vs: f64| {
        let s = normal_shock(eos, P1, T1, U_WALL + vs).expect("normal shock");
        (s.u2 - vs, s)
    };
    let (mut a, mut b) = (200.0, 2000.0);
    let (mut fa, mut fb) = (resid(a).0, resid(b).0);
    for _ in 0..100 {
        let c = b - fb * (b - a) / (fb - fa);
        let (fc, s) = resid(c);
        if fc.abs() < 1e-9 {
            return (c, s);
        }
        (a, fa, b, fb) = (b, fb, c, fc);
    }
    panic!("reflected-shock speed did not converge");
}

struct Run {
    x: Vec<f64>,
    prim: Vec<PrimVar>,
    steps: u64,
    time: f64,
    seconds: f64,
}

fn run(eos: Arc<dyn Eos>) -> Run {
    // EquilibriumEos recomputes density from (p, T); the ideal gas reads it.
    let rho1 = P1 / (IdealGas::air().r_gas * T1);
    let inflow = eos.prim_to_cons(&PrimVar::new(rho1, -U_WALL, 0.0, 0.0, P1, T1));
    let mesh = UniformMeshConfig {
        nx: CELLS,
        ny: 1,
        nz: 1,
        ng: 2,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [LENGTH, 0.01, 0.01],
        y_edges: None,
        z_edges: None,
    };
    let bcs = BoundaryRegistry::default()
        .with(BoundarySide::XLo, SlipWall)
        .with(
            BoundarySide::XHi,
            FnBoundary(move |_: &ConsVar, _: &_, _: &dyn Eos| inflow),
        )
        .with_axis(1, SlipWall)
        .with_axis(2, SlipWall);

    let mut app = App::new();
    app.add_plugins(FieldDefaultPlugins { mesh }).add_plugins(
        CfdDefaultPlugins::new(move |_| inflow)
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
    // The only change from an ideal-gas case: replace the EOS resource.
    app.add_resource(EosResource(eos.clone()));
    app.prepare();

    let start = Instant::now();
    loop {
        let t = app.get_resource_ref::<SolverState>().unwrap().time;
        if t >= T_END {
            break;
        }
        for _ in 0..3 {
            app.run(); // one RK3 stage per call
        }
    }
    let seconds = start.elapsed().as_secs_f64();

    let mesh = app.get_resource_ref::<UniformMesh>().unwrap();
    let reg = app.get_resource_ref::<FieldRegistry>().unwrap();
    let state = reg.get::<CfdState>().unwrap();
    let ss = app.get_resource_ref::<SolverState>().unwrap();
    let (x, prim) = (0..CELLS)
        .map(|i| {
            let c = mesh.idx(i, 0, 0);
            (mesh.cell_centroid(c)[0], eos.cons_to_prim(&state.u[c]))
        })
        .unzip();
    Run {
        x,
        prim,
        steps: ss.step,
        time: ss.time,
        seconds,
    }
}

/// Shock position (where pressure crosses the mid-value) and the mean plateau
/// state over the middle half of the region between wall and shock.
fn measure(r: &Run, p2_guess: f64) -> (f64, PrimVar) {
    let mid = 0.5 * (P1 + p2_guess);
    let i_s = (0..r.x.len())
        .rev()
        .find(|&i| r.prim[i].p > mid)
        .expect("no shock found");
    let w = (r.prim[i_s].p - mid) / (r.prim[i_s].p - r.prim[i_s + 1].p);
    let xs = r.x[i_s] + w * (r.x[i_s + 1] - r.x[i_s]);
    let (lo, hi) = (i_s / 4, 3 * i_s / 4);
    let n = (hi - lo) as f64;
    let mut m = PrimVar::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in &r.prim[lo..hi] {
        m.rho += p.rho / n;
        m.u += p.u / n;
        m.p += p.p / n;
        m.t += p.t / n;
    }
    (xs, m)
}

fn main() {
    let eq = Arc::new(EquilibriumEos::new("air_11").expect("air_11 (set MPP_DATA_DIRECTORY)"));
    let (vs, ex) = exact(&eq);
    println!("Reflected shock in equilibrium air: {P1} Pa, {T1} K, {U_WALL} m/s into a wall");
    println!(
        "exact (equilibrium RH):  Vs = {vs:7.1} m/s  p2 = {:8.0} Pa  T2 = {:6.0} K  rho2 = {:.4e}",
        ex.post.p, ex.post.t, ex.post.rho
    );

    let r = run(eq.clone());
    let (xs, m) = measure(&r, ex.post.p);
    let vs_num = xs / r.time;
    println!(
        "efvm + Mutation++:       Vs = {vs_num:7.1} m/s  p2 = {:8.0} Pa  T2 = {:6.0} K  rho2 = {:.4e}  u2 = {:.1} m/s",
        m.p, m.t, m.rho, m.u
    );
    println!(
        "  {CELLS} cells, {} steps to t = {:.2e} s in {:.1} s ({:.1} us per cell-step)",
        r.steps,
        r.time,
        r.seconds,
        1e6 * r.seconds / (r.steps as f64 * CELLS as f64)
    );

    // Cost of the same run with efvm's ideal gas (its shock is faster and
    // leaves this short domain, so only the timing is used).
    let ri = run(Arc::new(IdealGas::air()));
    println!(
        "  ideal-gas run of the same case: {} steps in {:.3} s ({:.2} us per cell-step), so Mutation++ costs {:.0}x",
        ri.steps,
        ri.seconds,
        1e6 * ri.seconds / (ri.steps as f64 * CELLS as f64),
        (r.seconds / r.steps as f64) / (ri.seconds / ri.steps as f64)
    );

    // Contrast: the same reflected shock in efvm's perfect gas (closed form).
    // In the incoming gas's frame this is a piston at U_WALL, whose shock moves
    // at W = (γ+1)/4·U + sqrt(((γ+1)/4·U)² + a1²).
    let ideal = IdealGas::air();
    let (g, a1) = (ideal.gamma, (ideal.gamma * ideal.r_gas * T1).sqrt());
    let w = (g + 1.0) / 4.0 * U_WALL + (((g + 1.0) / 4.0 * U_WALL).powi(2) + a1 * a1).sqrt();
    let m2 = (w / a1).powi(2);
    let p2i = P1 * (1.0 + 2.0 * g / (g + 1.0) * (m2 - 1.0));
    let rr = (g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0);
    let t2i = T1 * (p2i / P1) / rr;
    println!(
        "perfect gas (exact):     Vs = {:7.1} m/s  p2 = {p2i:8.0} Pa  T2 = {t2i:6.0} K  rho2 = {:.4e}",
        w - U_WALL,
        P1 / (ideal.r_gas * T1) * rr
    );

    let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
    let checks = [
        ("shock speed", rel(vs_num, vs), 0.02),
        ("pressure", rel(m.p, ex.post.p), 0.01),
        ("temperature", rel(m.t, ex.post.t), 0.01),
        ("density", rel(m.rho, ex.post.rho), 0.02),
    ];
    let mut ok = m.u.abs() < 0.01 * U_WALL;
    println!("  post-shock velocity {:.1} m/s (should be ~0)", m.u);
    for (name, err, tol) in checks {
        let pass = err <= tol;
        ok &= pass;
        println!(
            "  {name:12} error {:6.3}%  (tolerance {:.0}%) {}",
            100.0 * err,
            100.0 * tol,
            if pass { "ok" } else { "FAIL" }
        );
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    if !ok {
        std::process::exit(1);
    }
}
