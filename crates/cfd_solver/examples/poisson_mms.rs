//! **FIELD global-solve proof: Poisson MMS with PCG + geometric multigrid.**
//!
//! This example validates that the FIELD substrate exposes enough equation-
//! agnostic structure to host long-range/global mesh physics above it. The solver
//! code lives here, in the consumer example: `field_core` supplies only
//! `CellVector`, `LinearOperator`, `Vector`, and `MeshHierarchy`.
//!
//! The manufactured reference is analytic, not a self-comparison:
//!
//! ```text
//! u(x,y) = sin(pi x / L) sin(pi y / L)
//! -lap u = 2(pi/L)^2 u,        u = 0 on the boundary
//! ```
//!
//! The run sweeps a declared refinement sequence, solves each grid with
//! preconditioned conjugate gradient using a geometric-multigrid V-cycle
//! preconditioner, and fails unless the L2 error decreases at second order.
//!
//! ```text
//! cargo run --release -p cfd_solver --example poisson_mms -- \
//!     crates/cfd_solver/examples/poisson_mms.toml
//! ```

use std::f64::consts::PI;

use field_core::{
    CellVector, FvMesh, LinearOperator, MeshHierarchy, StructuredMesh, UniformMesh,
    UniformMeshConfig, Vector,
};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
struct Config {
    problem: ProblemCfg,
    solver: SolverCfg,
    convergence: ConvergenceCfg,
}

#[derive(Debug, Clone, Deserialize)]
struct ProblemCfg {
    /// Square domain side length.
    domain: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct SolverCfg {
    max_iters: usize,
    relative_residual_tol: f64,
    jacobi_omega: f64,
    pre_smooth: usize,
    post_smooth: usize,
    coarse_smooth: usize,
    /// Coarsening stops when either active axis would fall below this count.
    min_coarse_cells: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct ConvergenceCfg {
    /// Cells per side for each grid in the refinement sequence.
    resolutions: Vec<usize>,
    expected_order: f64,
    order_tol: f64,
    /// Absolute L2 error limit on the finest grid.
    finest_l2_max: f64,
}

/// Matrix-free SPD operator for the cell-centered finite-volume discretization
/// of `A = -lap` on a square 2-D grid with homogeneous Dirichlet faces.
struct DirichletPoisson<'m> {
    mesh: &'m UniformMesh,
}

impl<'m> DirichletPoisson<'m> {
    fn diagonal_at(&self, i: usize, j: usize) -> f64 {
        let hx = self.mesh.spacing(0, i);
        let hy = self.mesh.spacing(1, j);
        let d = self.mesh.dims();
        let x_faces = if i == 0 { 2.0 } else { 1.0 } + if i + 1 == d[0] { 2.0 } else { 1.0 };
        let y_faces = if j == 0 { 2.0 } else { 1.0 } + if j + 1 == d[1] { 2.0 } else { 1.0 };
        x_faces / (hx * hx) + y_faces / (hy * hy)
    }
}

impl<'m> LinearOperator for DirichletPoisson<'m> {
    type Vector = CellVector;

    fn apply(&self, x: &CellVector, y: &mut CellVector) {
        let m = self.mesh;
        let d = m.dims();
        y.fill(0.0);
        for j in 0..d[1] {
            for i in 0..d[0] {
                let c = m.idx(i, j, 0);
                let uc = x.get(c);
                let hx = m.spacing(0, i);
                let hy = m.spacing(1, j);

                let mut au = 0.0;
                au += if i == 0 {
                    2.0 * uc / (hx * hx)
                } else {
                    (uc - x.get(m.idx(i - 1, j, 0))) / (hx * hx)
                };
                au += if i + 1 == d[0] {
                    2.0 * uc / (hx * hx)
                } else {
                    (uc - x.get(m.idx(i + 1, j, 0))) / (hx * hx)
                };
                au += if j == 0 {
                    2.0 * uc / (hy * hy)
                } else {
                    (uc - x.get(m.idx(i, j - 1, 0))) / (hy * hy)
                };
                au += if j + 1 == d[1] {
                    2.0 * uc / (hy * hy)
                } else {
                    (uc - x.get(m.idx(i, j + 1, 0))) / (hy * hy)
                };
                y.set(c, au);
            }
        }
    }
}

fn exact_u(domain: f64, x: f64, y: f64) -> f64 {
    (PI * x / domain).sin() * (PI * y / domain).sin()
}

fn exact_rhs(domain: f64, x: f64, y: f64) -> f64 {
    2.0 * (PI / domain).powi(2) * exact_u(domain, x, y)
}

fn base_config(n: usize, domain: f64) -> UniformMeshConfig {
    UniformMeshConfig {
        nx: n,
        ny: n,
        nz: 1,
        ng: 1,
        bounds_lo: [0.0, 0.0, 0.0],
        bounds_hi: [domain, domain, domain / n as f64],
        y_edges: None,
        z_edges: None,
    }
}

fn hierarchy_levels(n: usize, min_coarse_cells: usize) -> usize {
    let mut levels = 1usize;
    let mut cur = n;
    while cur > min_coarse_cells && cur.is_multiple_of(2) {
        cur /= 2;
        levels += 1;
    }
    levels
}

fn fill_rhs(mesh: &UniformMesh, domain: f64) -> CellVector {
    let mut b = CellVector::from_mesh(mesh);
    let d = mesh.dims();
    for j in 0..d[1] {
        for i in 0..d[0] {
            let c = mesh.idx(i, j, 0);
            let p = mesh.cell_centroid(c);
            b.set(c, exact_rhs(domain, p[0], p[1]));
        }
    }
    b
}

fn residual(op: &DirichletPoisson<'_>, b: &CellVector, x: &CellVector) -> CellVector {
    let mut ax = CellVector::zeros_like(x);
    op.apply(x, &mut ax);
    let mut r = CellVector::zeros_like(x);
    r.copy_from(b);
    r.axpy(-1.0, &ax);
    r
}

fn smooth_jacobi(
    op: &DirichletPoisson<'_>,
    b: &CellVector,
    x: &mut CellVector,
    omega: f64,
    sweeps: usize,
) {
    let d = op.mesh.dims();
    for _ in 0..sweeps {
        let r = residual(op, b, x);
        for j in 0..d[1] {
            for i in 0..d[0] {
                let c = op.mesh.idx(i, j, 0);
                x.set(c, x.get(c) + omega * r.get(c) / op.diagonal_at(i, j));
            }
        }
    }
}

fn v_cycle(
    level: usize,
    hierarchy: &MeshHierarchy,
    b: &CellVector,
    x: &mut CellVector,
    cfg: &SolverCfg,
) {
    let op = DirichletPoisson {
        mesh: hierarchy.level(level),
    };
    if level + 1 == hierarchy.n_levels() {
        smooth_jacobi(&op, b, x, cfg.jacobi_omega, cfg.coarse_smooth);
        return;
    }

    smooth_jacobi(&op, b, x, cfg.jacobi_omega, cfg.pre_smooth);

    let r = residual(&op, b, x);
    let coarse_mesh = hierarchy.level(level + 1);
    let mut coarse_b = CellVector::from_mesh(coarse_mesh);
    hierarchy.restrict(level, r.as_slice(), coarse_b.as_mut_slice());

    let mut coarse_x = CellVector::zeros_like(&coarse_b);
    v_cycle(level + 1, hierarchy, &coarse_b, &mut coarse_x, cfg);

    let mut correction = CellVector::zeros_like(x);
    hierarchy.prolong(level, coarse_x.as_slice(), correction.as_mut_slice());
    x.axpy(1.0, &correction);

    smooth_jacobi(&op, b, x, cfg.jacobi_omega, cfg.post_smooth);
}

fn apply_preconditioner(hierarchy: &MeshHierarchy, r: &CellVector, cfg: &SolverCfg) -> CellVector {
    let mut z = CellVector::zeros_like(r);
    v_cycle(0, hierarchy, r, &mut z, cfg);
    z
}

fn pcg(
    op: &DirichletPoisson<'_>,
    hierarchy: &MeshHierarchy,
    b: &CellVector,
    cfg: &SolverCfg,
) -> (CellVector, usize, f64) {
    let mut x = CellVector::zeros_like(b);
    let mut r = residual(op, b, &x);
    let b_norm = b.norm().max(1e-300);
    let mut rel = r.norm() / b_norm;
    if rel <= cfg.relative_residual_tol {
        return (x, 0, rel);
    }

    let mut z = apply_preconditioner(hierarchy, &r, cfg);
    let mut p = CellVector::zeros_like(&z);
    p.copy_from(&z);
    let mut rz = r.dot(&z);
    let mut ap = CellVector::zeros_like(b);

    for iter in 1..=cfg.max_iters {
        op.apply(&p, &mut ap);
        let denom = p.dot(&ap);
        assert!(denom > 0.0, "PCG encountered non-SPD search direction");
        let alpha = rz / denom;
        x.axpy(alpha, &p);
        r.axpy(-alpha, &ap);
        rel = r.norm() / b_norm;
        if rel <= cfg.relative_residual_tol {
            return (x, iter, rel);
        }

        z = apply_preconditioner(hierarchy, &r, cfg);
        let rz_new = r.dot(&z);
        let beta = rz_new / rz;
        p.scale(beta);
        p.axpy(1.0, &z);
        rz = rz_new;
    }

    (x, cfg.max_iters, rel)
}

fn l2_error(mesh: &UniformMesh, domain: f64, x: &CellVector) -> f64 {
    let d = mesh.dims();
    let (mut num, mut vol) = (0.0, 0.0);
    for j in 0..d[1] {
        for i in 0..d[0] {
            let c = mesh.idx(i, j, 0);
            let p = mesh.cell_centroid(c);
            let e = x.get(c) - exact_u(domain, p[0], p[1]);
            let cv = mesh.cell_volume(c);
            num += e * e * cv;
            vol += cv;
        }
    }
    (num / vol).sqrt()
}

fn fitted_order(ns: &[usize], errs: &[f64]) -> f64 {
    let xs: Vec<f64> = ns.iter().map(|&n| (n as f64).ln()).collect();
    let ys: Vec<f64> = errs.iter().map(|&e| e.ln()).collect();
    let m = xs.len() as f64;
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| x * y).sum();
    -(m * sxy - sx * sy) / (m * sxx - sx * sx)
}

fn run_case(cfg: &Config, n: usize) -> (f64, usize, f64, usize) {
    let levels = hierarchy_levels(n, cfg.solver.min_coarse_cells);
    let base = base_config(n, cfg.problem.domain);
    let hierarchy = MeshHierarchy::from_base(&base, levels);
    let mesh = hierarchy.finest();
    let op = DirichletPoisson { mesh };
    let b = fill_rhs(mesh, cfg.problem.domain);
    let (x, iters, rel) = pcg(&op, &hierarchy, &b, &cfg.solver);
    assert!(
        rel <= cfg.solver.relative_residual_tol,
        "N={n} PCG relative residual {rel:e} exceeded tolerance {:e}",
        cfg.solver.relative_residual_tol
    );
    (l2_error(mesh, cfg.problem.domain, &x), iters, rel, levels)
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: poisson_mms <config.toml>");
    let text = std::fs::read_to_string(&path).expect("read config");
    let cfg: Config = toml::from_str(&text).expect("parse config");

    let mut ns = cfg.convergence.resolutions.clone();
    ns.sort_unstable();
    assert!(
        ns.len() >= 3,
        "need at least 3 resolutions for an order fit"
    );

    println!("FIELD Poisson MMS: PCG + geometric-multigrid preconditioner");
    println!(
        "analytic reference: u=sin(pi x/L) sin(pi y/L), L={}",
        cfg.problem.domain
    );
    println!("   N   levels   iters   rel_resid       L2 error       order");

    let mut errs: Vec<f64> = Vec::with_capacity(ns.len());
    for (k, &n) in ns.iter().enumerate() {
        let (err, iters, rel, levels) = run_case(&cfg, n);
        if k == 0 {
            println!("{n:4}   {levels:6}   {iters:5}   {rel:10.3e}   {err:12.6e}      --");
        } else {
            let p = (errs[k - 1] / err).log2();
            println!("{n:4}   {levels:6}   {iters:5}   {rel:10.3e}   {err:12.6e}   {p:6.3}");
        }
        errs.push(err);
    }

    let order = fitted_order(&ns, &errs);
    let lo = cfg.convergence.expected_order - cfg.convergence.order_tol;
    let hi = cfg.convergence.expected_order + cfg.convergence.order_tol;
    let finest = *errs.last().unwrap();
    let pass = errs.windows(2).all(|w| w[1] < w[0])
        && (lo..=hi).contains(&order)
        && finest <= cfg.convergence.finest_l2_max;

    println!(
        "\nfitted order = {:.3} (expected {} +/- {}), finest L2 = {:.6e} (limit {:.6e}) -> {}",
        order,
        cfg.convergence.expected_order,
        cfg.convergence.order_tol,
        finest,
        cfg.convergence.finest_l2_max,
        if pass { "PASS" } else { "FAIL" }
    );

    if !pass {
        std::process::exit(1);
    }
}
