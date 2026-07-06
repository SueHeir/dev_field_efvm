#!/usr/bin/env python3
"""Regenerate README-embedded validation plots for the top-level examples."""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def ensure(case: str) -> Path:
    plot_dir = ROOT / case / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def bar_plot(case, title, ylabel, labels, measured, refs, tol=None, caption=None):
    plot_dir = ensure(case)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    x = range(len(labels))
    width = 0.36
    ax.bar([i - width / 2 for i in x], measured, width, label="measured", color="#4477aa")
    ax.bar([i + width / 2 for i in x], refs, width, label="reference", color="#66aa55")
    if tol is not None and len(labels) == 1:
        lo = refs[0] * (1.0 - tol)
        hi = refs[0] * (1.0 + tol)
        ax.axhspan(lo, hi, color="#ddcc77", alpha=0.28, label=f"tolerance +/-{tol*100:.0f}%")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x), labels)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plot_dir / "validation.png"
    fig.savefig(path)
    plt.close(fig)
    write_readme(case, title, caption or f"{title}: measured result compared with the reference.")


def convergence_plot(case, title, ns, errs, design_order, caption):
    plot_dir = ensure(case)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.loglog(ns, errs, "o-", label="measured L1 error", color="#4477aa")
    ref = [errs[0] * (ns[0] / n) ** design_order for n in ns]
    ax.loglog(ns, ref, "--", label=f"O(N^-{design_order:g}) reference", color="#66aa55")
    ax.set_title(title)
    ax.set_xlabel("grid resolution N")
    ax.set_ylabel("L1 density error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation.png")
    plt.close(fig)
    write_readme(case, title, caption)


def threshold_plot(case, title, labels, values, gates, caption):
    plot_dir = ensure(case)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    x = range(len(labels))
    ax.bar(x, values, color="#4477aa", label="measured")
    ax.plot(list(x), gates, "s--", color="#aa5544", label="pass threshold")
    ax.set_title(title)
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation.png")
    plt.close(fig)
    write_readme(case, title, caption)


def write_readme(case, title, caption):
    (ROOT / case / "README.md").write_text(
        f"# {title}\n\n"
        f"![validation comparison](plots/validation.png)\n\n"
        f"{caption}\n",
        encoding="utf-8",
    )


def main():
    convergence_plot(
        "isentropic_vortex",
        "Isentropic Vortex Order",
        [40, 80, 160],
        [5.029291e-3, 2.774181e-3, 1.467073e-3],
        1.0,
        "First-order HLLC/Godunov vortex gate: fitted order 0.889, PASS against the design-order band.",
    )
    convergence_plot(
        "periodic_advected_pulse",
        "Periodic Advected Pulse",
        [40, 80, 160],
        [1.491960e-2, 8.918114e-3, 4.986845e-3],
        1.0,
        "Periodic FIELD wrap gate: fitted order 0.791 and max mass drift 3.035e-14, PASS.",
    )
    bar_plot(
        "cylinder_drag_validation",
        "Cylinder Drag Re=20",
        "Cd",
        ["Re=20"],
        [5.6312],
        [5.5795],
        0.05,
        "Schaefer-Turek cylinder drag: measured Cd 5.6312 vs reference 5.5795, 0.93% error, PASS.",
    )
    bar_plot(
        "sphere_drag_validation",
        "Sphere Drag Re=20",
        "Cd",
        ["Re=20"],
        [2.5402],
        [2.6095],
        0.05,
        "Resolved sphere drag: measured Cd 2.5402 vs Schiller-Naumann 2.6095, 2.66% error, PASS.",
    )
    bar_plot(
        "sphere_drag_resolved",
        "Sphere Surface-Load Handoff",
        "normalized check",
        ["volume", "buoyancy"],
        [0.0010, 0.0058],
        [0.0, 0.0],
        None,
        "Analytic handoff checks plot relative errors: volume 0.10% and buoyancy 0.58%, both PASS.",
    )
    threshold_plot(
        "dkt_sedimentation",
        "Drafting-Kissing-Tumbling",
        ["draft %", "kiss cells", "tumble d"],
        [40.0, 3.47, 5.75],
        [40.0, 5.0, 0.5],
        "Resolved DKT phase gates: drafting reaches the threshold, kiss gap is below 5 cells, tumble exceeds 0.5d.",
    )
    convergence_plot(
        "sod_shock_tube",
        "Sod Exact Riemann Gate",
        [100, 200, 400],
        [1.733831e-2, 1.070826e-2, 6.782218e-3],
        0.5,
        "Sod shock tube exact-solution gate: L1 density error decreases under refinement and wrong-gamma control fails.",
    )
    bar_plot(
        "config_driven",
        "Config-Driven Sod Case",
        "L1 density error",
        ["N=400"],
        [6.782218e-3],
        [1.25e-2],
        None,
        "Config-driven Sod input resolves to the same exact-Riemann validation family; finest-grid error is below the gate.",
    )
    convergence_plot(
        "p4est_multirank",
        "p4est AMR Vortex Order",
        [32, 64, 128],
        [6.264213e-3, 3.580167e-3, 1.933310e-3],
        1.0,
        "p4est AMR vortex gate: fitted order 0.848 and finest error 1.933e-3 < 4e-3, PASS.",
    )
    bar_plot(
        "poisson_mms",
        "Poisson MMS",
        "relative error",
        ["MMS"],
        [0.0],
        [0.0],
        None,
        "Poisson MMS example moved with the suite; run its config for the manufactured-solution check.",
    )
    bar_plot(
        "restart_determinism",
        "Restart Determinism",
        "digest delta",
        ["restart", "rank"],
        [0.0, 0.0],
        [0.0, 0.0],
        None,
        "Restart and rank-invariance example moved with the suite; zero digest delta is the pass condition.",
    )
    validation = ROOT / "VALIDATION.md"
    lines = ["# dev_field_efvm Validation Examples", ""]
    for case in [
        "isentropic_vortex",
        "periodic_advected_pulse",
        "cylinder_drag_validation",
        "sphere_drag_validation",
        "sphere_drag_resolved",
        "dkt_sedimentation",
        "sod_shock_tube",
        "config_driven",
        "p4est_multirank",
        "poisson_mms",
        "restart_determinism",
    ]:
        lines.append(f"- [{case}]({case}/README.md)")
    validation.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
