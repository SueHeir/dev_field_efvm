#!/usr/bin/env python3
"""Run the periodic pulse validation and regenerate its README plot."""

from __future__ import annotations

import re
import subprocess
import sys
from math import log2
from pathlib import Path

import matplotlib.pyplot as plt

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "periodic_advected_pulse"
CONFIG = CASE / "config.toml"


def load_thresholds() -> tuple[float, float, float]:
    """Read, rather than duplicate, the executable's declared acceptance limits."""
    with CONFIG.open("rb") as f:
        cfg = tomllib.load(f)
    convergence = cfg["convergence"]
    return (
        float(convergence["design_order"]),
        float(convergence["order_tol"]),
        float(cfg["conservation"]["mass_tol"]),
    )


def regression_order(ns: list[int], errors: list[float]) -> float:
    """Independently recompute the log-log least-squares slope used by the gate."""
    xs = [log2(n) for n in ns]
    # Resolution doubles in the declared case, so log2(N) keeps this calculation
    # simple while giving the same least-squares slope as natural logarithms.
    ys = [log2(e) for e in errors]
    count = len(xs)
    sx, sy = sum(xs), sum(ys)
    return -(count * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / (
        count * sum(x * x for x in xs) - sx * sx
    )


def main() -> int:
    cmd = [
        "cargo",
        "run",
        "--release",
        "-p",
        "dev_field_efvm",
        "--bin",
        "periodic_advected_pulse",
        "--",
        str(CONFIG.relative_to(ROOT)),
    ]
    run = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(run.stdout, end="")
    if run.returncode != 0:
        return run.returncode

    rows = []
    for line in run.stdout.splitlines():
        m = re.match(r"\s*(\d+)\s+([0-9.eE+-]+)\s+(?:-|[0-9.]+)\s+([0-9.eE+-]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    order_m = re.search(r"fitted order = ([0-9.]+).*max mass drift = ([0-9.eE+-]+)", run.stdout)
    if len(rows) < 2 or not order_m:
        raise SystemExit("could not parse validation output")

    ns = [r[0] for r in rows]
    errs = [r[1] for r in rows]
    mass = [r[2] for r in rows]
    order = float(order_m.group(1))
    max_mass = float(order_m.group(2))
    design_order, order_tol, mass_tol = load_thresholds()
    recomputed_order = regression_order(ns, errs)
    monotone = all(b < a for a, b in zip(errs, errs[1:]))
    independent_pass = (
        monotone
        and design_order - order_tol <= recomputed_order <= design_order + order_tol
        and max(mass) <= mass_tol
    )
    if not independent_pass:
        raise SystemExit(
            "independent plot-gate check failed: "
            f"monotone={monotone}, order={recomputed_order:.3f}, max_mass={max(mass):.3e}"
        )

    plot_dir = CASE / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, (ax, mass_ax) = plt.subplots(1, 2, figsize=(9.0, 3.5), dpi=150)
    ax.loglog(ns, errs, "o-", label="measured L1 density error", color="#4477aa")
    ref = [errs[0] * (ns[0] / n) for n in ns]
    lo_ref = [errs[0] * (ns[0] / n) ** (design_order - order_tol) for n in ns]
    hi_ref = [errs[0] * (ns[0] / n) ** (design_order + order_tol) for n in ns]
    ax.loglog(ns, ref, "--", label=f"design O(N^-{design_order:g})", color="#66aa55")
    ax.fill_between(ns, lo_ref, hi_ref, color="#66aa55", alpha=0.16,
                    label=f"accepted slope {design_order - order_tol:.2f}–{design_order + order_tol:.2f}")
    ax.set_title("Accuracy gate")
    ax.set_xlabel("grid resolution N")
    ax.set_ylabel("L1 density error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    ax.text(
        0.04,
        0.05,
        f"fit {order:.3f}; independently recomputed {recomputed_order:.3f}\n"
        f"monotone refinement: PASS\n"
        f"order gate: PASS",
        transform=ax.transAxes,
        fontsize=7.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    mass_ax.semilogy(ns, mass, "o-", color="#cc6677", label="measured relative mass drift")
    mass_ax.axhline(mass_tol, color="#332288", linestyle="--", label=f"tolerance {mass_tol:.0e}")
    mass_ax.fill_between(ns, mass_tol, [mass_tol * 10 for _ in ns], color="#cc6677", alpha=0.12)
    mass_ax.set_title("Conservation gate")
    mass_ax.set_xlabel("grid resolution N")
    mass_ax.set_ylabel("relative mass drift")
    mass_ax.grid(True, which="both", alpha=0.25)
    mass_ax.legend(loc="lower left", fontsize=7.5)
    mass_ax.text(
        0.04,
        0.92,
        f"max {max_mass:.3e} ≤ {mass_tol:.0e}: PASS",
        transform=mass_ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "validation.png")
    plt.close(fig)

    (CASE / "README.md").write_text(
        "# Periodic Advected Pulse\n\n"
        "![validation comparison](plots/validation.png)\n\n"
        "The left panel compares the measured L1 error with the declared first-order "
        "acceptance band (including the monotone-refinement check); the right panel compares "
        "each measured mass drift with its declared tolerance.\n\n"
        f"Periodic FIELD wrap gate: fitted order {order:.3f} (independently recomputed "
        f"{recomputed_order:.3f}) against {design_order:.1f} +/- {order_tol:.2f}, max mass drift "
        f"{max_mass:.3e} <= {mass_tol:.1e}; PASS.\n\n"
        "## Reference and limits\n\n"
        "The transported smooth contact/entropy wave is the analytical Euler reference, as "
        "described for linear advection and contact waves in R. J. LeVeque, *Finite Volume "
        "Methods for Hyperbolic Problems* (Cambridge, 2002), DOI: "
        "[10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253). "
        "This check exercises a serial structured-mesh wrap only; it does not validate AMR, "
        "MPI-rank periodic exchange, viscous reconstruction, or corner ghosts. The implementation "
        "and validation harness were produced with AI assistance; the exact-solution comparison, "
        "independent sweep-side recomputation, and review are the available checks, not a claim "
        "of broad solver verification.\n",
        encoding="utf-8",
    )
    print(f"ALL CHECKS PASSED: order={order:.3f}, max_mass_drift={max_mass:.3e}, rows={rows}")
    _ = mass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
