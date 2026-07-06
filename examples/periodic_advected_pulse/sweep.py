#!/usr/bin/env python3
"""Run the periodic pulse validation and regenerate its README plot."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "examples" / "periodic_advected_pulse"
CONFIG = CASE / "config.toml"


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

    plot_dir = CASE / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=150)
    ax.loglog(ns, errs, "o-", label="measured L1 density error", color="#4477aa")
    ref = [errs[0] * (ns[0] / n) for n in ns]
    ax.loglog(ns, ref, "--", label="O(N^-1) reference", color="#66aa55")
    ax.set_title("Periodic Advected Pulse")
    ax.set_xlabel("grid resolution N")
    ax.set_ylabel("L1 density error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    txt = f"fitted order {order:.3f}\nmax mass drift {max_mass:.1e}"
    ax.text(0.04, 0.06, txt, transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation.png")
    plt.close(fig)

    (CASE / "README.md").write_text(
        "# Periodic Advected Pulse\n\n"
        "![validation comparison](plots/validation.png)\n\n"
        f"Periodic FIELD wrap gate: fitted order {order:.3f} against the first-order reference, "
        f"max mass drift {max_mass:.3e}; PASS.\n",
        encoding="utf-8",
    )
    print(f"ALL CHECKS PASSED: order={order:.3f}, max_mass_drift={max_mass:.3e}, rows={rows}")
    _ = mass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
