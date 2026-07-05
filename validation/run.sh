#!/usr/bin/env bash
# dev_field_efvm validation harness.
#
#   ./validation/run.sh          # smoke: fast analytic/manufactured gates (every CI push/PR)
#   ./validation/run.sh full     # smoke + heavy wait-for-plateau gates (scheduled / on-demand)
#
# The gates below are the maintained validation set documented in README.md
# ("cases that currently deserve to be called validation"). The cylinder-drag
# IBM gate is intentionally heavier — it waits for the drag to plateau (multi-
# minute) — so it is a `full`-mode gate run on a schedule rather than on every
# push, while still receiving automated coverage.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-smoke}"

echo "=== dev_field_efvm validation set (mode=$MODE) ==="

# --- Smoke gates: fast, analytic/manufactured references. Always run. ---

# Isentropic-vortex order of accuracy, first-order HLLC/Godunov path.
cargo test -p cfd_solver --test isentropic_vortex_convergence -- --nocapture

# Isentropic-vortex order of accuracy, MUSCL path (fitted order >= 1.8).
cargo test -p cfd_solver --test isentropic_vortex_muscl_convergence -- --nocapture

# Analytic plane-Poiseuille velocity profile.
cargo test -p cfd_solver --test poiseuille_channel_validation -- --nocapture

# Stokes terminal velocity, low-Reynolds settling sphere (IBM).
cargo run --release -p cfd_ibm --example settling_sphere -- crates/cfd_ibm/examples/settling_sphere_stokes.toml

if [ "$MODE" = "full" ]; then
  # --- Full gates: heavy, wait-for-plateau. Scheduled / on-demand only. ---

  # Schaefer-Turek DFG 2D-1 flow-around-a-cylinder, Re=20, ghost-cell IBM.
  # Gates the marker-integrated drag coefficient against Cd = 5.5795, a flat
  # steady plateau, and lift. Heavier than a unit test by design.
  cargo run --release -p cfd_ibm --example cylinder_drag_validation -- crates/cfd_ibm/examples/cylinder_drag_re20.toml
fi

echo "ALL VALIDATIONS PASSED (mode=$MODE)"
