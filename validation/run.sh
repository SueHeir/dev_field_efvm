#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== dev_field_efvm validation set ==="
cargo test -p cfd_solver --test isentropic_vortex_convergence -- --nocapture
cargo test -p cfd_solver --test poiseuille_channel_validation -- --nocapture
cargo run --release -p cfd_ibm --example settling_sphere -- crates/cfd_ibm/examples/settling_sphere_stokes.toml
echo "ALL VALIDATIONS PASSED"
