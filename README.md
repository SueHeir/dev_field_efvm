# dev_field_efvm

> **Development-stage CFD tier. Not peer-reviewed.**
>
> This repository is a readable, inspectable proof that a compressible finite-volume
> CFD physics layer can ride the FIELD mesh substrate on GRASS. It is useful as a
> place to learn, test, and extend the stack. It is not a production CFD package,
> not a peer-reviewed solver, and not a source of benchmark claims beyond the
> checked cases below.

`dev_field_efvm` is the Eulerian finite-volume CFD physics tier in the GRASS
ecosystem. FIELD owns the mesh, field storage, registry, and halo exchange;
this repo owns the CFD state, equation of state, Riemann fluxes, boundary
conditions, explicit time integration, viscous terms, output, and immersed-boundary
physics.

```text
GRASS            App, plugins, scheduler, coupling, MPI
  FIELD          mesh substrate: FvMesh, FieldData, FieldRegistry, halo exchange
    dev_field_efvm  CFD physics: ConsVar, EOS, fluxes, BCs, RK, viscosity, IBM
```

The intended path is Rust first: read a small example, wire the plugins you need,
and write the missing physics as systems. Declarative TOML cases exist for some
examples, but they are a convenience layer over the Rust code, not the main model
of use.

## From a CFD App to a coupled CFD participant

The CFD code stays an ordinary GRASS App: FIELD resources hold the mesh and
fields, and CFD plugins schedule fluxes, boundaries, and integration. A coupling
repo mounts that App beside another solver and owns the exchange systems; this
repo does not acquire a particle dependency or contain DEM/SPH knowledge.

See the complete solver construction in the DEM-CFD lesson's
[`setup.rs`](https://github.com/SueHeir/dev_couple_dem_cfd/blob/main/examples/hello_dem_cfd/setup.rs)
and the coupling-only composition in
[`main.rs`](https://github.com/SueHeir/dev_couple_dem_cfd/blob/main/examples/hello_dem_cfd/main.rs).
For split MPI, the same binary is assigned a CFD role by TOML and FIELD's
partition directory maps positions to CFD owners; the runnable 3+2 test is
[`routed_3x2.rs`](https://github.com/SueHeir/dev_couple_dem_cfd/blob/main/crates/dem_cfd/tests/routed_3x2.rs).

## What Is Here

- Compressible Euler/Navier-Stokes finite-volume pieces on FIELD.
- Ideal-gas thermodynamics and transport helpers.
- Rusanov, HLLC, and MUSCL reconstruction support.
- Explicit Euler / SSP-RK3 stepping through FIELD schedule phases.
- Sharp-interface ghost-cell immersed-boundary machinery for analytic bodies.
- Top-level validation examples under `examples/<case>/`.

The crate split mirrors the physics responsibilities:

| crate | role |
|---|---|
| `cfd_state` | conserved and primitive CFD state (`ConsVar`, `PrimVar`) |
| `cfd_eos` | equation of state and transport helpers |
| `cfd_riemann` | inviscid numerical fluxes |
| `cfd_boundary` | boundary-condition traits and registry |
| `cfd_solver` | flux assembly, CFL, RK integration, MUSCL, viscous terms |
| `cfd_ibm` | ghost-cell immersed boundary and marker loads |
| `cfd_output` | CFD-specific VTK fields |
| `test_cfd` | top-level bundle and p4est-oriented tests |

## Validated Cases

These are the cases that currently deserve to be called validation, with their
limits attached.

**Isentropic vortex order of accuracy.** The solver advects the smooth Shu
isentropic vortex, compares density against the exact translated solution, and
fits the mesh-refinement order. The first-order HLLC/Godunov path is pinned by
`crates/cfd_solver/tests/isentropic_vortex_convergence.rs`; the MUSCL path is
pinned by `crates/cfd_solver/tests/isentropic_vortex_muscl_convergence.rs` and
requires a fitted order of at least 1.8. The runnable TOML example is
`examples/isentropic_vortex/config.toml`.

**Immersed-boundary cylinder drag.** The ghost-cell IBM runs the Schaefer-Turek
DFG 2D-1 flow-around-a-cylinder benchmark at `Re = 20`, using the marker-integrated
drag coefficient against `Cd = 5.5795`. The case gates on drag accuracy, a flat
steady plateau, and lift. The case lives in
`examples/cylinder_drag_validation/` with the declarative input
`examples/cylinder_drag_validation/config.toml`.

Both cases are intentionally narrow. The vortex is a smooth manufactured-style
Euler check, not a turbulence or shock validation. The cylinder is one steady,
low-Re, 2-D immersed-boundary drag check with a first-order staircase/ghost-cell
geometry error; it does not prove moving-body FSI, high-Re shedding, wall-modeling,
or general industrial IBM accuracy.

## How To Run The Checks

From this repo, with the sibling `grass`, `field`, and `soil` checkouts present:

```bash
source ~/projects/.build-env
cargo test -p cfd_solver --test isentropic_vortex_convergence -- --nocapture
cargo test -p cfd_solver --test isentropic_vortex_muscl_convergence -- --nocapture
cargo run --release -p dev_field_efvm --bin cylinder_drag_validation -- \
  examples/cylinder_drag_validation/config.toml
```

The cylinder case is intentionally heavier than a unit test because it waits for
the drag to plateau; a transient pass-through value is not accepted as validation.

## What Is Not Proven Yet

- No peer review, no independent production-code comparison suite, and no claim
  of general CFD fidelity.
- The Sod shock-tube examples are **DEMO (not a validation)** cases: they exercise
  the end-to-end FIELD/CFD path and VTK output, but do not compare against an exact
  Riemann solution or print a PASS/FAIL verdict.
- AMR/p4est support exists in older tests and examples, but the default validation
  story above is the uniform FIELD path.
- The current validated IBM drag story is a stationary analytic cylinder at one
  steady Reynolds number, not a complete FSI solver.
- DEM-CFD and SPH-CFD seams live in dedicated `dev_couple_*` repositories.
- Most examples are written to expose the mechanism plainly. Expect to read and
  change Rust, not only edit input files.

## More Documentation

- [docs/README.md](docs/README.md) indexes reference material.
- [docs/reference/conserved-variables.md](docs/reference/conserved-variables.md)
  explains the conserved state vector and how it moves through FIELD.
- [PLAN-dev_field_efvm.md](PLAN-dev_field_efvm.md) records the design plan and
  stack placement.
- [DISCLAIMER.md](DISCLAIMER.md) gives the broader ecosystem disclaimer.
