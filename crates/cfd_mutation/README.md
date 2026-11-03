# cfd_mutation

[Mutation++](https://github.com/mutationpp/Mutationpp) thermochemistry behind
dev_field_efvm's `Eos` trait. With it, the existing solver runs on air in local
chemical equilibrium (dissociation and ionization included) by swapping one
resource:

```rust
let air = Arc::new(EquilibriumEos::new("air_11")?);
app.add_resource(EosResource(air));
```

The fluxes, boundary conditions and VTK output need no other change.

## What is here

| file | lines | what |
|---|---:|---|
| `shim/mpp_shim.cpp` | 85 | flat C interface; C++ exceptions become status codes |
| `build.rs` | 22 | compiles the shim, links `libmutation++` |
| `src/lib.rs` | 377 | safe `Mixture` owner, `EquilibriumEos`, a robust (ρ, e) inversion, `normal_shock` |
| `tests/equilibrium_air.rs` | 90 | checks against Mutation++'s `mppequil` and `mppshock` |
| `examples/equilibrium_wall_shock.rs` | 165 | end-to-end validation through the efvm solver |

## Build and run

Mutation++ is not needed to build the rest of dev_field_efvm. This crate is
its own workspace and is excluded from the parent one.

```sh
# Mutation++ (once)
git clone https://github.com/mutationpp/Mutationpp && cd Mutationpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/mpp && cmake --build build -j --target install

# this crate
export MUTATIONPP_DIR=$HOME/mpp
export MPP_DATA_DIRECTORY=/path/to/Mutationpp/data
cd crates/cfd_mutation
cargo test --release
cargo run --release --example equilibrium_wall_shock
```

## Validation: reflected shock in equilibrium air

Air at 100 Pa and 300 K flows at 5 km/s into a wall. The exact answer is the
equilibrium Rankine–Hugoniot jump that brings the gas to rest. `normal_shock`
computes it outside the solver, and the tests check it against Mutation++'s
`mppshock`.

```
exact (equilibrium RH):  Vs =   435.0 m/s  p2 =    31532 Pa  T2 =   5699 K  rho2 = 1.4450e-2
efvm + Mutation++:       Vs =   434.4 m/s  p2 =    31532 Pa  T2 =   5699 K  rho2 = 1.4450e-2  u2 = -0.0 m/s
  60 cells, 1224 steps to t = 9.00e-5 s in 56.1 s (763.3 us per cell-step)
  ideal-gas run of the same case: 1127 steps in 0.122 s (1.80 us per cell-step), so Mutation++ costs 423x
perfect gas (exact):     Vs =  1020.0 m/s  p2 =    35052 Pa  T2 =  17818 K  rho2 = 6.8532e-3
PASS
```

The perfect gas puts the post-shock gas at 17,800 K. Equilibrium air spends
that energy dissociating O₂ and N₂ and sits at 5,700 K. The plateau matches
the exact jump to 0.001% and the shock speed to 0.15%.

## What a CFD developer should know

* **It is about 400x slower than the ideal gas.** Every flux evaluation
  inverts (ρ, e) through an equilibrium solve. Production codes tabulate
  (ρ, e) → (p, T, a) once. That table is the obvious next step.
* **Mutation++'s own (ρ, e) setter is not used.** Its Newton iteration starts
  from the mixture's previous state and has no iteration cap. In a flow solver,
  where calls jump between cold and hot cells, it looped forever. This crate
  brackets T from a coarse e(T) table and runs a safeguarded secant instead.
  (`mppshock`'s default Newton solver also hung on this case. Its bisection
  option works.)
* **Mutation++ is not thread-safe.** Separate mixtures used from different
  threads at once hung in testing. The types are not `Send`, and the tests run
  on one thread (`.cargo/config.toml`). efvm is one thread per rank today.
* **Viscous terms still use the ideal gas.** `cfd_solver::viscous_system`
  reads `Res<IdealGas>` and Sutherland's law directly, not the `Eos`. Mutation++
  transport (`EquilibriumEos::transport_pt`) is available, but efvm's `Eos`
  needs a transport method before the viscous path can use it.
* **`SupersonicInflow` assumes air's gas constant.** It derives T from
  `p / (ρ · 287.058)`, and `EquilibriumEos::prim_to_cons` takes (p, T), so the
  inflow density comes out 0.4% off for `air_11` (no argon). Use an
  `FnBoundary` with a state from `EquilibriumEos::prim_pt` until that BC takes
  T directly.
* **Equilibrium only.** Finite-rate chemistry (Mutation++'s `ChemNonEq1T`,
  two-temperature models) needs species densities in efvm's conserved state.
  That change belongs to efvm, and the `Mixture` wrapper here already accepts
  any state model through `set_state_raw`.
