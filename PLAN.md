# test-cfd — Restructuring Plan

Migrating `toy-cfd` into a clean two-repo stack on the **GRASS** framework, mirroring
the proven **GRASS → SOIL → DIRT** architecture on the mesh/Eulerian side.

---

## 0. The architecture we're matching

GRASS is a method-agnostic framework (App + Plugin + Scheduler, IO, MPI coupling). It
knows nothing about particles *or* mesh. SOIL is the **particle substrate** that rides
GRASS; DIRT is the **DEM physics** that rides SOIL. We are building the **second branch**
of the substrate layer plus its physics:

```
GRASS    framework      App + Plugin + Scheduler, IO, MPI coupling   (no particles/mesh/physics)
  ├─ SOIL    particle substrate   Atom, domain decomp, comm, neighbor lists ── DIRT      (DEM physics)
  └─ FIELD   mesh substrate       cells/faces/nodes, partition, halo, geometry ── test-cfd (CFD physics)
                                                                                  ^ this repo
```

- **FIELD** = the SOIL-equivalent for meshes. New repo (sibling of `soil/`).
- **test-cfd** = the DIRT-equivalent: compressible-Navier–Stokes physics as GRASS
  plugins, zero infrastructure of its own. This repo.

**Naming convention (resolve the "field" collision up front):** the substrate is
`FIELD`, crates are prefixed `field_*`. Solution variables (ρ, ρu, ρE, …) are referred
to everywhere as **state** or **vars**, never "fields", so `FIELD` (the mesh layer) and
`state` (the numbers living on it) never get confused in prose or code. Optional
backronym in the spirit of SOIL ("Substrate for Off-lattice Interacting Lagrangians"):
*Framework for Indexed Eulerian Lattice Discretization*.

---

## 1. Why the substrate inverts (the one thing to internalize)

Coming from particle methods, SOIL's whole reason to exist is that **carriers move**:
particles carry state through space, so *who-is-near-whom is dynamic*. Hence neighbor
lists rebuilt from positions, atom migration between ranks, ghosts rebuilt each step.

In the mesh/Eulerian world the carriers (cells) are **fixed**; data flows through a
**static topology**. The substrate's job inverts:

| SOIL (particle) | FIELD (mesh) | What changes |
|---|---|---|
| Dynamic neighbor list, rebuilt on displacement | **Static connectivity** (cell→face→cell), computed once | no rebuild loop; adjacency is a property of the mesh |
| Atom migration between ranks | **Static graph partition** (METIS/Scotch-style), once | cells never migrate; partition frozen for the run |
| Ghosts rebuilt each step | **Halo cells** — fixed set, *values* exchanged each step | topology frozen, only numbers move |
| `Atom` SoA | **state-on-mesh SoA** (cell/face/node-centered) | same idea, indexed by mesh entity |
| `AtomData` `#[forward]/#[reverse]/#[zero]` | **`FieldData` halo contract** (already exists in toy-cfd) | near-1:1; see §2 |
| particle = point + radius (trivial geometry) | **geometry/metrics**: volumes, face areas+normals, centroids | NEW responsibility, no particle analog (§3) |
| (none) | **AMR**: octree refine/coarsen, coarse↔fine reflux, transfer | NEW; toy-cfd already does it via p4est |
| (none — DEM/MD/SPH are matrix-free) | **sparse matrix + linear-solver seam** | NEW; the deepest divergence — deferred to M3 (§6) |

Two responsibilities FIELD owns that SOIL never needed: **geometry as a first-class
module**, and (eventually) **the linear-algebra seam**. toy-cfd is fully explicit today,
so M1 can skip linear solvers exactly as SOIL does.

---

## 2. The `FieldData` contract (the single physics seam)

This is the direct sibling of SOIL's `AtomData`, and **toy-cfd already implements it**
(`cfd_state` + `cfd_derive`'s `#[derive(FieldData)]`). Lift it nearly verbatim.

| SOIL `AtomData` | FIELD `FieldData` | Meaning on a mesh |
|---|---|---|
| `#[forward]` (owner→ghost, overwrite) | `#[forward]` (interior→halo, overwrite) | halo cells get owner values before flux eval |
| `#[reverse]` (ghost→owner, `+=`) | `#[reverse]` (halo→owner, `+=`) | accumulate flux/face/node contributions back to owners (FVM flux sum, FEM assembly) |
| `#[zero]` (per-step reset) | `#[zero]` (per-step reset) | reset the RHS accumulator each substep |
| (no attr) migrates with atom | (no attr) lives on the cell | per-cell state with no halo traffic |

Downstream physics registers state once (`register_field_data!`, mirroring
`register_atom_data!`) and the substrate carries the columns through every halo exchange,
partition, AMR transfer, and restart automatically — **test-cfd writes no comm code**,
exactly as DIRT writes none.

---

## 3. FIELD crate layout (mirror SOIL's six-crate shape)

```
field/  (new repo, sibling of soil/)
  crates/
    field_core      mesh entities (cells/faces/nodes), connectivity/adjacency,
                    state SoA + FieldData registry, partition + halo exchange,
                    Domain/boundary kinds, schedule set (mirrors soil_core)
    field_geometry  *** NEW vs SOIL *** cell volumes, face areas + normals,
                    centroids, cell-face distances, stretched/non-uniform metrics
    field_derive    #[derive(FieldData)] proc macro (from cfd_derive; mirrors soil_derive)
    field_amr       octree refine/coarsen, coarse↔fine flux matching (reflux),
                    solution transfer  (p4est-backed; mirrors nothing in SOIL)
    field_integrate generic explicit time integration (Euler, RK; mirror soil_verlet —
                    decide in §4 whether RK lives here or in physics)
    field_print     thermo/dump/restart + VTK writer (generic parts of cfd_output)
```

Substrate boundary rule, copied from SOIL: **nothing in `field_*` may depend on a
physics crate.** `field_*` depends on `grass_*` only; physics depends on `field_*`.

---

## 4. What lifts from toy-cfd, crate by crate

toy-cfd's 12 crates already cleave along the substrate/physics line — the split *is* the
migration:

| toy-cfd crate | → destination | notes |
|---|---|---|
| `cfd_grid` (UniformGrid, MPI pencil decomp, ghost layers) | **field_core** | topology + partition + halo |
| `cfd_state` (FieldData trait + registry) | **field_core** | the AtomData-analog storage |
| `cfd_derive` (`#[derive(FieldData)]`) | **field_derive** | sibling of `soil_derive` |
| `cfd_p4est` (forest topology, leaf/neighbor queries) | **field_amr** | substrate-side AMR topology |
| `cfd_p4est_solver` | **split** | coarse-fine *flux matching/reflux* → field_amr; the *physics* (MUSCL, cut-cell drag) → test-cfd |
| `cfd_output` (VTK) | **split** | generic dump → field_print; CFD-specific state → test-cfd |
| `cfd_integrate` (Euler, RK3) | **field_integrate** (lean) | SOIL keeps Verlet in substrate → mirror: generic RK in substrate, physics picks scheme |
| `cfd_eos` (ideal gas, Sutherland) | **test-cfd** | constitutive (DIRT material-model analog) |
| `cfd_riemann` (Rusanov, HLLC+Thornber) | **test-cfd** | flux constitutive laws (DIRT contact-model analog) |
| `cfd_boundary` (BC trait + registry) | **test-cfd** | physics BCs (SOIL keeps method-agnostic `pin`; DEM BCs live in DIRT) |
| `cfd_solver` (RHS assembly, CFL, IBM) | **test-cfd** | the heart — twin of DIRT's fused `contact.rs` loop |
| `toy_cfd` (prelude/plugin bundle) | **test-cfd** | becomes `CfdDefaultPlugins` (cf. `GranularDefaultPlugins`) |

---

## 5. test-cfd physics layer (the DIRT-twin)

Pure constitutive + discretization physics as GRASS plugins, riding FIELD:

- **State extension:** register the compressible 5-tuple `[ρ, ρu, ρv, ρw, ρE]` via
  `register_field_data!` (cf. DIRT's `DemAtom`).
- **EOS plugin:** ideal gas, Sutherland viscosity, Prandtl conductivity (trait-based,
  like DIRT's `MaterialTable`).
- **Flux plugin:** Rusanov / HLLC+Thornber behind a `RiemannFlux` trait (the constitutive
  law — DIRT's contact model).
- **Integrator selection:** Euler / TVD-RK3.
- **Boundary plugin:** per-face BC registry (periodic, sup/sub in/outflow, slip/symmetry).
- **Solver plugin:** fused inviscid+viscous RHS sweep, CFL timestep, ghost fills.
- **IBM plugin:** sharp-interface ghost-cell IBM (signed distance + image-point). **This
  is the coupling seam to the particle stack** — sphere geometry from MDDEM atoms is the
  GRASS multi-app (`grass_multi`) coupling story; long-term argument for both substrates
  being GRASS-native.
- **Bundle:** `CfdDefaultPlugins` = FIELD infra + the physics plugins, TOML-selected.

Dependency conventions (copy DIRT exactly): FIELD as git submodule + `[patch]` path deps
to the local `field/` checkout; `dirt-mpi`/`dirt-dev-gpu`-style variant branches later.

---

## 6. Milestones

- **M1 — Explicit parity (no linear algebra).** FIELD + test-cfd reproduce toy-cfd's
  `UniformGrid` examples (Sod, pore flow, cylinder drag Re-sweep) on structured Cartesian.
  Fully explicit; mirrors SOIL's matrix-free scope. *Definition of done: cylinder-drag
  validation matches toy-cfd numbers; MPI ghost tests pass.*
- **M2 — AMR.** `field_amr` carries octree refine/coarsen + reflux + transfer; test-cfd
  reproduces the forest examples (cylinder Re40 AMR, regrid demo, spheres-in-flow).
- **M3 — Implicit / linear-solver seam (the "serious CFD substrate" milestone).** Add the
  sparse-matrix + iterative-solver interface FIELD deliberately lacks at M1. Decision to
  make here: substrate owns a thin matrix/solver trait delegating to `nalgebra`/`faer`/
  PETSc, vs. physics brings its own. Unlocks incompressible (pressure-Poisson) and
  implicit time stepping. This is the capability the established players (PETSc/hypre
  multigrid) have that M1/M2 won't.

---

## 7. Landscape position (for reference)

- Physics content (compressible FVM, Rusanov/HLLC, Thornber low-Mach, octree AMR,
  ghost-cell/cut-cell IBM) sits at **SU2 ∩ AMReX** (density-based compressible aero +
  block-structured/octree AMR; p4est is the same octree lineage AMReX-style codes use).
- Architecture (physics-agnostic substrate + plugin physics + scheduler) is the
  **MOOSE/libMesh** and **PETSc DMPlex + TS** design philosophy — re-expressed as
  GRASS + FIELD + test-cfd, but in Rust with a Bevy-style ECS scheduler, which essentially
  no established CFD code does.
- The one mature capability the incumbents have that this stack defers to M3 is the
  **implicit linear-solver stack** (PETSc/hypre). Being explicit lets M1 ship coherent.
