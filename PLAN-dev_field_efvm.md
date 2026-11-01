# PLAN — `dev_field_efvm`: the CFD physics layer on the FIELD substrate

The DIRT-equivalent for the mesh/Eulerian branch. `dev_field_efvm` is compressible
Navier–Stokes physics expressed as GRASS plugins riding the **FIELD** mesh
substrate, the way DIRT rides SOIL.

```
GRASS    framework: App + Plugin + Scheduler + IO + MPI
  └─ FIELD   substrate: FvMesh / StructuredMesh, FieldData, FieldRegistry, HaloPlan, MeshScheduleSet
       └─ dev_field_efvm  physics: ConsVar state, EOS, Riemann flux, BCs, viscous, RK, IBM   ← this repo
```

This plan ports `toy-cfd`'s physics crates onto FIELD's contracts. The substrate
half of `toy-cfd` (`cfd_grid`, `cfd_state`'s `FieldData`/`FieldRegistry`,
`cfd_derive`) is **already provided by FIELD** (`UniformMesh`, `FieldData` +
`FieldRegistry`, `field_derive`), so `dev_field_efvm` writes *zero* mesh, storage, or
halo code — it registers state and provides physics systems, exactly as DIRT
writes no neighbor-list or migration code.

All Rust below is illustrative pseudocode in this plan; **no Rust is to be
written as part of this planning task.**

---

## 0. The single most important structural change vs. toy-cfd

`toy-cfd` fuses an entire physics step — MPI ghost exchange, BC fill, IBM ghost
fill, scratch refresh, inviscid RHS, viscous RHS, solid-RHS zeroing, and the RK
substep — into **one monolithic system** (`do_full_step`, dispatched from
`system_step_rusanov`/`_hllc` in a single `CfdScheduleSet::Step` phase). One App
iter = one full physics step; the RK substep loop lives *inside* the system body.

FIELD instead ships a finer, generic phase ordering — `MeshScheduleSet`
(`Setup → PreHalo → Halo → PostHalo → PreBoundary → Boundary → PreFlux → Flux →
PostFlux → PreUpdate → Update → PostUpdate → Output`) — that is the mesh
counterpart to SOIL's velocity-Verlet schedule set. **The port decomposes
`do_full_step` across these phases**, with each phase a separate system. This is
the DIRT idiom (DIRT splits force from integration across `Force` /
`InitialIntegration` / `FinalIntegration`), and it buys:

- Halo exchange becomes the substrate's generic `halo_exchange_forward_all`
  (registry-driven), not physics-hardcoded `Grid::ghost_exchange`.
- BCs, IBM ghost fill, flux, and update become independently observable and
  independently swappable plugins.
- The RK substage loop is driven by **re-running `MeshScheduleSet` N times per
  step** (§6), the way the schedule-set doc intends ("Multi-stage integrators
  (RK) drive this set repeatedly per step").

The one cost is that the per-substep work is no longer in a single inlined
function; §3 and §10 address keeping the structured fast path for performance.

---

## 1. Crate layout

Mirror DIRT's split: thin leaf crates for constitutive pieces, a heart crate for
the assembly loop, and a top-level convenience crate exporting a default bundle +
prelude. Every crate depends on `field_core` + `grass_*` only; **none depends on
another *physics-policy* crate in a way that inverts the substrate rule**, and
nothing in `field_*` ever depends on `dev_field_efvm`.

| crate | responsibility | key types | FIELD contract it touches |
|---|---|---|---|
| `cfd_state` | conserved-variable type + `Component`/`FieldData` columns | `ConsVar` (WIDTH 5), `PrimVar`, the `#[derive(FieldData)] CfdState` store | `Component`, `field_derive::FieldData`, `register_field_data!` |
| `cfd_eos` | constitutive thermodynamics + transport | `Eos` trait, `IdealGas`, `sutherland_viscosity`, `thermal_conductivity` | none directly (pure functions of `ConsVar`/`PrimVar`) — DIRT's `MaterialTable` analog |
| `cfd_riemann` | numerical flux laws | `RiemannFlux` trait, `FluxScheme`, `Rusanov`, `Hllc` (+Thornber), `physical_flux_inline` | consumes `ConsVar`; **gains a face-normal API** (§3) |
| `cfd_boundary` | per-patch BC trait + registry | `BoundaryCondition` trait, `BoundaryRegistry` (indexed by `BoundarySide`), `Periodic`/`SlipWall`/`SupersonicInflow`/… | `BoundarySide`/`PatchId`, `Face.patch`, fills ghost cells in `Boundary` phase |
| `cfd_solver` | RHS assembly, CFL dt, time stepping, phase systems | `CfdState`-driven `flux_system`, `cfl_system`, `update_system`, `SolverConfig`, `SolverState`, `RkState` | **the heart**: `FvMesh::for_each_face`, `StructuredMesh` fast path, `cell_volume`, `halo_plan`, `MeshScheduleSet` |
| `cfd_ibm` | sharp-interface ghost-cell IBM + diagnostics | `SolidMask` (`#[forward] Vec<bool>`), `IbmGhostCells`, `IbmMode`, signed-distance geometry sources, marker loadings | `Component for bool`, `FieldData`, `StructuredMesh` (image-point sampling), `halo` |
| `test_cfd` | top-level: bundle, prelude, config wiring | `CfdDefaultPlugins` (`PluginGroup`), `prelude`, schedule glue, examples | re-exports `FieldRegistryPlugin`/`UniformMeshPlugin` + all CFD plugins |

Optional later crates (out of M1 scope, noted for symmetry with DIRT's leaf
crates and the existing repo `PLAN.md`): `cfd_amr` (rides a future
`field_amr::ForestMesh` + `AdaptiveMesh`, `PostFlux` reflux via `#[reverse]`),
and generic VTK output which belongs in a substrate-side `field_print`, not here
(only CFD-specific scalar derivation stays in `dev_field_efvm`).

### What changed vs. toy-cfd's crate split

- **`cfd_grid` is deleted** — `UniformMesh` is FIELD's. `dev_field_efvm` never sees a
  grid struct; it sees `&M: FvMesh (+ StructuredMesh)`.
- **`cfd_state`'s `FieldData`/`FieldRegistry` definitions are deleted** — those
  are FIELD's. `cfd_state` keeps only `ConsVar`/`PrimVar` and now *implements*
  `field_core::Component` for `ConsVar` and derives `FieldData` for the state
  store.
- **`cfd_derive` is deleted** — replaced by `field_derive`.
- **`cfd_integrate` folds into `cfd_solver`** (or stays a thin sibling). FIELD's
  schedule-set doc says the integrator is physics layered on the generic phases;
  there is no `field_integrate` at M1, so the RK state machine lives in
  `cfd_solver` next to the `Update` system. (Contrast: SOIL keeps Verlet in
  `soil_verlet`; FIELD deliberately did *not* ship `field_integrate`, so the
  integrator is physics here.)
- **`cfd_solver` loses its grid/ghost knowledge** — no `Grid::ghost_exchange`,
  no `FIELDS_PER_CELL = 5`; halo is `halo_exchange_forward_all` over the
  registry.
- **IBM is promoted to its own `cfd_ibm` crate** (it was `toy_cfd::ibm`), since
  it now owns a registered `FieldData` (the solid mask) and is the FSI coupling
  seam.

---

## 2. State as `FieldData`

### 2.1 `ConsVar` becomes a `Component`

`ConsVar` keeps its five `f64` fields and operator overloads
(`Add`/`Sub`/`Mul<f64>`/`AddAssign`/`ZERO`/`velocity()`/`kinetic_energy()`), and
gains a `Component` impl (WIDTH 5) so a `Vec<ConsVar>` halo-exchanges directly —
this is exactly the example FIELD's `component.rs` doc-comments anticipate:

```rust,ignore
impl field_core::Component for ConsVar {
    const WIDTH: usize = 5;
    fn write(&self, b: &mut Vec<f64>) {
        b.extend_from_slice(&[self.rho, self.rho_u, self.rho_v, self.rho_w, self.rho_e]);
    }
    fn read(b: &[f64]) -> Self {
        Self { rho: b[0], rho_u: b[1], rho_v: b[2], rho_w: b[3], rho_e: b[4] }
    }
    fn add_from(&mut self, b: &[f64]) {
        self.rho += b[0]; self.rho_u += b[1]; self.rho_v += b[2];
        self.rho_w += b[3]; self.rho_e += b[4];
    }
    fn zero() -> Self { Self::ZERO }
}
```

### 2.2 The state store(s)

Derive `FieldData` on a struct of `Vec<C: Component>` columns, tagged per FIELD's
`#[forward]`/`#[reverse]`/`#[zero]` contract:

```rust,ignore
#[derive(field_core::FieldData, Default)]
pub struct CfdState {
    #[forward] pub u: Vec<ConsVar>,    // conserved state: owner → ghost, overwrite
    #[zero]    pub rhs: Vec<ConsVar>,  // flux-divergence accumulator, reset each substage
}
```

- **`u` is `#[forward]`** — the everyday FVM halo. Before flux assembly the
  substrate copies each owner's `u` into its neighbor ghosts via
  `halo_exchange_forward_all` (which reads `forward_size()` from the registry,
  so it moves exactly these 5 f64/cell, no hardcoding).
- **`rhs` is `#[zero]`** — reset to `ConsVar::ZERO` each RK substage by
  `FieldRegistry::zero_all`/`zero_cell`. It is *not* forward-exchanged (it is
  accumulated locally from faces; cross-rank face contributions arrive through
  `u`'s ghosts, not through `rhs`). It could optionally be `#[reverse]` *if* we
  ever scatter flux onto ghost owners across ranks, but the standard structured
  decomposition computes every interior owner's RHS locally because each rank
  owns a full ghost layer — so M1 keeps `rhs` as plain `#[zero]` and relies on
  `u`-halo. (`#[reverse]` is reserved for AMR reflux on `field_amr` — §10.)

### 2.3 RK backup `u0`: plain column, not exchanged

toy-cfd's `FlowField` carried `u0: Vec<ConsVar>` for RK stage backup. Two
choices:

- **Recommended:** add `u0` as an *untagged* `Vec<ConsVar>` column in `CfdState`
  (no `#[forward]`/`#[reverse]`/`#[zero]` attribute). FIELD's derive resizes it
  but never exchanges or zeroes it — exactly the "no-attr lives on the cell"
  row of the contract. `save_state()` (copy `u → u0`) runs once in `Setup` at
  the start of a full step; the RK update reads `u0`.
- Alternative: a separate `RkBackup { u0: Vec<ConsVar> }` `FieldData` store. This
  is cleaner if multiple integrators want different backup shapes, but adds a
  second registry entry. **Pick the single-store option for M1.**

### 2.4 Solid mask

The IBM solid mask is `#[forward] Vec<bool>` (FIELD already implements
`Component for bool`, `WIDTH 1`, with `add_from` = logical OR). It is a separate
`FieldData` store owned by `cfd_ibm` (so non-IBM runs never register it):

```rust,ignore
#[derive(field_core::FieldData, Default)]
pub struct SolidMask { #[forward] pub solid: Vec<bool> }
```

Forward-exchanging the mask replaces toy-cfd's bespoke `Grid::solid_exchange`;
it rides the *same* `halo_exchange_forward_all` pass as `u` (registry packs both
in declaration order). See §7.

---

## 3. The flux loop — `for_each_face` mapping and the structured fast path

### 3.1 Generic path (correct first; works on any `FvMesh`)

The `Flux` phase system is generic `<M: FvMesh>`. After `u` halos are filled and
BC ghosts are set, it zeroes `rhs` (via the registry's `#[zero]`) and runs:

```text
mesh.for_each_face(|f: &Face| {
    let uL = state.u[f.owner];
    let uR = state.u[f.other];                 // interior, halo-ghost, or BC-ghost
    // 1. project to face-normal frame
    let area = |f.area_normal|;                // magnitude
    let n̂    = f.area_normal / area;           // unit outward normal, owner→other
    // 2. numerical flux in the face-normal direction (scalar normal flux)
    let Fn = flux.flux_normal(&uL, &uR, eos, n̂);  // ConsVar, the flux·n̂ vector
    // 3. scatter, conservatively, scaled by 1/volume
    state.rhs[f.owner] -= Fn * (area / mesh.cell_volume(f.owner));
    if f.patch.is_none() {                      // don't write into BC ghosts
        state.rhs[f.other] += Fn * (area / mesh.cell_volume(f.other));
    }
});
```

**Per-face math, made concrete.** FIELD's `Face` carries `area_normal = A·n̂`,
`n̂` pointing `owner → other`. toy-cfd's Riemann solvers take a *direction index*
`dir ∈ {0,1,2}`, not a normal vector. Two equivalent ways to bridge:

- **Axis-aligned shortcut (UniformMesh, M1):** every `area_normal` is a single
  nonzero axis component; recover `dir = argmax|area_normal|` and the sign.
  Momentum flux already lands on the correct axis component, so toy-cfd's
  `physical_flux_inline(u, eos, dir)` and `Rusanov`/`Hllc` work *unchanged*; the
  only addition is dividing by `area` ⋅ accounting for the `±` sign of the
  low-side faces (FIELD emits `area_normal = [-A,0,0]` for the `-x` boundary
  face). This keeps M1 numerically identical to toy-cfd.
- **General normal frame (unstructured/rotated faces, post-M1):** add a
  `flux_normal(uL, uR, eos, n̂)` method to `RiemannFlux`/`FluxScheme` that rotates
  momentum into the face frame, calls the 1-D solver in the normal direction,
  and rotates the momentum flux back. This is the OpenFOAM face-reduction the
  `FvMesh` trait is built around; **add the method now (axis-aligned `n̂` falls
  back to the `dir` path), implement true rotation when a non-axis-aligned mesh
  lands.**

`area_normal` magnitude replaces toy-cfd's per-face `dx*dz`/`dy*dz` area; the
`1/cell_volume` replaces toy-cfd's `dx_inv`/`dy_inv[j]`/`dz_inv[k]` scaling —
both come straight from the `FvMesh` API, so the loop is metric-agnostic.

### 3.2 The structured fast path — keep it, behind the same plugin

`for_each_face` visits faces through a `&mut dyn FnMut` closure; per FIELD's own
`mesh.rs` note, performance-critical physics "should instead be generic
`<M: FvMesh>` … and the structured fast path avoids dynamic dispatch entirely."
toy-cfd's `compute_inviscid_rhs` is a fused `(i,j,k)` sweep with a stripe-based
solid-free fast path and `flux_scheme_cached` (precomputed `prim`/`sound_speed`
in a `SolverScratch`), which is where its performance lives.

**Decision:** ship *both*, selected at runtime by a `flux_path` config and by a
`M: StructuredMesh` bound:

- **Generic `for_each_face`** path — correctness baseline, runs on every
  `FvMesh`, used to validate and as the only path on future unstructured meshes.
- **Structured fused sweep** — a second `Flux`-phase system bounded
  `<M: FvMesh + StructuredMesh>`. It uses `dims()`/`idx()`/`n_ghost()`/`spacing()`
  to walk pencils, keeps a `SolverScratch { prim: Vec<PrimVar>, a: Vec<f64> }`
  populated once per substage (in `PreFlux`), and calls `flux_scheme_cached`. The
  stripe `stripe_has_solid[j*nk+k]` fast/slow split ports directly using the
  registered `SolidMask`. This is the twin of DIRT's fused `contact.rs` loop.

Tradeoff to record (and §10): the generic path pays a `dyn FnMut` indirect call
per face plus recomputes sound speeds; the structured path inlines and caches but
only exists for `StructuredMesh`. M1 default = structured fast path on
`UniformMesh`; the generic path is always available and is what the Sod
validation runs first.

### 3.3 Boundary faces

When `f.patch == Some(side)`, `f.other` is the BC ghost cell. The flux loop does
**not** branch on the BC — by the time `Flux` runs, the `Boundary` phase has
already written the correct ghost state into `state.u[f.other]` (§4), so the same
`flux.flux_normal(uL, uR, …)` reproduces the wall/inflow/outflow flux. The only
flux-loop rule is "don't scatter `rhs` into a `patch.is_some()` ghost"
(`f.other` is not an owned interior cell). This mirrors toy-cfd's design note in
`ibm.rs`: "With ghost cells filled this way, the regular flux at fluid↔ghost
faces reproduces the wall BC — no special `wall_flux` is needed."

(The `IbmMode::Staircase` *pressure-only wall flux* is the one exception that
needs a branch; it lives in `cfd_ibm`/`cfd_solver`'s slow-path face handler, not
in the generic loop — see §7.)

---

## 4. Boundary conditions

Lift toy-cfd's `cfd_boundary` almost verbatim; the only change is *how a face is
identified*. toy-cfd used `Face { dim, is_lo }` and indexed a 6-field
`BoundaryRegistry` by hand; FIELD already gives us `BoundarySide` (XLo..ZHi as
`PatchId 0..5`) on `Face.patch`.

- **`BoundaryCondition` trait** (unchanged shape):
  `fn fill(&self, ghost: GhostView, eos: &dyn Eos) -> ConsVar`. Stock impls
  port directly: `Periodic`, `SupersonicOutflow` (zero-gradient),
  `SupersonicInflow`, `SlipWall` (reflect normal momentum), `Symmetry`,
  `SubsonicInflow`, `PressureOutlet`.
- **`BoundaryRegistry`** holds `[Arc<dyn BoundaryCondition>; 6]` indexed by
  `BoundarySide as usize`; `.set(side, bc)` / `.with(side, bc)`. Default all
  sides `SupersonicOutflow`.
- **`GhostView`** replaces toy-cfd's `GhostCell` and is built from FIELD
  primitives: it needs `nearest_interior()` (the owner cell across the boundary
  face) and `periodic_mirror()` (opposite-side interior). On a `StructuredMesh`
  these come from `idx()` arithmetic; on a generic `FvMesh` the BC system can
  read `Face.owner` directly (the nearest interior is `f.owner`), which covers
  every BC except `Periodic` (periodic is naturally a halo-exchange wrap on a
  decomposed mesh, not a physical patch — note this in §10).

### The `Boundary` phase system

A system that runs in `MeshScheduleSet::Boundary`:

```text
for each face f with f.patch == Some(side):
    let bc = registry[side];
    state.u[f.other] = bc.fill(GhostView{ owner: f.owner, side, .. }, eos);
```

The simplest implementation iterates `mesh.for_each_face` and acts only on
`patch.is_some()` faces (cheap; boundary faces are O(N^{2/3})). A `StructuredMesh`
variant can fill whole ghost layers (`ng` deep) via `idx()` for multi-layer
stencils — needed because the viscous central differences (§5) read two ghost
layers. Physical-boundary faces only exist where there is no MPI neighbor (FIELD
sets `patch = None` and lets halo fill the ghost when a neighbor rank exists), so
the `Boundary` system never fights the `Halo` phase.

---

## 5. Viscous terms — bounded to `StructuredMesh`

toy-cfd's `compute_viscous_rhs` uses **3-point centered second differences** for
the velocity Laplacian and temperature Laplacian, with precomputed non-uniform
y/z stencil weights, no-slip (`u=0`) and adiabatic (`dT/dn=0`) at solid faces,
and adds `μ∇²u + κ∇²T` (with `κ = μ·cp/Pr`, `cp = γR/(γ−1)`, `Pr = 0.72`,
`μ` from Sutherland) into `rhs`. This is a **wide directional stencil** ("the
cell two to the left") that the universal `for_each_face` view deliberately
cannot express — FIELD's `mesh.rs` calls this out explicitly as the reason
`StructuredMesh` exists.

**Decision:** the viscous plugin's system is bounded
`<M: FvMesh + StructuredMesh>` and uses `idx(i±1, j, k)` / `spacing(axis, n)` for
the central differences. This **bounds viscous physics to structured meshes for
M1**, which is acceptable and explicitly anticipated: AMR/unstructured viscous
terms (gradient reconstruction, least-squares) are an M2+ concern. The inviscid
solver remains mesh-agnostic; only the viscous *add-on* requires
`StructuredMesh`.

- Runs in `MeshScheduleSet::Flux` (after, or fused with, the inviscid sweep),
  reading the same `SolverScratch` `prim`.
- A `SolverConfig.viscous` flag gates registration of the viscous system —
  inviscid Euler runs (Sod) skip it entirely, matching toy-cfd.
- The viscous CFL term (`2ν(1/dx²+1/dy²+1/dz²)`) is added in the `cfl_system`
  when viscous is on (§6).

---

## 6. Time integration

### 6.1 Schemes (physics, in `cfd_solver`)

Port toy-cfd's `TimeIntegrator` trait, `RkStage{First,Second,Third}`,
`SubstepResult{next_stage, completed_step}`, `EulerIntegrator` (1 substep),
`Rk3Integrator` (Shu–Osher TVD-RK3, 3 substeps). The RK3 stage formulas are
unchanged and operate on the registered `CfdState` columns `u`, `rhs`, `u0`:

```text
save:    u0 ← u                                         (once, Setup, start of step)
stage 1: u  ← u0 + dt·rhs(u0)
stage 2: u  ← ¾·u0 + ¼·(u + dt·rhs(u))
stage 3: u  ← ⅓·u0 + ⅔·(u + dt·rhs(u))
```

### 6.2 Driving `MeshScheduleSet` repeatedly per step

The key interleave: **one full time step = `n_substeps` passes of the
`Halo→Boundary→Flux→Update` portion of `MeshScheduleSet`.** A `RkState` resource
holds the current `RkStage`. The schedule is driven so that each pass does, in
FIELD phase order:

| phase | system | per substage? |
|---|---|---|
| `Setup` | step bookkeeping; **on substage 0 only:** `save_state` (u0←u) and `cfl_system` (compute + all-reduce `dt`) | dt/save: once per step |
| `PreHalo` | (nothing for M1) | — |
| `Halo` | `halo_exchange_forward_all(mesh.halo_plan(), registry, comm)` — moves `u` **and** `SolidMask` in one pass | every substage |
| `PostHalo` | (IBM ghost mirror, §7) | every substage |
| `PreBoundary` | — | — |
| `Boundary` | BC ghost fill (§4) | every substage |
| `PreFlux` | `scratch.refresh` (prim, sound speed); `registry.zero_all` to reset `#[zero] rhs` | every substage |
| `Flux` | inviscid RHS (+ viscous if enabled) into `rhs` | every substage |
| `PostFlux` | zero `rhs` on solid cells; (AMR reflux later) | every substage |
| `PreUpdate` | — | — |
| `Update` | `integrator.substep(state, dt, rk_stage)` → advance `u`, return `SubstepResult` | every substage |
| `PostUpdate` | advance `RkState`; on `completed_step` advance `time`/`step` | every substage |
| `Output` | diagnostics, VTK dump, drag report, termination check | once per step (gated on `completed_step`) |

The "repeat N times" loop is realized with the scheduler's loop constructs
(the schedule-set doc: "Multi-stage integrators (RK) drive this set repeatedly
per step via the scheduler's loop constructs — the set itself describes one
sub-stage"). Concretely: a substage counter in `RkState`; the scheduler re-runs
the `Halo..PostUpdate` band until `completed_step` is true, then runs `Output`
once. **The `#[zero]` reset, the halo exchange, and the BC fill all happen
*inside* the per-substage band**, so every RK stage sees fresh ghosts and a fresh
RHS — the same ordering `do_full_step` enforced internally, now expressed in
phases.

### 6.3 CFL `dt`

`cfl_system` runs in `Setup` on substage 0. It computes the per-cell spectral
radius (skipping solid cells):

```text
λ = (|u|+a)/dx + (|v|+a)/dy + (|w|+a)/dz      [+ 2ν(1/dx²+1/dy²+1/dz²) if viscous]
dt_local = min over local cells of (cfl / λ)
dt       = comm.all_reduce_min_f64(dt_local)
```

Spacings come from `StructuredMesh::spacing(axis, n)` (or `cell_volume`-derived
on generic meshes). The cross-rank reduction uses GRASS's
`CommResource::all_reduce_min_f64` (sibling of the `all_reduce_sum_f64` DIRT uses
for KE / drag), so `dt` is global. The result is stored in `SolverState.dt` and
held constant across that step's substages.

---

## 7. Ghost-cell IBM (`cfd_ibm`)

toy-cfd's sharp-interface ghost-cell IBM (Mittal & Iaccarino 2005) ports with the
solid mask and ghost-cell list becoming FIELD-native.

### 7.1 Data

- **`SolidMask { #[forward] solid: Vec<bool> }`** — registered `FieldData`
  (§2.4). `make_cell_solid`/`make_cell_fluid` become helpers that flip the mask
  bit and set the `CfdState.u` cell to a stagnant/ambient `ConsVar`. The mask
  forward-exchanges via the registry, replacing `Grid::solid_exchange`.
- **`IbmGhostCells { cells: Vec<(usize, IbmGhostInfo)> }`** — a *resource* (not
  `FieldData`): an ephemeral per-step list of `(interior_cell_idx, info)` where
  `IbmGhostInfo { cell_center, image_point, normal, body_id, wall_velocity }`.
  Rebuilt every step for moving geometry. Not exchanged (it indexes local cells).
- **`IbmMode { Staircase, GhostCell }`** resource, read by the flux slow path.

### 7.2 Geometry → mask + ghosts (in `Setup`/`UpdateGeometry`)

A geometry source system (e.g. the analytic `Cylinder` with
`signed_distance(point) = √(dx²+dy²) − r` and `normal`) runs early each step:

1. **Mask refresh:** for each interior cell, `inside = sd(cell_centroid) < 0`;
   flip the registered `SolidMask`. Uses `FvMesh::cell_centroid`.
2. **Ghost-cell list:** for each solid cell with ≥1 fluid face-neighbor, find the
   closest surface (min `|sd|`), set `image_point = cell_center + 2|sd|·n̂`,
   push `(idx, info)`.
3. **Markers** (diagnostics): analytic surface markers with exact normals + areas
   for refinement-independent drag.

### 7.3 Per-substage ghost mirror (in `PostHalo`)

`apply_ibm_ghost_bc` runs **every substage, after the halo** (so image points can
sample real fluid in the MPI ghost layer): for each `(idx, info)`,
**trilinearly interpolate** primitives at `image_point` from the 8 bracketing
cells (raw/ghost-inclusive indices via `StructuredMesh::idx`, dropping solid /
ρ≤0 cells and renormalizing weights; nearest-cell fallback), then write the
mirrored state into `u[idx]`:

```text
u_ghost   = 2·u_wall − u_image      (no-slip; u_wall = info.wall_velocity, 0 for static)
T_ghost   = T_image                 (adiabatic)
p_ghost   = p_image                 (zero pressure gradient)
ρ_ghost   = p_ghost / (R · T_ghost) (ideal gas)
u[idx]    = eos.prim_to_cons(ghost-prim)
```

Because trilinear sampling needs neighbors across rank boundaries, this requires
`StructuredMesh` and a populated `u`-halo — hence `PostHalo`, after `Halo`, and
hence the IBM plugin is **structured-only at M1** (same bound as viscous, §5).

### 7.4 Flux coupling and the two modes

- **`GhostCell` mode:** solid cells are filled with mirrored fluid each substage;
  fluid↔ghost faces use the *regular* `flux_normal` — no special handling in the
  generic loop. This is the accurate mode (drag within a few % at moderate Re).
- **`Staircase` mode:** solid cells stay stagnant; fluid↔solid faces use a
  **pressure-only wall flux** (`F = [0, p·n̂_x, p·n̂_y, p·n̂_z, 0]`) handled in the
  structured *slow path* face dispatcher (the stripe-touches-solid branch), with
  solid↔solid faces contributing zero. Cheap and robust.

`PostFlux` zeroes `rhs` on solid cells either way, so solid cells never advance.

### 7.5 Wall loadings + the coupling seam to the particle stack (future note)

`compute_marker_wall_loadings` / `compute_ibm_wall_loadings` /
`aggregate_loadings_per_body` port unchanged (pressure traction `−p·n̂·A`, viscous
traction `μ·u_t/offset·A`, torque about body center), with the per-body
force/torque all-reduced via `all_reduce_sum_f64`. **The FSI seam:** when the
geometry source is MDDEM/DIRT spheres rather than an analytic cylinder, the
`body_id` maps to a particle atom id, `wall_velocity = u_body + ω×r`, and the
aggregated `BodyLoad` feeds back to particle dynamics. Because both substrates
(SOIL particles, FIELD mesh) are GRASS-native, this coupling is a **`grass_multi`
multi-app** story — two Apps sharing a comm, exchanging sphere geometry → mask
and wall loadings → forces. This is the long-term argument for both substrates
riding GRASS; it is explicitly **out of M1 scope** and recorded here as the
design target `cfd_ibm` is shaped for (note the `body_id`/`wall_velocity` fields
already in `IbmGhostInfo`).

---

## 8. Plugin wiring

Mirror DIRT exactly: a `PluginGroup` bundle built via
`PluginGroupBuilder::start::<Self>().add(...)`, each plugin declaring `provides`
/ `requires` (string capability labels) and `dependencies`
(`grass_app::type_ids![...]` hard plugin-type deps), registering its `FieldData`
and config in `build()`.

### 8.1 The plugins and their contracts

| plugin | crate | `build()` does | `provides` | `requires` | `dependencies` |
|---|---|---|---|---|---|
| `CfdStatePlugin` | cfd_state | `register_field_data!(app, CfdState::default())`; init `u` from an init closure | `["cfd_state"]` | `["field_registry","mesh"]` | `type_ids![FieldRegistryPlugin]` |
| `IdealGasPlugin` | cfd_eos | `Config::load::<EosConfig>`; add `IdealGas` resource | `["eos"]` | — | — |
| `BoundaryPlugin` | cfd_boundary | load `[boundary]`; build `BoundaryRegistry`; add `Boundary`-phase fill system | `["boundary_conditions"]` | `["cfd_state","eos","mesh"]` | — |
| `FluxPlugin` (→ `Rusanov`/`Hllc`) | cfd_riemann + cfd_solver | read `[solver].flux_scheme`; add flux resource + `Flux`-phase RHS system (structured fast path or generic) | `["riemann_flux"]` | `["cfd_state","eos","mesh","boundary_conditions"]` | — |
| `ViscousPlugin` | cfd_solver | if `[solver].viscous`, add `Flux`-phase viscous system (`StructuredMesh`-bounded) | `["viscous"]` | `["cfd_state","eos","structured_mesh","riemann_flux"]` | — |
| `IntegratorPlugin` (→ `Euler`/`Rk3`) | cfd_solver | read `[solver].time_scheme`; add `TimeIntegratorResource`; add `Setup` save/cfl + `Update`/`PostUpdate` systems; init `RkState`/`SolverState` | `["time_integrator"]` | `["cfd_state","mesh"]` | — |
| `HaloPlugin` | cfd_solver | add `Halo`-phase `halo_exchange_forward_all` system | `["halo"]` | `["field_registry","mesh"]` | — |
| `GhostCellIbmPlugin` | cfd_ibm | `register_field_data!(app, SolidMask::default())`; add `IbmGhostCells`/`IbmMode` resources; add `PostHalo` mirror system | `["ibm_ghost_cell"]` | `["cfd_state","eos","structured_mesh"]` | `type_ids![CfdStatePlugin]` |
| `CylinderIbmPlugin` | cfd_ibm | add cylinder geometry + mask-refresh/ghost-compute/marker systems (`Setup`) + drag diagnostic (`Output`) | `["cylinder_ibm"]` | `["ibm_ghost_cell","cfd_state","mesh","communication"]` | — |
| `OutputPlugin` | test_cfd | VTK dump / term-out columns in `Output` | `["output"]` | `["cfd_state","mesh"]` | — |

`mesh`/`structured_mesh` are the labels `UniformMeshPlugin` already `provides`;
`field_registry` is what `FieldRegistryPlugin` provides — so the CFD plugins'
`requires` slot straight into the FIELD substrate plugins, the way DIRT's
`HertzMindlinContactPlugin` requires `"dem_particles"`/`"neighbor_list"`.

### 8.2 The `CfdDefaultPlugins` bundle (twin of `GranularDefaultPlugins`)

```rust,ignore
pub struct CfdDefaultPlugins {
    pub init_fn: Option<Box<InitFn>>,   // cell_centroid → PrimVar, like CfdCorePlugins::with_init
    pub solid:   Option<SolidSpec>,
}

impl PluginGroup for CfdDefaultPlugins {
    fn build(self) -> PluginGroupBuilder {
        let mut b = PluginGroupBuilder::start::<Self>()
            .add(CfdStatePlugin::new(self.init_fn))   // register CfdState, init u
            .add(IdealGasPlugin)
            .add(BoundaryPlugin)
            .add(HaloPlugin)
            .add(FluxPlugin)                           // reads flux_scheme
            .add(ViscousPlugin)                        // self-gates on [solver].viscous
            .add(IntegratorPlugin);                    // reads time_scheme
        if let Some(spec) = self.solid {
            b = b.add(GhostCellIbmPlugin::new(spec.mode))
                 .add(spec.geometry_plugin());         // e.g. CylinderIbmPlugin
        }
        b.add(OutputPlugin)
    }
}
```

### 8.3 `main`/example assembly sketch

```rust,ignore
use test_cfd::prelude::*;     // glob re-exports field_core::prelude + CFD types
use field_core::prelude::*;   // FieldRegistryPlugin, UniformMeshPlugin, UniformMeshConfig

fn main() {
    let mut app = App::new();
    app.add_plugins(grass_io::InputPlugin)                    // TOML config
       .add_plugins(CorePlugins)                              // GRASS: clock, comm, run, term-out
       .add_plugins(FieldRegistryPlugin)                      // FIELD: the registry  (provides "field_registry")
       .add_plugins(UniformMeshPlugin::new(mesh_cfg))         // FIELD: the mesh      (provides "mesh","structured_mesh")
       .add_plugins(CfdDefaultPlugins::new()                  // the CFD physics
            .with_init(|x| sod_initial_state(x))
            .with_solid(SolidSpec::cylinder(/* center, radius, IbmMode::GhostCell */)));
    app.start();
}
```

This is the line-for-line analog of DIRT's `App::new().add_plugins(CorePlugins)
.add_plugins(GranularDefaultPlugins).add_plugins(WallPlugin)`.

### 8.4 Prelude

`test_cfd::prelude` re-exports (mirroring `toy_cfd::prelude` / `dirt_core::prelude`):
`CfdDefaultPlugins`, the individual plugins, `ConsVar`/`PrimVar`/`CfdState`,
`Eos`/`IdealGas`, `RiemannFlux`/`Rusanov`/`Hllc`, `BoundaryRegistry` + stock BCs,
`SolverConfig`/`SolverState`, IBM types, and a glob of `field_core::prelude::*`
(which brings `BoundarySide`, `FvMesh`, `StructuredMesh`, `MeshScheduleSet`,
`register_field_data!`, `UniformMeshPlugin`, …).

---

## 9. Milestone M1 — validation and build order

**Targets to reproduce (toy-cfd parity):**
1. **Sod shock tube** (1-D Riemann problem in a 3-D box) — inviscid Euler;
   validates state registration, halo, BCs, flux, RK3 against the analytic Sod
   solution.
2. **Laminar cylinder drag Re-sweep** (Henderson 1995 `Cd(Re)`) — viscous +
   ghost-cell IBM; the definition-of-done from the repo `PLAN.md`
   ("cylinder-drag validation matches toy-cfd numbers; MPI ghost tests pass").

**Build order — get something running end-to-end ASAP, then add layers:**

1. **`cfd_state` + `cfd_eos`.** `ConsVar: Component`, `CfdState` derives
   `FieldData`, `IdealGas`. Unit-test `Component` round-trip and registration via
   `FieldRegistry` (FIELD's own tests are the template). Nothing runs yet.
2. **`cfd_riemann` + generic `Flux` system on `UniformMesh` via `for_each_face`.**
   Inviscid Euler, Rusanov, Euler integrator, `Halo` via
   `halo_exchange_forward_all`. Hardcode periodic/outflow BCs inline. **First
   end-to-end run:** a uniform-flow / advection sanity case.
3. **`cfd_boundary` + the `Boundary` phase.** `BoundaryRegistry` indexed by
   `BoundarySide`; ghost fill before flux. Now Sod's reflecting/outflow ends work.
4. **`cfd_solver` RK3 + CFL.** Swap Euler→TVD-RK3, add `cfl_system` with
   `all_reduce_min_f64`, the `RkState` substage loop over `MeshScheduleSet`.
   **Validate Sod** against the analytic solution (serial, then 2-rank MPI to
   exercise the halo).
5. **Structured fast path.** Add the fused `<M: StructuredMesh>` flux sweep +
   `SolverScratch` + `flux_scheme_cached`; check bit-for-bit-ish agreement with
   the generic path on Sod; benchmark the speedup.
6. **`ViscousPlugin`** (`StructuredMesh`-bounded central differences) + viscous
   CFL term. Validate on a Couette/Poiseuille or Blasius-type case.
7. **`cfd_ibm`** — `SolidMask` (`#[forward] bool`), analytic cylinder geometry,
   `PostHalo` ghost mirror, marker loadings, drag diagnostic. **Validate the
   cylinder Re-sweep `Cd(Re)`** against toy-cfd and Henderson. Add `Hllc`+Thornber
   if the wake needs it.

Each step leaves the build green and a runnable example; the order front-loads
the cheapest path to a validated physics result (Sod by step 4).

---

## 10. Open questions / risks

1. **Generic `for_each_face` dynamic-dispatch cost.** The `&mut dyn FnMut(&Face)`
   indirect call per face + recomputed sound speeds is the perf risk FIELD's own
   docs flag. *Mitigation:* the `StructuredMesh` fused fast path (§3.2) is the
   default for production; the generic path is correctness/portability only.
   Open: whether to also offer a generic-but-cached path (precompute `prim` for
   all cells, pass `&Scratch` into the closure) for unstructured meshes — likely
   yes at M2.
2. **`RiemannFlux` is direction-indexed, faces are normal-vectored.** M1 leans on
   axis-aligned normals (recover `dir` from `area_normal`); a true
   `flux_normal(uL,uR,eos,n̂)` with momentum rotation is needed before any rotated
   or unstructured mesh. *Decision:* add the `flux_normal` method now with an
   axis-aligned fast path; implement rotation when needed. Risk: subtle bugs in
   the tangential-momentum rotation for HLLC's star state.
3. **`u0` RK backup placement.** Untagged column vs. separate store (§2.3).
   **RESOLVED:** `field_derive` resizes *every* column but only exchanges/zeroes
   *tagged* ones, so an untagged `Vec<ConsVar>` is resize-only — exactly what `u0`
   needs. Already test-covered (`field_core/tests/derive.rs`, the `backup` column).
   Use the single-store `u0` option (§2.3 recommended).
4. **`rhs` exchange semantics.** M1 assumes every interior owner's RHS is
   computed locally (full ghost layer ⇒ no cross-rank flux scatter needed), so
   `rhs` is `#[zero]` only. If a future scheme scatters flux onto ghost owners
   (e.g. node-centered or AMR fine→coarse), `rhs` must become `#[reverse]` and
   `PostFlux` must run `halo_exchange_reverse_all`. FIELD kept the
   forward/reverse split precisely for this; flagged for AMR (M2).
5. **Periodic BC on a decomposed mesh.** Periodicity is naturally a halo wrap, not
   a physical `patch`. On a single rank it is a `Boundary`-phase
   `periodic_mirror`; under MPI it should be expressed as a halo link
   (`HaloPlan` wrap), which `UniformMesh::build_halo_plan` does *not* currently
   emit for periodic directions. *Open:* either FIELD's `UniformMesh` grows
   periodic halo links (substrate change), or `cfd_boundary` keeps a structured
   periodic ghost-copy for M1. Lean on the latter for M1, request the former for
   M2. **RESOLVED:** FIELD's `UniformMesh` now emits periodic halo links from a
   `MeshPeriodicity` resource, which `StructuredBoundaryPlugin` adds for
   `Periodic` axes; the local ghost-copy is gone and `tests/periodic_mpi.rs`
   checks 1–4 ranks agree.
6. **Scheduler loop construct for RK substages.** §6.2 assumes GRASS's scheduler
   can re-run a band of phases N times per iter (the schedule-set doc promises
   "loop constructs"). **RESOLVED:** GRASS provides
   `Schedule::builder().loop_until(until, max_iters, OnMax, |body| …)` (a
   `ScheduleNode::Loop` re-executing its body until the condition flips), plus
   `then_variant` for a Save-before / Check-after stage, installed via
   `App::set_schedule`. Put the `Halo..PostUpdate` band in the loop body,
   `save_state` (u0←u) + `cfl` in a `then_variant` before it, and `Output` after.
   The phase-decomposition approach is feasible; the toy-cfd fallback is not needed.
7. **Multi-layer ghost fill for wide stencils.** The viscous stencil reads two
   ghost layers; the `Boundary` system must fill all `ng` layers, not just the
   face-adjacent one. The generic `for_each_face` boundary iteration only exposes
   the first ghost; use the `StructuredMesh` layer-fill variant (§4) when
   `ng > 1`. Ensure `UniformMeshConfig.ng` is ≥ the stencil half-width.
8. **Potential FIELD contract extensions surfaced by this port** (candidates to
   feed back upstream): `flux_normal` is physics-side, fine; but (a) periodic
   halo links (#5), (b) a `for_each_boundary_face`/layered-ghost accessor so BCs
   don't depend on `StructuredMesh`, and (c) eventually `AdaptiveMesh` reflux are
   the three places `dev_field_efvm` will press on the substrate. None block M1.
