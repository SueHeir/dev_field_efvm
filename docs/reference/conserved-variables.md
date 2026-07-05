# `ConsVar` — the conserved-variable state vector

`ConsVar` is the per-cell primary unknown of the compressible flow solver: the
five-component **vector of conserved variables** that the finite-volume method
advances in time. It is defined in
[`crates/cfd_state/src/lib.rs:18`](../../crates/cfd_state/src/lib.rs) and is the
single most connected type in the codebase — nearly every physics crate
(`cfd_eos`, `cfd_riemann`, `cfd_solver`, `cfd_boundary`, `cfd_ibm`, `cfd_output`)
reads or writes it. This page explains what it is, why the solver uses this
particular set of variables, the units of each component, and how a `ConsVar`
flows through one time step.

```rust
pub struct ConsVar {
    pub rho:   f64,  // ρ    — mass density
    pub rho_u: f64,  // ρu   — x-momentum density
    pub rho_v: f64,  // ρv   — y-momentum density
    pub rho_w: f64,  // ρw   — z-momentum density
    pub rho_e: f64,  // ρE   — total energy per unit volume
}
```

---

## 1. Why *conserved* variables?

The solver integrates the **compressible Euler / Navier–Stokes equations**, which
are conservation laws for mass, momentum, and energy. Written in *conservation
(divergence) form* for an inviscid gas in three dimensions, they are

$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla\cdot\mathbf{F}(\mathbf{U}) = 0,
\qquad
\mathbf{U} =
\begin{pmatrix} \rho \\ \rho u \\ \rho v \\ \rho w \\ \rho E \end{pmatrix},
$$

where $\mathbf{U}$ is exactly the vector of **conserved variables** and
$\mathbf{F}$ is the flux tensor (§3). This is the standard form given in
Toro (2009, §1.1 and §3.1), LeVeque (2002, §1.1), Hirsch (2007, §1.2), and
Anderson (1995, §2.5).

The choice of $(\rho,\ \rho u,\ \rho v,\ \rho w,\ \rho E)$ — rather than the
"primitive" variables $(\rho,\ u,\ v,\ w,\ p)$ — is deliberate and physically
essential:

- **Discrete conservation.** A finite-volume update of the conserved variables
  changes a cell's mass/momentum/energy only by the fluxes through its faces
  (§4). Because an interior face's outgoing flux from one cell is the incoming
  flux to its neighbour, the totals are conserved to machine precision. This is
  the property that lets the scheme capture shocks at the correct speed
  (the Lax–Wendroff theorem requires conservation form; LeVeque 2002, §12.1).
- **Correct weak solutions.** Only the conservation form admits the correct
  Rankine–Hugoniot jump conditions across discontinuities (Toro 2009, §2.4.2).
  A non-conservative (primitive) discretisation can converge to a *wrong* shock
  speed.

The primitive variables are recovered from `ConsVar` on demand by the equation of
state (§2); the solver stores and transports only the conserved set.

> **Naming note.** The field is called `rho_e`, but it holds the **total** energy
> density $\rho E$ (internal + kinetic), *not* the internal energy $\rho e$.
> The split is $\rho E = \rho e + \tfrac12\rho\lVert\mathbf{u}\rVert^2$; see §2.
> `ConsVar::kinetic_energy()` returns the second term and `IdealGas::pressure`
> subtracts it to recover the internal part.

---

## 2. What each component is (and its SI units)

Let $\mathbf{u}=(u,v,w)$ be the flow velocity and $E$ the **specific total
energy** (total energy per unit *mass*). The five components are *volumetric
densities* of the conserved quantities:

| Field    | Symbol   | Quantity                        | SI units                          |
|----------|----------|---------------------------------|-----------------------------------|
| `rho`    | $\rho$   | mass density                    | $\mathrm{kg\,m^{-3}}$             |
| `rho_u`  | $\rho u$ | $x$-momentum density            | $\mathrm{kg\,m^{-2}\,s^{-1}}$     |
| `rho_v`  | $\rho v$ | $y$-momentum density            | $\mathrm{kg\,m^{-2}\,s^{-1}}$     |
| `rho_w`  | $\rho w$ | $z$-momentum density            | $\mathrm{kg\,m^{-2}\,s^{-1}}$     |
| `rho_e`  | $\rho E$ | total energy per unit volume    | $\mathrm{J\,m^{-3}=kg\,m^{-1}\,s^{-2}}$ |

The total energy density decomposes into internal and kinetic parts:

$$
\rho E \;=\; \rho e \;+\; \tfrac12\,\rho\lVert\mathbf{u}\rVert^2,
$$

with $e$ the specific internal energy (Toro 2009, §1.1; Anderson 1995, §2.3).
The velocity is recovered by dividing out the density,
$\mathbf{u} = (\rho\mathbf{u})/\rho$ — this is `ConsVar::velocity()` — and the
kinetic term $\tfrac12\rho\lVert\mathbf{u}\rVert^2$ is `ConsVar::kinetic_energy()`.

### Closure: the equation of state

The system has five equations but six thermodynamic unknowns
$(\rho, u, v, w, p, e)$; an **equation of state** closes it. For a calorically
perfect ideal gas ([`cfd_eos::IdealGas`](../../crates/cfd_eos/src/lib.rs)) with
ratio of specific heats $\gamma$ and specific gas constant $R$:

$$
p = (\gamma-1)\Bigl(\rho E - \tfrac12\rho\lVert\mathbf{u}\rVert^2\Bigr),
\qquad
T = \frac{p}{\rho R},
\qquad
a = \sqrt{\frac{\gamma p}{\rho}} .
$$

These are the standard ideal-gas pressure, temperature, and sound-speed relations
(Toro 2009, §1.2.3, eqs. 1.20–1.24; Anderson 1995, §2.3). They are implemented
verbatim in `IdealGas::pressure`, `IdealGas::cons_to_prim`, and
`IdealGas::sound_speed`. The default gas is air, $\gamma=1.4$,
$R=287.058\ \mathrm{J\,kg^{-1}\,K^{-1}}$.

`PrimVar` $(\rho, u, v, w, p, T)$ is the primitive counterpart used only where
pressure/temperature are needed directly (boundary conditions, viscous stresses,
output). `cons_to_prim` / `prim_to_cons` convert between the two and are exact
inverses (tested in `cfd_eos`).

---

## 3. The physical flux

For the Euler equations the flux in coordinate direction $d\in\{x,y,z\}$, with
$u_d$ the velocity component along $d$, is

$$
\mathbf{F}_d(\mathbf{U}) =
\begin{pmatrix}
\rho u_d \\
\rho u\,u_d + p\,\delta_{dx} \\
\rho v\,u_d + p\,\delta_{dy} \\
\rho w\,u_d + p\,\delta_{dz} \\
(\rho E + p)\,u_d
\end{pmatrix},
$$

where $\delta_{dx}$ etc. is the Kronecker delta (the pressure adds only to the
momentum component *along* $d$). This is the standard compressible-Euler flux
(Toro 2009, §3.1.1, eqs. 3.3–3.5; LeVeque 2002, §14.3). It is implemented exactly
by `cfd_riemann::physical_flux_inline`: mass is advected as $\rho u_d$, momentum
as $\rho\mathbf{u}\,u_d$ plus the pressure term, and energy as the
**enthalpy-weighted** flux $(\rho E + p)u_d$ (note the $+p$: the energy flux
carries the flow work, Toro 2009, eq. 3.5).

At a cell interface the two neighbouring `ConsVar` states generally differ, so the
face flux is a **numerical flux** $\hat{\mathbf F}$ that solves (exactly or
approximately) the local Riemann problem. `cfd_riemann` ships two, both functions
of the left/right `ConsVar`:

- **Rusanov / local Lax–Friedrichs:**
  $\hat{\mathbf F} = \tfrac12(\mathbf F_L + \mathbf F_R)
  - \tfrac12\,\lambda_{\max}(\mathbf U_R - \mathbf U_L)$,
  with $\lambda_{\max}=\max(|u_L|+a_L,\ |u_R|+a_R)$ (Toro 2009, §10.5.1).
- **HLLC** three-wave solver (Toro 2009, ch. 10), optionally with the Thornber
  et al. (2008) low-Mach correction.

---

## 4. How `ConsVar` flows through the solver

`ConsVar` is stored as the columns of
[`CfdState`](../../crates/cfd_state/src/lib.rs) — one `ConsVar` per mesh cell:

| Column | Role | `FieldData` tag |
|--------|------|-----------------|
| `u`   | the conserved state $\mathbf U$ | `#[forward]` — copied owner→ghost each substage |
| `rhs` | flux-divergence accumulator $\mathrm{d}\mathbf U/\mathrm dt$ | `#[zero]` — reset each substage |
| `u0`  | Runge–Kutta stage backup of $\mathbf U^n$ | untagged — resized with the mesh only |

Because `ConsVar` implements the `field_core::Component` trait with `WIDTH = 5`,
a `Vec<ConsVar>` serialises to a flat buffer of 5 `f64`/cell and **halo-exchanges
across MPI ranks with no hand-written packing** — the mesh substrate moves the
state generically. The trait, and a worked `ConsVar` `Component` example, live in
the sibling **FIELD** substrate repo at
[`field_core/src/component.rs:12`](../../../field/crates/field_core/src/component.rs)
(path-dependency `../field/crates/field_core` in this repo's `Cargo.toml`); the
concrete `impl Component for ConsVar` is at
[`crates/cfd_state/src/lib.rs:54`](../../crates/cfd_state/src/lib.rs).

### The semi-discrete finite-volume update

Integrating the conservation law over a cell $i$ of volume $V_i$ and applying the
divergence theorem gives the standard finite-volume semi-discretisation
(LeVeque 2002, §19.1; Toro 2009, §16.1):

$$
\frac{\mathrm d \mathbf U_i}{\mathrm d t}
= -\frac{1}{V_i}\sum_{f\,\in\,\partial i}
   \hat{\mathbf F}(\mathbf U_L,\mathbf U_R;\hat{\mathbf n}_f)\,A_f ,
$$

the sum running over the cell's faces, each with area $A_f$ and outward unit
normal $\hat{\mathbf n}_f$. The right-hand side is precisely the `rhs` column.
`cfd_solver::flux_system` assembles it face-by-face: it subtracts
$\hat{\mathbf F}\,A_f/V_\text{owner}$ from the owner's `rhs` and adds the same to
the neighbour's — the two-sided scatter that makes the scheme discretely
conservative.

### One time step, phase by phase

The per-step work is split across `field_core`'s `MeshScheduleSet` phases; every
phase reads or mutates `ConsVar`:

1. **`Setup`** — `cfl_system` scans `u`, converts each cell to primitives, and
   sets the timestep from the CFL condition
   $\Delta t = \mathrm{CFL}/\max_i \lambda_i$ (Toro 2009, §6.3.2; the
   viscous case adds the parabolic spectral radius).
2. **`Halo`** — `halo_system` forward-exchanges the `#[forward]` column `u` so
   each rank's ghost cells hold neighbour-rank state.
3. **`Boundary`** — `cfd_boundary` fills physical-boundary ghost cells: given the
   interior `ConsVar`, each BC returns the ghost `ConsVar`.
4. **`PreFlux`** — `zero_rhs_system` resets the `#[zero]` column `rhs` to
   `ConsVar::ZERO`; with MUSCL enabled, `compute_slopes_system` also builds
   minmod-limited slopes of `u` for 2nd-order face reconstruction.
5. **`Flux`** — `flux_system` (or the MUSCL / structured-fast variants) computes
   the numerical flux across every face and accumulates the flux divergence into
   `rhs`; the optional `viscous_system` adds the Navier–Stokes $\mu\nabla^2\mathbf
   u$ and $\kappa\nabla^2 T$ terms to `rhs`.
6. **`Update`** — `update_system` calls the time integrator, which advances
   `u` from `rhs` (§below).
7. **`Output`** — `cfd_output` converts each `ConsVar` back to primitives and
   writes density, pressure, temperature, velocity, and Mach number to VTK.

### Time integration

The integrator advances the conserved state using the operators on `ConsVar`
(`Add`, `Sub`, `Mul<f64>`, `AddAssign` — the vector-space algebra defined
alongside the struct). Two schemes are provided:

- **Forward Euler:** $\mathbf U^{n+1} = \mathbf U^n + \Delta t\,\mathrm{RHS}$.
- **SSP-RK3 (Shu–Osher):** the strong-stability-preserving third-order scheme
  (Gottlieb & Shu 1998, eq. 4.2; Shu & Osher 1988):
  $$
  \begin{aligned}
  \mathbf U^{(1)} &= \mathbf U^n + \Delta t\,\mathrm{RHS}(\mathbf U^n),\\
  \mathbf U^{(2)} &= \tfrac34\mathbf U^n + \tfrac14\bigl(\mathbf U^{(1)} + \Delta t\,\mathrm{RHS}(\mathbf U^{(1)})\bigr),\\
  \mathbf U^{n+1} &= \tfrac13\mathbf U^n + \tfrac23\bigl(\mathbf U^{(2)} + \Delta t\,\mathrm{RHS}(\mathbf U^{(2)})\bigr).
  \end{aligned}
  $$
  The backup $\mathbf U^n$ is the `u0` column, saved once per step by
  `CfdState::save_state`.

---

## 5. Summary

`ConsVar` is the discrete embodiment of the compressible flow's conserved
quantities — mass, three momenta, and total energy, all as per-unit-volume
densities. Storing and transporting *this* set (rather than primitives) is what
makes the finite-volume solver discretely conservative and shock-capturing. Every
stage of a time step — CFL, halo exchange, boundary conditions, flux assembly,
integration, output — is an operation on `ConsVar`, which is why it is the central
type of the solver.

---

## References

1. E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
   3rd ed., Springer, 2009. (Conservation form §1.1, §3.1; ideal-gas EOS §1.2.3;
   Euler flux §3.1.1; Rusanov §10.5.1; HLLC ch. 10; CFL §6.3.2; Rankine–Hugoniot
   §2.4.2.)
2. R. J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge
   University Press, 2002. (Conservation laws §1.1; finite-volume method §19.1;
   Lax–Wendroff / conservation-form necessity §12.1; Euler equations §14.3.)
3. C. Hirsch, *Numerical Computation of Internal and External Flows*, Vol. 1,
   2nd ed., Butterworth-Heinemann, 2007. (Governing equations and conservative
   variables §1.2.)
4. J. D. Anderson, *Computational Fluid Dynamics: The Basics with Applications*,
   McGraw-Hill, 1995. (Conservation form of the governing equations §2.5; energy
   and thermodynamic relations §2.3.)
5. S. Gottlieb and C.-W. Shu, "Total Variation Diminishing Runge–Kutta Schemes,"
   *Mathematics of Computation*, 67(221):73–85, 1998. (SSP-RK3, eq. 4.2.)
6. C.-W. Shu and S. Osher, "Efficient Implementation of Essentially
   Non-oscillatory Shock-Capturing Schemes," *Journal of Computational Physics*,
   77(2):439–471, 1988. (Original SSP-RK3 form.)
7. B. Thornber, A. Mosedale, D. Drikakis, D. Youngs, and R. J. R. Williams, "An
   improved reconstruction method for compressible flows with low Mach number
   features," *Journal of Computational Physics*, 227(10):4873–4894, 2008.
   (Low-Mach correction used in the HLLC flux.)
</content>
</invoke>
