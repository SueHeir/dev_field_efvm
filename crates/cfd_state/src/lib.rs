//! Conserved/primitive variables and the registered per-cell CFD state.
//!
//! This is the `cfd_state` of toy-cfd minus the parts FIELD now owns: the
//! `FieldData`/`FieldRegistry` definitions are gone (they live in `field_core`),
//! and storage is no longer a hand-rolled `FlowField`. Instead [`ConsVar`]
//! implements [`field_core::Component`] (WIDTH 5) so a `Vec<ConsVar>` halo-
//! exchanges directly, and [`CfdState`] derives `FieldData` so the substrate
//! moves/resets it generically.
//!
//! This crate is the mesh-agnostic, EOS-agnostic core: just the variable types
//! and their algebra. It is the dev_field_efvm analog of DIRT's `dirt_atom` (which
//! defines `DemAtom` over SOIL's `Atom`).

use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// Conservative variables: density, momentum (3), total energy per unit volume.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConsVar {
    pub rho: f64,
    pub rho_u: f64,
    pub rho_v: f64,
    pub rho_w: f64,
    pub rho_e: f64,
}

impl ConsVar {
    pub const ZERO: ConsVar = ConsVar {
        rho: 0.0,
        rho_u: 0.0,
        rho_v: 0.0,
        rho_w: 0.0,
        rho_e: 0.0,
    };

    pub fn new(rho: f64, rho_u: f64, rho_v: f64, rho_w: f64, rho_e: f64) -> Self {
        ConsVar { rho, rho_u, rho_v, rho_w, rho_e }
    }

    /// Velocity components `(u, v, w)`.
    pub fn velocity(&self) -> [f64; 3] {
        let inv_rho = 1.0 / self.rho;
        [self.rho_u * inv_rho, self.rho_v * inv_rho, self.rho_w * inv_rho]
    }

    /// Kinetic energy per unit volume: `0.5 * rho * |u|^2`.
    pub fn kinetic_energy(&self) -> f64 {
        0.5 * (self.rho_u * self.rho_u + self.rho_v * self.rho_v + self.rho_w * self.rho_w)
            / self.rho
    }
}

/// `ConsVar` is the per-cell state component: it serializes to its 5 `f64`s, so
/// `#[forward] u: Vec<ConsVar>` halo-exchanges with no hand-written packing.
impl field_core::Component for ConsVar {
    const WIDTH: usize = 5;
    #[inline]
    fn write(&self, buf: &mut Vec<f64>) {
        buf.extend_from_slice(&[self.rho, self.rho_u, self.rho_v, self.rho_w, self.rho_e]);
    }
    #[inline]
    fn read(buf: &[f64]) -> Self {
        ConsVar { rho: buf[0], rho_u: buf[1], rho_v: buf[2], rho_w: buf[3], rho_e: buf[4] }
    }
    #[inline]
    fn add_from(&mut self, buf: &[f64]) {
        self.rho += buf[0];
        self.rho_u += buf[1];
        self.rho_v += buf[2];
        self.rho_w += buf[3];
        self.rho_e += buf[4];
    }
    #[inline]
    fn zero() -> Self {
        ConsVar::ZERO
    }
}

impl Add for ConsVar {
    type Output = ConsVar;
    #[inline(always)]
    fn add(self, rhs: ConsVar) -> ConsVar {
        ConsVar {
            rho: self.rho + rhs.rho,
            rho_u: self.rho_u + rhs.rho_u,
            rho_v: self.rho_v + rhs.rho_v,
            rho_w: self.rho_w + rhs.rho_w,
            rho_e: self.rho_e + rhs.rho_e,
        }
    }
}

impl Sub for ConsVar {
    type Output = ConsVar;
    #[inline(always)]
    fn sub(self, rhs: ConsVar) -> ConsVar {
        ConsVar {
            rho: self.rho - rhs.rho,
            rho_u: self.rho_u - rhs.rho_u,
            rho_v: self.rho_v - rhs.rho_v,
            rho_w: self.rho_w - rhs.rho_w,
            rho_e: self.rho_e - rhs.rho_e,
        }
    }
}

impl Mul<f64> for ConsVar {
    type Output = ConsVar;
    #[inline(always)]
    fn mul(self, s: f64) -> ConsVar {
        ConsVar {
            rho: self.rho * s,
            rho_u: self.rho_u * s,
            rho_v: self.rho_v * s,
            rho_w: self.rho_w * s,
            rho_e: self.rho_e * s,
        }
    }
}

impl Mul<ConsVar> for f64 {
    type Output = ConsVar;
    #[inline(always)]
    fn mul(self, rhs: ConsVar) -> ConsVar {
        rhs * self
    }
}

impl AddAssign for ConsVar {
    #[inline(always)]
    fn add_assign(&mut self, rhs: ConsVar) {
        self.rho += rhs.rho;
        self.rho_u += rhs.rho_u;
        self.rho_v += rhs.rho_v;
        self.rho_w += rhs.rho_w;
        self.rho_e += rhs.rho_e;
    }
}

impl SubAssign for ConsVar {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: ConsVar) {
        self.rho -= rhs.rho;
        self.rho_u -= rhs.rho_u;
        self.rho_v -= rhs.rho_v;
        self.rho_w -= rhs.rho_w;
        self.rho_e -= rhs.rho_e;
    }
}

/// Primitive variables: density, velocity (3), pressure, temperature.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PrimVar {
    pub rho: f64,
    pub u: f64,
    pub v: f64,
    pub w: f64,
    pub p: f64,
    pub t: f64,
}

impl PrimVar {
    pub fn new(rho: f64, u: f64, v: f64, w: f64, p: f64, t: f64) -> Self {
        PrimVar { rho, u, v, w, p, t }
    }

    pub fn velocity_magnitude(&self) -> f64 {
        (self.u * self.u + self.v * self.v + self.w * self.w).sqrt()
    }
}

/// The registered per-cell CFD state store.
///
/// - `u`  — conserved state. `#[forward]`: copied owner→ghost each substage so
///   boundary fluxes see neighbor state.
/// - `rhs` — flux-divergence accumulator. `#[zero]`: reset to `ConsVar::ZERO`
///   each substage by the registry before flux assembly.
/// - `u0` — RK stage backup. *Untagged*: resized with the mesh but never
///   exchanged or zeroed (the "lives on the cell" row of the `FieldData`
///   contract). Written once per full step by [`CfdState::save_state`].
#[derive(field_core::FieldData, Default)]
pub struct CfdState {
    #[forward]
    pub u: Vec<ConsVar>,
    #[zero]
    pub rhs: Vec<ConsVar>,
    pub u0: Vec<ConsVar>,
}

impl CfdState {
    /// Snapshot `u → u0` for multi-stage integration. Call once at the start of a
    /// full step (the `Setup` phase), before any RK substage runs.
    pub fn save_state(&mut self) {
        self.u0.copy_from_slice(&self.u);
    }
}

/// Per-cell solid flag for immersed boundaries. `#[forward]` so a rank's halo
/// learns which neighbor-rank cells are solid (image-point samples must skip
/// another rank's solids). Lives here, with the other registered per-cell state,
/// so both `cfd_ibm` and the viscous solver can read it without depending on
/// each other.
#[derive(field_core::FieldData, Default)]
pub struct SolidMask {
    #[forward]
    pub solid: Vec<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use field_core::{Component, FieldRegistry};

    #[test]
    fn cons_var_arithmetic() {
        let a = ConsVar::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let b = ConsVar::new(0.5, 1.0, 1.5, 2.0, 2.5);
        assert_eq!((a + b).rho_e, 7.5);
        assert_eq!((a - b).rho, 0.5);
        assert_eq!((a * 2.0).rho_e, 10.0);
        assert_eq!((3.0 * b).rho, 1.5);
    }

    #[test]
    fn cons_var_velocity_and_ke() {
        let u = ConsVar::new(2.0, 4.0, 6.0, 8.0, 100.0);
        assert_eq!(u.velocity(), [2.0, 3.0, 4.0]);
        // 0.5 * (16+36+64)/2 = 0.5 * 116/2 = 29
        assert!((u.kinetic_energy() - 29.0).abs() < 1e-12);
    }

    #[test]
    fn consvar_component_round_trip() {
        let c = ConsVar::new(1.0, -2.0, 3.0, -4.0, 5.0);
        let mut buf = Vec::new();
        c.write(&mut buf);
        assert_eq!(buf, vec![1.0, -2.0, 3.0, -4.0, 5.0]);
        assert_eq!(ConsVar::read(&buf), c);
        assert_eq!(<ConsVar as Component>::WIDTH, 5);
    }

    #[test]
    fn cfdstate_columns_behave() {
        let mut reg = FieldRegistry::new();
        reg.register(CfdState::default());
        reg.resize_all(4);
        // u is forward (5 f64/cell); rhs is zero-only; u0 untagged → no exchange.
        assert_eq!(reg.forward_size(), 5);
        {
            let mut s = reg.get_mut::<CfdState>().unwrap();
            s.u[0] = ConsVar::new(1.0, 0.0, 0.0, 0.0, 2.0);
            s.rhs[0] = ConsVar::new(9.0, 9.0, 9.0, 9.0, 9.0);
            assert_eq!(s.u0.len(), 4); // untagged column still resized
            s.save_state();
            assert_eq!(s.u0[0], s.u[0]);
        }
        // forward pack moves exactly u's 5 components.
        let mut buf = Vec::new();
        reg.pack_forward_all(0, &mut buf);
        assert_eq!(buf, vec![1.0, 0.0, 0.0, 0.0, 2.0]);
        // zero resets rhs only.
        reg.zero_all(4);
        assert_eq!(reg.get::<CfdState>().unwrap().rhs[0], ConsVar::ZERO);
    }
}
