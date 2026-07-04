//! Equation of state and transport: ideal gas with Sutherland viscosity.
//!
//! Pure functions of [`ConsVar`]/[`PrimVar`] — no mesh, no substrate types. This
//! is the test-cfd analog of DIRT's material/constitutive layer (`MaterialTable`):
//! the physics of "given state, what are pressure / temperature / sound speed /
//! transport coefficients", swappable behind the [`Eos`] trait.

use cfd_state::{ConsVar, PrimVar};
use std::sync::Arc;

/// The active equation of state as a GRASS resource (`Res<EosResource>`). Lives
/// here, with the EOS, so both the solver and the boundary conditions can read it
/// without either depending on the other.
pub struct EosResource(pub Arc<dyn Eos>);

/// Equation of state interface. New gas models (real gas, multi-species) add
/// another impl without touching the solver.
pub trait Eos {
    fn cons_to_prim(&self, u: &ConsVar) -> PrimVar;
    fn prim_to_cons(&self, p: &PrimVar) -> ConsVar;
    fn pressure(&self, u: &ConsVar) -> f64;
    fn sound_speed(&self, u: &ConsVar) -> f64;
}

/// Calorically perfect ideal gas.
#[derive(Debug, Clone, Copy)]
pub struct IdealGas {
    /// Ratio of specific heats (cp/cv).
    pub gamma: f64,
    /// Specific gas constant R = cp - cv [J/(kg·K)].
    pub r_gas: f64,
}

impl IdealGas {
    pub fn air() -> Self {
        IdealGas {
            gamma: 1.4,
            r_gas: 287.058,
        }
    }
}

impl Eos for IdealGas {
    fn cons_to_prim(&self, u: &ConsVar) -> PrimVar {
        let inv_rho = 1.0 / u.rho;
        let vel_u = u.rho_u * inv_rho;
        let vel_v = u.rho_v * inv_rho;
        let vel_w = u.rho_w * inv_rho;
        let ke = 0.5 * u.rho * (vel_u * vel_u + vel_v * vel_v + vel_w * vel_w);
        let p = (self.gamma - 1.0) * (u.rho_e - ke);
        let t = p * inv_rho / self.r_gas;
        PrimVar::new(u.rho, vel_u, vel_v, vel_w, p, t)
    }

    fn prim_to_cons(&self, p: &PrimVar) -> ConsVar {
        let ke = 0.5 * p.rho * (p.u * p.u + p.v * p.v + p.w * p.w);
        let rho_e = p.p / (self.gamma - 1.0) + ke;
        ConsVar::new(p.rho, p.rho * p.u, p.rho * p.v, p.rho * p.w, rho_e)
    }

    fn pressure(&self, u: &ConsVar) -> f64 {
        (self.gamma - 1.0) * (u.rho_e - u.kinetic_energy())
    }

    fn sound_speed(&self, u: &ConsVar) -> f64 {
        let p = self.pressure(u);
        (self.gamma * p / u.rho).sqrt()
    }
}

/// Sutherland's law for dynamic viscosity of air.
///
/// `mu(T) = mu_ref * (T / T_ref)^(3/2) * (T_ref + S) / (T + S)`,
/// with `T_ref = 291.15 K`, `mu_ref = 1.827e-5 Pa·s`, `S = 120.0 K`.
pub fn sutherland_viscosity(t: f64) -> f64 {
    const MU_REF: f64 = 1.827e-5;
    const T_REF: f64 = 291.15;
    const S: f64 = 120.0;
    MU_REF * (t / T_REF).powf(1.5) * (T_REF + S) / (t + S)
}

/// Thermal conductivity from the Prandtl number: `k = mu * cp / Pr`, `Pr = 0.72`.
pub fn thermal_conductivity(mu: f64, gamma: f64, r_gas: f64) -> f64 {
    let cp = gamma * r_gas / (gamma - 1.0);
    const PR: f64 = 0.72;
    mu * cp / PR
}

/// Dynamic-viscosity transport model, selected per case as a GRASS resource
/// (`Res<Viscosity>`). `Sutherland` is air's physical `μ(T)`; `Constant` pins `μ`
/// so a case can target a chosen Reynolds number `Re = ρ U L / μ` directly.
///
/// The constant option is what makes incompressible-limit validations tractable in
/// a *compressible* solver: to resolve, say, `Re = 20` past a cylinder at a sane
/// Mach number you need `μ ≈ O(0.1)`, ~10⁴× air's physical `1.8e-5 Pa·s`. It is
/// equally the knob the DEM–CFD drag closure needs to set a particle Reynolds
/// number. Default is [`Viscosity::Sutherland`], so existing air cases are
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Viscosity {
    /// Air's Sutherland law `μ(T)` (the physical default).
    #[default]
    Sutherland,
    /// A fixed dynamic viscosity `μ` [Pa·s], independent of temperature.
    Constant(f64),
}

impl Viscosity {
    /// Dynamic viscosity `μ` at temperature `t` [K].
    #[inline]
    pub fn mu(&self, t: f64) -> f64 {
        match self {
            Viscosity::Sutherland => sutherland_viscosity(t),
            Viscosity::Constant(mu) => *mu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cons_prim_roundtrip() {
        let eos = IdealGas::air();
        let t = 101325.0 / (1.225 * 287.058);
        let prim = PrimVar::new(1.225, 100.0, 0.0, 0.0, 101325.0, t);
        let cons = eos.prim_to_cons(&prim);
        let prim2 = eos.cons_to_prim(&cons);
        assert!((prim2.rho - prim.rho).abs() < 1e-10);
        assert!((prim2.u - prim.u).abs() < 1e-10);
        assert!((prim2.p - prim.p).abs() < 1e-6);
        assert!((prim2.t - prim.t).abs() < 1e-6);
    }

    #[test]
    fn sound_speed_air() {
        let eos = IdealGas::air();
        let prim = PrimVar::new(1.225, 0.0, 0.0, 0.0, 101325.0, 288.15);
        let cons = eos.prim_to_cons(&prim);
        assert!((eos.sound_speed(&cons) - 340.3).abs() < 1.0);
    }

    #[test]
    fn sutherland_and_conductivity() {
        let mu = sutherland_viscosity(288.15);
        assert!((mu - 1.8e-5).abs() < 1e-6);
        let k = thermal_conductivity(mu, 1.4, 287.058);
        assert!(k > 0.02 && k < 0.03);
    }
}
