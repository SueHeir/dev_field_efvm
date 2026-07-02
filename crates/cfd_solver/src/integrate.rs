//! Time integration: trait + Euler / TVD-RK3, operating on the registered
//! [`CfdState`] columns (`u`, `rhs`, `u0`). Ported from toy-cfd's `cfd_integrate`
//! with `FlowField` → `CfdState`; FIELD's schedule-set doc places the integrator
//! in physics (there is no `field_integrate`), so it lives here next to `Update`.

use cfd_state::CfdState;
use std::sync::Arc;

/// RK3 substep marker — also the "where am I in the step?" flag the solver reads
/// to gate once-per-step work (CFL, save-state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RkStage {
    First,
    Second,
    Third,
}

/// Result of one substep.
#[derive(Debug, Clone, Copy)]
pub struct SubstepResult {
    pub next_stage: RkStage,
    pub completed_step: bool,
}

/// One time-integration scheme. Each call advances one substep; Euler completes a
/// step every call, RK3 every third.
pub trait TimeIntegrator: Send + Sync {
    fn n_substeps(&self) -> u8;
    fn substep(&self, state: &mut CfdState, dt: f64, current: RkStage) -> SubstepResult;
    fn name(&self) -> &'static str {
        "<unnamed>"
    }
}

/// Resource holding the active integrator.
#[derive(Clone)]
pub struct TimeIntegratorResource(pub Arc<dyn TimeIntegrator>);

impl TimeIntegratorResource {
    pub fn integrator(&self) -> &dyn TimeIntegrator {
        &*self.0
    }
}

/// Forward Euler: `U^{n+1} = U^n + dt * RHS`.
pub struct EulerIntegrator;

impl TimeIntegrator for EulerIntegrator {
    fn n_substeps(&self) -> u8 {
        1
    }
    fn substep(&self, state: &mut CfdState, dt: f64, _current: RkStage) -> SubstepResult {
        for (u, rhs) in state.u.iter_mut().zip(state.rhs.iter()) {
            *u += *rhs * dt;
        }
        SubstepResult { next_stage: RkStage::First, completed_step: true }
    }
    fn name(&self) -> &'static str {
        "Euler"
    }
}

/// Shu–Osher TVD-RK3:
/// ```text
/// U^(1)   = U^n + dt·RHS(U^n)
/// U^(2)   = 3/4·U^n + 1/4·(U^(1) + dt·RHS(U^(1)))
/// U^{n+1} = 1/3·U^n + 2/3·(U^(2) + dt·RHS(U^(2)))
/// ```
pub struct Rk3Integrator;

impl TimeIntegrator for Rk3Integrator {
    fn n_substeps(&self) -> u8 {
        3
    }
    fn substep(&self, state: &mut CfdState, dt: f64, current: RkStage) -> SubstepResult {
        match current {
            RkStage::First => {
                state.save_state();
                for (u, rhs) in state.u.iter_mut().zip(state.rhs.iter()) {
                    *u += *rhs * dt;
                }
                SubstepResult { next_stage: RkStage::Second, completed_step: false }
            }
            RkStage::Second => {
                for ((u, u0), rhs) in
                    state.u.iter_mut().zip(state.u0.iter()).zip(state.rhs.iter())
                {
                    *u = 0.75 * *u0 + 0.25 * (*u + *rhs * dt);
                }
                SubstepResult { next_stage: RkStage::Third, completed_step: false }
            }
            RkStage::Third => {
                for ((u, u0), rhs) in
                    state.u.iter_mut().zip(state.u0.iter()).zip(state.rhs.iter())
                {
                    *u = (1.0 / 3.0) * *u0 + (2.0 / 3.0) * (*u + *rhs * dt);
                }
                SubstepResult { next_stage: RkStage::First, completed_step: true }
            }
        }
    }
    fn name(&self) -> &'static str {
        "Rk3"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_state::ConsVar;

    fn state_with(n: usize, u: ConsVar, rhs: ConsVar) -> CfdState {
        let mut s = CfdState::default();
        use field_core::FieldData;
        s.resize(n);
        for i in 0..n {
            s.u[i] = u;
            s.rhs[i] = rhs;
        }
        s
    }

    #[test]
    fn euler_step_basic() {
        let mut s = state_with(1, ConsVar::new(1.0, 0.0, 0.0, 0.0, 100.0), ConsVar::new(0.1, 0.0, 0.0, 0.0, 10.0));
        let r = EulerIntegrator.substep(&mut s, 0.5, RkStage::First);
        assert!(r.completed_step);
        assert!((s.u[0].rho - 1.05).abs() < 1e-15);
        assert!((s.u[0].rho_e - 105.0).abs() < 1e-15);
    }

    #[test]
    fn rk3_constant_rhs_matches_euler_total() {
        let u0 = ConsVar::new(1.0, 0.0, 0.0, 0.0, 100.0);
        let rhs = ConsVar::new(0.1, 0.0, 0.0, 0.0, 10.0);
        let dt = 0.5;
        let mut s = state_with(1, u0, rhs);
        let mut stage = RkStage::First;
        let mut completed = false;
        for _ in 0..3 {
            for r in s.rhs.iter_mut() {
                *r = rhs;
            }
            let res = Rk3Integrator.substep(&mut s, dt, stage);
            stage = res.next_stage;
            completed = res.completed_step;
        }
        assert!(completed);
        // For constant RHS, RK3 reduces to U + dt·RHS.
        assert!((s.u[0].rho - (u0.rho + dt * rhs.rho)).abs() < 1e-12);
        assert!((s.u[0].rho_e - (u0.rho_e + dt * rhs.rho_e)).abs() < 1e-12);
    }
}
