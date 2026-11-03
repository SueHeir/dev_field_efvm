//! [Mutation++](https://github.com/mutationpp/Mutationpp) thermochemistry for
//! dev_field_efvm.
//!
//! Two layers:
//!
//! * [`Mixture`]: a safe owner of one Mutation++ mixture, over the C shim in
//!   `shim/mpp_shim.cpp`. Nothing in it is CFD-specific.
//! * [`EquilibriumEos`]: an [`Eos`] for a gas in local chemical equilibrium
//!   (for example 11-species air). It drops in wherever efvm uses
//!   `EosResource`, so the inviscid solver, boundary conditions and output
//!   run on real-gas air with no other change.
//!
//! Requirements: build with `MUTATIONPP_DIR` set to the Mutation++ install
//! prefix, and run with `MPP_DATA_DIRECTORY` pointing at Mutation++'s `data`
//! directory (Mutation++ reads its species and mixture files from there).
//!
//! Mutation++ is not thread-safe: separate mixtures used from different threads
//! at once hung in testing. Neither type is `Send` or `Sync`, and the tests run
//! on one thread (`.cargo/config.toml`). efvm runs one thread per rank, so this
//! costs nothing today, but a threaded solver would need a lock or processes.

use cfd_eos::Eos;
use cfd_state::{ConsVar, PrimVar};
use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr::NonNull;

mod ffi {
    use super::*;
    extern "C" {
        pub fn mpp_last_error(buf: *mut c_char, len: c_int);
        pub fn mpp_create(mixture: *const c_char, state_model: *const c_char) -> *mut c_void;
        pub fn mpp_destroy(m: *mut c_void);
        pub fn mpp_n_species(m: *mut c_void) -> c_int;
        pub fn mpp_set_state(m: *mut c_void, v1: *const f64, v2: *const f64, vars: c_int) -> c_int;
        pub fn mpp_get_state(m: *mut c_void, out: *mut f64) -> c_int;
        pub fn mpp_get_basic(m: *mut c_void, out: *mut f64) -> c_int;
        pub fn mpp_get_transport(m: *mut c_void, out: *mut f64) -> c_int;
        pub fn mpp_mole_fractions(m: *mut c_void, x: *mut f64) -> c_int;
        pub fn mpp_species_name(m: *mut c_void, i: c_int, buf: *mut c_char, len: c_int);
    }
}

fn last_error() -> String {
    let mut buf = [0 as c_char; 512];
    unsafe { ffi::mpp_last_error(buf.as_mut_ptr(), buf.len() as c_int) };
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn check(rc: c_int) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(last_error())
    }
}

/// Mixture state after a `set_state_*` call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermoState {
    /// Temperature [K].
    pub t: f64,
    /// Pressure [Pa].
    pub p: f64,
    /// Density [kg/m³].
    pub rho: f64,
    /// Specific internal energy [J/kg], including heats of formation
    /// (Mutation++'s reference: elements in their standard state at 298.15 K).
    pub e: f64,
    /// Frozen sound speed [m/s].
    pub a_frozen: f64,
    /// Equilibrium sound speed [m/s].
    pub a_eq: f64,
    /// Mixture molar mass [kg/mol].
    pub mw: f64,
    /// Frozen specific heat at constant pressure [J/(kg·K)].
    pub cp_frozen: f64,
}

/// Transport coefficients at the current state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransportState {
    /// Dynamic viscosity [Pa·s].
    pub mu: f64,
    /// Frozen thermal conductivity [W/(m·K)].
    pub k_frozen: f64,
    /// Equilibrium thermal conductivity, including reactive conduction [W/(m·K)].
    pub k_eq: f64,
}

/// One Mutation++ mixture (for example `"air_11"` with the `"Equil"` state model).
pub struct Mixture {
    h: NonNull<c_void>,
    ns: usize,
}

impl Mixture {
    /// Loads `mixture` (a file name under `$MPP_DATA_DIRECTORY/mixtures`, without
    /// `.xml`) with the given Mutation++ state model.
    pub fn new(mixture: &str, state_model: &str) -> Result<Self, String> {
        let name = CString::new(mixture).map_err(|e| e.to_string())?;
        let model = CString::new(state_model).map_err(|e| e.to_string())?;
        let raw = unsafe { ffi::mpp_create(name.as_ptr(), model.as_ptr()) };
        let h = NonNull::new(raw).ok_or_else(|| {
            format!(
                "Mutation++ mixture '{mixture}' ({state_model}): {}",
                last_error()
            )
        })?;
        let ns = unsafe { ffi::mpp_n_species(h.as_ptr()) } as usize;
        Ok(Mixture { h, ns })
    }

    pub fn n_species(&self) -> usize {
        self.ns
    }

    pub fn species_names(&self) -> Vec<String> {
        (0..self.ns)
            .map(|i| {
                let mut buf = [0 as c_char; 64];
                unsafe { ffi::mpp_species_name(self.h.as_ptr(), i as c_int, buf.as_mut_ptr(), 64) };
                let bytes: Vec<u8> = buf
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect();
                String::from_utf8_lossy(&bytes).into_owned()
            })
            .collect()
    }

    /// Mutation++'s raw `setState(v1, v2, vars)`. The meaning of `v1`, `v2` and
    /// `vars` depends on the state model; prefer the typed setters below.
    pub fn set_state_raw(&mut self, v1: &[f64], v2: &[f64], vars: i32) -> Result<(), String> {
        check(unsafe { ffi::mpp_set_state(self.h.as_ptr(), v1.as_ptr(), v2.as_ptr(), vars) })
    }

    /// Equilibrium state model: set from density [kg/m³] and internal energy
    /// density ρe [J/m³].
    pub fn set_state_rho_e(&mut self, rho: f64, rho_e: f64) -> Result<(), String> {
        self.set_state_raw(&[rho], &[rho_e], 0)
    }

    /// Equilibrium state model: set from pressure [Pa] and temperature [K].
    pub fn set_state_pt(&mut self, p: f64, t: f64) -> Result<(), String> {
        self.set_state_raw(&[p], &[t], 1)
    }

    pub fn state(&self) -> ThermoState {
        let mut o = [0.0; 8];
        check(unsafe { ffi::mpp_get_state(self.h.as_ptr(), o.as_mut_ptr()) })
            .unwrap_or_else(|e| panic!("Mutation++ state query failed: {e}"));
        ThermoState {
            t: o[0],
            p: o[1],
            rho: o[2],
            e: o[3],
            a_frozen: o[4],
            a_eq: o[5],
            mw: o[6],
            cp_frozen: o[7],
        }
    }

    /// (T, p, ρ, e, molar mass) only: skips the sound speeds, which cost a
    /// composition-derivative solve.
    pub fn basic(&self) -> (f64, f64, f64, f64, f64) {
        let mut o = [0.0; 5];
        check(unsafe { ffi::mpp_get_basic(self.h.as_ptr(), o.as_mut_ptr()) })
            .unwrap_or_else(|e| panic!("Mutation++ state query failed: {e}"));
        (o[0], o[1], o[2], o[3], o[4])
    }

    pub fn transport(&self) -> TransportState {
        let mut o = [0.0; 3];
        check(unsafe { ffi::mpp_get_transport(self.h.as_ptr(), o.as_mut_ptr()) })
            .unwrap_or_else(|e| panic!("Mutation++ transport query failed: {e}"));
        TransportState {
            mu: o[0],
            k_frozen: o[1],
            k_eq: o[2],
        }
    }

    pub fn mole_fractions(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.ns];
        check(unsafe { ffi::mpp_mole_fractions(self.h.as_ptr(), x.as_mut_ptr()) })
            .unwrap_or_else(|e| panic!("Mutation++ mole fraction query failed: {e}"));
        x
    }
}

impl Drop for Mixture {
    fn drop(&mut self) {
        unsafe { ffi::mpp_destroy(self.h.as_ptr()) }
    }
}

/// Which sound speed the flux functions see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SoundSpeed {
    /// The characteristic speed of the equilibrium Euler equations (default).
    #[default]
    Equilibrium,
    /// Frozen sound speed. Slightly larger, so a more conservative wave-speed
    /// bound for Rusanov/HLLC estimates.
    Frozen,
}

/// Equation of state for a gas in local chemical equilibrium, from Mutation++.
///
/// `ConsVar::rho_e` carries Mutation++'s internal energy (heats of formation
/// included) plus kinetic energy. Shifting the energy reference does not change
/// the Euler equations, so the solver needs no change.
///
/// [`Eos::prim_to_cons`] uses **pressure and temperature** from the `PrimVar`
/// and recomputes density from them; its `rho` field is ignored. Use
/// [`EquilibriumEos::prim_pt`] to build a consistent `PrimVar`.
///
/// Each conversion runs Mutation++'s equilibrium solve (a Newton iteration on
/// temperature around a Gibbs minimisation), so it costs far more than the
/// ideal gas. A production code would tabulate it.
pub struct EquilibriumEos {
    mix: RefCell<Mixture>,
    sound: SoundSpeed,
    /// Coarse e(T) curves at fixed densities, for the inversion's first guess.
    table: GuessTable,
    /// The solver asks for pressure, sound speed and primitives of the same
    /// cell in a row; remember the last (ρ, e) answer.
    last: Cell<Option<(f64, f64, ThermoState)>>,
}

impl EquilibriumEos {
    /// `mixture` is a Mutation++ mixture name such as `"air_11"` or `"air_5"`.
    pub fn new(mixture: &str) -> Result<Self, String> {
        let mut eos = EquilibriumEos {
            mix: RefCell::new(Mixture::new(mixture, "Equil")?),
            sound: SoundSpeed::default(),
            table: GuessTable::default(),
            last: Cell::new(None),
        };
        eos.table = GuessTable::build(&eos)?;
        Ok(eos)
    }

    pub fn with_sound_speed(mut self, s: SoundSpeed) -> Self {
        self.sound = s;
        self
    }

    /// Full equilibrium state at (p, T).
    pub fn state_pt(&self, p: f64, t: f64) -> ThermoState {
        let mut m = self.mix.borrow_mut();
        m.set_state_pt(p, t)
            .unwrap_or_else(|e| panic!("equilibrium state at p={p}, T={t}: {e}"));
        m.state()
    }

    /// Full equilibrium state at density `rho` [kg/m³] and specific internal
    /// energy `e` [J/kg].
    ///
    /// This does **not** use Mutation++'s own (ρ, ρe) state setter. That Newton
    /// iteration starts from the mixture's previous state and has no iteration
    /// cap, so in a flow solver, where consecutive calls jump between cold and
    /// hot cells, it can loop forever. Instead this brackets the temperature
    /// and runs a safeguarded secant on `e(T)` at fixed density, with each
    /// evaluation an equilibrium (p, T) state whose pressure is iterated to
    /// match ρ. The answer agrees with Mutation++'s to round-off where both
    /// converge (see the tests).
    pub fn state_rho_e(&self, rho: f64, e: f64) -> ThermoState {
        if let Some((r, ee, s)) = self.last.get() {
            if r == rho && ee == e {
                return s;
            }
        }
        let (t, p) = self
            .invert_rho_e(rho, e)
            .unwrap_or_else(|err| panic!("equilibrium state at rho={rho}, e={e}: {err}"));
        let s = {
            let mut m = self.mix.borrow_mut();
            m.set_state_pt(p, t)
                .unwrap_or_else(|err| panic!("equilibrium state at p={p}, T={t}: {err}"));
            m.state()
        };
        self.last.set(Some((rho, e, s)));
        s
    }

    /// Equilibrium (e, p) at density `rho` and temperature `t`, starting the
    /// pressure iteration from `p`. ρ ∝ p at fixed T and composition, so the
    /// pressure is rescaled by the density ratio until they agree (this avoids
    /// depending on which gas constant Mutation++ was built with).
    fn e_at_rho_t(&self, rho: f64, t: f64, mut p: f64) -> Result<(f64, f64), String> {
        let mut m = self.mix.borrow_mut();
        for _ in 0..30 {
            m.set_state_pt(p, t)?;
            let (_, _, r, e, _) = m.basic();
            let p_new = p * rho / r;
            if (p_new - p).abs() <= 1e-13 * p {
                return Ok((e, p_new));
            }
            p = p_new;
        }
        Err(format!(
            "pressure iteration at rho={rho}, T={t} did not converge"
        ))
    }

    /// A `PrimVar` whose density is consistent with (p, T).
    pub fn prim_pt(&self, u: f64, v: f64, w: f64, p: f64, t: f64) -> PrimVar {
        let s = self.state_pt(p, t);
        PrimVar::new(s.rho, u, v, w, p, t)
    }

    /// Transport at (p, T): viscosity and conductivities.
    pub fn transport_pt(&self, p: f64, t: f64) -> TransportState {
        let mut m = self.mix.borrow_mut();
        m.set_state_pt(p, t)
            .unwrap_or_else(|e| panic!("equilibrium state at p={p}, T={t}: {e}"));
        m.transport()
    }

    /// Species names and equilibrium mole fractions at (p, T).
    pub fn composition_pt(&self, p: f64, t: f64) -> Vec<(String, f64)> {
        let mut m = self.mix.borrow_mut();
        m.set_state_pt(p, t)
            .unwrap_or_else(|e| panic!("equilibrium state at p={p}, T={t}: {e}"));
        m.species_names()
            .into_iter()
            .zip(m.mole_fractions())
            .collect()
    }

    fn from_cons(&self, u: &ConsVar) -> ThermoState {
        let e = (u.rho_e - u.kinetic_energy()) / u.rho;
        self.state_rho_e(u.rho, e)
    }

    /// Temperature and pressure at (ρ, e): bracket from the table's guess, then
    /// a safeguarded (Illinois) secant on e(T) − e at fixed ρ.
    fn invert_rho_e(&self, rho: f64, e: f64) -> Result<(f64, f64), String> {
        const T_MIN: f64 = 50.0;
        const T_MAX: f64 = 60_000.0;
        const R_AIR: f64 = 288.0; // only seeds the pressure iteration
        let t0 = self.table.guess(rho, e).clamp(T_MIN, T_MAX);
        let mut p = rho * R_AIR * t0;
        let mut f = |t: f64| -> Result<f64, String> {
            let (et, pt) = self.e_at_rho_t(rho, t, p)?;
            p = pt;
            Ok(et - e)
        };
        let tol = |t: f64, ft: f64, a: f64, b: f64| {
            ft.abs() <= 1e-12 * e.abs().max(1e3) || (b - a).abs() <= 1e-12 * t
        };

        let (mut a, mut fa) = (t0, f(t0)?);
        if fa == 0.0 {
            return Ok((a, p));
        }
        let up = fa < 0.0;
        let (mut b, mut fb);
        let mut step = 1.01;
        loop {
            b = if up {
                (a * step).min(T_MAX)
            } else {
                (a / step).max(T_MIN)
            };
            fb = f(b)?;
            if fb == 0.0 {
                return Ok((b, p));
            }
            if (fb < 0.0) != up {
                break;
            }
            if b == T_MAX || b == T_MIN {
                return Err(format!(
                    "e = {e} J/kg is outside the {T_MIN}–{T_MAX} K range"
                ));
            }
            (a, fa) = (b, fb);
            step *= step;
        }
        let mut side = 0;
        for _ in 0..100 {
            let mut t = b - fb * (b - a) / (fb - fa);
            if !(t > a.min(b) && t < a.max(b)) {
                t = 0.5 * (a + b);
            }
            let ft = f(t)?;
            if tol(t, ft, a, b) {
                return Ok((t, p));
            }
            if (ft < 0.0) == (fb < 0.0) {
                (b, fb) = (t, ft);
                if side == -1 {
                    fa *= 0.5;
                }
                side = -1;
            } else {
                (a, fa) = (b, fb);
                (b, fb) = (t, ft);
                side = 1;
            }
        }
        Err("no convergence".into())
    }
}

/// e(T) on a log-spaced temperature grid for a set of densities, used only to
/// start the (ρ, e) inversion close to the answer.
#[derive(Default)]
struct GuessTable {
    log_rho: Vec<f64>,
    t: Vec<f64>,
    /// e[row][k] at density 10^log_rho[row] and temperature t[k].
    e: Vec<Vec<f64>>,
}

impl GuessTable {
    fn build(eos: &EquilibriumEos) -> Result<Self, String> {
        let log_rho: Vec<f64> = (0..=32).map(|i| -7.0 + 0.25 * i as f64).collect();
        let t: Vec<f64> = (0..=80)
            .map(|k| 100.0 * (400.0_f64).powf(k as f64 / 80.0))
            .collect();
        let mut e = Vec::with_capacity(log_rho.len());
        for &lr in &log_rho {
            let rho = 10f64.powf(lr);
            let mut row = Vec::with_capacity(t.len());
            for &tk in &t {
                row.push(eos.e_at_rho_t(rho, tk, rho * 288.0 * tk)?.0);
            }
            e.push(row);
        }
        Ok(GuessTable { log_rho, t, e })
    }

    fn guess(&self, rho: f64, e: f64) -> f64 {
        if self.t.is_empty() {
            return 1000.0;
        }
        let lr = rho.log10();
        let i = ((lr - self.log_rho[0]) / 0.25)
            .round()
            .clamp(0.0, (self.log_rho.len() - 1) as f64) as usize;
        let row = &self.e[i];
        let k = row.partition_point(|&ek| ek < e);
        if k == 0 {
            return self.t[0];
        }
        if k >= row.len() {
            return self.t[self.t.len() - 1];
        }
        let w = (e - row[k - 1]) / (row[k] - row[k - 1]);
        self.t[k - 1] + w * (self.t[k] - self.t[k - 1])
    }
}

impl Eos for EquilibriumEos {
    fn cons_to_prim(&self, u: &ConsVar) -> PrimVar {
        let s = self.from_cons(u);
        let inv = 1.0 / u.rho;
        PrimVar::new(u.rho, u.rho_u * inv, u.rho_v * inv, u.rho_w * inv, s.p, s.t)
    }

    fn prim_to_cons(&self, p: &PrimVar) -> ConsVar {
        let s = self.state_pt(p.p, p.t);
        let ke = 0.5 * s.rho * (p.u * p.u + p.v * p.v + p.w * p.w);
        ConsVar::new(
            s.rho,
            s.rho * p.u,
            s.rho * p.v,
            s.rho * p.w,
            s.rho * s.e + ke,
        )
    }

    fn pressure(&self, u: &ConsVar) -> f64 {
        self.from_cons(u).p
    }

    fn sound_speed(&self, u: &ConsVar) -> f64 {
        let s = self.from_cons(u);
        match self.sound {
            SoundSpeed::Equilibrium => s.a_eq,
            SoundSpeed::Frozen => s.a_frozen,
        }
    }
}

/// Post-shock equilibrium state behind a stationary normal shock.
#[derive(Debug, Clone, Copy)]
pub struct NormalShock {
    /// Equilibrium state behind the shock.
    pub post: ThermoState,
    /// Post-shock velocity in the shock frame [m/s].
    pub u2: f64,
}

/// Equilibrium Rankine–Hugoniot jump for a gas at (p1, T1) entering a normal
/// shock at `u1` [m/s] in the shock frame.
///
/// Iterates on the density ratio ε = ρ1/ρ2: momentum gives p2, energy gives
/// h2, and the equilibrium state at (p2, h2) gives the next ρ2. This is the
/// same jump Mutation++'s `mppshock` reports as "Equilibrated".
pub fn normal_shock(
    eos: &EquilibriumEos,
    p1: f64,
    t1: f64,
    u1: f64,
) -> Result<NormalShock, String> {
    let s1 = eos.state_pt(p1, t1);
    let h1 = s1.e + p1 / s1.rho;
    let mut eps = 0.1;
    for _ in 0..200 {
        let p2 = p1 + s1.rho * u1 * u1 * (1.0 - eps);
        let h2 = h1 + 0.5 * u1 * u1 * (1.0 - eps * eps);
        let post = state_ph(eos, p2, h2)?;
        let eps_new = s1.rho / post.rho;
        if (eps_new - eps).abs() < 1e-12 {
            return Ok(NormalShock {
                post,
                u2: u1 * eps_new,
            });
        }
        eps = 0.5 * (eps + eps_new); // damped: the plain update can oscillate
    }
    Err(format!(
        "normal_shock: no convergence for p1={p1}, T1={t1}, u1={u1}"
    ))
}

/// Equilibrium state at pressure `p` and specific enthalpy `h`, by bracketed
/// secant/bisection on temperature (h rises monotonically with T at fixed p).
pub fn state_ph(eos: &EquilibriumEos, p: f64, h: f64) -> Result<ThermoState, String> {
    let h_at = |t: f64| {
        let s = eos.state_pt(p, t);
        (s.e + p / s.rho - h, s)
    };
    let (mut lo, mut hi) = (100.0_f64, 60_000.0_f64);
    let (mut f_lo, _) = h_at(lo);
    let (f_hi, _) = h_at(hi);
    if f_lo > 0.0 || f_hi < 0.0 {
        return Err(format!(
            "state_ph: h={h} J/kg at p={p} Pa is outside 100–60000 K"
        ));
    }
    let mut f_hi = f_hi;
    for _ in 0..200 {
        // Secant step, falling back to bisection when it leaves the bracket.
        let mut t = hi - f_hi * (hi - lo) / (f_hi - f_lo);
        if !(t > lo && t < hi) || (hi - lo) > 0.5 * hi {
            t = 0.5 * (lo + hi);
        }
        let (f, s) = h_at(t);
        if f.abs() <= 1e-12 * h.abs().max(1.0) || (hi - lo) < 1e-10 * t {
            return Ok(s);
        }
        if f < 0.0 {
            lo = t;
            f_lo = f;
        } else {
            hi = t;
            f_hi = f;
        }
    }
    Err(format!("state_ph: no convergence for p={p}, h={h}"))
}
