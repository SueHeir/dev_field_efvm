//! Checks the wrapper against Mutation++'s own `mppequil` tool.
//!
//! Reference values were produced with Mutation++ (commit 4026765), e.g.
//! `mppequil --no-header -T 6000 -P 1000 -m 0,1,3,31,41,42 --scientific 8 air_11`.
//! Run with `MPP_DATA_DIRECTORY` set to Mutation++'s `data` directory.

use cfd_eos::{Eos, IdealGas};
use cfd_mutation::EquilibriumEos;
use cfd_state::PrimVar;

fn air() -> EquilibriumEos {
    EquilibriumEos::new("air_11").expect("load air_11 (is MPP_DATA_DIRECTORY set?)")
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
}

/// (T [K], rho [kg/m3], e [J/kg], a_frozen [m/s], a_eq [m/s]) at p = 1000 Pa.
const MPPEQUIL_1000PA: [(f64, f64, f64, f64, f64); 5] = [
    (
        300.0,
        1.15663131e-02,
        -8.45881008e+04,
        3.47764811e+02,
        3.47764811e+02,
    ),
    (
        3000.0,
        1.01015351e-03,
        4.89503791e+06,
        1.15200786e+03,
        1.06635108e+03,
    ),
    (
        6000.0,
        3.12101821e-04,
        2.98515897e+07,
        2.17601161e+03,
        1.90791496e+03,
    ),
    (
        9000.0,
        1.77785417e-04,
        4.71960367e+07,
        2.83830732e+03,
        2.53312997e+03,
    ),
    (
        12000.0,
        8.21058328e-05,
        1.22291373e+08,
        4.33681228e+03,
        3.76177964e+03,
    ),
];

#[test]
fn pressure_temperature_state_matches_mppequil() {
    let eos = air();
    for &(t, rho, e, af, aeq) in &MPPEQUIL_1000PA {
        let s = eos.state_pt(1000.0, t);
        assert!(close(s.rho, rho, 1e-7), "rho at {t} K: {} vs {rho}", s.rho);
        assert!(close(s.e, e, 1e-7), "e at {t} K: {} vs {e}", s.e);
        assert!(
            close(s.a_frozen, af, 1e-7),
            "a_f at {t} K: {} vs {af}",
            s.a_frozen
        );
        assert!(
            close(s.a_eq, aeq, 1e-6),
            "a_eq at {t} K: {} vs {aeq}",
            s.a_eq
        );
    }
}

#[test]
fn density_energy_inversion_recovers_temperature_and_pressure() {
    // The solver only ever holds (rho, rho*e): this is the path cons_to_prim takes.
    let eos = air();
    for &(t, rho, e, _, _) in &MPPEQUIL_1000PA {
        let s = eos.state_rho_e(rho, e);
        assert!(close(s.t, t, 1e-6), "T from (rho, e): {} vs {t}", s.t);
        assert!(
            close(s.p, 1000.0, 1e-6),
            "p from (rho, e) at {t} K: {}",
            s.p
        );
    }
}

#[test]
fn eos_round_trip_with_velocity() {
    let eos = air();
    for &t in &[300.0, 2500.0, 7000.0, 11000.0] {
        for &p in &[100.0, 1.0e4, 1.0e6] {
            let prim = eos.prim_pt(3000.0, -200.0, 50.0, p, t);
            let cons = eos.prim_to_cons(&prim);
            let back = eos.cons_to_prim(&cons);
            assert!(
                close(back.t, t, 1e-7),
                "T round trip at p={p}: {} vs {t}",
                back.t
            );
            assert!(
                close(back.p, p, 1e-7),
                "p round trip at T={t}: {} vs {p}",
                back.p
            );
            assert!(close(back.u, 3000.0, 1e-12) && close(back.v, -200.0, 1e-12));
            assert!(close(eos.pressure(&cons), p, 1e-7));
        }
    }
}

#[test]
fn prim_to_cons_ignores_inconsistent_density() {
    let eos = air();
    let wrong = PrimVar::new(123.0, 0.0, 0.0, 0.0, 101325.0, 300.0);
    let cons = eos.prim_to_cons(&wrong);
    assert!(
        close(cons.rho, 1.17195667, 1e-7),
        "rho recomputed from (p, T): {}",
        cons.rho
    );
}

#[test]
fn transport_matches_mppequil() {
    // mppequil -m 32 (viscosity) at 1 atm.
    let eos = air();
    assert!(close(
        eos.transport_pt(101325.0, 300.0).mu,
        1.94239659e-05,
        1e-6
    ));
    assert!(close(
        eos.transport_pt(101325.0, 5000.0).mu,
        1.43239402e-04,
        1e-6
    ));
}

#[test]
fn cold_air_matches_the_ideal_gas() {
    // Below ~800 K nothing dissociates, so equilibrium air is the perfect gas
    // the rest of efvm assumes. air_11 is 79/21 N2/O2 with no argon, so its gas
    // constant is 288.2 rather than 287.06: allow 0.5%.
    let eos = air();
    let ideal = IdealGas::air();
    let prim = eos.prim_pt(0.0, 0.0, 0.0, 101325.0, 300.0);
    let cons = eos.prim_to_cons(&prim);
    let r_eff = prim.p / (prim.rho * prim.t);
    assert!(
        close(r_eff, ideal.r_gas, 5e-3),
        "R_eff {r_eff} vs {}",
        ideal.r_gas
    );
    let a_ideal = (ideal.gamma * ideal.r_gas * 300.0).sqrt();
    assert!(close(eos.sound_speed(&cons), a_ideal, 5e-3));
}

#[test]
fn hot_air_dissociates() {
    let eos = air();
    let x: std::collections::HashMap<_, _> =
        eos.composition_pt(1000.0, 6000.0).into_iter().collect();
    assert!(
        x["O"] > 0.2,
        "O2 should be mostly dissociated at 6000 K, 1000 Pa: X_O = {}",
        x["O"]
    );
    assert!(x["N"] > 0.05, "N2 dissociation starting: X_N = {}", x["N"]);
}

#[test]
fn normal_shock_matches_mppshock() {
    // `mppshock -P 100 -T 300 -V 6000 -m 0 air_11`, "Equilibrated" row:
    //   p = 38673.3 Pa, u2 = 441.7 m/s, T = 6067.1 K
    let eos = air();
    let s = cfd_mutation::normal_shock(&eos, 100.0, 300.0, 6000.0).unwrap();
    assert!(close(s.post.p, 38673.3, 2e-5), "p2 {}", s.post.p);
    assert!(close(s.u2, 441.7, 2e-4), "u2 {}", s.u2);
    assert!(close(s.post.t, 6067.1, 2e-5), "T2 {}", s.post.t);
}
