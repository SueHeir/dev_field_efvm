//! A case picks its gas when it builds the App.

use cfd_eos::{EosResource, IdealGas};
use cfd_solver::GasPlugin;
use cfd_state::PrimVar;
use grass_app::prelude::*;

#[test]
fn gas_plugin_installs_the_chosen_gas() {
    let argon = IdealGas {
        gamma: 5.0 / 3.0,
        r_gas: 208.1,
    };
    let mut app = App::new();
    app.add_plugins(GasPlugin(argon));
    let gas = *app.get_resource_ref::<IdealGas>().unwrap();
    assert_eq!(gas.gamma, 5.0 / 3.0);
    // The dyn EOS the flux and boundaries read is the same gas.
    let eos = app.get_resource_ref::<EosResource>().unwrap();
    let p = PrimVar::new(1.0, 0.0, 0.0, 0.0, 1e5, 1e5 / argon.r_gas);
    let a = eos.0.sound_speed(&eos.0.prim_to_cons(&p));
    assert!((a - (argon.gamma * 1e5f64).sqrt()).abs() < 1e-9 * a);
}
