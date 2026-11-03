// Flat C interface over a Mutation++ mixture, so Rust can drive it without
// C++ name mangling. Every function catches C++ exceptions and turns them into
// a status code: an exception must never unwind into Rust.
#include <mutation++.h>

#include <cstring>
#include <exception>
#include <string>

using Mutation::Mixture;
using Mutation::MixtureOptions;

namespace {
thread_local std::string g_last_error;

template <class F>
int guarded(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_last_error = e.what();
    } catch (...) {
        g_last_error = "unknown C++ exception";
    }
    return 1;
}
}  // namespace

extern "C" {

// Copies the last error message (NUL-terminated, truncated to len) into buf.
void mpp_last_error(char* buf, int len) {
    if (len <= 0) return;
    std::strncpy(buf, g_last_error.c_str(), static_cast<size_t>(len - 1));
    buf[len - 1] = '\0';
}

// Creates a mixture by name (e.g. "air_11") with the given state model
// (e.g. "Equil", "ChemNonEq1T"). Returns null on failure.
void* mpp_create(const char* mixture, const char* state_model) {
    Mixture* m = nullptr;
    int rc = guarded([&] {
        MixtureOptions opts(mixture);
        opts.setStateModel(state_model);
        m = new Mixture(opts);
    });
    return rc == 0 ? m : nullptr;
}

void mpp_destroy(void* m) { delete static_cast<Mixture*>(m); }

int mpp_n_species(void* m) { return static_cast<int>(static_cast<Mixture*>(m)->nSpecies()); }

// Mutation++'s variable sets: for "Equil", vars = 0 takes (rho, rho*e) and
// vars = 1 takes (p, T).
int mpp_set_state(void* m, const double* v1, const double* v2, int vars) {
    return guarded([&] { static_cast<Mixture*>(m)->setState(v1, v2, vars); });
}

// Scalar state after set_state:
// out = [T, p, rho, e (J/kg), frozen sound speed, equilibrium sound speed,
//        mixture molar mass (kg/mol), frozen cp (J/kg/K)]
int mpp_get_state(void* m, double* out) {
    return guarded([&] {
        Mixture* x = static_cast<Mixture*>(m);
        out[0] = x->T();
        out[1] = x->P();
        out[2] = x->density();
        out[3] = x->mixtureEnergyMass();
        out[4] = x->frozenSoundSpeed();
        out[5] = x->equilibriumSoundSpeed();
        out[6] = x->mixtureMw();
        out[7] = x->mixtureFrozenCpMass();
    });
}

// The cheap subset of mpp_get_state: out = [T, p, rho, e (J/kg), molar mass].
// The equilibrium sound speed needs composition derivatives, so iterative
// callers use this and ask for the full state once at the end.
int mpp_get_basic(void* m, double* out) {
    return guarded([&] {
        Mixture* x = static_cast<Mixture*>(m);
        out[0] = x->T();
        out[1] = x->P();
        out[2] = x->density();
        out[3] = x->mixtureEnergyMass();
        out[4] = x->mixtureMw();
    });
}

// Transport at the current state: out = [viscosity (Pa s),
// frozen conductivity, equilibrium conductivity (W/m/K)].
int mpp_get_transport(void* m, double* out) {
    return guarded([&] {
        Mixture* x = static_cast<Mixture*>(m);
        out[0] = x->viscosity();
        out[1] = x->frozenThermalConductivity();
        out[2] = x->equilibriumThermalConductivity();
    });
}

// Species mole fractions at the current state (length n_species).
int mpp_mole_fractions(void* m, double* x_out) {
    return guarded([&] {
        Mixture* x = static_cast<Mixture*>(m);
        std::memcpy(x_out, x->X(), sizeof(double) * x->nSpecies());
    });
}

// Species name i, NUL-terminated and truncated to len.
void mpp_species_name(void* m, int i, char* buf, int len) {
    if (len <= 0) return;
    const std::string& s = static_cast<Mixture*>(m)->speciesName(i);
    std::strncpy(buf, s.c_str(), static_cast<size_t>(len - 1));
    buf[len - 1] = '\0';
}

}  // extern "C"
