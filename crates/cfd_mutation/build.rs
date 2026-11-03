//! Compiles the C shim against an installed Mutation++ and links it.
//!
//! Set `MUTATIONPP_DIR` to the Mutation++ install prefix (the directory that
//! holds `include/mutation++` and `lib/libmutation++.so`).
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=shim/mpp_shim.cpp");
    println!("cargo:rerun-if-env-changed=MUTATIONPP_DIR");
    let prefix = PathBuf::from(env::var("MUTATIONPP_DIR").unwrap_or_else(|_| {
        panic!("cfd_mutation: set MUTATIONPP_DIR to your Mutation++ install prefix")
    }));
    let inc = prefix.join("include");
    let lib = prefix.join("lib");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("shim/mpp_shim.cpp")
        .include(inc.join("mutation++"))
        .include(inc.join("eigen3"))
        .warnings(false)
        .compile("mpp_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=mutation++");
    // So test and example binaries find libmutation++.so without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
