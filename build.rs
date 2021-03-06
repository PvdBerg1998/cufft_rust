use std::path::PathBuf;

fn main() {
    let (bin_path, header_path) = paths();

    println!("cargo:rustc-link-search={}", bin_path.to_str().unwrap());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    //println!("cargo:rustc-link-lib=static=legacy_stdio_definitions");

    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", header_path.to_str().unwrap()))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-Wno-everything")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .allowlist_function("cufft.*")
        .allowlist_type("cufft.*")
        .allowlist_var("cufft.*")
        .allowlist_function("cuda.*")
        .allowlist_type("cuda.*")
        .allowlist_var("cuda.*")
        .generate()
        .expect("Unable to generate bindings");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let out = out.join("bindings.rs");
    bindings
        .write_to_file(out)
        .expect("Couldn't write bindings!");
}

#[cfg(windows)]
fn paths() -> (PathBuf, PathBuf) {
    let base_path = PathBuf::from(std::env::var("CUDA_PATH").unwrap());
    (base_path.join("lib/x64"), base_path.join("include"))
}

#[cfg(unix)]
fn paths() -> (PathBuf, PathBuf) {
    pkg_config::Config::new()
        .atleast_version("11.0")
        .probe("cuda")
        .ok()
        .and_then(|mut lib| Some((lib.link_paths.pop()?, lib.include_paths.pop()?)))
        .unwrap_or_else(|| {
            // Guess default locations
            (
                PathBuf::from("/usr/local/cuda/lib64"),
                PathBuf::from("/usr/local/cuda/include"),
            )
        })
}

#[cfg(all(not(windows), not(unix)))]
fn paths() -> ! {
    unimplemented!("Unsupported target");
}
