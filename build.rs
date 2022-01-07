use std::env;
use std::path::PathBuf;

fn main() {
    let nvidia_path = PathBuf::from(env::var("CUDA_PATH").unwrap());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    //println!("cargo:rustc-link-lib=static=legacy_stdio_definitions");

    if cfg!(windows) {
        println!(
            "cargo:rustc-link-search={}",
            nvidia_path.join("lib/x64/").to_str().unwrap()
        );
    } else {
        todo!("Unsupported platform");
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!(
            "-I{}",
            nvidia_path.join("include").to_str().unwrap()
        ))
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

    bindings
        .write_to_file("bindings.rs")
        .expect("Couldn't write bindings!");
}
