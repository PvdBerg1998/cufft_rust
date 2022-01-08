use std::env;
use std::path::PathBuf;

fn main() {
    let (bin_path, header_path) = if cfg!(windows) {
        let base_path = PathBuf::from(env::var("CUDA_PATH").unwrap());
        (base_path.join("lib/x64"), base_path.join("include"))
    } else if cfg!(unix) {
        if cfg!(target_os = "arch") && cfg!(target_arch = "x86_64") {
            let base_path = PathBuf::from("/opt/cuda/targets/x86_64-linux/");
            (base_path.join("lib"), base_path.join("include"))
        } else {
            todo!("Unsupported OS");
        }
    } else {
        todo!("Unsupported platform")
    };

    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cufft");
    //println!("cargo:rustc-link-lib=static=legacy_stdio_definitions");

    println!("cargo:rustc-link-search={}", bin_path.to_str().unwrap());

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

    bindings
        .write_to_file("bindings.rs")
        .expect("Couldn't write bindings!");
}
