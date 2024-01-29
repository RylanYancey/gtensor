
use std::env;
use std::path::PathBuf;

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let cuda_root = match env::var("GT_CUDA_SRC") {
        Ok(v) => v,
        Err(e) => {
            panic!("Failed to get GT_CUDA_SRC variable!")
        }
    };

    println!("cargo:rustc-link-search={}/lib64/", cuda_root);
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=nvrtc");

    driver_bindings(cuda_root.clone());
}

pub fn driver_bindings(cuda_root: String) {
    let bindings = bindgen::Builder::default()
        .header(cuda_root.clone() + "/include/cuda.h")
        .generate()
        .expect("Unable to generate cuda driver bindings!");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
    .write_to_file(out_path.join("driver_bindings.rs"))
        .expect("Couldn't write cuda driver bindings!");
}

pub fn compile(root: String, sm: &str) {
    use std::fs::read_dir;

    let nvcc = root.clone() + "/bin/nvcc";
    let include = root.clone() + "/include/";
    let path = "src/gpu/kernels/".to_owned() + sm;

    let dir = match std::fs::read_dir(&path) {
        Ok(v) => v,
        Err(e) => {
            panic!("src/gpu/kernels/{} not found!", sm)
        },
    };

    for entry in dir {
        let entry = entry.expect("Failed to get entry!");
        let path = path.clone() + entry.file_name().to_str().unwrap();
        // nvcc --ptx --output-directory=src/gpu/kernels/ptx/ -arch=sm_80 -I$GT_CUDA_SRC/include/ src/gpu/kernels/sm_80/mulbf16.cu
        Command::new("sh")
            .args(&[
                &nvcc, 
                "--ptx", 
                "--output-directory=src/gpu/kernels/ptx/",
                &format!("-I{}", include), 
                &format!("-arch={}", sm),
                &path
            ]);
    }
}