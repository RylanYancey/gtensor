
use std::env;
use std::path::PathBuf;
use std::process::Command;

// The Directory where the Kernels are stored
const KERNEL_DIR: &str = "src/gpu/kernels/";

fn main() {
    // make sure this build script will be reran upon changes
    println!("cargo:rerun-if-changed=build.rs");

    // Get the GT_CUDA_SRC env variable
    let root = env::var("GT_CUDA_SRC")
        .expect("Failed to get GT_CUDA_SRC");

    // tell cargo where to look for compiled cuda binaries
    println!("cargo:rustc-link-search={}/lib64/", root);
    println!("cargo:rustc-link-lib=cuda");

    // Generate the bindings for $GT_CUDA_SRC/include/cuda.h
    let bindings = bindgen::Builder::default()
        .header(root.clone() + "/include/cuda.h")
        .generate()
        .expect("Unable to generate cuda driver bindings!");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
    .write_to_file(out_path.join("driver_bindings.rs"))
        .expect("Couldn't write cuda driver bindings!");

    // Create a directory for compiled PTXs'
    let _ = std::fs::create_dir(out_path.join("ptx/"));

    // Location of NVCC Installation
    let nvcc = &format!("{}/bin/nvcc", root.clone());
    // Output directory for the compiled PTXs
    let out_dir = out_path.join("ptx/").to_str().unwrap().to_owned();
    // Include directory to search for headers
    let include = root.clone() + "/include/";

    // Compile for all gpus sm_52+
    #[cfg(feature = "gpu")]
    compile("sm_52", &include, &out_dir, nvcc);

    // f16 is a feature of sm_70+
    #[cfg(feature = "f16")]
    compile("sm_70", &include, &out_dir, nvcc);

    // bf16 is a feature of sm_80+
    #[cfg(feature = "bf16")]
    compile("sm_80", &include, &out_dir, nvcc);
}

/// Compile sm_52, sm_70, or sm_80.
pub fn compile(sm: &str, include: &str, out_dir: &str, nvcc: &str) {

    // The Directory that the Kernels are stored
    let read_dir = std::fs::read_dir(format!("{}{}", KERNEL_DIR, sm))
        .expect("Failed to read directory to compile kernels");

    // For every file in the dir,
    for file in read_dir {
        // get the path
        let file = file.unwrap();
        let path = file.path();

        // Launch a bash process to run NVCC
        // nvcc --ptx -arch=sm_xx -I$GT_CUDA_SRC/include/ -odir $OUT_DIR/ptx/ <name>
        let status = Command::new(nvcc)
            .arg("--ptx")
            .arg(format!("-arch={}",sm))
            .arg(format!("-I{}", include))
            .arg(format!("-odir {}", out_dir))
            .arg(path)
            .status();

        // send an error if compilation fails.
        if let Err(e) = status {
            panic!("Failed to compile with error: {e}")
        }
    }
}
