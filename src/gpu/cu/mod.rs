
//! # Cuda Driver API v11.8.0 
//! 
//! official cuda docs: https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM

pub mod types;
pub mod device;
pub mod sys;
pub mod ctx;
pub mod module;
pub mod link;
pub mod array;
pub mod ipc;
pub mod mem;
pub mod mipmapped;
pub mod pointer;
pub mod stream;
pub mod thread;
pub mod event;
pub mod external;
pub mod func;
pub mod launch;
pub mod graph;
pub mod user;
pub mod occupancy;

pub use types::*;

use anyhow::Result;

/// Initialize the CUDA driver API Initializes 
/// the driver API and must be called before any 
/// other function from the driver API in the 
/// current process.
pub fn init() -> Result<()> {
    unsafe {
        check(sys::cuInit(0))
    }
}

/// Returns the latest CUDA version supported by the driver. 
/// 
/// # Description
/// 
/// Returns in *driverVersion the version of CUDA supported by the driver. 
/// The version is returned as (1000 major + 10 minor). 
/// For example, CUDA 9.2 would be represented by 9020.
pub fn driver_get_version() -> Result<i32> {
    let mut v: i32 = 0;

    unsafe {
        check(sys::cuDriverGetVersion(&mut v))?;
    }

    Ok(v)
}

/// CuResult handler
fn check(code: u32) -> Result<()> {
    if code == 0 {
        return Ok(())
    }

    let name = name(code);
    let desc = desc(code);

    return Err(anyhow::anyhow!(
        "Driver returned error with code: {code}, name: {name}, desc: {desc}."
    ));

    fn name(result: u32) -> String {
        let mut ptr = std::ptr::null::<std::ffi::c_char>();
    
        let name = unsafe {
            sys::cuGetErrorName(result, &mut ptr);
            std::ffi::CStr::from_ptr(ptr)
        };
    
        name.to_str().unwrap().to_owned()
    }
    
    fn desc(result: u32) -> String {
        let mut ptr = std::ptr::null::<std::ffi::c_char>();
    
        let desc = unsafe {
            sys::cuGetErrorString(result, &mut ptr);
            std::ffi::CStr::from_ptr(ptr)
        };
    
        desc.to_str().unwrap().to_owned()
    }
    
}