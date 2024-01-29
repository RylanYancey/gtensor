
//! 6.19 Execution Control

use std::ffi::c_void;

use super::*;

#[cfg(feature = "show_unimplemented")]
pub fn cooperative_kernel() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cooperative_kernel_multi_device() {

}

#[cfg(feature = "show_unimplemented")]
pub fn host_func() {

}

/// Launches a Cuda Kernel
pub fn kernel(
    function: &Function, 
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: &Stream,
    args: &[*mut c_void],
    extras: &[*mut c_void],
) -> Result<()> {
    unsafe {
        check(sys::cuLaunchKernel(
            function.ptr,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem_bytes,
            stream.ptr,
            &mut args.as_ptr().cast_mut().cast::<c_void>(),
            &mut extras.as_ptr().cast_mut().cast::<c_void>(),
        ))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn kernel_ex() {
    
}