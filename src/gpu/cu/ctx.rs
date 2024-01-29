
//! 6.8 Context Management

use anyhow::Result;

use super::*;
use super::types::*;

/// Create a CUDA context.
pub fn create(device: &Device) -> Result<Context> {
    let mut context: sys::CUcontext = std::ptr::null_mut();

    unsafe {
        check(sys::cuCtxCreate_v2(&mut context, 0, device.ptr))?;
    }

    Ok(Context {
        ptr: context,
    })
}

/// Destroy a CUDA context.
pub fn destroy(ctx: Context) -> Result<()> {
    unsafe {
        check(sys::cuCtxDestroy_v2(ctx.ptr))
    }
}

/// Gets the context's API Version.
/// 
/// Returns a version number in version corresponding to the 
/// capabilities of the context (e.g. 3010 or 3020), which library 
/// developers can use to direct callers to a specific API version. 
/// If ctx is NULL, returns the API version used to create the currently bound context.
///
/// Note that new API versions are only introduced when 
/// context capabilities are changed that break binary 
/// compatibility, so the API version and driver version 
/// may be different. For example, it is valid for the 
/// API version to be 3020 while the driver version is 4020
pub fn get_api_version(ctx: &Context) -> Result<usize> {
    let mut version: u32 = 0;

    unsafe {
        check(sys::cuCtxGetApiVersion(ctx.ptr, &mut version))?;
    }

    Ok(version as usize)
}

/// Returns the CUDA context bound to the calling CPU thread. 
/// 
/// This function may result in a null context pointer. 
/// It is recommended to use `ctx::try_get_current` instead.
///  
/// If no context is bound to the calling CPU thread then a null 
/// context pointer is returned.
pub fn get_current() -> Result<Context> {
    let mut context: sys::CUcontext = std::ptr::null_mut();

    unsafe {
        check(sys::cuCtxGetCurrent(&mut context))?;
    }

    Ok(Context {
        ptr: context,
    })
}

#[cfg(feature = "show_unimplemented")]
pub fn get_cache_config() {
    
}

/// Returns the CUDA context bound to the calling CPU thread. 
/// 
/// If no context is bound to the calling thread than a None 
/// value is returned.
pub fn try_get_current() -> Result<Option<Context>> {
    let mut context: sys::CUcontext = std::ptr::null_mut();

    unsafe {
        check(sys::cuCtxGetCurrent(&mut context))?;
    }

    if context.is_null() {
        return Ok(None)
    }

    Ok(Some(Context {
        ptr: context,
    }))
}

/// Returns the device ID for the current context. 
pub fn get_device() -> Result<Device> {
    let mut device: sys::CUdevice = 0;
    
    unsafe {
        check(sys::cuCtxGetDevice(&mut device))?
    }

    Ok(
        Device {
            ptr: device
        }
    )
} 

#[cfg(feature = "show_unimplemented")]
pub fn get_exec_affinity() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn get_flags() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn get_id() {
    
}

/// Query Resources Limits.
pub fn get_limit(limit: CtxLimit) -> Result<usize> {
    let mut value: usize = 0;

    unsafe {
        check(sys::cuCtxGetLimit(&mut value, limit.0))?;
    }

    Ok(value)
}

/// Returns the current shared memory bank size. (4 or 8 bytes)
pub fn get_shared_mem_config() -> Result<usize> {
    let mut config: sys::CUsharedconfig = 0;

    unsafe {
        check(sys::cuCtxGetSharedMemConfig(&mut config))?;
    }

    Ok((config * 4) as usize)
}

#[cfg(feature = "show_unimplemented")]
pub fn get_stream_priority_range() {
    unimplemented!()
}

#[cfg(feature = "show_unimplemented")]
pub fn pop_current() {
    unimplemented!()
}

#[cfg(feature = "show_unimplemented")]
pub fn push_current() {
    unimplemented!()
}

#[cfg(feature = "show_unimplemented")]
pub fn reset_persisting_l2_cache() {
    unimplemented!()
}

#[cfg(feature = "show_unimplemented")]
pub fn set_cache_config() {
    unimplemented!()
}

/// Binds the specified CUDA Context to the calling CPU Thread. 
pub fn set_current(ctx: &Context) -> Result<()> {
    unsafe {
        check(sys::cuCtxSetCurrent(ctx.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn set_flags() {
    unimplemented!()
}

#[cfg(feature = "show_unimplemented")]
pub fn set_limit() {
    unimplemented!()
}

#[cfg(feature = "show_unimplemented")]
pub fn set_shared_mem_config() {
    unimplemented!()
}

/// Blocks until the device has completed all preceding requested tasks. 
pub fn synchronize() -> Result<()> {
    unsafe {
        check(sys::cuCtxSynchronize())
    }
}

pub struct CtxLimit(u32);
impl CtxLimit {
    /// Stack size in bytes of each GPU Thread.
    pub const LIMIT_STACK_SIZE: Self = Self(0);
    /// Size in bytes of the FIFO used by the printf() device system call.
    pub const PRINTF_FIFO_SIZE: Self = Self(1);
    /// Size in bytes of the heap used by the malloc() and free() device system calls.
    pub const MALLOC_HEAP_SIZE: Self = Self(2);
    /// Maximum grid dpeth at which a thread can issue the device runtime
    /// call cudaDeviceSynchronize() to wait on child grid launches to complete.
    pub const DEV_RUNTIME_SYNC_DEPTH: Self = Self(3);
    /// Maximum number of outstanding device runtime launches that can be made from this context.
    pub const DEV_RUNTIME_PENDING_LAUNCH_COUNT: Self = Self(4);
    /// L2 cache fetch granularity.
    pub const MAX_L2_FETCH_GRANULARITY: Self = Self(5);
    /// Persisting L2 cache size in bytes.
    pub const PERSISTING_L2_CACHE_SIZE: Self = Self(6);
}

pub struct CtxFlags;
impl CtxFlags {
    pub const SCHED_SPIN: u32 = 1;
    pub const SCHED_YIELD: u32 = 2;
    pub const SCHED_BLOCKING_SYNC: u32 = 4;
    pub const SCHED_AUTO: u32 = 0;
}