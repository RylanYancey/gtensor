
//! # 6.11 Memory Management
//! 
//! This section describes the memory management functions 
//! of the low-level CUDA driver application programming interface. 
//! 
//! # 6.12 Virtual Memory Management
//! 
//! This section describes the virtual memory management functions of the 
//! low-level CUDA driver application programming interface. 
//! 
//! # 6.13 Stream Ordered Memory Allocator
//! 
//! This section describes the stream ordered memory allocator exposed
//! by the low-level cuda driver application programming interface.
//! 
//! The asynchronous allocator allows the user to allocate and free in stream order.
//! All asynchronous accesses of the allocation must happen between the stream executions
//! of the allocation and the free. If the memory is accessed outside of the promised stream order,
//! a use before allocation / use after free error will cause undefined behaviour.
//! 
//! The allocator is free to reallocate the memory as long as it can guarantee that compliant
//! memory accesses will not overlap temporarily. The allocator may refer to internal 
//! stream ordering as well as inter-stream dependencies (such as CUDA events and null 
//! stream dependencies) when establishing the temporal guarantee. The allocator may
//! also insert inter-stream dependencies to establish the temporal guarantee.
//!
//! # 6.14 Unified Addressing
//! 
//! This section describes the unified addressing functions of the low-level CUDA driver
//! application programming interface.

use super::*;
use super::types::*;

/// Allocates Device Memory
pub fn alloc<T>(len: usize) -> Result<DevicePtr> {
    let mut ptr: sys::CUdeviceptr = 0;
    let size = std::mem::size_of::<T>();

    unsafe {
        check(sys::cuMemAlloc_v2(&mut ptr, len * size))?;
    }

    Ok(DevicePtr {
        ptr,
    })
}

#[cfg(feature = "show_unimplemented")]
pub fn alloc_host() {

}

#[cfg(feature = "show_unimplemented")]
pub fn alloc_managed() {

}

#[cfg(feature = "show_unimplemented")]
pub fn alloc_pitch() {

}

/// Frees device memory
pub fn free(ptr: DevicePtr) -> Result<()> {
    unsafe {
        check(sys::cuMemFree_v2(ptr.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn free_host() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_address_range() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_handle_for_address_range() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_info() {

}

#[cfg(feature = "show_unimplemented")]
pub fn host_alloc() {

}

#[cfg(feature = "show_unimplemented")]
pub fn host_get_device_pointer() {

}

#[cfg(feature = "show_unimplemented")]
pub fn host_get_flags() {

}

#[cfg(feature = "show_unimplemented")]
pub fn host_register() {

}

#[cfg(feature = "show_unimplemented")]
pub fn host_unregister() {

}

/// Copies memory.
pub fn cpy<T>(dst: &DevicePtr, src: &DevicePtr, len: usize) -> Result<()> {
    let size = std::mem::size_of::<T>();

    unsafe {
        check(sys::cuMemcpy(src.ptr, dst.ptr, len * size))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_2d() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_2d_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_2d_unaligned() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_3d() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_3d_async() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_3d_peer() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_3d_peer_async() {

}

/// Copies Memory Asynchronously
/// 
/// # Description
/// 
/// Copies data between two pointers. dst and src are base pointers of the 
/// destination and source, respectively. ByteCount specifies the number of 
/// bytes to copy. Note that this function infers the type of the transfer 
/// (host to host, host to device, device to device, or device to host) 
/// from the pointer values. This function is only allowed in contexts 
/// which support unified addressing. 
pub fn cpy_async<T>(dst: &DevicePtr, src: &DevicePtr, len: usize, stream: &Stream) -> Result<()> {
    let bytes = std::mem::size_of::<T>() * len;

    unsafe {
        check(sys::cuMemcpyAsync(dst.ptr, src.ptr, bytes, stream.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_a_to_a() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_a_to_d() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_a_to_h() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_a_to_h_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_d_to_a() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_d_to_d() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_d_to_d_async() {

}

/// Copies memory from device to host.
pub fn cpy_d_to_h<T>(dst_host: *mut T, src_device: &DevicePtr, len: usize) -> Result<()> {
    let bytes = std::mem::size_of::<T>() * len;

    unsafe {
        check(sys::cuMemcpyDtoH_v2(dst_host as *mut std::ffi::c_void, src_device.ptr, bytes))
    }
}

/// Copies memory asyncronously from device to host.
pub fn cpy_d_to_h_async<T>(dst_host: *mut T, src_device: &DevicePtr, len: usize, stream: &Stream) -> Result<()> {
    let bytes = std::mem::size_of::<T>() * len;

    unsafe {
        check(sys::cuMemcpyDtoHAsync_v2(dst_host as *mut std::ffi::c_void, src_device.ptr, bytes, stream.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_h_to_a() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_h_to_a_async() {
    
}

/// Copies memory frmo Host to Device
pub fn cpy_h_to_d<T>(dst: &DevicePtr, src: *const T, len: usize) -> Result<()> {
    let bytes = std::mem::size_of::<T>() * len;

    unsafe {
        check(sys::cuMemcpyHtoD_v2(dst.ptr, src as *const std::ffi::c_void, bytes))
    }
}

/// Copies memory asyncronously from host to device.
pub fn cpy_h_to_d_async<T>(dst: &mut DevicePtr, src: *const T, len: usize, stream: &Stream) -> Result<()> {
    let bytes = std::mem::size_of::<T>() * len;

    unsafe {
        check(sys::cuMemcpyHtoDAsync_v2(dst.ptr, src as *const std::ffi::c_void, bytes, stream.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_peer() {

}

#[cfg(feature = "show_unimplemented")]
pub fn cpy_peer_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d16() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d16_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d2d16() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d2d16_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d2d32() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d2_d32_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d2_d8() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_d8_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn address_free() {

}

#[cfg(feature = "show_unimplemented")]
pub fn address_reserve() {

}

#[cfg(feature = "show_unimplemented")]
pub fn create() {

}

#[cfg(feature = "show_unimplemented")]
pub fn export_to_shareable_handle() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_access() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_allocation_granularity() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_allocation_properties_from_handle() {

}

#[cfg(feature = "show_unimplemented")]
pub fn import_from_shareable_handle() {

}

#[cfg(feature = "show_unimplemented")]
pub fn map() {

}

#[cfg(feature = "show_unimplemented")]
pub fn map_array_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn release() {

}

#[cfg(feature = "show_unimplemented")]
pub fn retain_allocation_handle() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_access() {

}

#[cfg(feature = "show_unimplemented")]
pub fn unmap() {

}

#[cfg(feature = "show_unimplemented")]
pub fn alloc_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn alloc_from_pool_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn free_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_create() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_destroy() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_export_pointer() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_export_to_shareable_handle() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_get_access() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_get_attribute() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_import_from_shareable_handle() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_import_pointer() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_set_access() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_set_attribute() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pool_trim_to() {

}

#[cfg(feature = "show_unimplemented")]
pub fn advise() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn prefetch_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn range_get_attribute() {

}

#[cfg(feature = "show_unimplemented")]
pub fn pointer_get_attribute() {

}
