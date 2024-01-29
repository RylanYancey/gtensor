
//! # 6.5 Device Management
//! # 6.11 Memory Management
//! # 6.12 Graph Management

use std::ffi::CString;

use anyhow::Result;

use super::*;
use super::types::*;

/// Returns a handle to a compute device.
/// 
/// # Parameters
/// 
/// * `ordinal`: Device number to get handle for.
pub fn get(ordinal: usize) -> Result<Device> {
    let mut device: sys::CUdevice = 0;
    
    unsafe {
        check(sys::cuDeviceGet(&mut device, ordinal as i32))?
    }

    Ok(
        Device {
            ptr: device
        }
    )
}

/// Returns information about the device.
/// 
/// # Parameters
/// 
/// * attr: Device attribute to query.
/// * device: Device handle.
pub fn get_attribute(attr: DeviceAttribute, device: &Device) -> Result<i32> {
    let mut pi: i32 = 0;

    unsafe {
        check(sys::cuDeviceGetAttribute(&mut pi, attr as u32, device.ptr))?;
    }

    Ok(pi)
}

/// Returns the number of compute-capable devices.
pub fn get_count() -> Result<usize> {
    let mut count: i32 = 0;

    unsafe {
        check(sys::cuDeviceGetCount(&mut count))?
    }

    Ok(count as usize)
}

/// Returns the default mempool of a device.
pub fn get_default_mem_pool(device: &Device) -> Result<MemoryPool> {
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

    unsafe {
        check(sys::cuDeviceGetDefaultMemPool(&mut pool, device.ptr))?;
    }

    Ok(MemoryPool {
        ptr: pool,
    })
}

#[cfg(feature = "show_unimplemented")]
pub fn get_luid() {

}

/// Returns the current mempool for a device.
pub fn get_mem_pool(device: &Device) -> Result<MemoryPool> {
    let mut pool: sys::CUmemoryPool = std::ptr::null_mut();

    unsafe {
        check(sys::cuDeviceGetMemPool(&mut pool, device.ptr))?;
    }

    Ok(
        MemoryPool {
            ptr: pool,
        }
    )
}

/// Returns an identifier string for the device.
pub fn get_name(device: &Device) -> Result<String> {
    let str = String::from_iter((0..100).map(|_| ' '));
    let name = CString::new(str)?;

    unsafe {
        sys::cuDeviceGetName(name.as_ptr().cast_mut(), 100, device.ptr);
    }

    Ok(
        name.to_str()?.to_owned().trim().to_owned()
    )
}

#[cfg(feature = "show_unimplemented")]
pub fn get_nv_sci_sync_attributes() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_texture_1d_linear_max_width() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_uuid() {
    
}

/// Sets the current memory pool of a device.
pub fn set_mem_pool(device: &Device, pool: &MemoryPool) -> Result<()> {
    unsafe {
        return check(sys::cuDeviceSetMemPool(device.ptr, pool.ptr))
    }
}

/// Returns the total amount of memory on the device.
pub fn total_mem(device: &Device) -> Result<usize> {
    let mut bytes: usize = 0;

    unsafe {
        check(sys::cuDeviceTotalMem_v2(&mut bytes, device.ptr))?;
    }

    Ok(bytes)
}

#[cfg(feature = "show_unimplemented")]
pub fn flush_gpu_direct_rdma_writes() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_by_pci_bus_id() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_pci_bus_id() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_graph_mem_attribute() {

}

#[cfg(feature = "show_unimplemented")]
pub fn graph_mem_trim() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_graph_mem_attribute() {
    
}