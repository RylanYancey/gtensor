
//! 6.10 Module Management

use super::*;
use super::types::*;

/// Returns a function handle
/// 
/// # Description
/// 
/// Returns the handle of the function with `name` located in the provided module.
/// If no function name exists, cuModuleGetFunction() returns CUDA_ERROR_NOT_FOUND. 
pub fn get_function(module: &Module, name: &str) -> Result<Function> {
    let mut function: sys::CUfunction = std::ptr::null_mut();

    let name = std::ffi::CString::new(name).unwrap();

    unsafe {
        check(sys::cuModuleGetFunction(&mut function, module.ptr, name.as_ptr()))?;
    }

    Ok(
        Function {
            ptr: function,
        }
    )
}

#[cfg(feature = "show_unimplemented")]
pub fn get_global() {
    
}

/// Returns the loading mode of the module, controlled by the CUDA_MODULE_LOADING env variable.
pub fn get_loading_mode() -> Result<ModuleLoadingMode> {
    let mut mode: u32 = 0;

    unsafe {
        check(sys::cuModuleGetLoadingMode(&mut mode))?;
    }

    Ok(
        if mode == 2 {
            ModuleLoadingMode::Lazy
        } else {
            ModuleLoadingMode::Eager
        }
    )
}

/// Loads a Compute Module
/// 
/// # Description
///
/// Takes a filename and loads the corresponding module 
/// into the current context. The CUDA driver API does 
/// not attempt to lazily allocate the resources needed by a 
/// module; if the memory for functions and data (constant and global) 
/// needed by the module cannot be allocated, cuModuleLoad() fails. 
/// The file should be a cubin file as output by nvcc, or a PTX 
/// file either as output by nvcc or handwritten, or a fatbin file 
/// as output by nvcc from toolchain 4.0 or later. 
pub fn load(filename: &str) -> Result<Module> {
    let mut module: sys::CUmodule = std::ptr::null_mut();

    let name = std::ffi::CString::new(filename).unwrap();

    unsafe {
        check(sys::cuModuleLoad(&mut module, name.as_ptr()))?;
    }

    Ok(Module {
        ptr: module,
    })
}

/// Load a modules's data.
pub fn load_data(ptx: &str) -> Result<Module> {
    let mut module: sys::CUmodule = std::ptr::null_mut();

    let cstr = std::ffi::CString::new(ptx).unwrap();

    unsafe {
        check(sys::cuModuleLoadData(&mut module, cstr.as_ptr() as *const std::ffi::c_void))?;
    }

    Ok(
        Module {
            ptr: module,
        }
    )
}

/// Load a modules data, with a CString
pub fn load_data_cstr(ptx: &std::ffi::CString) -> Result<Module> {
    let mut module: sys::CUmodule = std::ptr::null_mut();

    unsafe {
        check(sys::cuModuleLoadData(&mut module, ptx.as_ptr() as *const std::ffi::c_void))?;
    }

    Ok(
        Module {
            ptr: module,
        }
    )
}

#[cfg(feature = "show_unimplemented")]
pub fn load_data_ex() {
    
}

#[cfg(feature = "show_unimplemented")]
pub fn load_fat_binary() {
    
}

/// Unloads a module.
pub fn unload(module: Module) -> Result<()> {
    unsafe {
        check(sys::cuModuleUnload(module.ptr))
    }
}

pub enum ModuleLoadingMode {
    Lazy = 2,
    Eager = 1,
}