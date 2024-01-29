

use anyhow::*;

/// GTE001: Env Variable GT_CUDA_SRC not set.
pub fn gte_001(msg: String) -> String {
    format!("GTE001: Env Variable GT_CUDA_SRC not set. msg: {}", msg)
}

/// GTE002: Cuda Initialization Failed
pub fn gte_002(msg: String) -> String {
    format!("GTE002: Cuda Initialization Failed. msg: {}", msg)
}

/// GTE003: Failed to get cuda device count.
pub fn gte_003(msg: String) -> String {
    format!("GTE003: Failed to get Cuda Device Count during initialization. msg: {}", msg)
}

/// GTE004: Failed to get cuda device from driver.
pub fn gte_004(ordinal: usize, msg: String) -> String {
    format!("GTE004: Failed to get cuda device at ordinal {} from device driver. msg: {}", ordinal, msg)
}

/// GTE005: Failed to get cuda device attribute from driver.
pub fn gte_005(msg: String, attr: &str) -> String {
    format!("GTE005: Failed to get cuda device attribute {}. msg: {}", attr, msg)
}

/// GTE006: Ordinal out of range.
pub fn gte_006(ordinal: usize, len: usize) -> String {
    format!("GTE006: Attempted to get cuda device at ordinal index: {}, but the len is {}!", ordinal, len)
}

/// GTE007: Failed to create Cuda Context
pub fn gte_007(ordinal: usize, msg: String) -> String {
    format!("GTE007: Attempted to create Cuda Context for device with ordinal {}, but it failed. msg: {}", ordinal, msg)
}

/// GTE008: Failed to bind Cuda Context to thread.
pub fn gte_008(msg: String) -> String {
    format!("GTE008: Attempted to bind Cuda Context to thread, but it failed. msg: {}", msg)
}

/// GTE009: No Devices Found
pub fn gte_009() -> String {
    format!("GTE009: No Cuda-Capable Devices Found")
}

/// GTE010: Failed to Create Cuda Program from src.
pub fn gte_010(msg: String, name: &str) -> String {
    format!("GTE_010: Failed to create cuda program with name {}. msg: {}", name, msg)
}

/// GTE011: Failed to Compile Cuda Program from src.
pub fn gte_011(msg: String, name: &str) -> String {
    format!("GTE_011: Failed to COMPILE cuda program with name {}. msg: {}", name, msg)
}

/// GTE012: Failed to get ptx src from nvrtc program. 
pub fn gte_012(msg: String, name: &str) -> String {
    format!("GTE_012: Failed to get PTX src from nvrtc program with name {}. msg: {}", name, msg)
}

/// GTE013: Failed to load module.
pub fn gte_013(msg: String, name: &str) -> String {
    format!("GTE_013: Failed to load module from ptx with name: {}. msg: {}", name, msg)
} 