
//! # 6.15 Stream Management
//! # 6.18 Stream Memory Operations

use super::*;

#[cfg(feature = "show_unimplemented")]
pub fn add_callback() {

}

#[cfg(feature = "show_unimplemented")]
pub fn attach_mem_async() {

}

#[cfg(feature = "show_unimplemented")]
pub fn begin_capture() {

}

#[cfg(feature = "show_unimplemented")]
pub fn copy_attributes() {

}

/// Creates a Stream
pub fn create(is_non_blocking: bool) -> Result<Stream> {
    let mut stream: sys::CUstream = std::ptr::null_mut();
    let flags = if is_non_blocking { 0 } else { 1 };

    unsafe {
        check(sys::cuStreamCreate(&mut stream, flags))?;
    }

    Ok(Stream {
        ptr: stream
    })
}

/// Create a stream with a set priority. Lower numbers have higher priority.
pub fn create_with_priority(priority: usize, is_non_blocking: bool) -> Result<Stream> {
    let mut stream: sys::CUstream = std::ptr::null_mut();
    let flags = if is_non_blocking { 0 } else { 1 };

    unsafe {
        check(sys::cuStreamCreateWithPriority(&mut stream, flags, priority as i32))?;
    }

    Ok(Stream {
        ptr: stream
    })
}

/// Destroy a stream.
pub fn destroy(stream: Stream) -> Result<()> {
    unsafe {
        check(sys::cuStreamDestroy_v2(stream.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn end_capture() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_attribute() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_capture_info() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_ctx() {

}

#[cfg(feature = "show_unimplemented")]
pub fn get_flags() {

}

#[cfg(feature = "show_unimplemented")]
pub fn is_capturing() {

}

#[cfg(feature = "show_unimplemented")]
pub fn query() {

}

#[cfg(feature = "show_unimplemented")]
pub fn set_attribute() {

}

/// Wait until all of a streams' tasks are completed.
pub fn synchronize(stream: &Stream) -> Result<()> {
    unsafe {
        check(sys::cuStreamSynchronize(stream.ptr))
    }
}

#[cfg(feature = "show_unimplemented")]
pub fn update_capture_dependencies() {

}

#[cfg(feature = "show_unimplemented")]
pub fn wait_event() {

}

#[cfg(feature = "show_unimplemented")]
pub fn batch_mem_op() {

}

#[cfg(feature = "show_unimplemented")]
pub fn wait_value_32() {

}

#[cfg(feature = "show_unimplemented")]
pub fn wait_value_64() {

}

#[cfg(feature = "show_unimplemented")]
pub fn write_value_32() {

}

#[cfg(feature = "show_unimplemented")]
pub fn write_value_64() {
    
}