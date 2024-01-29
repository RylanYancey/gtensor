
use std::sync::Arc;

use once_cell::sync::Lazy;

use super::device::Device;
use super::cu;

static DEFAULT_DEVICE: Lazy<Arc<Device>> = Lazy::new(|| {
    cu::init()
        .expect("Failed to initialize Cuda!");

    let dev = Device::pick_strongest()
        .expect("Failed to create global device!");

    dev
});

pub fn get_default_device() -> Arc<Device> {
    DEFAULT_DEVICE.clone()
}