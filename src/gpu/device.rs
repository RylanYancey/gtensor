
use hashbrown::HashMap;
use anyhow::{Result, anyhow};

use std::sync::RwLock;
use std::sync::Arc;

use super::cu;
use crate::error;
use super::stream::Stream;
use super::kernel::Kernel;

pub struct Device {
    context: cu::Context, 
    modules: RwLock<HashMap<String, cu::Module>>,
}

impl Device {
    /// Create a new device. By default, this function will pick the strongest device.
    pub fn new() -> Result<Arc<Self>> {
        Self::pick_strongest()
    }

    /// Pick the device at the provided ordinal index.
    pub fn pick_ordinal(ordinal: usize) -> Result<Arc<Self>> {
        let n = cu::device::get_count()?;

        if ordinal > n {
            return Err(anyhow!(error::gte_006(ordinal, n)));
        }

        let device = cu::device::get(ordinal)?;
        let context = cu::ctx::create(&device)?;
        cu::ctx::set_current(&context)?;

        Ok(Arc::new(Self {
            context,
            modules: RwLock::new(HashMap::new()),
        }))
    }

    /// Pick the strongest device with the highest compute capability.
    pub fn pick_strongest() -> Result<Arc<Self>> {
        let n = cu::device::get_count()?;

        let mut device = cu::device::get(0)?;
        let mut version = 
            cu::device::get_attribute(cu::DeviceAttribute::COMPUTE_CAPABILITY_MAJOR, &device)? * 100 +
            cu::device::get_attribute(cu::DeviceAttribute::COMPUTE_CAPABILITY_MINOR, &device)?;

        for i in 1..n {
            let dev = cu::device::get(i)?;
            let ver = 
                cu::device::get_attribute(cu::DeviceAttribute::COMPUTE_CAPABILITY_MAJOR, &device)? * 100 +
                cu::device::get_attribute(cu::DeviceAttribute::COMPUTE_CAPABILITY_MINOR, &device)?;

            if ver > version {
                device = dev;
                version = ver;
            }
        }

        let context = cu::ctx::create(&device)?;
        cu::ctx::set_current(&context)?;

        Ok(Arc::new(Self {
            context,
            modules: RwLock::new(HashMap::new()),
        }))
    }

    /// Bind this devices' context to the current thread.
    pub fn bind_to_thread(self: &Arc<Self>) -> Result<()> {
        cu::ctx::set_current(&self.context)
    }

    /// Create a new stream
    pub fn fork(self: &Arc<Self>) -> Result<Stream> {
        Ok(Stream {
            stream: cu::stream::create(false)?
        })
    }

    /// Create a new stream with a priority
    pub fn fork_priority(self: &Arc<Self>, priority: usize) -> Result<Stream> {
        Ok(Stream {
            stream: cu::stream::create_with_priority(priority, false)?
        })
    }

    /// Load a Compiled PTX Module into device memory. Does nothing if a module with `name` already exists.
    pub fn load_module(self: &Arc<Self>, name: &str, ptx: &str) -> Result<()> {
        let mut lock = self.modules.write().unwrap();

        if !lock.contains_key(name) {
            let module = cu::module::load_data(ptx)?;
            lock.insert(name.to_owned(), module);
        }

        Ok(())
    }

    /// Get a Kernel from a loaded module. Returns an error if the module or function is not found.
    pub fn get_kernel(self: &Arc<Self>, module: &str, kernel: &str) -> Result<Kernel> {
        let lock = self.modules.read().unwrap();

        if let Some(module) = lock.get(module) {
            let function = cu::module::get_function(module, kernel)?;
            Ok(Kernel::from(kernel.to_owned(), function))
        } else {
            Err(anyhow!("No module with name: {} was loaded.", module))
        }
    }

    /// check if a module is loaded
    pub fn is_module_loaded(self: &Arc<Self>, module: &str) -> bool {
        let lock = self.modules.read().unwrap();
        lock.contains_key(module)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let lock = self.modules.write().unwrap();

        for (_, module) in lock.iter() {
            cu::module::unload(*module).unwrap();
        }

        cu::ctx::destroy(self.context).unwrap();
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        let lock = self.modules.read().unwrap();

        Self {
            context: self.context,
            modules: RwLock::new(lock.clone())
        }
    }
}