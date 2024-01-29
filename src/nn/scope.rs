
use std::cell::RefCell;
use std::sync::Arc;

use crate::storage::Storage;
use super::node::Node;
use crate::gpu::device::Device;

pub struct Scope<S: Storage> {
    nodes: Vec<Node<S>>,
}

pub struct ScopeBuilder<S: Storage> {
    device: Arc<Device>,
    scope: RefCell<Scope<S>>,
}

impl<S: Storage> ScopeBuilder<S> {
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}