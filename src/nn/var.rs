
use std::sync::Arc;

use crate::storage::Storage;
use super::scope::ScopeBuilder;
use crate::storage::{Cpu, Gpu, Float};
use crate::storage::Shape;
use crate::gpu::device::Device;

pub struct Var<'s, S: Storage> {
    scope: &'s ScopeBuilder<S>,
    shape: Shape,
    level: usize,
}

impl<'s, S: Storage> Var<'s, S> {

}

impl<'s, S: Storage> std::ops::Deref for Var<'s, S> {
    type Target = ScopeBuilder<S>;

    fn deref(&self) -> &Self::Target {
        &self.scope
    }
}