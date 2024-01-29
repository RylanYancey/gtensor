
use anyhow::Result;
use anyhow::anyhow;
use itertools::multizip;

use crate::storage::Storage;
use super::node::Node;
use crate::storage::Gpu;
use crate::storage::Cpu;
use crate::storage::Float;
use super::var::Var;
use crate::storage::StorageInfo;
use super::node::NodeBuilder;

mod mul;

#[allow(unused_variables)]
pub trait Operator<S: Storage> {
    fn forward(&mut self, node: &Node<S>) -> Result<()>;
    fn reshape(&mut self, node: &Node<S>) -> Result<()>;
    fn wrt_x1(&self, node: &Node<S>) -> Result<()> { Ok(()) }
    fn wrt_x2(&self, node: &Node<S>) -> Result<()> { Ok(()) }
    fn wrt_x3(&self, node: &Node<S>) -> Result<()> { Ok(()) }
    fn wrt_x4(&self, node: &Node<S>) -> Result<()> { Ok(()) }
    fn wrt_x5(&self, node: &Node<S>) -> Result<()> { Ok(()) }
}