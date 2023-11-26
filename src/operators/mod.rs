
use std::fmt::Display;
use std::fmt::Formatter;

use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};

use crate::graph::node::Node;
use crate::graph::var::Var;
use crate::graph::node::NodeBuilder;
use crate::tensor::shape::ToShape;
use crate::tensor::shape::Shape;
use crate::tensor::axis::ToAxis;

pub trait Operator: 
    serde_traitobject::Serialize + 
    serde_traitobject::Deserialize + 
    dyn_clone::DynClone + 
    Display + 'static 
{
    fn forward(&mut self, node: &Node) -> Result<()>;
    fn backward(&mut self, node: &Node) -> Result<()>;
    fn reshape(&mut self, _new: Shape) {}
}

dyn_clone::clone_trait_object!(Operator);

#[derive(Copy, Clone)]
pub struct PoolParams {
    pub kernel: [usize; 2],
    pub stride: [usize; 2],
    pub padh: [usize; 2],
    pub padw: [usize; 2],
}

impl Default for PoolParams {
    fn default() -> Self {
        Self {
            kernel: [2,2],
            stride: [2,2],
            padh: [1,1],
            padw: [1,1],
        }
    }
}

#[derive(Copy, Clone)]
pub struct ConvParams {
    pub kernel: [usize; 4],
    pub stride: [usize; 2],
    pub padh: [usize; 2],
    pub padw: [usize; 2],
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            kernel: [3,3,3,3],
            stride: [1,1],
            padh: [1,1],
            padw: [1,1]
        }
    }
}

pub(crate) mod input;

mod matmul;
mod tanh;
mod sigmoid;
mod softmax;
mod relu;
mod reshape;
mod max_pool;
mod avg_pool;
mod lrn;
mod im2col;
mod dropout;
mod axis_add;
mod flatten;

pub mod op {
    use super::*;

    pub use super::PoolParams;
    pub use super::ConvParams;

    pub use matmul::matmul;
    pub use tanh::tanh;
    pub use sigmoid::sigmoid;
    pub use softmax::softmax;
    pub use relu::relu;
    pub use reshape::reshape;
    pub use max_pool::max_pool;
    pub use avg_pool::avg_pool;
    pub use lrn::lrn;
    pub use im2col::im2col;
    pub use dropout::dropout;
    pub use axis_add::axis_add;
    pub use flatten::flatten;
}