
use std::fmt::Display;
use std::fmt::Formatter;

use serde::{Serialize, Deserialize};
use anyhow::Result;

use crate::operators::Operator;
use crate::tensor::shape::Shape;
use crate::graph::node::Node;

pub trait Optimizer: Operator {
    fn to_operator(&self, shape: Shape) -> Box<dyn Operator>;
}

mod sgd;
mod adam;
mod momentum;

pub mod opt {
    use super::*;

    pub use sgd::sgd;
    pub use adam::adam;
    pub use momentum::momentum;
}