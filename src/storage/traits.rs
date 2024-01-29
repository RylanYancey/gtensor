
use ndarray::Array4;

use super::shape::Shape;
use super::float::Float;

pub trait Storage {
    type F: Float;

    fn shape(&self) -> &Shape;
    fn fill(&mut self, v: Self::F);
    fn clone_from(&mut self, data: &[Self::F]);
    fn as_ndarray(&self) -> Array4<Self::F>;
    fn clone_into(&mut self, array: Array4<Self::F>);
    fn len(&self) -> usize;
}

pub trait StorageInfo: Storage {
    /// The Type of Storage this is.
    /// Either `gpu` or `cpu`. 
    const TYPE: &'static str;

    /// The Float type this storage contains.
    /// Could be `f64`, `f16`, `f32`, or `bf16`. 
    const FLOAT: &'static str;
}