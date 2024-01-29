
use ndarray::Array4;

use super::shape::Shape;
use super::traits::Storage;
use super::cpu::Cpu;
use super::gpu::Gpu;
use super::float::Float;

pub struct Tensor<S: Storage>(S);

impl<S: Storage> Tensor<S> 
where
    S: for<'a> From<&'a Array4<S::F>>
{
    pub fn from_ndarray(array: &Array4<S::F>) -> Self {
        Self(S::from(array))
    }
}

impl<S: Storage> Tensor<S> 
where
    S: From<Shape>
{
    pub fn new(shape: Shape) -> Self {
        Self(S::from(shape))
    }
}

impl<S: Storage> std::ops::Deref for Tensor<S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: Storage> std::ops::DerefMut for Tensor<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
