
use std::marker::PhantomData;

use ndarray::Array4;

use crate::gpu::cu;
use super::shape::Shape;
use super::traits::{Storage, StorageInfo};
use super::float::Float;

pub struct Gpu<T: Float> {
    _type: PhantomData<T>,
    data: cu::DevicePtr,
    shape: Shape,
}

impl<T: Float> Gpu<T> {
    pub fn new(shape: Shape) -> Self {
        let _ = crate::gpu::get_default_device();
        let len = shape.len();
        let ptr = cu::mem::alloc::<T>(len).unwrap();
        let zeroes = vec![T::zero(); len];
        cu::mem::cpy_h_to_d(&ptr, zeroes.as_ptr(), len)
            .expect("Failed to copy data to the gpu!");

        Self {
            _type: PhantomData,
            data: ptr,
            shape,
        }
    }

    pub fn as_ptr(&self) -> cu::DevicePtr {
        self.data
    }
}

impl<T: Float> Storage for Gpu<T> {
    type F = T;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn fill(&mut self, v: T) {
        let len = self.shape.len();
        let vec = vec![v; len];

        cu::mem::cpy_h_to_d(&self.data, vec.as_ptr(), len)
            .expect("Failed to copy data to the gpu!")
    }

    fn clone_from(&mut self, data: &[T]) {
        let len = self.shape.len();

        if len > data.len() {
            panic!("Length of data is not the same as the inner data!")
        }

        cu::mem::cpy_h_to_d(&self.data, data.as_ptr(), len)
            .expect("Failed to copy data to the gpu!");
    }

    fn as_ndarray(&self) -> Array4<T> {
        let len = self.shape.len();
        let mut vec = vec![T::zero(); len];

        cu::mem::cpy_d_to_h(vec.as_mut_ptr(), &self.data, len)
            .expect("Failed to copy from device to host!");

        Array4::from_shape_vec(self.shape.as_array4(), vec)
            .expect("Failed to create Array4 from Tensor!")
    }

    fn clone_into(&mut self, array: Array4<Self::F>) {
        todo!()
    }

    fn len(&self) -> usize {
        self.shape.len()
    }
}

impl<T: Float> StorageInfo for Gpu<T> {
    const TYPE: &'static str = "gpu";
    const FLOAT: &'static str = T::NAME;
}

impl<T: Float> From<Shape> for Gpu<T> {
    fn from(value: Shape) -> Self {
        Self::new(value)
    }
}

impl<T: Float> From<&Array4<T>> for Gpu<T> {
    fn from(value: &Array4<T>) -> Self {
        let shape: Shape = value.shape().into();
    
        let mut out = Self::new(shape);
        out.clone_from(value.as_slice().unwrap());
    
        out
    }
}

impl<T: Float> Drop for Gpu<T> {
    fn drop(&mut self) {
        cu::mem::free(self.data)
            .expect("Failed to free memory!");
    }
}