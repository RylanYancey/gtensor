
use std::alloc::Layout;
use std::alloc;
use std::any::TypeId;

use half::{f16, bf16};
use ndarray::Array4;

use super::float::Float;
use super::shape::Shape;
use super::traits::Storage;
use super::traits::StorageInfo;

pub struct Cpu<T: Float> {
    data: *mut T,
    shape: Shape,
}

impl<T: Float> Cpu<T> {
    pub fn new(shape: Shape) -> Self {
        let layout = Layout::array::<T>(shape.len()).unwrap();

        let ptr = unsafe {
            alloc::alloc_zeroed(layout).cast::<T>()
        };

        Self {
            data: ptr,
            shape,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        let len = self.shape.len();
        unsafe {
            std::slice::from_raw_parts(self.data, len)
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut[T] {
        let len = self.shape.len();
        unsafe {
            std::slice::from_raw_parts_mut(self.data, len)
        }
    }
}

impl<T: Float> Storage for Cpu<T> {
    type F = T;

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn fill(&mut self, v: T) {
        unsafe {
            std::slice::from_raw_parts_mut(self.data, self.shape.len())
                .iter_mut().for_each(|x| *x = v)
        };
    }

    fn clone_from(&mut self, data: &[T]) {
        let len = self.shape.len();

        if len > data.len() {
            panic!("Length of data is not the same as the inner data!")
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.data, len)
        }
    }

    fn as_ndarray(&self) -> Array4<T> {
        let len = self.shape.len();
        let vec = unsafe {
            Vec::from_raw_parts(self.data, len, len)
        };
        Array4::from_shape_vec(self.shape.as_array4(), vec).unwrap()
    }

    fn clone_into(&mut self, array: Array4<Self::F>) {
        
    }

    fn len(&self) -> usize {
        self.shape.len()
    }
}

impl<T: Float> StorageInfo for Cpu<T> {
    const TYPE: &'static str = "cpu";
    const FLOAT: &'static str = T::NAME;
}

impl<T: Float> From<Shape> for Cpu<T> {
    fn from(value: Shape) -> Self {
        Cpu::new(value)
    }
}

impl<T: Float> From<&Array4<T>> for Cpu<T> {
    fn from(value: &Array4<T>) -> Self {
        let shape: Shape = value.shape().into();

        let mut out = Cpu::new(shape);
        out.clone_from(value.as_slice().unwrap());

        out
    }
}

impl<T: Float> Drop for Cpu<T> {
    fn drop(&mut self) {
        let len = self.shape.len();
        unsafe {
            std::alloc::dealloc(self.data as *mut u8, Layout::array::<T>(len).unwrap());
        }
    }
}