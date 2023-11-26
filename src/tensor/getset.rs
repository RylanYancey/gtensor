
use anyhow::{Result, anyhow};

use super::shape::{Shape, ToShape};
use super::axis::ToAxis;
use super::Tensor;
use super::slice::TensorSlice;

impl Tensor {
    pub fn from_data(shape: impl ToShape, data: Vec<f32>) -> Self {
        Self::try_from_data(shape, data).unwrap()
    }

    pub fn try_from_data(shape: impl ToShape, data: Vec<f32>) -> Result<Self> {
        let shape = shape.to_shape();

        if shape.len() != data.len() {
            return Err(anyhow!("
                Provided data and shape to do not have the same length!
                Data Len: ({}), Shape: ({}, len: {}).
            ", data.len(), shape, shape.len()))
        } else {
            Ok(
                Self { data , shape}
            )
        }
    }

    pub fn from_iter(shape: impl ToShape, iter: impl Iterator<Item=f32>) -> Self {
        Self::try_from_data(shape, iter.collect::<Vec<f32>>()).unwrap()
    }

    pub fn try_from_iter(shape: impl ToShape, iter: impl Iterator<Item=f32>) -> Result<Self> {
        Self::try_from_data(shape.to_shape(), iter.collect::<Vec<f32>>())
    }

    pub fn from_slice(shape: impl ToShape, slice: &[f32]) -> Self {
        Self::try_from_slice(shape, slice).unwrap()
    }

    pub fn try_from_slice(shape: impl ToShape, slice: &[f32]) -> Result<Self> {
        let shape = shape.to_shape();

        if shape.len() != slice.len() {
            return Err(anyhow!("
                Provided slice and shape to do not have the same length!
                slice Len: ({}), Shape: ({}, len: {}).
            ", slice.len(), shape, shape.len()))
        } else {
            Ok(
                Self { data: slice.to_vec() , shape}
            )
        }
    }

    pub fn from_fill(shape: impl ToShape, value: f32) -> Self {
        let shape = shape.to_shape();

        Self {
            data: vec![value; shape.len()], shape
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn try_reshape(&mut self, new: impl ToShape) -> Result<()> {
        self.shape.try_reshape(new)
    }

    pub fn reshape(&mut self, new: impl ToShape) {
        self.shape.try_reshape(new).unwrap()
    }

    pub fn axis(&self, axis: impl ToAxis) -> usize {
        self.shape[axis.to_axis()]
    }

    pub fn to_vec(self) -> Vec<f32> {
        self.data
    }

    pub fn as_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn slice_inner_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn slice(&self) -> TensorSlice {
        TensorSlice {
            data: &self.data,
            shape: self.shape,
        }
    }
}

use std::ops::Index;
use std::ops::IndexMut;

impl Index<usize> for Tensor {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Index<(usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (n, c) = index;
        &self.data[n*self.shape[1]*c]
    }
}

impl IndexMut<(usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (n, c) = index;
        &mut self.data[n*self.shape[1]*c]
    }
}

impl Index<(usize, usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        let (n,c,h) = index;
        let shape = self.shape;
        &self.data[n*shape[1]*shape[2]+c*shape[2]+h]
    }
}

impl IndexMut<(usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        let (n,c,h) = index;
        let shape = self.shape;
        &mut self.data[n*shape[1]*shape[2]+c*shape[2]*h]
    }
}

impl Index<(usize, usize, usize, usize)> for Tensor {
    type Output = f32;

    fn index(&self, index: (usize, usize, usize, usize)) -> &Self::Output {
        let (n,c,h,w) = index;
        let shape = self.shape;
        &self.data[n*shape[1]*shape[2]*shape[3]+c*shape[2]*shape[3]+h*shape[3]+w]
    }
}

impl IndexMut<(usize, usize, usize, usize)> for Tensor {
    fn index_mut(&mut self, index: (usize, usize, usize, usize)) -> &mut Self::Output {
        let (n,c,h,w) = index;
        let shape = self.shape;
        &mut self.data[n*shape[1]*shape[2]*shape[3]+c*shape[2]*shape[3]+h*shape[3]+w]
    }
}