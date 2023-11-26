
use super::shape::Shape;
use super::axis::ToAxis;
use super::axis::AxisIter;

#[derive(Copy, Clone)]
pub struct TensorSlice<'a> {
    pub data: &'a [f32],
    pub shape: Shape,
}

impl<'a> TensorSlice<'a> {
    pub fn iter(&self) -> std::slice::Iter<'a, f32> {
        self.data.iter()
    }

    /// Each element in the tensor added together
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn iter_axis(&self, axis: impl ToAxis) -> AxisIter {
        let axis = axis.to_axis();

        let mut axis_shape = self.shape;
        axis_shape[axis] = 1;

        AxisIter {
            data: self.data,
            curr: 0,
            len: self.shape[axis],
            shape: axis_shape,
        }
    }
}

impl<'a> std::ops::Index<usize> for TensorSlice<'a> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}