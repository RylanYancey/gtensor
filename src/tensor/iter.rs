
use super::Tensor;
// use super::shape::Shape;
// use super::slice::TensorSlice;

impl Tensor {
    pub fn iter(&self) -> core::slice::Iter<'_, f32> {
        self.data.iter()
    }
}

// impl Tensor {
//     pub fn iter(&self) -> std::slice::Iter<'_, f32> {
//         self.data.iter()
//     }

//     pub fn axis_iter(&self) -> AxisIter<'_> {
//         todo!()
//     }
// }

// pub struct AxisIter<'a> {
//     data: &'a Vec<f32>,
//     shape: Shape,
//     curr: usize,
//     step: usize,
//     len: usize,
// }

// impl<'a> Iterator for AxisIter<'a> {
//     type Item = TensorSlice<'a>;

//     fn next(&mut self) -> Option<Self::Item> {
//         todo!()
//     }
// }