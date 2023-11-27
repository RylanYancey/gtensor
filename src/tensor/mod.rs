
pub mod getset;
pub mod shape;
pub mod axis;
pub mod slice;
pub mod iter;
pub mod math;

use shape::Shape;

pub struct Tensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Shape,
}
