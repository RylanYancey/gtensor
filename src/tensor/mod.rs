
pub mod getset;
pub mod shape;
pub mod axis;
pub mod slice;
pub mod iter;
pub mod math;

use shape::Shape;

pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}
