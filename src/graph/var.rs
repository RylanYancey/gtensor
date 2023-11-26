
use super::builder::TapeBuilder;
use super::node::NodeBuilder;
use crate::tensor::shape::Shape;

pub struct Var<'t> {
    pub tape: &'t TapeBuilder,
    pub shape: Shape,
    pub index: usize,
    pub is_batched: bool,
}

impl<'t> Var<'t> {
    pub fn extend(&self, builder: NodeBuilder) -> Var<'t> {
        self.tape.extend(builder)
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn shape2(&self) -> [usize; 2] {
        [self.shape[0], self.shape[1]]
    }

    pub fn shape4(&self) -> [usize; 4] {
        [self.shape[0], self.shape[1], self.shape[2], self.shape[3]]
    }
}