
use super::*;

struct Constant {
    v: f32,
}

impl Initializer for Constant {
    fn init_vec(&self, shape: Shape) -> Vec<f32> {
        vec![self.v; shape.len()]
    }

    fn _clone(&self) -> Box<dyn Initializer> {
        Box::new(Constant { v: self.v })
    }
}

pub fn constant(v: f32) -> Box<dyn Initializer> {
    Box::new(Constant { v })
}