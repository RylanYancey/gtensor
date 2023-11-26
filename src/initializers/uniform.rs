
use super::*;

struct Uniform {
    value: f32,
}

impl Initializer for Uniform {
    fn init_vec(&self, shape: Shape) -> Vec<f32> {
        vec![self.value; shape.len()]
    }

    fn _clone(&self) -> Box<dyn Initializer> {
        Box::new(Uniform { value: self.value })
    }
}

pub fn uniform(value: f32) -> Box<dyn Initializer> {
    Box::new(Uniform {value})
}