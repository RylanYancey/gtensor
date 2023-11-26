
use super::*;

#[derive(Clone)]
struct Random {
    max: f32, min: f32
}

impl Initializer for Random {
    fn init_vec(&self, shape: Shape) -> Vec<f32> {
        let mut rng = rand::thread_rng();

        (0..shape.len()).map(|_| rng.gen_range(self.min..self.max)).collect()
    }

    fn _clone(&self) -> Box<dyn Initializer> {
        Box::new(self.clone())
    }
}

pub fn random(max: f32, min: f32) -> Box<dyn Initializer> {
    Box::new(Random { max, min })
}