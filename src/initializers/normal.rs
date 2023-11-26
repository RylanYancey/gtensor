
use super::*;

#[derive(Clone)]
struct Normal {
    mean: f32,
    stdev: f32,
}

impl Initializer for Normal {
    fn init_vec(&self, shape: Shape) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..shape.len())
            .map(|_| rng.sample::<f32, StandardNormal>(StandardNormal) + self.mean * self.stdev)
            .collect::<Vec<f32>>()
    }

    fn _clone(&self) -> Box<dyn Initializer> {
        Box::new(self.clone())
    }
}

pub fn normal(mean: f32, stdev: f32) -> Box<dyn Initializer> {
    Box::new(Normal { mean, stdev })
}