
use super::*;

struct Xavier {
    gain: f32,
    mode: bool,
}

impl Initializer for Xavier {
    fn init_vec(&self, shape: Shape) -> Vec<f32> {
        if self.mode {
            let field_size = shape[2] * shape[3];
            let fan_in = shape[1] * field_size;
            let fan_out = shape[0] * field_size;
            let std = self.gain * f32::sqrt(2.0 / (fan_in + fan_out) as f32);

            let mut rng = rand::thread_rng();
            (0..shape.len())
                .map(|_| rng.sample::<f32, StandardNormal>(StandardNormal) * std)
                .collect::<Vec<f32>>()
        } else {
            let field_size = shape[2] * shape[3];
            let fan_in = shape[1] * field_size;
            let fan_out = shape[0] * field_size;
            let std = self.gain * f32::sqrt(2.0 / (fan_in + fan_out) as f32);
            let a = f32::sqrt(3.0) * std;

            let mut rng = rand::thread_rng();
            (0..shape.len())
                .map(|_| rng.gen_range(a..-a))
                .collect::<Vec<f32>>()
        }
    }

    fn _clone(&self) -> Box<dyn Initializer> {
        Box::new(Xavier {
            gain: self.gain, mode: self.mode
        })
    }
}

/// Actv can be "sigmoid", "linear", "tanh", "relu", or "selu".
/// mode can be either "fan_in" or "fan_out".
pub fn xavier_normal(actv: &str, mode: &str) -> Box<dyn Initializer> {
    Box::new(Xavier { gain: gain(actv), mode: if mode == "fan_in" {true} else {false}})
}

/// Actv can be "sigmoid", "linear", "tanh", "relu", or "selu".
/// mode can be either "fan_in" or "fan_out".
pub fn xavier_uniform(actv: &str, mode: &str) -> Box<dyn Initializer> {
    Box::new(Xavier { gain: gain(actv), mode: if mode == "fan_in" {true} else {false}})
}