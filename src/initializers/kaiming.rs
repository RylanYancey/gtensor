
use super::*;

struct Kaiming {
    gain: f32,
    mode: bool,
    norm: bool,
}

impl Initializer for Kaiming {
    fn init_vec(&self, shape: Shape) -> Vec<f32> {
            // calculate standard deviation
            let field_size = shape[2] * shape[3];
            let fan = if self.mode { shape[1] * field_size } else { shape[0] * field_size };
            let std = self.gain / f32::sqrt(fan as f32);

            // fill
            let mut rng = rand::thread_rng();
            (0..shape.len())
                .map(|_| {
                    if self.norm {
                        rng.sample::<f32, StandardNormal>(StandardNormal) * std
                    } else {
                        rng.gen_range(-std..std)
                    }
            }).collect::<Vec<f32>>()
    }

    fn _clone(&self) -> Box<dyn Initializer> {
        Box::new(
            Kaiming {
                gain: self.gain,
                mode: self.mode,
                norm: self.norm
            }
        )
    }
}

/// Actv can be "sigmoid", "linear", "tanh", "relu", or "selu".
/// mode can be either "fan_in" or "fan_out".
pub fn kaiming_uniform(actv: &str, mode: &str) -> Box<dyn Initializer> {
    Box::new( Kaiming {
        gain: gain(actv),
        mode: if mode == "fan_in" { false } else { true }, 
        norm: false,
    })
}

/// Actv can be "sigmoid", "linear", "tanh", "relu", or "selu".
/// mode can be either "fan_in" or "fan_out".
pub fn kaiming_normal(actv: &str, mode: &str) -> Box<dyn Initializer> {
    Box::new( Kaiming {
        gain: gain(actv),
        mode: if mode == "fan_in" { false } else { true },
        norm: true,
    })
}