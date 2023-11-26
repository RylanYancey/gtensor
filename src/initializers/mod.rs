
use rand::Rng;
use rand_distr::StandardNormal;

use crate::tensor::shape::Shape;

pub trait Initializer {
    fn init_vec(&self, shape: Shape) -> Vec<f32>;   
    fn _clone(&self) -> Box<dyn Initializer>;
}

pub fn gain(nonlinearity: &str) -> f32 {
    match nonlinearity {
        "linear" | "sigmoid" => 1.,
        "tanh" => 5. / 3.,
        "relu" => f32::sqrt(2.),
        "selu" => 3. / 4.,
        _ => panic!("Unrecognized Nonlinearity.")
    }
}

mod random;
mod uniform;
mod kaiming;
mod xavier;
mod constant;
mod normal;

pub mod init {
    use super::*;

    pub use random::random;
    pub use uniform::uniform;
    pub use kaiming::kaiming_normal;
    pub use kaiming::kaiming_uniform;
    pub use xavier::xavier_normal;
    pub use xavier::xavier_uniform;
    pub use constant::constant;
    pub use normal::normal;
}