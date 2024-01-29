
use std::ops::*;

use num_traits::Zero;
use num_traits::FromPrimitive;
use num_traits::AsPrimitive;
use half::{bf16, f16};

pub trait Float
    : Copy 
    + Clone 
    + Zero 
    + FromPrimitive 
    + AsPrimitive<f32>
    + AsPrimitive<f64>
    + Add<Output=Self>
    + Mul<Output=Self>
    + Div<Output=Self>
    + Sub<Output=Self>
{
    const NAME: &'static str;
}
        
impl Float for f64 {
    const NAME: &'static str = "f64";
}

impl Float for f32 {
    const NAME: &'static str = "f32";
}

impl Float for bf16 {
    const NAME: &'static str = "bf16";
}

impl Float for f16 {
    const NAME: &'static str = "f16";
}