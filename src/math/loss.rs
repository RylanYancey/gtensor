
use crate::Tensor;
use crate::tensor::slice::TensorSlice;

pub fn mse(
    t: TensorSlice,
    p: TensorSlice,
    e: &mut Tensor,
    g: &mut Tensor,
) {
    bmls::mse(
        t.data,
        p.data,
        &mut e.data,
        &mut g.data,
        t.shape.as_array2(),
    ).unwrap()
}