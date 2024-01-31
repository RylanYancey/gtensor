
mod gpu;
mod util;
mod storage;
mod nn;

#[test]
fn test_gpu() {
    use ndarray::Array4;
    use crate::storage::Tensor;
    use crate::storage::Gpu;
    use crate::storage::Storage;

    let array = Array4::from_shape_vec([1, 1, 1, 1], vec![0.0]).unwrap();

    let mut tensor: Tensor<Gpu<f32>> = Tensor::from_ndarray(&array);
    tensor.fill(1.0);

    let array = tensor.as_ndarray();

    for i in array.iter() {
        assert_eq!(*i, 1.0);
    }
}