
#![feature(slice_pattern)]

pub mod tensor;
pub mod graph;
pub mod operators;
pub mod initializers;
pub mod optimizers;
pub mod math;
pub mod data;
pub mod plot;
pub mod alg;

pub use optimizers::opt;
pub use initializers::init;
pub use operators::op;

pub use graph::tape::Tape;
pub use tensor::Tensor;
pub use data::dataset::Dataset;
pub use alg::knn;