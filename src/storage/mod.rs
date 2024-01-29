
mod shape;
mod tensor;
mod traits;
mod cpu;
mod gpu;
mod float;

pub use float::Float;
pub use tensor::Tensor;
pub use gpu::Gpu;
pub use traits::Storage;
pub use shape::Shape;
pub use cpu::Cpu;
pub use traits::StorageInfo;