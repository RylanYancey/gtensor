
use std::sync::Arc;

use upto::UpTo;
use ndarray::Array4;

use crate::util::Unsafe;
use crate::storage::Storage;
use super::operators::Operator;
use crate::storage::Tensor;
use crate::storage::Shape;
use crate::gpu::Kernel;
use crate::gpu::Stream;
use crate::gpu::Device;
use crate::storage::Float;

pub struct Node<S: Storage> {
    operator: Box<dyn Operator<S>>,
    dependencies: UpTo<6, Arc<Dependency<S>>>,
    kernels: UpTo<6, Kernel>,
    stream: Stream,
}

impl<S: Storage> Node<S>
where
    S: From<Shape> 
{
    pub fn reshape(&self, shape: Shape) {
        *self.dependencies[0].input.get_mut() = Tensor::new(shape);
    }
}

impl<S: Storage> Node<S> {
    pub fn stream(&self) -> Stream {
        self.stream
    }

    pub fn kernel(&self, index: usize) -> &Kernel {
        &self.kernels[index]
    }

    pub fn build() -> NodeBuilder<S> {
        todo!()
    }

    pub fn y(&self) -> &mut Tensor<S> {
        self.dependencies[0].input.get_mut()
    }

    pub fn gy(&self) -> &Tensor<S> {
        self.dependencies[0].gradient.get()
    }

    pub fn x1(&self) -> &Tensor<S> {
        self.dependencies[1].input.get()
    }

    pub fn x2(&self) -> &Tensor<S> {
        self.dependencies[2].input.get()
    }

    pub fn x3(&self) -> &Tensor<S> {
        self.dependencies[3].input.get()
    }

    pub fn x4(&self) -> &Tensor<S> {
        self.dependencies[4].input.get()
    }

    pub fn x5(&self) -> &Tensor<S> {
        self.dependencies[5].input.get()
    }

    pub fn g1(&self) -> &mut Tensor<S> {
        self.dependencies[1].gradient.get_mut()
    }

    pub fn g2(&self) -> &mut Tensor<S> {
        self.dependencies[2].gradient.get_mut()
    }

    pub fn g3(&self) -> &mut Tensor<S> {
        self.dependencies[3].gradient.get_mut()
    }

    pub fn g4(&self) -> &mut Tensor<S> {
        self.dependencies[4].gradient.get_mut()
    }

    pub fn g5(&self) -> &mut Tensor<S> {
        self.dependencies[5].gradient.get_mut()
    }
}

struct Dependency<S: Storage> {
    pub input: Unsafe<Tensor<S>>,
    pub gradient: Unsafe<Tensor<S>>,
    pub index: usize,
}

pub struct NodeBuilder<S: Storage> {
    data: Array4<S::F>,
    deps: Vec<usize>,
    kern: Vec<Kernel>,
}

impl<S: Storage> NodeBuilder<S> {
    pub fn with_kernel(mut self, dev: &Arc<Device>, module: &str, kern: &str) -> Self {
        let full_kernel_name = kern.to_owned() + "_" + S::F::NAME;

        let kern = 
        if dev.is_module_loaded(module) {
            dev.get_kernel(module, &full_kernel_name)   
                .expect(&format!("Failed to load kernel {} from module {}!", module, full_kernel_name));
        } else {
            
        };

        self
    }
}