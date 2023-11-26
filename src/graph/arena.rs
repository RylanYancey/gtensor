
use std::sync::Arc;
use std::sync::RwLock;

use serde::{Serialize, Deserialize};

use super::gtensor::GTensor;
use super::serde::GTensorSerde;
use crate::tensor::shape::Shape;
use crate::initializers::Initializer;

pub struct Arena {
    /// outputs are neither saved nor cleared.
    pub(in super) outputs: Vec<Arc<RwLock<Vec<f32>>>>,
    /// Gradients are not saved, but get cleared in the backwards step.
    pub(in super) gradients: Vec<Arc<RwLock<Vec<f32>>>>,
    /// Parameters are saved.
    pub(in super) parameters: Vec<Arc<RwLock<Vec<f32>>>>
}

impl Arena {
    pub fn new() -> Self {
        Self {
            outputs: Vec::new(),
            gradients: Vec::new(),
            parameters: Vec::new(),
        }
    }

    /// Fill All Gradient Tensors with Zeros.
    /// This is performed before the backwards pass starts,
    /// and is necessary since all backwards operations add to
    /// the gradient tensor rather than assigning. 
    pub fn clear_gradients(&mut self) {
        for i in 0..self.gradients.len() {
            self.gradients[i].write().unwrap().fill(0.);
        }
    }

    /// Allocation for data in an Operator. Allocates two tensors,
    /// one for the output Y and one for the gradient GY. 
    /// 
    /// Returns (output, gradient)
    pub fn alloc(&mut self, shape: Shape, batched: bool) -> (GTensor, GTensor) {
        // allocate the data
        let output = Arc::new(RwLock::new(vec![0.0; shape.len()]));
        let gradient = Arc::new(RwLock::new(vec![0.0; shape.len()]));
        // Initialize the Tensors
        let t1 = GTensor::new(output.clone(), shape, ArenaIndex::new('O', self.outputs.len()), batched);
        let t2 = GTensor::new(gradient.clone(), shape, ArenaIndex::new('G', self.gradients.len()), batched);
        // store the data in the arena
        self.outputs.push(output);
        self.gradients.push(gradient);
        // return the allocated tensors.
        (t1, t2)
    }

    /// Allocation for data in an Parameter. Allocates two tensors,
    /// one for the weight/output Y and one for the gradient GY. 
    /// 
    /// Returns (parameter, gradient)
    pub fn alloc_parameter(&mut self, shape: Shape, init: Box<dyn Initializer>, batched: bool) -> (GTensor, GTensor) {
        // allocate the data, using the provided initializer 
        // to fill the parameter data. 
        let parameter = Arc::new(RwLock::new(init.init_vec(shape)));
        let gradient = Arc::new(RwLock::new(vec![0.0; shape.len()]));
        // Initialize the Tensors
        let t1 = GTensor::new(parameter.clone(), shape, ArenaIndex::new('P', self.parameters.len()), batched);
        let t2 = GTensor::new(gradient.clone(), shape, ArenaIndex::new('G', self.gradients.len()), batched);
        // store the data in the arena
        self.parameters.push(parameter);
        self.gradients.push(gradient);
        // return the allocated tensors
        (t1, t2)
    }

    pub fn load(&self, tensor: GTensorSerde) -> GTensor {
        // load the data using the provided Vector ID.
        let data = 
        match tensor.index.vec {
            'O' => self.outputs[tensor.index.index].clone(),
            'G' => self.gradients[tensor.index.index].clone(),
            'P' => self.parameters[tensor.index.index].clone(),
            _ => panic!("Invalid ArenaIndex Vec ID! id: {}", tensor.index.vec)
        };

        {
            // make sure tensor lengths match
            let len = data.read().unwrap().len();
            if len != tensor.shape.len() {
                panic!("Cannot load tensor data of length {} into a tensor with shape {}! (length mismatch)",
                    len, tensor.shape)
            }
        }

        // return the constructed GTensor
        GTensor {
            data, shape: tensor.shape, index: tensor.index, is_batched: tensor.is_batched
        }
    }
}

/// The Produced GTensors store an Index to their data
/// in the Arena. The vec (output, gradient, or parameter)
/// is indicated by 'O', 'G', or 'P', and the index is the
/// index of the Vec the data is stored at. 
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct ArenaIndex {
    /// 'O', 'G', or 'P'.
    vec: char,
    index: usize,
}

impl ArenaIndex {
    pub fn new(vec: char, index: usize) -> Self {
        Self {
            vec, index,
        }
    }
}