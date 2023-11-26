
use std::sync::Arc;
use std::sync::RwLock;

use crate::tensor::shape::Shape;
use super::arena::ArenaIndex;

pub type Write<'a, T> = std::sync::RwLockWriteGuard<'a, T>;
pub type Read<'a, T> = std::sync::RwLockReadGuard<'a, T>;

/// GTensors are allocated by the Tape Arena. Internally, they
/// are Arc<RwLocks to Vectors. They have shared data, but they
/// have a unique shape. This allows us to perform operations like Reshape 
/// and debug with Zero cost. 
#[derive(Clone)]
pub struct GTensor {
    /// Locked reference to data allocated by the Arena. 
    pub(crate) data: Arc<RwLock<Vec<f32>>>,
    /// The unique shape of this data.
    pub(crate) shape: Shape,
    /// The location of the data in the Arena. 
    pub(crate) index: ArenaIndex,
    /// Whether or not this tensor is batched.
    pub(crate) is_batched: bool,
}

impl GTensor {
    pub fn new(data: Arc<RwLock<Vec<f32>>>, shape: Shape, index: ArenaIndex, batched: bool) -> Self {
        {
            let data = data.read().unwrap();
            if data.len() != shape.len() {
                panic!("Cannot create GTensor from vec with len {} and shape with len {}!", data.len(), shape.len())
            }
        }

        Self {
            data, 
            shape,
            index,
            is_batched: batched,
        }
    }

    pub fn clone_reshape(&self, shape: Shape, batched: bool) -> Self {
        if shape.len() != self.shape.len() {
            panic!("Cannot Clone-Reshape a tensor of shape {:?} into a tensor of shape {:?}.", shape, self.shape)
        }

        Self {
            data: self.data.clone(),
            shape,
            index: self.index,
            is_batched: batched,
        }
    }

    pub fn clone_with_batched(&self, batched: bool) -> Self {
        Self {
            data: self.data.clone(), 
            shape: self.shape,
            index: self.index,
            is_batched: batched,
        }
    }

    pub fn read(&self) -> Read<Vec<f32>> {
        self.data.read().unwrap()
    }

    pub fn write(&self) -> Write<Vec<f32>> {
        self.data.write().unwrap()
    }

    pub fn index(&self) -> ArenaIndex {
        self.index
    }

    pub fn len(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn shape2(&self) -> [usize; 2] {
        [self.shape[0], self.shape[1]]
    }

    pub fn shape4(&self) -> [usize; 4] {
        [self.shape[0], self.shape[1], self.shape[2], self.shape[3]]
    }

    pub fn slice(&self) -> TensorGuard {
        TensorGuard {
            data: self.read(),
            shape: self.shape,
        }
    }
}

pub struct TensorGuard<'a> {
    pub data: Read<'a, Vec<f32>>,
    pub shape: Shape,
}