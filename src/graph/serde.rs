
use std::sync::Arc;
use std::sync::RwLock;

use serde::{Serialize, Deserialize};

use super::arena::ArenaIndex;
use super::arena::Arena;
use super::node::Node;
use crate::tensor::shape::Shape;
use crate::operators::Operator;

// Arena Serialization / Deserialization

#[derive(Serialize, Deserialize)]
pub struct ArenaSerde {
    outputs: Vec<usize>,
    gradients: Vec<usize>,
    parameters: Vec<Vec<f32>>,
}

impl ArenaSerde {
    pub fn from_arena(arena: &Arena) -> Self {
        Self {
            outputs: arena.outputs.iter().map(|data| data.read().unwrap().len()).collect(),
            gradients: arena.gradients.iter().map(|data| data.read().unwrap().len()).collect(),
            parameters: arena.parameters.iter().map(|data| data.read().unwrap().clone()).collect(),
        }
    }

    pub fn to_arena(&self) -> Arena {
        Arena {
            outputs: self.outputs.iter().map(|len| Arc::new(RwLock::new(vec![0.0; *len]))).collect(),
            gradients: self.gradients.iter().map(|len| Arc::new(RwLock::new(vec![0.0; *len]))).collect(),
            parameters: self.parameters.iter().map(|data| Arc::new(RwLock::new(data.clone()))).collect(),
        }
    }
}

// Node Serialization / Deserialization 

#[derive(Serialize, Deserialize)]
pub struct NodeSerde {
    y: GTensorSerde,
    gy: GTensorSerde,
    x: Vec<GTensorSerde>,
    gx: Vec<GTensorSerde>,
    is_batched: bool,
}

impl NodeSerde {
    pub fn from_node(node: &Node) -> Self {
        Self{
            y: GTensorSerde {
                index: node.y.index,
                shape: node.y.shape,
                is_batched: node.is_batched,
            },
            gy: GTensorSerde {
                index: node.gy.index,
                shape: node.gy.shape,
                is_batched: node.is_batched,
            },
            x: node.x.iter().map(|gt| GTensorSerde {
                index: gt.index,
                shape: gt.shape,
                is_batched: gt.is_batched,
            }).collect(),
            gx: node.gx.iter().map(|gt| GTensorSerde {
                index: gt.index,
                shape: gt.shape,
                is_batched: gt.is_batched,
            }).collect(),
            is_batched: node.is_batched,
        }
    }

    pub fn to_node(&self, arena: &Arena) -> Node {
        Node {
            y: arena.load(self.y),
            gy: arena.load(self.gy),
            x: self.x.iter().map(|gt| arena.load(*gt)).collect(),
            gx: self.gx.iter().map(|gt| arena.load(*gt)).collect(),
            is_batched: self.is_batched,
        }
    }
}

// GTensor Serialization / Deserialization

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct GTensorSerde {
    pub index: ArenaIndex,
    pub shape: Shape,
    pub is_batched: bool,
}

// Operator Serialization / Deserialization

#[derive(Serialize, Deserialize)]
pub struct OperatorSerde {
    #[serde(with = "serde_traitobject")]
    op: Box<dyn Operator>,
}

impl OperatorSerde {
    pub fn to_op(self) -> Box<dyn Operator> {
        self.op
    }

    pub fn from_op(op: &Box<dyn Operator>) -> Self {
        Self { op: dyn_clone::clone_box(op.as_ref()) }
    }
}