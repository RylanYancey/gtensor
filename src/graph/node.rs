
use super::gtensor::GTensor;
use crate::operators::Operator;
use crate::tensor::shape::Shape;
use crate::initializers::Initializer;

pub struct Node {
    pub(crate) y: GTensor,
    pub(crate) gy: GTensor,
    pub(crate) x: Vec<GTensor>,
    pub(crate) gx: Vec<GTensor>,
    pub(crate) is_batched: bool,
}

impl Node {
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.y.shape[0] = batch_size;
        self.gy.shape[0] = batch_size;

        let mut y = self.y.write();
        let mut gy = self.gy.write();

        if self.y.shape.len() != y.len() {
            if self.y.is_batched {
                y.resize(self.y.shape.len(), 0.0)
            }
        }

        if self.gy.shape.len() != gy.len() {
            if self.gy.is_batched {
                gy.resize(self.gy.shape.len(), 0.0)
            }
        }

        self.x.iter_mut().for_each(|x| if x.is_batched { x.shape[0] = batch_size });
        self.gx.iter_mut().for_each(|gx| if gx.is_batched { gx.shape[0] = batch_size });
    }

    pub fn y(&self) -> (&GTensor, &GTensor) {
        (&self.y, &self.gy)
    }

    pub fn x(&self, index: usize) -> (&GTensor, &GTensor) {
        (&self.x[index - 1], &self.gx[index - 1])
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // collect the shapes of all X into a Vec<String>
        let x = self.x.iter().enumerate().map(|(i, t)| {
            format!("{} shape: {}", i, t.shape())
        }).collect::<Vec<String>>();

        // collect the shapes of all GX into a Vec<String>,
        let gx = self.gx.iter().enumerate().map(|(i, t)| {
            format!("{} shape: {}", i, t.shape())
        }).collect::<Vec<String>>();

        write!(f, "{}", format!("
            Node Data: 
            -   batched: {}
            -   y_shape: {},
            -   gy_shape: {},
            -   x: {:?},
            -   gx: {:?}
        ", self.is_batched, self.y.shape(), self.gy.shape(), x, gx))
    }
}

pub struct NodeBuilder {
    pub op: Box<dyn Operator>,
    pub deps: Vec<usize>,
    pub shape: Shape,
    pub skip: bool,
    pub init: Option<Box<dyn Initializer>>,
    pub is_batched: bool,
}