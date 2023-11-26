
use crate::tensor::shape::Shape;
use crate::tensor::shape::ToShape;

pub struct Dataset {
    pub(crate) features: Vec<f32>,
    pub(crate) labels: Vec<f32>,
    pub(crate) feature_shape: Shape,
    pub(crate) label_shape: Shape,
}

impl Dataset {
    pub fn new(feature_shape: impl ToShape, label_shape: impl ToShape) -> Self {
        Self {
            features: Vec::new(),
            labels: Vec::new(),
            feature_shape: feature_shape.to_shape().add_batch(1),
            label_shape: label_shape.to_shape().add_batch(1),
        }
    }

    pub fn len(&self) -> usize {
        self.features.len() / self.feature_shape.len()
    }

    // pub fn split(mut self, percentage: f32) -> (Self, Self) {
    //     let num = ((self.features.len() / self.feature_shape.len()) as f32 * (percentage / 100.)) 
    //         as usize * self.feature_shape.len();

            
    //     todo!()
    // }
}



