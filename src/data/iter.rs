
use super::dataset::Dataset;
use crate::tensor::shape::Shape;
use crate::tensor::slice::TensorSlice;

impl Dataset {  
    pub fn iter<'a>(&'a self) -> DatasetIter<'a> {
        self.iter_batched(1)
    }

    pub fn iter_batched<'a>(&'a self, batch_size: usize) -> DatasetIter<'a> {
        let mut feature_shape = self.feature_shape;
        feature_shape[0] = batch_size;

        let mut label_shape = self.label_shape;
        label_shape[0] = batch_size;

        if self.features.len() % feature_shape.len() != 0 {
            panic!("Cannot iterate dataset batched; provided batch size is not valid! 
                (Number of features must be evenly divisible by the batch size), feature shape: {}, feature vector len: {}",
                feature_shape, self.features.len())
        }

        DatasetIter {
            features: &self.features,
            labels: &self.labels,
            feature_shape,
            feature_len: feature_shape.len(),
            label_shape,
            label_len: label_shape.len(),
            feature_curr: 0,
            label_curr: 0,
        }
    }
}

pub struct DatasetIter<'a> {
    features: &'a Vec<f32>,
    labels: &'a Vec<f32>,
    feature_shape: Shape,
    feature_len: usize,
    label_shape: Shape,
    label_len: usize,
    feature_curr: usize,
    label_curr: usize,
}

impl<'a> Iterator for DatasetIter<'a> {
    type Item = (TensorSlice<'a>, TensorSlice<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.feature_curr >= self.features.len() {
            None
        } else {
            let out = Some((
                TensorSlice {
                    data: &self.features[self.feature_curr..self.feature_curr+self.feature_len],
                    shape: self.feature_shape,
                }, 
                TensorSlice {
                    data: &self.labels[self.label_curr..self.label_curr+self.label_len],
                    shape: self.label_shape,
                }
            ));

            self.feature_curr += self.feature_len;
            self.label_curr += self.label_len;

            out
        }
    }
}