
use core::slice::SlicePattern;

// imports par_sort_by for KNN.
use rayon::slice::ParallelSliceMut;
use rayon::prelude::*;

use crate::Dataset;
use crate::tensor::slice::TensorSlice;

pub fn knn(dataset: &Dataset, feature: &TensorSlice, k: usize, metric: &impl KnnMetric) -> Vec<f32> {
    if dataset.feature_shape.len() != feature.shape.len() / feature.shape[0] {
        panic!("Dataset feature len does not match provided feature len.")
    }

    feature.data.par_chunks(dataset.feature_shape.len()).map(|chunk| {
        let slice = TensorSlice {
            data: chunk.as_slice(),
            shape: dataset.feature_shape,
        };

        knn_internal(dataset, &slice, k, metric)
    }).collect()
}

fn knn_internal(dataset: &Dataset, feature: &TensorSlice, k: usize, metric: &impl KnnMetric) -> f32 {

    // Vec<(Distance, Label)>. 
    let mut dist_label: Vec<(f32, f32)> = Vec::with_capacity(dataset.len());

    // collect all of the distances and labels into a vector.
    for(f, label) in dataset.iter() {
        dist_label.push((metric.distance(&f, feature), label[0]));
    }

    // sort by distance, ignoring the label since it isn't important for the sort.
    dist_label.par_sort_by(|(a,_), (b,_)| a.partial_cmp(b).unwrap());

    // resize to length K. 
    // This leaves us with the K nearest points.
    dist_label.resize(k, (0.0, 0.0));

    // find the most frequent value (count, label)
    let mut most = (0, 0.0);
    dist_label.iter().for_each(|(d,l)| {
        // count the number of time `d` appears in the vec.
        let count = dist_label.iter().filter(|(d2,_)| d2 == d).count();

        // if the count is greater than the existing, set it.
        if count > most.0 {
            most = (count, *l)
        }
    });

    // return the prediction.
    most.1
}

pub trait KnnMetric: Send+Sync {
    fn distance(&self, left: &TensorSlice, right: &TensorSlice) -> f32;
}

pub mod metric {
    use super::KnnMetric;
    use crate::tensor::slice::TensorSlice;

    pub struct Euclidian;

    impl KnnMetric for Euclidian {
        fn distance(&self, left: &TensorSlice, right: &TensorSlice) -> f32 {
            f32::sqrt(left.iter().zip(right.iter()).map(|(x,y)| f32::powi(*y-*x, 2)).sum())
        }
    }

    pub struct Manhattan;

    impl KnnMetric for Manhattan {
        fn distance(&self, left: &TensorSlice, right: &TensorSlice) -> f32 {
            left.iter().zip(right.iter()).map(|(x,y)| f32::abs(*x-*y)).sum()
        }
    }

    /// The provided value is the p-value defined by the Minkowsky function.
    pub struct Minkowsky(pub f32);

    impl KnnMetric for Minkowsky {
        fn distance(&self, left: &TensorSlice, right: &TensorSlice) -> f32 {
            f32::powf(left.iter().zip(right.iter()).map(|(x,y)| f32::abs(*x-*y)).sum(), 1./self.0)
        }
    }

    pub struct Hamming;

    impl KnnMetric for Hamming {
        fn distance(&self, left: &TensorSlice, right: &TensorSlice) -> f32 {
            left.iter().zip(right.iter()).map(|(x,y)| if *x!=*y {1.0} else {0.0}).sum()
        }
    }
}

pub mod dist {
    /// Standard Euclidian Distance.
    #[inline]
    pub fn distance(left: &[f32], right: &[f32]) -> f32 {
        f32::sqrt(left.iter().zip(right.iter()).map(|(x,y)| f32::powi(*y-*x, 2)).sum())
    }
        
    /// Manhattan Distance. Measures sum of the the absolute values of Xi - Yi.
    #[inline]
    pub fn manhattan(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right.iter()).map(|(x,y)| f32::abs(*x-*y)).sum()
    }
        
    /// Minkowsky Distance. Measures the sum of the absolute values of Xi - Yi raised to (1./p). 
    #[inline]
    pub fn minkowsky(left: &[f32], right: &[f32], p: f32) -> f32 {
        f32::powf(left.iter().zip(right.iter()).map(|(x,y)| f32::abs(*x-*y)).sum(), 1./p)
    }
        
    /// Hamming distance. Measures the number of values that differ between the two points.
    #[inline]
    pub fn hamming(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right.iter()).map(|(x,y)| if *x!=*y {1.0} else {0.0}).sum()
    }
}