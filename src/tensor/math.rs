
use super::Tensor;

impl Tensor {
    /// Each element in the tensor added together
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Standard Euclidian Distance.
    pub fn distance(&self, other: &Tensor) -> f32 {
        f32::sqrt(self.iter().zip(other.iter()).map(|(x,y)| f32::powi(*y-*x, 2)).sum())
    }

    /// Manhattan Distance. Measures sum of the the absolute values of Xi - Yi.
    pub fn manhattan_distance(&self, other: &Tensor) -> f32 {
        self.iter().zip(other.iter()).map(|(x,y)| f32::abs(*x-*y)).sum()
    }

    /// Minkowsky Distance. Measures the sum of the absolute values of Xi - Yi raised to (1./p). 
    pub fn minkowsky_distance(&self, other: &Tensor, p: f32) -> f32 {
        f32::powf(self.iter().zip(other.iter()).map(|(x,y)| f32::abs(*x-*y)).sum(), 1./p)
    }

    /// Hamming distance. Measures the number of values that differ between the two points.
    pub fn hamming_distance(&self, other: &Tensor) -> f32 {
        self.iter().zip(other.iter()).map(|(x,y)| if *x!=*y {1.0} else {0.0}).sum()
    }
}