
use std::fmt::Display;

use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Shape([usize; 6]);

impl Shape {
    pub fn len(&self) -> usize {
        self[0]*self[1]*self[2]*self[3]*self[4]*self[5]
    }

    pub fn reshape(&mut self, new: impl ToShape) {
        self.try_reshape(new).unwrap()
    }

    pub fn add_batch(mut self, batch_size: usize) -> Self {
        for i in (1..6).rev() {
            self[i] = self[i-1]
        }

        self[0] = batch_size;
        self
    }

    pub fn try_reshape(&mut self, new: impl ToShape) -> Result<()> {
        let new = new.to_shape();

        if new.len() == self.len() {
            Ok(())
        } else {
            return Err(anyhow!("
                Length of Old Shape does not match length of New Shape! (Old: {}), (New: {})
            ", self, new))
        }
    }

    pub fn as_array2(&self) -> [usize; 2] {
        [self[0],self[1]]
    }

    pub fn as_array3(&self) -> [usize; 3] {
        [self[0],self[1],self[2]]
    }

    pub fn as_array4(&self) -> [usize; 4] {
        [self[0],self[1],self[2],self[3]]
    }

    pub fn as_array5(&self) -> [usize; 5] {
        [self[0],self[1],self[2],self[3],self[4]]
    }

    pub fn as_array6(&self) -> [usize; 6] {
        [self[0],self[1],self[2],self[3],self[4],self[5]]
    }
}

impl Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}, {}, {}, {}, {}]", self[0], self[1], self[2], self[3], self[4], self[5])
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

pub trait ToShape {
    fn to_shape(self) -> Shape;
}

impl ToShape for Shape {
    fn to_shape(self) -> Shape {
        self
    }
}

macros::impl_shape!(1,2,3,4,5,6);

mod macros {
    macro_rules! impl_shape {
        ($($n:literal),*) => {
            $(
            impl ToShape for [usize; $n] {
                fn to_shape(self) -> Shape {
                    for i in self.iter() {
                        if *i == 0 {
                            panic!("Shape Axes must be non-zero!")
                        }
                    }

                    let mut arr = [1; 6];
                    arr.chunks_exact_mut($n).take(1).for_each(|slice| slice.copy_from_slice(&self));
                    Shape(arr)
                }
            }
            )*
        }
    }
    pub(super) use impl_shape;
}