
use std::ops::Index;

use upto::UpTo;

#[derive(Clone)]
pub struct Shape(UpTo<4, usize>);

impl Shape {
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    pub fn as_array4(&self) -> [usize; 4] {
        let mut out = [1usize; 4];

        for (i, v) in self.0.iter().enumerate() {
            out[i] = *v;
        }

        out
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.0.len() {
            &1
        } else {
            &self.0[index]
        }
    }
}

impl Index<char> for Shape {
    type Output = usize;

    fn index(&self, index: char) -> &Self::Output {
        self.index(match index {
            'N' | 'n' => 0,
            'C' | 'c' => 1,
            'H' | 'h' => 2,
            'W' | 'w' => 3,
            _ => panic!("Unrecognized index character")
        })
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self(UpTo::from_slice(&value))
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self(UpTo::from_slice(value))
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        for i in 0..4 {
            if self[i] != other[i] {
                return false
            }
        }

        true
    }
}