
use super::shape::Shape;
use super::slice::TensorSlice;

pub trait ToAxis {
    fn to_axis(self) -> usize;
}

impl ToAxis for usize {
    fn to_axis(self) -> usize {
        if self > 5 {
            panic!("Expected Axis less than 6, found: {}", self)
        }

        self
    }
}

impl ToAxis for char {
    fn to_axis(self) -> usize {
        match self {
            'N' => 0,
            'C' => 1,
            'H' => 2,
            'W' => 3,
            _ => panic!("
                    Cannot convert Char ({}) to Axis! 
                    Valid characters are N=0, C=1, H=2, and W=3.
                ", self)
        }
    }
}

pub struct AxisIter<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) curr: usize,
    pub(crate) len: usize,
    pub(crate) shape: Shape,
}

impl<'a> Iterator for AxisIter<'a> {
    type Item = TensorSlice<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == self.len {
            None
        } else {
            let out = Some(TensorSlice {
                data: &self.data[self.shape.len()*self.curr..self.shape.len()*(self.curr+1)],
                shape: self.shape,
            });

            self.curr += 1;

            out
        }
    }
}
