
use std::ffi::c_void;

use as_slice::AsSlice;

use super::cu;
use anyhow::Result;
use super::stream::Stream;

#[derive(Clone)]
pub struct Kernel {
    name: String,
    kernel: cu::Function,
}

impl Kernel {
    pub fn from(name: String, kernel: cu::Function) -> Self {
        Self {
            name, kernel,
        }
    }

    pub fn launch(
        &self, 
        grid: impl Into<GridDim>, 
        block: impl Into<BlockDim>, 
        stream: Stream, 
        params: impl ToKernelParams
    ) -> Result<()> {
        let grid = grid.into();
        let block = block.into();

        cu::launch::kernel(
            &self.kernel, 
            (grid.0, grid.1, grid.2), 
            (block.0, block.1, block.2), 
            0, 
            &stream.stream, 
            params.to_kernel_params().as_slice(), 
            &[]
        )
    }
}

pub struct GridDim(pub u32, pub u32, pub u32);

impl GridDim {
    pub fn x(&self) -> u32 {
        self.0
    }

    pub fn y(&self) -> u32 {
        self.1
    }

    pub fn z(&self) -> u32 {
        self.2
    }
}

impl From<[u32; 1]> for GridDim {
    fn from(value: [u32; 1]) -> Self {
        Self(value[0], 1, 1)
    }
}

impl From<[u32; 2]> for GridDim {
    fn from(value: [u32; 2]) -> Self {
        Self(value[0], value[1], 1)
    }
}

impl From<[u32; 3]> for GridDim {
    fn from(value: [u32; 3]) -> Self {
        Self(value[0], value[1], value[2])
    }
}

pub struct BlockDim(pub u32, pub u32, pub u32);

impl BlockDim {
    pub fn x(&self) -> u32 {
        self.0
    }

    pub fn y(&self) -> u32 {
        self.1
    }

    pub fn z(&self) -> u32 {
        self.2
    }
}

impl From<[u32; 1]> for BlockDim {
    fn from(value: [u32; 1]) -> Self {
        Self(value[0], 1, 1)
    }
}

impl From<[u32; 2]> for BlockDim {
    fn from(value: [u32; 2]) -> Self {
        Self(value[0], value[1], 1)
    }
}

impl From<[u32; 3]> for BlockDim {
    fn from(value: [u32; 3]) -> Self {
        Self(value[0], value[1], value[2])
    }
}


pub trait ToKernelParams {
    type Output: AsSlice<Element=*mut c_void>;

    fn to_kernel_params(&self) -> Self::Output;
}

impl<T> ToKernelParams for (T,) {
    type Output = [*mut c_void; 1];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0)
        ]
    }
}

impl<T1,T2> ToKernelParams for (T1,T2) {
    type Output = [*mut c_void; 2];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1)
        ]
    }
}

impl<T1,T2,T3> ToKernelParams for (T1,T2,T3) {
    type Output = [*mut c_void; 3];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1),
            to_c_void(&self.2),
        ]
    }
}

impl<T1,T2,T3,T4> ToKernelParams for (T1,T2,T3,T4) {
    type Output = [*mut c_void; 4];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1),
            to_c_void(&self.2),
            to_c_void(&self.3),
        ]
    }
}

impl<T1,T2,T3,T4,T5> ToKernelParams for (T1,T2,T3,T4,T5) {
    type Output = [*mut c_void; 5];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1),
            to_c_void(&self.2),
            to_c_void(&self.3),
            to_c_void(&self.4),
        ]
    }
}

impl<T1,T2,T3,T4,T5,T6> ToKernelParams for (T1,T2,T3,T4,T5,T6) {
    type Output = [*mut c_void; 6];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1),
            to_c_void(&self.2),
            to_c_void(&self.3),
            to_c_void(&self.4),
            to_c_void(&self.5),
        ]
    }
}

impl<T1,T2,T3,T4,T5,T6,T7> ToKernelParams for (T1,T2,T3,T4,T5,T6,T7) {
    type Output = [*mut c_void; 7];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1),
            to_c_void(&self.2),
            to_c_void(&self.3),
            to_c_void(&self.4),
            to_c_void(&self.5),
            to_c_void(&self.6),
        ]
    }
}

impl<T1,T2,T3,T4,T5,T6,T7,T8> ToKernelParams for (T1,T2,T3,T4,T5,T6,T7,T8) {
    type Output = [*mut c_void; 8];

    fn to_kernel_params(&self) -> Self::Output {
        [
            to_c_void(&self.0),
            to_c_void(&self.1),
            to_c_void(&self.2),
            to_c_void(&self.3),
            to_c_void(&self.4),
            to_c_void(&self.5),
            to_c_void(&self.6),
            to_c_void(&self.7),
        ]
    }
}

fn to_c_void<T>(item: &T) -> *mut c_void {
    item as *const T as *mut c_void
}