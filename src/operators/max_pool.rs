
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct MaxPool {
    kernel: [usize; 2],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
    indices: Vec<usize>,
}

impl Operator for MaxPool {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::max_pool(
            &x.read(), &mut y.write(), &mut self.indices,
            x.shape4(), self.kernel, self.stride, self.padh, self.padw
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::max_pool_wrt_a(
            &self.indices, &gy.read(), &mut gx.write()
        )?;

        Ok(())
    }

    fn reshape(&mut self, new: Shape) {
        self.indices.resize(new.len(), 0)
    }
}

impl Display for MaxPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Max Pool Operator. Kernel: {:?}, Stride: {:?}, padh: {:?}, padw: {:?}", 
            self.kernel, self.stride, self.padh, self.padw)
    }
}

pub fn max_pool<'t>(x: Var<'t>, p: PoolParams) -> Var<'t> {
    let h = ((x.shape4()[2]-p.kernel[0]+(p.padh[0]+p.padh[1])) / p.stride[0]) + 1;
    let w = ((x.shape4()[3]-p.kernel[1]+(p.padw[0]+p.padw[1])) / p.stride[1]) + 1;
    let shape = [x.shape()[0], x.shape()[1], h, w].to_shape();

    x.extend(NodeBuilder {
        op: Box::new(MaxPool {
            kernel: p.kernel,
            stride: p.stride,
            padh: p.padh,
            padw: p.padw,
            indices: vec![0; shape.len()]
        }),
        deps: vec![x.index],
        shape: shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}