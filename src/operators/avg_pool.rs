
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct AvgPool {
    kernel: [usize; 2],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
}

impl Operator for AvgPool {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::avg_pool(
            &x.read(), &mut y.write(), x.shape4(), 
            self.stride, self.kernel, self.padh, self.padw
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::avg_pool_wrt_x(
            &gy.read(), &mut gx.write(), gx.shape4(),
            self.stride, self.kernel, self.padh, self.padw,
        )?;

        Ok(())
    }
}

impl Display for AvgPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Avg Pool Operator. Kernel: {:?}, Stride: {:?}, padh: {:?}, padw: {:?}", 
            self.kernel, self.stride, self.padh, self.padw)
    }
}

pub fn avg_pool<'t>(x: Var<'t>, p: PoolParams) -> Var<'t> {
    let h = ((x.shape4()[2]-p.kernel[0]+(p.padh[0]+p.padh[1])) / p.stride[0]) + 1;
    let w = ((x.shape4()[3]-p.kernel[1]+(p.padw[0]+p.padw[1])) / p.stride[1]) + 1;
    let shape = [x.shape()[0], x.shape()[1], h, w].to_shape();

    x.extend(NodeBuilder {
        op: Box::new(AvgPool {
            kernel: p.kernel,
            stride: p.stride,
            padh: p.padh,
            padw: p.padw,
        }),
        deps: vec![x.index],
        shape: shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}

