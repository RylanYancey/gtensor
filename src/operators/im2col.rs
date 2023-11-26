
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Im2Col {
    kernel: [usize; 4],
    stride: [usize; 2],
    padh: [usize; 2],
    padw: [usize; 2],
}

impl Operator for Im2Col {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::im2col(
            &x.read(), &mut y.write(), x.shape4(),
            self.kernel, self.stride, self.padh, self.padw
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::im2col_wrt_x(
            &gy.read(), &mut gx.write(), gx.shape4(),
            self.kernel, self.stride, self.padh, self.padw,
        )?;

        Ok(())
    }
}

impl Display for Im2Col {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Im2Col Operator")
    }
}

pub fn im2col<'t>(x: Var<'t>, p: ConvParams) -> Var<'t> {
    let h = p.kernel[1]*p.kernel[2]*p.kernel[3];
    let w = (((x.shape()[2]-p.kernel[2]+(p.padh[0]+p.padh[1]))/p.stride[0]) + 1) * 
            (((x.shape()[3]-p.kernel[3]+(p.padw[0]+p.padw[1]))/p.stride[1]) + 1) * 
            x.shape()[0];
    let shape = [h,w,1,1].to_shape();

    x.extend(NodeBuilder {
        op: Box::new(Im2Col {
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