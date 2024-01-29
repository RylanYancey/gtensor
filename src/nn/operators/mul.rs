
use super::*;

struct Mul;

impl<T: Float> Operator<Cpu<T>> for Mul {
    fn forward(&mut self, node: &Node<Cpu<T>>) -> Result<()> {
        let y = node.y().as_slice_mut();
        let x1 = node.x1().as_slice();
        let x2 = node.x2().as_slice();

        for (y, x1, x2) in multizip((y, x1, x2)) {
            *y = *x1 * *x2
        }

        Ok(())
    }

    fn reshape(&mut self, node: &Node<Cpu<T>>) -> Result<()> {
        let x1 = node.x1().shape();
        let x2 = node.x2().shape();

        if x1 != x2 {
            return Err(anyhow!("X1 and X2 shape must match!"))
        }

        node.reshape(x1.clone());

        Ok(())
    }

    fn wrt_x1(&self, node: &Node<Cpu<T>>) -> Result<()> {
        let gy = node.gy().as_slice();
        let x2 = node.x2().as_slice();
        let g1 = node.g1().as_slice_mut();

        for (gy, x2, g1) in multizip((gy, x2, g1)) {
            *g1 = *gy * *x2
        }

        Ok(())
    }

    fn wrt_x2(&self, node: &Node<Cpu<T>>) -> Result<()> {
        let gy = node.gy().as_slice();
        let x1 = node.x1().as_slice();
        let g2 = node.g2().as_slice_mut();

        for (gy, x1, g2) in multizip((gy, x1, g2)) {
            *g2 = *gy * *x1
        }

        Ok(())
    }
}

impl<T: Float> Operator<Gpu<T>> for Mul {
    fn forward(&mut self, node: &Node<Gpu<T>>) -> Result<()> {
        let y = node.y();
        let x1 = node.x1();
        let x2 = node.x2();

        let len = y.len() as u32;

        node.kernel(0).launch(
            [len / 1024], 
            [1024], 
            node.stream(),
            (
                y.as_ptr(),
                x1.as_ptr(),
                x2.as_ptr(),
            )
        )?;

        Ok(())
    }

    fn reshape(&mut self, node: &Node<Gpu<T>>) -> Result<()> {
        Ok(())
    }

    fn wrt_x1(&self, node: &Node<Gpu<T>>) -> Result<()> {
        Ok(())
    }

    fn wrt_x2(&self, node: &Node<Gpu<T>>) -> Result<()> {
        Ok(())
    }
}

pub fn mul<'s, S>(x1: Var<'s, S>, x2: Var<'s, S>) -> Var<'s, S> 
where
    S: StorageInfo,
{
    let node: NodeBuilder<S> = Node::build();

    if S::TYPE == "gpu" {
        
    }

    todo!()
}
