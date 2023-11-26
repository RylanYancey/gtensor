
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Matmul;

impl Operator for Matmul {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x1, _) = node.x(1);
        let (x2, _) = node.x(2);

        bmls::matmul(
            &x1.read(), &x2.read(), &mut y.write(),
            x1.shape2(), x2.shape2()
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (x1, g1) = node.x(1);
        let (x2, g2) = node.x(2);

        let gy = gy.read();

        bmls::matmul_wrt_a(
            &gy, &x2.read(), &mut g1.write(),
            x1.shape2(), x2.shape2(),
        )?;

        bmls::matmul_wrt_b(
            &x1.read(), &gy, &mut g2.write(),
            x1.shape2(), x2.shape2(),
        )?;

        Ok(())
    }
}

impl Display for Matmul {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matmul")
    }
}

pub fn matmul<'t>(x1: Var<'t>, x2: Var<'t>) -> Var<'t> {
    if x1.shape()[1] != x2.shape()[0] {
        panic!("X1 cols must be equal to X2 rows! X1: {}, X2: {}", x1.shape(), x2.shape())
    }

    let shape = [x1.shape()[0], x2.shape()[1], 1, 1].to_shape();

    x1.extend(
        NodeBuilder {
            op: Box::new(Matmul),
            deps: vec![x1.index, x2.index],
            shape: shape,
            skip: false,
            init: None,
            is_batched: x1.is_batched || x2.is_batched,
        }
    )
}