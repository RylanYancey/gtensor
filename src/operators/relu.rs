
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct ReLU;

impl Operator for ReLU {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::relu(
            &x.read(),
            &mut y.write(),
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (x, gx) = node.x(1);

        bmls::relu_wrt_x(
            &x.read(),
            &gy.read(),
            &mut gx.write()
        )?;

        Ok(())
    }
}

impl Display for ReLU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ReLU")
    }
}

pub fn relu<'t>(x: Var<'t>) -> Var<'t> {
    x.extend(NodeBuilder {
        op: Box::new(ReLU),
        deps: vec![x.index],
        shape: x.shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}