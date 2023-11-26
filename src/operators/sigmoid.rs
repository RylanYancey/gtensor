
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Sigmoid;

impl Operator for Sigmoid {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::sigmoid(
            &x.read(),
            &mut y.write(),
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (y, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::sigmoid_wrt_x(
            &y.read(),
            &gy.read(),
            &mut gx.write()
        )?;

        Ok(())
    }
}

impl Display for Sigmoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sigmoid")
    }
}

pub fn sigmoid<'t>(x: Var<'t>) -> Var<'t> {
    x.extend(NodeBuilder {
        op: Box::new(Sigmoid),
        deps: vec![x.index],
        shape: x.shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}