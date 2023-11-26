
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Tanh;

impl Operator for Tanh {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::tanh(
            &x.read(),
            &mut y.write(),
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (y, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::tanh_wrt_x(
            &y.read(),
            &gy.read(),
            &mut gx.write()
        )?;

        Ok(())
    }
}

impl Display for Tanh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tanh")
    }
}

pub fn tanh<'t>(x: Var<'t>) -> Var<'t> {
    x.extend(NodeBuilder {
        op: Box::new(Tanh),
        deps: vec![x.index],
        shape: x.shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}