
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Softmax;

impl Operator for Softmax {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::softmax(
            &x.read(),
            &mut y.write(),
            x.shape2(),
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (y, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::softmax_wrt_x(
            &y.read(),
            &gy.read(),
            &mut gx.write(),
            y.shape2(),
        )?;

        Ok(())
    }
}

impl Display for Softmax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Softmax")
    }
}

pub fn softmax<'t>(x: Var<'t>) -> Var<'t> {
    x.extend(NodeBuilder {
        op: Box::new(Softmax),
        deps: vec![x.index],
        shape: x.shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}