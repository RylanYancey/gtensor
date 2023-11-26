
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Dropout {
    rate: f32,
    rand: Vec<f32>,
}

impl Operator for Dropout {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::dropout(
            &x.read(),
            &mut self.rand,
            &mut y.write(),
            self.rate
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (_, gx) = node.x(1);

        bmls::dropout_wrt_x(
            &self.rand,
            &gy.read(),
            &mut gx.write(),
            self.rate
        )?;

        Ok(())
    }

    fn reshape(&mut self, new: Shape) {
        self.rand.resize(new.len(), 0.0)
    }
}

impl Display for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dropout Operator")
    }
}

pub fn dropout<'t>(x: Var<'t>, rate: f32) -> Var<'t> {
    x.extend(NodeBuilder {
        op: Box::new(Dropout {
            rate,
            rand: vec![0.0; x.shape().len()]
        }),
        deps: vec![x.index],
        shape: x.shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}