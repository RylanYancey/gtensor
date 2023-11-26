
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct SGD {
    alpha: f32,
}

impl Operator for SGD {
    fn forward(&mut self, _: &Node) -> Result<()> {
        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (y, gy) = node.y();

        bmls::sgd(
            &gy.read(), &mut y.write(), self.alpha
        )?;

        Ok(())
    }
}

impl Optimizer for SGD {
    fn to_operator(&self, _: Shape) -> Box<dyn Operator> {
        Box::new(SGD { alpha: self.alpha })
    }
}

impl Display for SGD {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SGD Operator. \n- alpha: {}", self.alpha)
    }
}

pub fn sgd(alpha: f32) -> Box<dyn Optimizer> {
    Box::new(SGD { alpha })
}