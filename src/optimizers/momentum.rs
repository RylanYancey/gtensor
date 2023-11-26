
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Momentum {
    alpha: f32,
    beta: f32,
    v: Vec<f32>,
}

impl Operator for Momentum {
    fn forward(&mut self, _: &Node) -> Result<()> {
        // nothing happens in the forward pass.
        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (y, gy) = node.y();

        bmls::momentum(
            &gy.read(), &mut self.v, &mut y.write(),
            self.alpha, self.beta,
        )?;

        Ok(())
    }
}

impl Optimizer for Momentum {
    fn to_operator(&self, shape: Shape) -> Box<dyn Operator> {
        Box::new(Momentum {
            alpha: self.alpha,
            beta: self.beta,
            v: vec![0.0; shape.len()]
        })
    }
}

impl Display for Momentum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Momentum Operator. Alpha: {}, Beta: {}, v_len: {}", 
            self.alpha, self.beta, self.v.len())
    }
}

pub fn momentum(alpha: f32, beta: f32) -> Box<dyn Optimizer> {
    Box::new(Momentum { alpha, beta, v: vec![0.0] })
}

