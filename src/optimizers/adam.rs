
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Adam {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    v: Vec<f32>,
    s: Vec<f32>,
}

impl Operator for Adam {
    fn forward(&mut self, _: &Node) -> Result<()> {
        // parameters have no forward operation.
        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (y, gy) = node.y();

        bmls::adam(
            &gy.read(), &mut self.v, &mut self.s, &mut y.write(),
            self.alpha, self.beta1, self.beta2,
        )?;

        Ok(())
    }
}

impl Optimizer for Adam {
    fn to_operator(&self, shape: Shape) -> Box<dyn Operator> {
        let v = vec![0.0; shape.len()];
        let s = vec![0.0; shape.len()];

        Box::new(
            Adam {
                alpha: self.alpha,
                beta1: self.beta1,
                beta2: self.beta2,
                v, s,
            }
        )
    }
}

impl Display for Adam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Adam Optimizer. Alpha: {}, Beta1: {}, Beta2: {}, VLen: {}, SLen: {}"
            ,self.alpha, self.beta1, self.beta2, self.v.len(), self.s.len())
    }
}

pub fn adam(alpha: f32, beta1: f32, beta2: f32) -> Box<dyn Optimizer> {
    Box::new(Adam { alpha, beta1, beta2, v: vec![0.0], s: vec![0.0] })
}