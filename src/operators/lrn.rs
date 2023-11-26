
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct LRN {
    n_size: usize,
    alpha: f32,
    beta: f32,
    inter: bool,
}

impl Operator for LRN {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x, _) = node.x(1);

        bmls::lrn(
            &x.read(), &mut y.write(), x.shape4(),
            self.n_size, self.alpha, self.beta, 0.00000001, self.inter
        )?;

        Ok(())
    }

    fn backward(&mut self, _: &Node) -> Result<()> {
        Ok(())
    }
}

impl Display for LRN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LRN Operator. NSize: {}, Alpha: {}, Beta: {}, Inter: {}"
            ,self.n_size, self.alpha, self.beta, self.inter)
    }
}

// if mode is true, the mode is inter-channel, else, intra-channel.
pub fn lrn<'t>(x: Var<'t>, n: usize, alpha: f32, beta: f32, mode: bool) -> Var<'t> {
    x.extend(NodeBuilder {
        op: Box::new(LRN {
            n_size: n,
            alpha, beta,
            inter: mode,
        }),
        deps: vec![x.index],
        shape: x.shape,
        skip: false,
        init: None,
        is_batched: x.is_batched,
    })
}