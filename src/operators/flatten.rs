
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Flatten;

impl Operator for Flatten {
    fn forward(&mut self, _: &Node) -> Result<()> {
        Ok(())
    }

    fn backward(&mut self, _: &Node) -> Result<()> {
        Ok(())
    }
}

impl Display for Flatten {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Flatten")
    }
}

pub fn flatten<'t>(x: Var<'t>) -> Var<'t> {
    let mut shape = x.shape();
    shape[1] = shape[1]*shape[2]*shape[3];

    if x.is_batched {
        shape = shape.add_batch(1);
    }

    if x.shape().len() != shape.len() {
        panic!("New shape len does not match old shape len! New: {}, Old: {}", shape, x.shape())
    }

    x.extend(NodeBuilder {
        op: Box::new(Flatten),
        deps: vec![x.index],
        shape: shape,
        skip: true,
        init: None,
        is_batched: x.is_batched,
    })
}