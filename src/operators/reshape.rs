
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct Reshape;

impl Operator for Reshape {
    fn forward(&mut self, _: &Node) -> Result<()> {
        // do nothing, the reshape op is skipped.
        Ok(())   
    }

    fn backward(&mut self, _: &Node) -> Result<()> {
        // do nothing, the reshape op is skipped.
        Ok(())
    }
}

impl Display for Reshape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Reshape Operator.")
    }
}

pub fn reshape<'t>(x: Var<'t>, shape: impl ToShape) -> Var<'t> {
    let mut shape = shape.to_shape();

    if x.is_batched {
        shape = shape.add_batch(1);
    }

    if x.shape().len() != shape.len() {
        panic!("New shape len does not match old shape len! New: {}, Old: {}", shape, x.shape())
    }

    x.extend(NodeBuilder {
        op: Box::new(Reshape),
        deps: vec![x.index],
        shape: shape,
        skip: true,
        init: None,
        is_batched: x.is_batched,
    })
}