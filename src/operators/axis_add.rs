
use super::*;

#[derive(Clone, Serialize, Deserialize)]
struct AxisAdd {
    // the axis to add to
    axis: usize,
}

impl Operator for AxisAdd {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x1, _) = node.x(1);
        let (x2, _) = node.x(2);

        bmls::axis_add(
            &x1.read(), &x2.read(), &mut y.write(),
            x1.shape4(), self.axis,
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (_, g1) = node.x(1);
        let (_, g2) = node.x(2);

        bmls::axis_add_wrt_x1(
            &gy.read(), &mut g1.write(),
        )?;

        bmls::axis_add_wrt_x2(
            &gy.read(), &mut g2.write(),
            gy.shape4(), self.axis,
        )?;

        Ok(())
    }
}

impl Display for AxisAdd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Axis Add Operator. Axis: {}", self.axis)
    }
}

pub fn axis_add<'t>(x1: Var<'t>, x2: Var<'t>, axis: impl ToAxis) -> Var<'t> {
    let axis = axis.to_axis();

    if x1.shape()[axis] != x2.shape().len() {
        panic!("
            Length of Axis {} of X1 must be equal to the length of X2. X1: {}, X2: {}, Axis: {}
            ", axis, x1.shape(), x2.shape(), axis)
    }

    x1.extend(NodeBuilder {
        op: Box::new(AxisAdd { axis }),
        deps: vec![x1.index, x2.index],
        shape: x1.shape,
        skip: false,
        init: None,
        is_batched: x1.is_batched || x2.is_batched,
    })
}