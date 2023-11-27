
# Implementing Custom Operators

GT provides the `Operator` trait in gt::operators. To implement a custom operator, create a new struct and derive Clone, Serialize, and Deserialize. We will be learning to implement the `Matmul` operator.

```rs
use gtensor as gt;

use std::fmt::Display;
use gt::operators::Operator;
use serde::{Serialize, Deserialize};
use anyhow::Result;

#[derive(Clone, Serialize, Deserialize)]
pub struct Matmul;
```

`Clone`, `Serialize`, and `Deserialize` allow us to seamlessly save and load the tape without needing to implement lots of custom behaviour. Side note, `Operator`s do not actually implement `Clone`, they implement `DynClone` from the `dyn-clone` crate. Next, our custom operator needs an implementation of `Operator` and `Display`.

```rs
use gt::operator::Operator;

impl Operator for Matmul {
    fn forward(&mut self, node: &Node) -> Result<()> {
        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        Ok(())
    }

    // Optional!
    fn reshape(&mut self, new: Shape) { }
}

impl Display for Matmul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matmul")
    }
}
```

The `forward` function provided by the `Operator` trait is executed during the forward pass of the neural network. The next step is to collect input and output tensors from the `Node`. We will also remove the `reshape` function since it is not needed for the Matmul op. We can collect node variables using the functions `node.x` and `node.y`.

```rs
impl Operator for Matmul {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y,  _) = node.y();
        let (x1, _) = node.x(1);
        let (x2, _) = node.x(2);

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (x1, g1) = node.x(1);
        let (x2, g2) = node.x(2);

        Ok(())
    }
}
```

The `x` and `y` functions return tuples with the type `(GTensor, GTensor)`, where the first `GTensor` is the output (in the case of `y`) or an input (in the case of `x`). The second `GTensor` is the respective gradient. When calling `x`, the index you provide should start at `1`. GTensors are internally a `Vec<f32>` and a `Shape`. To read or write the internal `Vec`, `GTensor`s provide the `read()` and `write()` functions.  The internal `Shape` is not read/write protected, and can be acquired by calling the GTensors `shape()` function.

Next, we will use BMLS to perform the matrix multiplication. BMLS will take care of the parallelization, error handling, and operation correctness for us, making this rather complex op trivial to implement. 

```rs
impl Operator for Matmul {
    fn forward(&mut self, node: &Node) -> Result<()> {
        let (y, _) = node.y();
        let (x1, _) = node.x(1);
        let (x2, _) = node.x(2);

        bmls::matmul(
            &x1.read(), &x2.read(), &mut y.write(),
            x1.shape2(), x2.shape2()
        )?;

        Ok(())
    }

    fn backward(&mut self, node: &Node) -> Result<()> {
        let (_, gy) = node.y();
        let (x1, g1) = node.x(1);
        let (x2, g2) = node.x(2);

        let gy = gy.read();

        bmls::matmul_wrt_a(
            &gy, &x2.read(), &mut g1.write(),
            x1.shape2(), x2.shape2(),
        )?;

        bmls::matmul_wrt_b(
            &x1.read(), &gy, &mut g2.write(),
            x1.shape2(), x2.shape2(),
        )?;

        Ok(())
    }
}
```

Finally, our custom operator needs a way to be constructed. Create a function to handle the creation and perform error checking. We will make use of `Var`s, which internally contain references to the `TapeBuilder`, an index corresponding an operator in the `Tape`, and a `Shape`. We will make sure the shape of the input `Var`s is valid for matrix multiplication by ensuring x1 cols == x2 rows. We will also compute the shape of the output `Var`. 

```rs
pub fn matmul<'t>(x1: Var<'t>, x2: Var<'t>) -> Var<'t> {
    if x1.shape()[1] != x2.shape()[0] {
        panic!("X1 cols must be equal to X2 rows! X1: {}, X2: {}", x1.shape(), x2.shape())
    }

    // make use of the ToShape trait to convert this to a shape.
    let shape = [x1.shape()[0], x2.shape()[1]].to_shape();

// ...continued
```

Now an end user will not have to guess if their matrix multiplication is valid, we will be able to know for sure. Next, we will create a `NodeBuilder` and pass it to the `TapeBuilder`. The `NodeBuilder` needs a `Box<dyn Operator>`, a vector of the indices of dependency variables (order matters!), the shape of the output of the `Node`s' forward operation, whether or not the node is skipped, an initializer (only used for parameters), and whether or not the `Node` is batched. 

```rs
    x1.extend(
        NodeBuilder {
            op: Box::new(Matmul),
            deps: vec![x1.index, x2.index],
            shape: shape,
            skip: false,
            init: None,
            is_batched: x1.is_batched || x2.is_batched,
        }
    )
}
```

`skip` specifies if the Node performs any operations on the data or not. For example, the `Reshape` operator is skipped, but the `Shape` is a new shape, which allows to skip cloning the tensor. `skip` nodes do not have `GTensor`s of their own, they have pointers to the `y` in the previous layer. 

`init` is an `Option<Box<dyn Initializer>>` and is only provided when the op is a parameter such as `SGD`, `Adam`, and `Momentum`. We won't be using it for this op. 

`is_batched` specifies whether or not the output shape of this `Node` is dependent on the batch size. When an `input` is created with `TapeBuilder.input()`, the resulting `Var`s' is_batched equals false, but if `TapeBuilder.parameter()` is used, the resulting `Var`s' is_batched equals true. We do this because we might want to change the batch size dynamically at runtime, but the shape of parameters is unchanging. The rule to follow is that if a `Var` has a dependency for which `is_batched == true`, then the new `Var` should have the same.

And... Thats it! You now have everything you need to create custom operators. 
