
use std::cell::RefCell;

use super::node::NodeBuilder;
use super::var::Var;
use super::node::Node;
use super::tape::Tape;
use crate::tensor::shape::ToShape;
use crate::operators::input::Input;
use crate::optimizers::Optimizer;
use crate::initializers::Initializer;

/// Used to record operators to a Computational Graph.
/// Use input, parameter, and provided 'op's, which internallly
/// use the 'extend' method. 
/// When you are done recording, call 'tape.finish()'
/// to produce the finished tape.
pub struct TapeBuilder {
    /// A RefCell is used because the references
    /// stored in Var<'t> must be immutable to comply
    /// to type safety rules. RefCell checks these
    /// rules at runtime to ensure there are no
    /// conflicts and allows pseudo-mutable reference to
    /// exist in multiple places at once.
    pub tape: RefCell<Tape>,
    pub opt: Box<dyn Optimizer>,
    pub init: Box<dyn Initializer>,
}

impl TapeBuilder {
    /// Create an input node for the tape.
    /// The Input node is always considered to be the first node
    /// in the tape. A tape can only have one input, and it must
    /// be the first node.
    pub fn input<'t>(&'t self, shape: impl ToShape) -> Var<'t> {
        let shape = shape.to_shape().add_batch(1);

        // make sure this is the first node.
        if self.tape.borrow().nodes.len() != 0 {
            panic!("Tapes cannot have multiple inputs! (Input Node Index was non-Zero)")
        }

        self.extend(NodeBuilder {
            op: Box::new(Input),
            deps: Vec::new(),
            shape, 
            skip: false,
            init: None,
            is_batched: true,
        })
    }

    /// Create a new trainable parameter on the tape.
    pub fn parameter<'t>(&'t self, shape: impl ToShape) -> Var<'t> {
        let shape = shape.to_shape();

        self.extend(NodeBuilder {
            op: self.opt.to_operator(shape),
            deps: vec![],
            shape,
            skip: false,
            init: Some(self.init._clone()),
            is_batched: false,
        })
    }

    pub fn extend<'t>(&'t self, builder: NodeBuilder) -> Var<'t> {
        let mut tape = self.tape.borrow_mut();

        let (output, gradient) = 
        if let Some(init) = builder.init {
            // if a weight initializer exists, this node is a parameter.
            // Use the initializer to build the parameter.
            tape.arena.alloc_parameter(builder.shape, init, builder.is_batched)
        } else {
            // If this Node is skipped, meaning it does not perform any operations,
            // just assign the tensorptrs directly to its dependency.
            if builder.skip {
                if builder.deps.len() != 1 {
                    panic!("A Skip node must have exactly 1 dependency!")
                }
                // the node is transparent, meaning reads or writes to its tensors will
                // read and write its dependency. 
                (
                    tape.nodes[builder.deps[0]].y.clone_reshape(builder.shape, builder.is_batched),
                    tape.nodes[builder.deps[0]].gy.clone_reshape(builder.shape, builder.is_batched),
                )
            } else {
                // if the node is not skip or a parameter, allocate the gradient and output tensors
                // normally, taking the reuse into account (handled by the arena).
                tape.arena.alloc(builder.shape, builder.is_batched)
            }
        };

        let node = Node {
            y: output,
            gy: gradient,
            x: builder.deps.iter().map(|i| tape.nodes[*i].y.clone()).collect(),
            gx: builder.deps.iter().map(|i| tape.nodes[*i].gy.clone()).collect(),
            is_batched: builder.is_batched,
        };

        tape.nodes.push(node);
        tape.ops.push(builder.op);

        Var {
            tape: self,
            shape: builder.shape,
            index: tape.nodes.len() - 1,
            is_batched: builder.is_batched,
        }
    }

    pub fn finish(self) -> Tape {
        self.tape.into_inner()
    }
}