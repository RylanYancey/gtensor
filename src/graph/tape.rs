
use std::cell::RefCell;

use anyhow::Result;

use super::builder::TapeBuilder;
use super::arena::Arena;
use super::node::Node;
use super::serde::ArenaSerde;
use super::serde::NodeSerde;
use super::serde::OperatorSerde;
use crate::operators::Operator;
use crate::initializers;
use crate::optimizers;
use crate::tensor::slice::TensorSlice;
use super::gtensor::TensorGuard;

/// The Main Datatype of gTensor. 
/// The Tape is a Computation Graph on which
/// operations can be performed on Tensors.
/// Each 'Node' in this graph represents an operation
/// to be done, such as matrix multiplication, tanh, etc.
pub struct Tape {
    /// Arena Allocator for the Tape.
    /// See [Arena] for more info. 
    pub(crate) arena: Arena,
    /// Nodes store pointers to Arena data.
    /// Each node has a output, an output gradient,
    /// 0+ inputs, and a gradient for each.
    pub(crate) nodes: Vec<Node>,
    /// Operators have a forward and backward
    /// function, and perform operations
    /// on the data pointed to by the nodes.
    /// There is exactly one operator for every node.
    pub(crate) ops: Vec<Box<dyn Operator>>,
}

impl Tape {
    pub fn builder() -> TapeBuilder {
        TapeBuilder {
            tape: RefCell::new(Tape {
                arena: Arena::new(),
                nodes: Vec::new(),
                ops: Vec::new(),
            }),
            // default initializer / optimizer.
            opt: optimizers::opt::sgd(0.04),
            init: initializers::init::random(1.0, 0.0),
        }
    }

    pub fn forward(&mut self, input: TensorSlice) -> TensorGuard {
        let first = &self.nodes.first().unwrap().y;

        // ensure compatability between the input data and the input node.
        if first.shape() != input.shape {
            panic!(
                "Shape Mismatch when cloning input data into Tape! Input: {}, First Node Shape: {}"
                ,input.shape, first.shape)
        }

        // clone the input data into the tape.
        input.data.clone_into(&mut first.write());

        // execute the forward pass for each node.
        for i in 1..self.nodes.len() {
            self.ops[i].forward(&self.nodes[i])
                .expect(&format!(
                    "Error in the Forward Pass at node index ({}) with name ({}) and data ({})."
                    , i, self.ops[i], self.nodes[i]))
        }

        // return the output of the last node.
        self.nodes.last().unwrap().y.slice()
    }

    pub fn backward(&mut self, gradient: TensorSlice) {
        let last = &self.nodes.last().unwrap().gy;

        // ensure compatability between gradient data and last node.
        if last.shape != gradient.shape {
            panic!(
                "Shape mismatch when cloning gradient data into Tape! Input: {}, Node: {}",
                gradient.shape, last.shape,
            )
        }

        // clear the allocated gradient tensors.
        // we need to do this since all operators
        // add to the existing values rather than overwriting them.
        self.arena.clear_gradients();

        // copy the gradient data into last.
        gradient.data.clone_into(&mut last.write());

        // execute the backward pass for each node.
        for i in (1..self.nodes.len()).rev() {
            self.ops[i].backward(&self.nodes[i])
                .expect(&format!("Error in the Backward Pass at node index ({}) with name ({}) and data ({})"
                , i, self.ops[i], self.nodes[i]))
        }
    }

    /// Reshape Batched Nodes with a new batch size.
    /// We know if a node is batched because it has the 'is_batched'
    /// field. This field is set during the creation of the node.
    /// If a Node is a parameter, is_batched is false. if the node is
    /// an input, is_batched is true. If a node depends on a batched node,
    /// the is_batched is true. If all dependencies are not batched, is_batched is false.
    pub fn set_batch_size(&mut self, batch_size: usize) {
        for i in 0..self.nodes.len() {
            // check if the node is batched...
            if self.nodes[i].is_batched {
                // ...if it is, change the batch size on the node.
                self.nodes[i].set_batch_size(batch_size);

                // get the shape we will be resizing to and
                // set the batch size accordingly.
                let mut new_shape = self.nodes[i].gy.shape;
                new_shape[0] = batch_size;

                // ...and inform the operator that the size has changed.
                // (important for some operators, like Max Pool and Dropout)
                self.ops[i].reshape(new_shape);
            }
        }
    }
 
    /// Creates a directory called "name.tape". 
    pub fn save(&mut self, name: &str) -> Result<()> {
        // Try to create the directory. If it already exists,
        // it will fail, but we don't unwrap it. This makes
        // sure the file exists, but also does not overwrite.
        let _ = std::fs::create_dir(name.to_owned() + ".tape");

        let mut i = 0;
        // Read every item in the directory...
        for item in std::fs::read_dir(name.to_owned() + ".tape").unwrap() {
            // ...parse the version number from the name...
            let name = item.unwrap().file_name();
            let name = name.to_str().to_owned().unwrap().replace("v", "");
            let version = name.parse::<usize>()?;

            // ...if the version number is higher, set it.
            if version > i {
                i = version;
            }
        }
        // We want one higher than the highest
        // existing version number.
        i+=1;

        // Create a directory for the version.
        let path = format!("{name}.tape/v{i}");
        std::fs::create_dir(path.clone())?;

        // Convert the Tape to something serializable.
        let arena = serde_json::to_string(&ArenaSerde::from_arena(&self.arena))?;
        let nodes = serde_json::to_string(&self.nodes.iter().map(|node| 
            NodeSerde::from_node(node)).collect::<Vec<NodeSerde>>())?;
        let ops = serde_json::to_string(&self.ops.iter().map(|op| 
            OperatorSerde::from_op(op)).collect::<Vec<OperatorSerde>>())?;

        // write the data into their respective files.
        std::fs::write(format!("{}/data.json", path.clone()), arena)?;
        std::fs::write(format!("{}/nodes.json", path.clone()), nodes)?;
        std::fs::write(format!("{}/ops.json", path.clone()), ops)?;

        Ok(())
    }

    /// Loads a tape from 'name.tape'.
    /// - name: The name of the tape to load.
    /// - version: The version of the tape to load.
    /// 
    /// If the version is greater than 0, it opens the version.
    /// If the version is -1, it finds and opens the latest version.
    /// 
    /// The loaded tape will have a batch size of 1.
    pub fn load(name: &str, mut version: i32) -> Result<Tape> {
        // if the version is -1, we want to find the highest version number....
        if version == -1 {
            let mut i = 0;
            // ...iterate the entries in the tape directory...
            for item in std::fs::read_dir(name.to_owned() + ".tape").unwrap() {
                // ...parse the version from the folder name...
                let name = item.unwrap().file_name();
                let name = name.to_str().to_owned().unwrap().replace("v", "");
                let version = name.parse::<usize>()?;
    
                // ...check if the version is the highest version number...
                if version > i {
                    i = version;
                }
            }

            // ...return the highest version found.
            version = i as i32;
        }

        // Read the data into a string...
        let arena = std::fs::read_to_string(format!("{name}.tape/v{version}/data.json"))?;
        let nodes = std::fs::read_to_string(format!("{name}.tape/v{version}/nodes.json"))?;
        let ops = std::fs::read_to_string(format!("{name}.tape/v{version}/ops.json"))?;

        // ...Parse the data from yaml format into the types...
        let arena = serde_json::from_str::<ArenaSerde>(&arena)?;
        let nodes = serde_json::from_str::<Vec<NodeSerde>>(&nodes)?;
        let mut ops = serde_json::from_str::<Vec<OperatorSerde>>(&ops)?;

        let arena = arena.to_arena();

        let mut operators = Vec::new();
        while let Some(op) = ops.pop() {
            operators.push(op.to_op())
        }

        let mut tape = Tape {
            nodes: nodes.iter().map(|node| node.to_node(&arena)).collect(),
            arena,
            ops: operators
        };

        // make sure the batch size is 1 to avoid confusion.
        tape.set_batch_size(1);

        // return the tape.
        Ok(tape)
    }
}