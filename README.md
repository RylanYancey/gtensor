
# Graph Tensor

A library for reverse-mode automatic differentiation of tensor operations on computational graphs for machine learning and more.

# Goals

The goal of gTensor is to create a general-purpose framework for machine learning with an emphasis on performance, flexibility, and documentation. We hope to span Deep, Convolutional, and Recurrent neural networks, unsupervised algorithms like KNN and clustering, and reinforcement algorithms like Deep Q-Learning. 

# Documenation

Extensive documentation is provided in the /docs/ folder. 

# Examples

Currently gT provides the `classification` example which shows how to load a dataset, build, train, and test a neural network, and save the network to disk. 

# Contribution

gTensor is in active, early development. Expect frequent, breaking changes. If you find gT is missing important features, feel free to create a pull request. 

# Constructing Computational Graphs

gT provides the `Tape` type, which can be used to record operators to the computational graph. The Example below constructs a computational graph with 2 hidden layers, each with 4 neurons, with the tanh activation function.

```
/// Record Operators to the Tape.
fn build_tape() -> gt::Tape {
    let mut tape = gt::Tape::builder();

    // set the optimizer and initializer for the weights.
    tape.opt = gt::opt::momentum(0.04, 0.9);
    tape.init = gt::init::normal(0.5, 1.0);

    // input
    let x = tape.input([2]);
    
    // first layer (2 inputs, 4 neurons)
    // 1. declare weight parameters (2x4)
    // 2. declare bias parameters (4)
    // 3. matmul x * w (Nx2 * 2x4 = Nx4)
    // 4. add bias to the channels
    // 5. activate with tanh
    let w = tape.parameter([2,4]);
    let b = tape.parameter([4]);
    let x = gt::op::matmul(x, w);
    let x = gt::op::axis_add(x, b, 'C');
    let x = gt::op::tanh(x);

    // second layer (4 inputs, 4 neurons)
    // 1. declare weight parameters (4x4)
    // 2. declare bias parameters (4)
    // 3. matmul x * w (Nx4 * 4x4 = Nx4)
    // 4. add bias to the channels
    // 5. activate with tanh
    let w = tape.parameter([4,4]);
    let b = tape.parameter([4]);
    let x = gt::op::matmul(x,w);
    let x = gt::op::axis_add(x, b, 'C');
    let x = gt::op::tanh(x);

    // output layer (4 inputs, 1 neuron)
    // 1. declare weight parameters (4x1)
    // 2. declare bias parameters (1)
    // 3. matmul x * w (Nx4 * 4x1 = Nx1)
    // 4. add bias to the channels
    // 5. activate with tanh
    let w = tape.parameter([4,1]);
    let b = tape.parameter([1]);
    let x = gt::op::matmul(x, w);
    let x = gt::op::axis_add(x, b, 'C');
    let _ = gt::op::tanh(x);

    tape.finish()
}
```