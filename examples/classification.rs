
use gtensor as gt;

use rand::Rng;

// batch size
const N: usize = 10;

fn main() {
    // load a train and test dataset
    let train = load_dataset();
    let test = load_dataset();

    // draw the datasets for demonstration purposes
    draw_dataset(&train, "examples/class-train-dataset.bmp");
    draw_dataset(&test, "examples/class-test-dataset.bmp");

    // construct the tape and configure the batch size to N. 
    let mut tape = build_tape();
    tape.set_batch_size(N);

    // declare tensors to store error and gradient data.
    let mut error = gt::Tensor::from_fill([N], 0.0);
    let mut grad = gt::Tensor::from_fill([N,1], 0.0);

    // Use the training dataset to train the network (a.k.a tape).
    // each epoch is a full iteration of the dataset.
    for epoch in 0..50 {
        let mut loss = 0.0;
        for (feature, label) in train.iter_batched(N) {
            // 'tape.forward' returns an RwLockReadGuard, so
            // we add a block here so the guard goes out of scope
            // before we call 'tape.backward'. If you remove the block,
            // the compiler will error on 'tape.backward'. 
            {
                // get the prediction from the network.
                let prediction = tape.forward(feature);

                // use BMLS to calculate the loss.
                // currently, GTensor does not
                // provide loss functions directly.
                // im still figuring out how to do this elegently.
                bmls::mse(
                    &label.data, &prediction.data, error.slice_inner_mut(), grad.slice_inner_mut(), [N,1]
                ).unwrap();
            }

            // add the loss for each batch to the total loss
            error.iter().for_each(|x| loss += *x);

            // Execute the backward pass
            tape.backward(grad.slice())
        }

        // Print the epoch number and the average loss, 
        // dividing by the number of features in the dataset.
        println!("Epoch: {epoch}, loss: {}", loss / 200.)
    }

    // Use the testing dataset to test the tape. 
    let mut loss = 0.0;
    for (feature, label) in test.iter_batched(N) {
        let prediction = tape.forward(feature);

        bmls::mse(
            &label.data, &prediction.data, error.slice_inner_mut(), grad.slice_inner_mut(), [N,1]
        ).unwrap();

        // add the loss for each batch to the total loss
        error.iter().for_each(|x| loss += *x);
    }

    println!("Test Loss: {}", loss / 200.);

    draw_prediction(&mut tape, &train, "examples/class-prediction.bmp");

    tape.save("examples/class").unwrap();
}

/// Create a dataset
fn load_dataset() -> gt::Dataset {
    // 2 inputs/features, 1 truth label
    let mut data = gt::Dataset::new([2], [1]);

    let mut rng = rand::thread_rng();

    for _ in 0..200 {
        // generate random points in the range (-1,1)
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);

        // distance from (x,y) to the origin at (0,0).
        let d = f32::sqrt((x*x)+(y*y));
        
        // discretive the labels for classification.
        // if the distance is greater than 0.6, the
        // label is orange (1.0) and blue (0.0) otherwise.
        let d = if d > 0.6 { 1.0 } else { 0.0 };

        // load the feature/label into the dataset
        data.load_feature(&[x,y], &[d]);
    }

    data
}

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

/// Create a graphic of the dataset (what the neural net trains on).
pub fn draw_dataset(dataset: &gt::Dataset, name: &str) {
    let mut img = bmp::Image::new(200,200);

    for x in 0..200 {
        for y in 0..200 {
            img.set_pixel(x, y, bmp::Pixel::new(255,255,255))
        }
    }

    for (feature, label) in dataset.iter_batched(1) {
        let color = 
        if label[0] < 0.5 {
            bmp::Pixel::new(60, 165, 255)
        } else {
            bmp::Pixel::new(255, 165, 0)
        };

        let x = ((feature[0] + 1.0) * 100.) as u32;
        let y = ((feature[1] + 1.0) * 100.) as u32;

        img.set_pixel(x, y, color)
    }

    img.save(name).unwrap()
}

/// Draw the neural networks' predictions.
pub fn draw_prediction(tape: &mut gt::Tape, dataset: &gt::Dataset, name: &str) {
    let mut img = bmp::Image::new(200,200);

    // change the batch size to 1 for easier iteration. 
    tape.set_batch_size(1);

    // run the neural network for every pixel
    for x in 0..200 {
        for y in 0..200 {
            // convert x and y to the range (-1,1)
            let xf = (x as f32 / 100.) - 1.;
            let yf = (y as f32 / 100.) - 1.;

            // convert inputs to tensor (for tape)
            let input = gt::Tensor::from_slice([1,2], &[xf,yf]);

            // get the prediction
            let prediction = tape.forward(input.slice());   

            // the color indicated by the prediction.
            let color = 
            if prediction.data[0] < 0.5 {
                bmp::Pixel::new(60, 165, 255)
            } else {
                bmp::Pixel::new(255, 165, 0)
            };
            
            img.set_pixel(x, y, color)
        }
    }

    // draw the training dataset slightly darker
    for (feature, label) in dataset.iter_batched(1) {
        let color = 
        if label[0] < 0.5 {
            bmp::Pixel::new(30, 135, 225)
        } else {
            bmp::Pixel::new(225, 135, 0)
        };

        let x = ((feature[0] + 1.0) * 100.) as u32;
        let y = ((feature[1] + 1.0) * 100.) as u32;

        img.set_pixel(x, y, color)
    }

    img.save(name).unwrap();
}