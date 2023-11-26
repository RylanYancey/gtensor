
use gtensor as gt;
use csv;

fn main() {
    println!("Warning: This example takes a while!");

    if let Err(_) = std::fs::metadata("examples/data/mnist_train.csv") {
        panic!("Please provide the MNist Digits dataset in the examples/data/ 
        folder as 'mnist_train.csv' and 'mnist_test.csv'. (file is too large to keep in repo)")
    }

    // We will use the Training Dataset as the reference dataset. 
    let train = load_data("examples/data/mnist_train.csv", 60000);

    // We will use the Test Set to test the accuracy of KNN for this dataset.
    // Limiting it to 250 since the actual dataset is 10000 long and it would
    // take too long to do all of them.
    let test  = load_data("examples/data/mnist_test.csv", 250);

    // Distance Metric for KNN. 
    let metric = gt::knn::metric::Euclidian;

    let mut num_correct = 0;

    for (feature, label) in test.iter_batched(10) {
        let prediction = gt::knn::knn(&train, &feature, 10, &metric);
        
        for (p, l) in prediction.iter().zip(label.iter()) {
            if p == l {
                num_correct += 1;
                println!("Label: {}, Predicted: {} ✅", l, p)
            } else {
                println!("Label: {}, Predicted: {} ❌", l, p)
            }
        }
    }

    println!("KNN Mnist Example correctly predicted {num_correct} labels out of {}", test.len());
}

fn load_data(path: &str, limit: usize) -> gt::Dataset {
    let mut dataset = gt::Dataset::new([1,28,28], [1]);

    let reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(path)
        .unwrap();

    let mut feature = vec![0.0;28*28];
    let mut label = 0.0;

    for (i, result) in reader.into_records().enumerate() {
        if i == limit {
            break;
        }

        for (i, e) in result.unwrap().iter().enumerate() {
            let v = e.parse::<f32>().unwrap();
        
            if i == 0 {
                label = v;
            } else {
                feature[i-1] = v;
            }
        }

        dataset.load_feature(&feature, &[label]);
    }

    dataset
}