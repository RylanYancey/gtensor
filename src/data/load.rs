
use super::dataset::Dataset;

impl Dataset {
    pub fn load_feature(&mut self, feature: &[f32], label: &[f32]) {
        if feature.len() != self.feature_shape.len() {
            panic!("new feature length does not match existing feature shape. New len: {}, feature shape: {}",
                feature.len(), self.feature_shape.len())
        }

        if label.len() != self.label_shape.len() {
            panic!("new label length does not match existing label shape. New len: {}, label shape: {}",
                label.len(), self.label_shape.len())
        }

        self.features.extend_from_slice(feature);
        self.labels.extend_from_slice(label);
    }
}