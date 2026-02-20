use crate::domain::TrainingStepResult;
use crate::domain::{ModelState, Prediction, error::NNError};
use crate::port::classifier::{
    DigitPredictor, DigitTrainer, ModelStateExporter, ModelStateImporter,
};

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

pub struct NdArrayEngine {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    lr: f32,
}

impl NdArrayEngine {
    pub fn new() -> Self {
        let he1 = (2.0f32 / 784.0).sqrt();
        let he2 = (2.0f32 / 128.0).sqrt();

        Self {
            w1: Array2::random((128, 784), Normal::new(0.0, he1).unwrap()),
            b1: Array1::zeros(128),
            w2: Array2::random((10, 128), Normal::new(0.0, he2).unwrap()),
            b2: Array1::zeros(10),
            lr: 0.01,
        }
    }

    fn normalize(pixels: &[u8]) -> Result<Array1<f32>, NNError> {
        if pixels.len() != 784 {
            return Err(NNError::InvalidInput);
        }

        Ok(Array1::from(
            pixels.iter().map(|p| *p as f32 / 255.0).collect::<Vec<_>>(),
        ))
    }

    fn relu(x: &Array1<f32>) -> Array1<f32> {
        x.map(|v| v.max(0.0))
    }

    fn relu_derivative(x: &Array1<f32>) -> Array1<f32> {
        x.map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
    }

    fn softmax(x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp = x.map(|v| (v - max).exp());
        let sum = exp.sum();
        exp / sum
    }

    fn cross_entropy(output: &Array1<f32>, label: u8) -> f32 {
        let p = output[label as usize].max(1e-7);
        -p.ln()
    }
}

impl DigitPredictor for NdArrayEngine {
    fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError> {
        let input = Self::normalize(pixels)?;

        let z1 = self.w1.dot(&input) + &self.b1;
        let a1 = Self::relu(&z1);

        let z2 = self.w2.dot(&a1) + &self.b2;
        let output = Self::softmax(&z2);

        let (digit, confidence) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        Ok(Prediction {
            digit: digit as u8,
            confidence: *confidence,
        })
    }
}

impl DigitTrainer for NdArrayEngine {
    fn train(&mut self, label: u8, pixels: &[u8]) -> Result<TrainingStepResult, NNError> {
        let input = Self::normalize(pixels)?;

        // -------- Forward pass --------
        let z1 = self.w1.dot(&input) + &self.b1;
        let a1 = Self::relu(&z1);

        let z2 = self.w2.dot(&a1) + &self.b2;
        let output = Self::softmax(&z2);

        // -------- Target vector --------
        let mut target = Array1::<f32>::zeros(10);
        target[label as usize] = 1.0;

        // -------- Backpropagation --------

        // Output error (softmax + cross entropy derivative)
        let dz2 = &output - &target;

        let dw2 = dz2
            .view()
            .insert_axis(Axis(1))
            .dot(&a1.view().insert_axis(Axis(0)));

        let db2 = dz2.clone();

        // Hidden layer error
        let da1 = self.w2.t().dot(&dz2);
        let dz1 = da1 * Self::relu_derivative(&z1);

        let dw1 = dz1
            .view()
            .insert_axis(Axis(1))
            .dot(&input.view().insert_axis(Axis(0)));

        let db1 = dz1.clone();

        // -------- Gradient descent --------

        self.w2 -= &(dw2 * self.lr);
        self.b2 -= &(db2 * self.lr);

        self.w1 -= &(dw1 * self.lr);
        self.b1 -= &(db1 * self.lr);

        let loss = Self::cross_entropy(&output, label);

        let predicted = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u8;

        let train_metrics = TrainingStepResult {
            loss,
            correct: predicted == label,
        };

        Ok(train_metrics)
    }
}

impl ModelStateExporter for NdArrayEngine {
    fn export_state(&self) -> Result<ModelState, NNError> {
        Ok(ModelState {
            w1: self.w1.iter().cloned().collect(),
            b1: self.b1.iter().cloned().collect(),
            w2: self.w2.iter().cloned().collect(),
            b2: self.b2.iter().cloned().collect(),
        })
    }
}

impl ModelStateImporter for NdArrayEngine {
    fn import_state(&mut self, state: ModelState) -> Result<(), NNError> {
        self.w1 = Array2::from_shape_vec((128, 784), state.w1)
            .map_err(|_| NNError::SerializationError)?;

        self.b1 = Array1::from_vec(state.b1);

        self.w2 =
            Array2::from_shape_vec((10, 128), state.w2).map_err(|_| NNError::SerializationError)?;

        self.b2 = Array1::from_vec(state.b2);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_train() {
        let mut engine = NdArrayEngine::new();
        let pixels = vec![255u8; 784];
        let _ = engine.train(9, &pixels).unwrap();
    }

    fn sample_pixels() -> Vec<u8> {
        vec![255u8; 784]
    }

    #[tokio::test]
    async fn test_predict_runs() {
        let engine = NdArrayEngine::new();
        let pixels = vec![0u8; 784];

        let result = engine.predict(&pixels).unwrap();

        assert!(result.digit <= 9);
        assert!(result.confidence >= 0.0);
    }

    #[tokio::test]
    async fn test_weights_change_after_training() {
        let mut engine = NdArrayEngine::new();
        let pixels = sample_pixels();

        let before = engine.export_state().unwrap();

        let _ = engine.train(3, &pixels);

        let after = engine.export_state().unwrap();

        assert_ne!(before.w1, after.w1);
        assert_ne!(before.w2, after.w2);
    }

    #[tokio::test]
    async fn test_loss_decreases_on_same_sample() {
        let mut engine = NdArrayEngine::new();
        let pixels = sample_pixels();

        let loss1 = engine.train(5, &pixels).unwrap().loss;
        let loss2 = engine.train(5, &pixels).unwrap().loss;
        let loss3 = engine.train(5, &pixels).unwrap().loss;

        assert!(loss3 <= loss1 || loss3 <= loss2);
    }

    #[tokio::test]
    async fn test_model_can_overfit_single_sample() {
        let mut engine = NdArrayEngine::new();
        let pixels = sample_pixels();

        for _ in 0..200 {
            engine.train(7, &pixels).unwrap();
        }

        let prediction = engine.predict(&pixels).unwrap();

        assert_eq!(prediction.digit, 7);
        assert!(prediction.confidence > 0.8);
    }

    #[tokio::test]
    async fn test_export_import_consistency() {
        let mut engine = NdArrayEngine::new();
        let pixels = sample_pixels();

        let _ = engine.train(2, &pixels);

        let state = engine.export_state().unwrap();

        let mut new_engine = NdArrayEngine::new();
        new_engine.import_state(state).unwrap();

        let p1 = engine.predict(&pixels).unwrap();
        let p2 = new_engine.predict(&pixels).unwrap();

        assert_eq!(p1.digit, p2.digit);
    }
}
