use crate::domain::{prediction::Prediction, error::NNError};
use crate::port::classifier::DigitClassifier;
use crate::domain::model_state::ModelState;

use async_trait::async_trait;
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct NdArrayEngine {
    w1: Arc<RwLock<Array2<f32>>>,
    b1: Arc<RwLock<Array1<f32>>>,
    w2: Arc<RwLock<Array2<f32>>>,
    b2: Arc<RwLock<Array1<f32>>>,
    lr: f32,
}

impl NdArrayEngine {
    pub fn new() -> Self {
        Self {
            w1: Arc::new(RwLock::new(Array2::random((128, 784), Uniform::new(-0.5, 0.5)))),
            b1: Arc::new(RwLock::new(Array1::zeros(128))),
            w2: Arc::new(RwLock::new(Array2::random((10, 128), Uniform::new(-0.5, 0.5)))),
            b2: Arc::new(RwLock::new(Array1::zeros(10))),
            lr: 0.01,
        }
    }

    pub async fn export_state(&self) -> Result<ModelState, NNError> {
        let w1 = self.w1.read().await;
        let b1 = self.b1.read().await;
        let w2 = self.w2.read().await;
        let b2 = self.b2.read().await;

        Ok(ModelState {
            w1: w1.iter().cloned().collect(),
            b1: b1.iter().cloned().collect(),
            w2: w2.iter().cloned().collect(),
            b2: b2.iter().cloned().collect(),
        })
    }

    pub async fn import_state(&self, state: ModelState) -> Result<(), NNError> {
        *self.w1.write().await =
            Array2::from_shape_vec((128, 784), state.w1)
                .map_err(|_| NNError::SerializationError)?;

        *self.b1.write().await =
            Array1::from_vec(state.b1);

        *self.w2.write().await =
            Array2::from_shape_vec((10, 128), state.w2)
                .map_err(|_| NNError::SerializationError)?;

        *self.b2.write().await =
            Array1::from_vec(state.b2);

        Ok(())
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
}

#[async_trait]
impl DigitClassifier for NdArrayEngine {

    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError> {
        let input = Self::normalize(pixels)?;

        let w1 = self.w1.read().await;
        let b1 = self.b1.read().await;
        let w2 = self.w2.read().await;
        let b2 = self.b2.read().await;

        let z1 = w1.dot(&input) + &*b1;
        let a1 = Self::relu(&z1);

        let z2 = w2.dot(&a1) + &*b2;
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

    async fn train(&self, label: u8, pixels: &[u8]) -> Result<(), NNError> {
        let input = Self::normalize(pixels)?;

        let mut w1 = self.w1.write().await;
        let mut b1 = self.b1.write().await;
        let mut w2 = self.w2.write().await;
        let mut b2 = self.b2.write().await;

        // -------- Forward pass --------
        let z1 = w1.dot(&input) + &*b1;
        let a1 = Self::relu(&z1);

        let z2 = w2.dot(&a1) + &*b2;
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
        let da1 = w2.t().dot(&dz2);
        let dz1 = da1 * Self::relu_derivative(&z1);

        let dw1 = dz1
            .view()
            .insert_axis(Axis(1))
            .dot(&input.view().insert_axis(Axis(0)));

        let db1 = dz1.clone();

        // -------- Gradient descent --------

        *w2 -= &(dw2 * self.lr);
        *b2 -= &(db2 * self.lr);

        *w1 -= &(dw1 * self.lr);
        *b1 -= &(db1 * self.lr);

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*; 

    #[tokio::test]
    async fn test_predict_runs() {
        let engine = NdArrayEngine::new();
        let pixels = vec![0u8; 784];

        let result = engine.predict(&pixels).await.unwrap();

        assert!(result.digit <= 9);
        assert!(result.confidence >= 0.0);
    }

    #[tokio::test]
    async fn test_train() {
        let engine = NdArrayEngine::new();
        let pixels = vec![255u8; 784];
        engine.train(9, &pixels).await.unwrap();
    }
}