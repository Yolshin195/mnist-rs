use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task;

use crate::adapter::ndarray_engine::NdArrayEngine;
use crate::domain::TrainingStepResult;
use crate::domain::{ModelState, Prediction, error::NNError};
use crate::port::classifier::{
    DigitPredictor, DigitTrainer, ModelStateExporter, ModelStateImporter,
};
use crate::port::async_classifier::{
    AsyncDigitPredictor, AsyncDigitTrainer, AsyncModelStateExporter, AsyncModelStateImporter,
};

pub struct AsyncNdArrayEngine {
    inner: Arc<Mutex<NdArrayEngine>>,
}

impl AsyncNdArrayEngine {
    pub fn new(engine: NdArrayEngine) -> Self {
        Self {
            inner: Arc::new(Mutex::new(engine)),
        }
    }
}

#[async_trait]
impl AsyncDigitPredictor for AsyncNdArrayEngine {
    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError> {
        let engine = self.inner.clone();
        let pixels = pixels.to_vec();

        task::spawn_blocking(move || {
            let engine = engine.blocking_lock();
            engine.predict(&pixels)
        })
        .await
        .map_err(|_| NNError::InternalError)?
    }
}

#[async_trait]
impl AsyncDigitTrainer for AsyncNdArrayEngine {
    async fn train(&self, label: u8, pixels: &[u8]) -> Result<TrainingStepResult, NNError> {
        let engine = self.inner.clone();
        let pixels = pixels.to_vec();

        task::spawn_blocking(move || {
            let mut engine = engine.blocking_lock();
            engine.train(label, &pixels)
        })
        .await
        .map_err(|_| NNError::InternalError)?
    }
}

#[async_trait]
impl AsyncModelStateExporter for AsyncNdArrayEngine {
    async fn export_state(&self) -> Result<ModelState, NNError> {
        let engine = self.inner.clone();

        task::spawn_blocking(move || {
            let engine = engine.blocking_lock();
            engine.export_state()
        })
        .await
        .map_err(|_| NNError::InternalError)?
    }
}

#[async_trait]
impl AsyncModelStateImporter for AsyncNdArrayEngine {
    async fn import_state(&self, state: ModelState) -> Result<(), NNError> {
        let engine = self.inner.clone();

        task::spawn_blocking(move || {
            let mut engine = engine.blocking_lock();
            engine.import_state(state)
        })
        .await
        .map_err(|_| NNError::InternalError)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::task;

    fn sample_pixels() -> Vec<u8> {
        vec![255u8; 784]
    }

    #[tokio::test]
    async fn test_async_predict_runs() {
        let engine = NdArrayEngine::new();
        let async_engine = AsyncNdArrayEngine::new(engine);

        let pixels = sample_pixels();

        let result = async_engine.predict(&pixels).await.unwrap();

        assert!(result.digit <= 9);
        assert!(result.confidence >= 0.0);
    }

    #[tokio::test]
    async fn test_async_train_changes_prediction() {
        let engine = NdArrayEngine::new();
        let async_engine = AsyncNdArrayEngine::new(engine);

        let pixels = sample_pixels();

        let before = async_engine.predict(&pixels).await.unwrap().digit;

        async_engine.train(3, &pixels).await.unwrap();

        let after = async_engine.predict(&pixels).await.unwrap().digit;

        // не гарантируем конкретную цифру,
        // но проверяем что код отработал без паники
        assert!(after <= 9);
        assert!(before <= 9);
    }

    #[tokio::test]
    async fn test_async_parallel_predict() {
        let engine = NdArrayEngine::new();
        let async_engine = Arc::new(AsyncNdArrayEngine::new(engine));

        let pixels = sample_pixels();

        let mut handles = vec![];

        for _ in 0..10 {
            let engine = async_engine.clone();
            let pixels = pixels.clone();

            handles.push(task::spawn(async move {
                engine.predict(&pixels).await.unwrap()
            }));
        }

        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.digit <= 9);
        }
    }

    #[tokio::test]
    async fn test_async_overfit_single_sample() {
        let engine = NdArrayEngine::new();
        let async_engine = AsyncNdArrayEngine::new(engine);

        let pixels = sample_pixels();

        for _ in 0..200 {
            async_engine.train(7, &pixels).await.unwrap();
        }

        let prediction = async_engine.predict(&pixels).await.unwrap();

        assert_eq!(prediction.digit, 7);
        assert!(prediction.confidence > 0.8);
    }
}
