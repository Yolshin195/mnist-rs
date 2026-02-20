use async_trait::async_trait;
use std::sync::Arc;

use crate::adapter::ndarray_engine::NdArrayEngine;
use crate::adapter::async_ndarray_engine::AsyncNdArrayEngine;
use crate::adapter::file_repository::FileModelRepository;
use crate::domain::Prediction;
use crate::domain::TrainingStepResult;
use crate::domain::error::NNError;
use crate::port::async_classifier::{
    AsyncDigitPredictor, AsyncDigitTrainer, AsyncModelStateExporter, AsyncModelStateImporter,
};
use crate::port::model_repository::ModelRepository;

pub struct DigitClassifierService {
    engine: AsyncNdArrayEngine,
    repo: Arc<dyn ModelRepository + Send + Sync>,
}

impl DigitClassifierService {
    pub fn new(engine: AsyncNdArrayEngine, repo: Arc<dyn ModelRepository + Send + Sync>) -> Self {
        Self { engine, repo }
    }

    pub async fn load_model(&self) -> Result<(), NNError> {
        let state = self.repo.load().await?;
        self.engine.import_state(state).await?;
        Ok(())
    }

    pub async fn save_model(&self) -> Result<(), NNError> {
        let state = self.engine.export_state().await?;
        self.repo.save(&state).await
    }
}

impl DigitClassifierService {
    pub fn from_path(path: impl Into<String>) -> Self {
        let engine = AsyncNdArrayEngine::new(NdArrayEngine::new());
        let repo = Arc::new(FileModelRepository::new(path));

        Self { engine, repo }
    }
}

#[async_trait]
impl AsyncDigitPredictor for DigitClassifierService {
    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError> {
        self.engine.predict(pixels).await
    }
}

#[async_trait]
impl AsyncDigitTrainer for DigitClassifierService {
    async fn train(&self, label: u8, pixels: &[u8]) -> Result<TrainingStepResult, NNError> {
        self.engine.train(label, pixels).await
    }
}
