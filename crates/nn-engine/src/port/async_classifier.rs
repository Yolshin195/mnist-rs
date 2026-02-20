use async_trait::async_trait;
use crate::domain::{ModelState, Prediction, TrainingStepResult, error::NNError};

#[async_trait]
pub trait AsyncDigitPredictor {
    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError>;
}

#[async_trait]
pub trait AsyncDigitTrainer {
    async fn train(&self, label: u8, pixels: &[u8]) -> Result<TrainingStepResult, NNError>;
}

#[async_trait]
pub trait AsyncModelStateExporter {
    async fn export_state(&self) -> Result<ModelState, NNError>;
}

#[async_trait]
pub trait AsyncModelStateImporter {
    async fn import_state(&self, state: ModelState) -> Result<(), NNError>;
}

#[async_trait]
pub trait AsyncDigitClassifier: AsyncDigitPredictor + AsyncDigitTrainer + Send + Sync {}

impl<T> AsyncDigitClassifier for T where T: AsyncDigitPredictor + AsyncDigitTrainer + Send + Sync {}
