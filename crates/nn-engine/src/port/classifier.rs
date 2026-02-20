use crate::domain::{ModelState, Prediction, TrainingStepResult, error::NNError};
use async_trait::async_trait;

#[async_trait]
pub trait DigitClassifier: Send + Sync {
    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError>;
    async fn train(&self, label: u8, pixels: &[u8]) -> Result<(), NNError>;
}

pub trait DigitPredictor {
    fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError>;
}

pub trait DigitTrainer {
    fn train(&mut self, label: u8, pixels: &[u8]) -> Result<TrainingStepResult, NNError>;
}

pub trait ModelStateExporter {
    fn export_state(&self) -> Result<ModelState, NNError>;
}

pub trait ModelStateImporter {
    fn import_state(&mut self, state: ModelState) -> Result<(), NNError>;
}

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
pub trait AsyncDigitClassifier:
    AsyncDigitPredictor + AsyncDigitTrainer + Send + Sync
{
}

impl<T> AsyncDigitClassifier for T
where
    T: AsyncDigitPredictor + AsyncDigitTrainer + Send + Sync,
{}