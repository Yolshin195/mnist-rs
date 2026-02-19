use async_trait::async_trait;
use crate::domain::{prediction::Prediction, error::NNError};

#[async_trait]
pub trait DigitClassifier: Send + Sync {
    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError>;
    async fn train(&self, label: u8, pixels: &[u8]) -> Result<(), NNError>;
}