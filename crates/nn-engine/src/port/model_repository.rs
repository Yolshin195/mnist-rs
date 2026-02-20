use crate::domain::{ModelState, error::NNError};
use async_trait::async_trait;

#[async_trait]
pub trait ModelRepository: Send + Sync {
    async fn save(&self, state: &ModelState) -> Result<(), NNError>;
    async fn load(&self) -> Result<ModelState, NNError>;
}
