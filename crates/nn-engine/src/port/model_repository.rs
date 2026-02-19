use async_trait::async_trait;
use crate::domain::{model_state::ModelState, error::NNError};


#[async_trait]
pub trait ModelRepository: Send + Sync {
    async fn save(&self, state: &ModelState) -> Result<(), NNError>;
    async fn load(&self) -> Result<ModelState, NNError>;
}
