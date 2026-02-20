use async_trait::async_trait;
use tokio::fs;

use crate::domain::{ModelState, error::NNError};
use crate::port::model_repository::ModelRepository;

pub struct FileModelRepository {
    path: String,
}

impl FileModelRepository {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl ModelRepository for FileModelRepository {
    async fn save(&self, state: &ModelState) -> Result<(), NNError> {
        let bytes = bincode::serialize(state).map_err(|_| NNError::PersistenceError)?;

        fs::write(&self.path, bytes)
            .await
            .map_err(|_| NNError::PersistenceError)
    }

    async fn load(&self) -> Result<ModelState, NNError> {
        let bytes = fs::read(&self.path)
            .await
            .map_err(|_| NNError::PersistenceError)?;

        bincode::deserialize(&bytes).map_err(|_| NNError::PersistenceError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;
    use tokio::fs;

    fn temp_file_path(name: &str) -> PathBuf {
        let mut path = env::temp_dir();
        path.push(name);
        path
    }

    fn test_state() -> ModelState {
        ModelState {
            w1: vec![0.1; 128 * 784],
            b1: vec![0.2; 128],
            w2: vec![0.3; 10 * 128],
            b2: vec![0.4; 10],
        }
    }

    #[tokio::test]
    async fn save_and_load_success() {
        let path = temp_file_path("model_test.bin");

        // гарантируем чистоту
        let _ = fs::remove_file(&path).await;

        let repo = FileModelRepository::new(path.to_str().unwrap());
        let state = test_state();

        repo.save(&state).await.unwrap();
        let loaded = repo.load().await.unwrap();

        assert_eq!(state.w1, loaded.w1);
        assert_eq!(state.b1, loaded.b1);
        assert_eq!(state.w2, loaded.w2);
        assert_eq!(state.b2, loaded.b2);

        let _ = fs::remove_file(&path).await;
    }

    #[tokio::test]
    async fn load_non_existing_file_returns_error() {
        let path = temp_file_path("non_existing_model.bin");
        let repo = FileModelRepository::new(path.to_str().unwrap());

        let result = repo.load().await;

        assert!(result.is_err());
    }
}
