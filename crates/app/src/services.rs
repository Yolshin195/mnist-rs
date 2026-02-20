use std::sync::Arc;
use nn_engine::{
    DigitClassifier, FileModelRepository, ModelRepository, NNError, NdArrayEngine, Prediction
};
use async_trait::async_trait;

pub struct DigitClassifierService {
    engine: Arc<NdArrayEngine>,
    repo: Arc<FileModelRepository>,
}


impl DigitClassifierService {
    pub async fn new(model_path: impl Into<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let engine = NdArrayEngine::new();
        let repo = FileModelRepository::new(model_path);

        // если модель существует → загружаем
        if let Ok(state) = repo.load().await {
            engine.import_state(state).await?;
            println!("📂 Model loaded from disk");
        } else {
            println!("🆕 No model found, starting fresh");
        }

        Ok(Self {
            engine: Arc::new(engine),
            repo: Arc::new(repo),
        })
    }
}


#[async_trait]
impl DigitClassifier for DigitClassifierService {
    async fn predict(&self, pixels: &[u8]) -> Result<Prediction, NNError> {
        self.engine.predict(pixels).await
    }

    async fn train(&self, label: u8, pixels: &[u8]) -> Result<(), NNError> {
        self.engine.train(label, pixels).await
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_train() {
        let model_path = format!(
            "{}/../../assets/models/default.bin",
            env!("CARGO_MANIFEST_DIR")
        );
        let service = DigitClassifierService::new(model_path).await.unwrap();
        
        let pixels: Vec<i32> = vec![
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,255,255,255,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,134,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,243,127,162,255,255,255,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,255,255,0,0,0,0,0,18,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,255,255,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,32,0,0,0,0,0,0,2,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,117,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,255,255,203,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,86,255,255,248,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,188,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,226,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,255,255,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,255,129,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,151,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        ];
        let pixels: Vec<u8> = pixels
            .into_iter()
            .map(|v| v as u8)
            .collect();

        let prediction = service.predict(&pixels).await.unwrap();

        println!("{}, {}", prediction.digit, prediction.confidence)
    }

}