use async_trait::async_trait;

use tokio::sync::Mutex;
use nn_engine;

pub struct MockDigitClassifier {
    last_trained: Mutex<Option<u8>>,
}

impl MockDigitClassifier {
    pub fn new() -> Self {
        Self {
            last_trained: Mutex::new(None),
        }
    }
}

#[async_trait]
impl nn_engine::DigitClassifier for MockDigitClassifier {
    async fn predict(&self, _pixels: &[u8]) -> Result<nn_engine::Prediction, nn_engine::NNError> {
        let last_trained = self.last_trained.lock().await;
        let digit = last_trained.unwrap_or(0); // Возвращаем последний обученный класс или 0 по умолчанию
        Ok(nn_engine::Prediction {
            digit,
            confidence: 0.98, // Фиксированная точность для демонстрации
        })
    }

    async fn train(&self, label: u8, _pixels: &[u8]) -> Result<(), nn_engine::NNError> {
        let mut last_trained = self.last_trained.lock().await;
        *last_trained = Some(label); // Сохраняем последний обученный класс
        Ok(())
    }
}
