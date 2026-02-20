mod services;
pub use services::DigitClassifierService;

use async_trait::async_trait;
// ==== DTO ====

#[derive(Clone, Debug)]
pub struct PredictDigitRequest {
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct PredictDigitResponse {
    pub digit: u8,
    pub confidence: f32,
}

#[derive(Clone, Debug)]
pub struct TrainDigitRequest {
    pub label: u8,
    pub pixels: Vec<u8>,
}

// ==== INPUT PORT ====


#[async_trait]
pub trait DigitClassifierUseCase: Send + Sync {
    async fn predict_digit(
        &self,
        request: PredictDigitRequest,
    ) -> PredictDigitResponse;

    async fn train_digit(
        &self,
        request: TrainDigitRequest,
    );
}
