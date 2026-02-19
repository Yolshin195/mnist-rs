use axum::{
    extract::{State, Json},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct PredictRequest {
    pub image: Vec<u8>, // 784 пикселя
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub digit: u8,
    pub confidence: f32,
}

pub async fn predict(
    State(state): State<AppState>,
    Json(payload): Json<PredictRequest>,
) -> impl IntoResponse {
    tracing::debug!("Received predict request with image of length: {}", payload.image.len());

    // ToDo - Правильно обрабатывать ошибки, а не просто unwrap
    let result = state.classifier.predict(&payload.image).await.unwrap();

    tracing::debug!("Prediction result: digit={}, confidence={}", result.digit, result.confidence);

    Json(PredictResponse {
        digit: result.digit,
        confidence: result.confidence,
    })
}

#[derive(Deserialize)]
pub struct TrainRequest {
    pub label: u8,
    pub image: Vec<u8>,
}

pub async fn train(
    State(state): State<AppState>,
    Json(payload): Json<TrainRequest>,
) -> impl IntoResponse {
    tracing::debug!("Received train request with image of length: {}, label: {}", payload.image.len(), payload.label);

    let result = state.classifier
        .train(payload.label, &payload.image)
        .await;

    match result {
        Ok(_) => tracing::debug!("Training successful for label: {}", payload.label),
        Err(e) => tracing::error!("Training failed for label: {}: {:?}", payload.label, e),
        
    }

    Json("ok")
}
