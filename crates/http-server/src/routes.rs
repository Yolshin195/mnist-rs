use crate::handlers;
use crate::state::AppState;
use axum::{
    Router,
    routing::{get, post},
};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::page::index))
        .route("/api/predict", post(handlers::api::predict))
        .route("/api/train", post(handlers::api::train))
}
