use axum::{routing::{get, post}, Router};
use crate::handlers;
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::page::index))
        .route("/api/predict", post(handlers::api::predict))
        .route("/api/train", post(handlers::api::train))
}


#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_main() {
        let response = router()
            .with_state(AppState {
                classifier: std::sync::Arc::new(crate::mock_classifier::MockDigitClassifier::new()),
            })
            .oneshot(Request::get("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}