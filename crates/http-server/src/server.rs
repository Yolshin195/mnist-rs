use crate::routes::router;

use crate::state::AppState;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use nn_engine::application::digit_classifier_service::DigitClassifierService;

pub async fn run() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("{}=debug", env!("CARGO_CRATE_NAME")).into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let model_path = format!(
        "{}/../../assets/models/default.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let service = DigitClassifierService::from_path(model_path);
    service.load_model().await.expect("Can't load model");

    let state = AppState {
        classifier: Arc::new(service),
    };

    // build our application with some routes
    let app = router().with_state(state);
    // run it
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    let _ = axum::serve(listener, app).await;
}
