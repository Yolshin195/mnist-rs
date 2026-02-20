use crate::routes::router;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use crate::state::AppState;
use app::DigitClassifierService;
use std::sync::Arc;


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
    let service = DigitClassifierService::new(model_path).await.expect("Error: DigitClassifierService new");
    let state = AppState { classifier: Arc::new(service) };

    // build our application with some routes
    let app = router()
        .with_state(state);;
    // run it
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    tracing::debug!("listening on {}", listener.local_addr().unwrap());
    let _ = axum::serve(listener, app).await;
}