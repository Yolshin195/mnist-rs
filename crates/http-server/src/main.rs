//! Run with
//!
//! ```not_rust
//! cargo run -p example-templates
//! ```
//! 

mod mock_classifier;
mod classifier;
mod routes;
mod handlers;
mod templates;
mod server;
mod state;

#[tokio::main]
async fn main() {
    server::run().await;
}
