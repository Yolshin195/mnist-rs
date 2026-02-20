mod classifier;
mod handlers;
mod routes;
mod server;
mod state;
mod templates;

#[tokio::main]
async fn main() {
    server::run().await;
}
