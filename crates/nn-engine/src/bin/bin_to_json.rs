use nn_engine::{FileModelRepository, JsonModelRepository, port::model_repository::ModelRepository};


const MODELS_DIR: &str = "assets/models";


#[tokio::main]
async fn main() {
    let file_repo = FileModelRepository::new(format!("{}/default.bin", MODELS_DIR));
    let json_repo = JsonModelRepository::new(format!("{}/default.json", MODELS_DIR));

    if let Ok(model) = file_repo.load().await {
        json_repo.save(&model).await.expect("Error writre in Json");
    }
}