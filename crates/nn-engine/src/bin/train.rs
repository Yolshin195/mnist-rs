use std::env;
use std::fs::{self, File};
use std::io::BufReader;

use csv::ReaderBuilder;
use tokio;

use nn_engine::{
    NdArrayEngine,
    DigitClassifier,
    ModelRepository,
    FileModelRepository,
};

const TRAIN_PATH: &str = "assets/mnist/mnist_train.csv";
const MODELS_DIR: &str = "assets/models";
const EPOCHS: usize = 3;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {

    // -------- CLI args --------
    let args: Vec<String> = env::args().collect();

    let version = args.iter()
        .position(|a| a == "--version")
        .and_then(|i| args.get(i + 1))
        .expect("Usage: --version <name>");

    let resume = args.contains(&"--resume".to_string());

    fs::create_dir_all(MODELS_DIR)?;

    let model_path = format!("{}/{}.bin", MODELS_DIR, version);

    println!("🚀 Training version: {}", version);

    let engine = NdArrayEngine::new();
    let repo = FileModelRepository::new(&model_path);

    // -------- Resume logic --------
    if resume {
        println!("📂 Loading existing model...");
        let state = repo.load().await?;
        engine.import_state(state).await?;
        println!("✅ Model loaded");
    }

    // -------- Training --------
    for epoch in 0..EPOCHS {
        println!("\n📚 Epoch {}/{}", epoch + 1, EPOCHS);
        train_epoch(&engine).await?;
    }

    // -------- Save --------
    println!("\n💾 Saving model...");
    let state = engine.export_state().await?;
    repo.save(&state).await?;

    println!("✅ Model saved to {}", model_path);

    Ok(())
}

async fn train_epoch(
    engine: &NdArrayEngine,
) -> Result<(), Box<dyn std::error::Error>> {

    let file = File::open(TRAIN_PATH)?;
    let reader = BufReader::new(file);

    let mut csv_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(reader);

    let mut count = 0usize;

    for result in csv_reader.records() {
        let record = result?;

        let label: u8 = record[0].parse()?;

        let pixels: Vec<u8> = record
            .iter()
            .skip(1)
            .map(|v| v.parse::<u8>().unwrap())
            .collect();

        engine.train(label, &pixels).await?;

        count += 1;

        if count % 5000 == 0 {
            println!("Trained on {} samples", count);
        }
    }

    Ok(())
}
