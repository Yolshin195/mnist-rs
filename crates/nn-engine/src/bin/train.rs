use std::env;
use std::fs::{self, File};
use std::io::BufReader;

use csv::ReaderBuilder;
use tokio;

use nn_engine::{
    FileModelRepository, ModelRepository, NdArrayEngine,
    port::classifier::{DigitTrainer, ModelStateExporter, ModelStateImporter},
};

const TRAIN_PATH: &str = "assets/mnist/mnist_train.csv";
const MODELS_DIR: &str = "assets/models";
const EPOCHS: usize = 3;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -------- CLI args --------
    let args: Vec<String> = env::args().collect();

    let version = args
        .iter()
        .position(|a| a == "--version")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("default");

    let resume = args.contains(&"--resume".to_string());

    fs::create_dir_all(MODELS_DIR)?;

    let model_path = format!("{}/{}.bin", MODELS_DIR, version);

    println!("🚀 Training version: {}", version);

    let mut engine = NdArrayEngine::new();
    let repo = FileModelRepository::new(&model_path);

    // -------- Resume --------
    if resume {
        println!("📂 Loading existing model...");
        let state = repo.load().await?;
        engine.import_state(state)?;
        println!("✅ Model loaded");
    }

    // -------- Training --------
    for epoch in 0..EPOCHS {
        println!("\n📚 Epoch {}/{}", epoch + 1, EPOCHS);
        train_epoch(&mut engine).await?;
    }

    // -------- Save --------
    println!("\n💾 Saving model...");
    let state = engine.export_state()?;
    repo.save(&state).await?;

    println!("✅ Model saved to {}", model_path);

    Ok(())
}

async fn train_epoch(engine: &mut NdArrayEngine) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(TRAIN_PATH)?;
    let reader = BufReader::new(file);

    let mut csv_reader = ReaderBuilder::new().has_headers(true).from_reader(reader);

    let mut total_loss = 0.0;
    let mut total_correct = 0usize;
    let mut count = 0usize;

    for result in csv_reader.records() {
        let record = result?;

        let label: u8 = record[0].parse()?;

        let pixels: Vec<u8> = record
            .iter()
            .skip(1)
            .map(|v| v.parse::<u8>().unwrap())
            .collect();

        let train_metrics = engine.train(label, &pixels)?;

        total_loss += train_metrics.loss;
        if train_metrics.correct {
            total_correct += 1;
        }

        count += 1;

        if count % 5000 == 0 {
            println!(
                "Samples: {} | Avg Loss: {:.4} | Accuracy: {:.2}%",
                count,
                total_loss / count as f32,
                100.0 * total_correct as f32 / count as f32
            );
        }
    }

    println!(
        "\n📊 Epoch Result → Loss: {:.4} | Accuracy: {:.2}%",
        total_loss / count as f32,
        100.0 * total_correct as f32 / count as f32
    );

    Ok(())
}
