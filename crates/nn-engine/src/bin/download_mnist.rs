use std::fs;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use reqwest::Client;

const TRAIN_URL: &str =
    "https://raw.githubusercontent.com/phoebetronic/mnist/master/mnist_train.csv.zip";

const TEST_URL: &str =
    "https://raw.githubusercontent.com/phoebetronic/mnist/master/mnist_test.csv.zip";

const OUTPUT_DIR: &str = "assets/mnist";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(OUTPUT_DIR)?;

    download_and_extract(TRAIN_URL, "mnist_train.csv").await?;
    download_and_extract(TEST_URL, "mnist_test.csv").await?;

    println!("\n✅ MNIST successfully downloaded to `{}`", OUTPUT_DIR);

    Ok(())
}

async fn download_and_extract(
    url: &str,
    output_filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⬇ Downloading: {}", url);

    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let response = client
        .get(url)
        .header("User-Agent", "rust-mnist-downloader")
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("Download failed: {}", response.status()).into());
    }

    let bytes = response.bytes().await?;

    if bytes.len() < 1_000_000 {
        return Err("Downloaded file too small — likely not a ZIP".into());
    }

    let temp_zip_path = format!("{}/temp_download.zip", OUTPUT_DIR);
    tokio::fs::write(&temp_zip_path, &bytes).await?;

    println!("📦 Extracting...");

    let zip_file = File::open(&temp_zip_path)?;
    let mut archive = zip::ZipArchive::new(zip_file)?;

    if archive.len() == 0 {
        return Err("ZIP archive is empty".into());
    }

    // Берём первый файл из архива
    let mut zipped_file = archive.by_index(0)?;

    let outpath = Path::new(OUTPUT_DIR).join(output_filename);
    let mut outfile = File::create(&outpath)?;
    io::copy(&mut zipped_file, &mut outfile)?;

    outfile.flush()?;
    fs::remove_file(&temp_zip_path)?;

    println!("✅ Saved {}", output_filename);

    Ok(())
}
