use std::{
    env,
    fs::{self, File},
    io::{self, copy},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use ureq::{http::Response, Body};

const TOKENIZER_FILE: &str = "tokenizer.json";

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");

    let build_phi = env::var("CARGO_FEATURE_PHI").is_ok();

    let model_dir: &'static str;
    let weights_file: &'static str;
    let tokenizer_url: &'static str;
    let weights_url: &'static str;

    if build_phi {
        model_dir = "microsoft--Phi-3-mini-4k-instruct-gguf";
        weights_file = "Phi-3-mini-4k-instruct-q4.gguf";
        tokenizer_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/0a67737cc96d2554230f90338b163bc6380a2a85/tokenizer.json";
        weights_url = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/999f761fe19e26cf1a339a5ec5f9f201301cbb83/Phi-3-mini-4k-instruct-q4.gguf";
    } else {
        model_dir = "bartowski--google_gemma-3-4b-it-qat-GGUF";
        weights_file = "google_gemma-3-4b-it-qat-Q4_0.gguf";
        tokenizer_url = "https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-unquantized/resolve/b02f2b2916142396dace61d2e8c3412cdcbd1559/tokenizer.json";
        weights_url = "https://huggingface.co/bartowski/google_gemma-3-4b-it-qat-GGUF/resolve/0e3202d4c2315e5d2796d83c476e0961e4680513/google_gemma-3-4b-it-qat-Q4_0.gguf";
    }

    let dest_path = PathBuf::from("models").join(model_dir);
    let files = [TOKENIZER_FILE, weights_file];

    download_model_files(&dest_path, &files, tokenizer_url, weights_url)?;

    Ok(())
}

fn download_model_files(
    dest_path: &Path,
    files: &[&str; 2],
    tokenizer_url: &str,
    weights_url: &str,
) -> Result<()> {
    let all_files_exist = files
        .iter()
        .all(|filename| dest_path.join(filename).exists());

    if all_files_exist {
        println!("cargo::warning=All files already exist, skipping download");
        return Ok(());
    }

    println!("cargo::warning=Downloading files...");

    let hf_token = get_hf_token()?;

    fs::create_dir_all(dest_path).context("Failed to create destination directory")?;

    for filename in files {
        let dest_file = dest_path.join(filename);

        if dest_file.exists() {
            println!("cargo::warning=File already exists, skipping: {}", filename);
            continue;
        }

        let url = match filename {
            &TOKENIZER_FILE => tokenizer_url,
            _ => weights_url,
        };

        println!("cargo::warning=Downloading {} from {}...", filename, url);

        let mut response = do_request(filename, url, &hf_token)?;
        let mut reader = response.body_mut().as_reader();
        let mut file =
            File::create(&dest_file).context(format!("Failed to create file {:?}", dest_file))?;

        let bytes_written =
            copy(&mut reader, &mut file).context(format!("Failed to write {}", filename))?;

        println!(
            "cargo::warning=Downloaded {} ({} bytes)",
            filename, bytes_written
        );
    }

    Ok(())
}

fn get_hf_token() -> Result<String> {
    let hf_token_filename = "token";

    let hf_cache_dir = get_hf_cache_dir()?;
    let token_path = hf_cache_dir.join(hf_token_filename);

    println!(
        "cargo::warning=Reading HF token from {}...",
        &token_path.to_string_lossy()
    );

    let token = fs::read_to_string(token_path)
        .context("Failed to read HF token.")?
        .trim()
        .to_string();

    Ok(token)
}

fn get_hf_cache_dir() -> Result<PathBuf> {
    let home_path = std::env::var("HOME")
        .map_err(|_| io::Error::new(io::ErrorKind::NotFound, "HOME directory not found"))?;

    let token_path = PathBuf::from(home_path).join(".cache").join("huggingface");

    Ok(token_path)
}

fn do_request(filename: &str, url: &str, hf_token: &str) -> Result<Response<Body>> {
    println!("cargo::warning=Downloading {} from {}...", filename, url);

    let response = ureq::get(url)
        .header("Authorization", &format!("Bearer {}", hf_token))
        .call()
        .context(format!("Failed to download {}", url))?;

    if response.status() != 200 {
        anyhow::bail!(
            "Download failed for {} with status: {}",
            filename,
            response.status()
        );
    }

    Ok(response)
}
