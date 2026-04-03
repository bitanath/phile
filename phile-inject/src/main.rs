use std::{
    env, fs,
    io::{self, Read, Seek, SeekFrom, Write},
    path::PathBuf,
    process,
};

use anyhow::{Context, Result};
use flate2::{write::GzEncoder, Compression};
use tar::Builder;

const MAGIC: &[u8] = b"PHILE_EMBED_V1";
const TRAILER_SIZE: usize = MAGIC.len() + 8; // magic + u64 archive length

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: phile-inject <path-to-compiled-binary>");
        eprintln!();
        eprintln!("Appends model weights to the compiled phile binary.");
        eprintln!("The binary must be built with the `embed` feature enabled.");
        process::exit(1);
    }

    let exe_path = PathBuf::from(&args[1]);

    if !exe_path.exists() {
        eprintln!("Error: Binary not found at: {:?}", exe_path);
        process::exit(1);
    }

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());

    let build_phi = env::var("CARGO_FEATURE_PHI").is_ok();

    let (model_dir, _weights_file) = if build_phi {
        (
            "microsoft--Phi-3-mini-4k-instruct-gguf",
            "Phi-3-mini-4k-instruct-q4.gguf",
        )
    } else {
        (
            "bartowski--google_gemma-3-4b-it-qat-GGUF",
            "google_gemma-3-4b-it-qat-Q4_0.gguf",
        )
    };

    let model_path = PathBuf::from(&manifest_dir).join("models").join(model_dir);

    let tokenizer_path = model_path.join("tokenizer.json");
    let weights_path = model_path.join("google_gemma-3-4b-it-qat-Q4_0.gguf");

    if !tokenizer_path.exists() {
        eprintln!("Error: Tokenizer not found at: {:?}", tokenizer_path);
        eprintln!("Run `cargo build` first to download model files.");
        process::exit(1);
    }

    if !weights_path.exists() {
        eprintln!("Error: Weights not found at: {:?}", weights_path);
        eprintln!("Run `cargo build` first to download model files.");
        process::exit(1);
    }

    println!("Creating compressed archive...");
    let archive_data = create_archive(&tokenizer_path, &weights_path)?;
    println!("Archive size: {:.2} GB", archive_data.len() as f64 / 1e9);

    println!("Appending archive to binary...");
    append_to_binary(&exe_path, &archive_data)?;

    println!("Successfully packed binary");
    println!(
        "Final size: {:.2} GB",
        exe_path
            .metadata()
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0)
    );

    Ok(())
}

fn create_archive(tokenizer_path: &PathBuf, weights_path: &PathBuf) -> Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());

    {
        let mut builder = Builder::new(&mut encoder);

        builder
            .append_path_with_name(tokenizer_path, "tokenizer.json")
            .context("Failed to add tokenizer to archive")?;

        builder
            .append_path_with_name(weights_path, "weights.gguf")
            .context("Failed to add weights to archive")?;

        builder.finish().context("Failed to finalize tar archive")?;
    }

    encoder.finish().context("Failed to compress archive")
}

fn append_to_binary(exe_path: &PathBuf, archive_data: &[u8]) -> Result<()> {
    let archive_len = archive_data.len() as u64;

    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(exe_path)
        .context("Failed to open binary for writing")?;

    // Read existing trailer info
    let file_len = file.metadata()?.len();

    if file_len >= TRAILER_SIZE as u64 {
        file.seek(SeekFrom::End(-(TRAILER_SIZE as i64)))?;
        let mut trailer = [0u8; TRAILER_SIZE];
        file.read_exact(&mut trailer)?;
        let mut magic = [0u8; MAGIC.len()];
        magic.copy_from_slice(&trailer[..MAGIC.len()]);

        if magic == MAGIC {
            let old_len = u64::from_le_bytes(trailer[MAGIC.len()..].try_into().unwrap());
            let old_end = file_len - TRAILER_SIZE as u64 - old_len;
            file.set_len(old_end)
                .context("Failed to remove old archive")?;
            println!("Removed previous embedded archive");
        }
    }

    // Seek to end and append
    file.seek(SeekFrom::End(0))?;
    file.write_all(archive_data)
        .context("Failed to write archive data")?;
    file.write_all(MAGIC)
        .context("Failed to write magic bytes")?;
    file.write_all(&archive_len.to_le_bytes())
        .context("Failed to write archive length")?;

    // Preserve executable permissions
    use std::os::unix::fs::PermissionsExt;
    let mut perms = file.metadata()?.permissions();
    perms.set_mode(0o755);
    file.set_permissions(perms)?;

    Ok(())
}
