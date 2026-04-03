#[allow(unused)]
use std::{
    fs,
    io::{Cursor, Read},
};

use anyhow::{Context, Result};
use candle_core::{
    quantized::gguf_file::{self},
    Device,
};
use tokenizers::Tokenizer;

#[cfg(feature = "gemma")]
use candle_transformers::models::quantized_gemma3::ModelWeights;

#[cfg(feature = "phi")]
use candle_transformers::models::quantized_phi3::ModelWeights;

const MAGIC: &[u8] = b"PHILE_EMBED_V1";
const TRAILER_SIZE: usize = MAGIC.len() + 8;

pub(crate) fn load_model(
    device: &Device,
    model_dir_name: &str,
    tokenizers_file: &str,
    weights_file: &str,
    verbose: bool,
) -> Result<(Tokenizer, ModelWeights)> {
    let start = std::time::Instant::now();

    let (tokenizer_data, weights_data) =
        load_model_files(model_dir_name, tokenizers_file, weights_file)?;

    let tokenizer = Tokenizer::from_bytes(tokenizer_data).map_err(anyhow::Error::msg)?;

    let mut model_dummy_file = Cursor::new(weights_data);
    let model_content = gguf_file::Content::read(&mut model_dummy_file)?;

    if verbose {
        print_model_info(&model_content, start.elapsed().as_secs_f32());
    }

    let model = init_model(model_content, &mut model_dummy_file, device)?;

    Ok((tokenizer, model))
}

#[allow(unused_variables)]
fn load_model_files(
    model_dir_name: &str,
    tokenizers_file: &str,
    weights_file: &str,
) -> Result<(Vec<u8>, Vec<u8>)> {
    #[cfg(not(feature = "embed"))]
    {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let models_dir = format!("{manifest_dir}/models");
        let model_dir = format!("{models_dir}/{model_dir_name}");

        let tokenizers_file_path = format!("{model_dir}/{tokenizers_file}");
        let tokenizer_data = fs::read(&tokenizers_file_path)
            .context(format!("Couldn't read file: {tokenizers_file_path:?}"))?;

        let weights_file_path = format!("{model_dir}/{weights_file}");
        let weights_data = fs::read(&weights_file_path)
            .context(format!("Couldn't read file: {weights_file_path:?}"))?;

        Ok((tokenizer_data, weights_data))
    }

    #[cfg(feature = "embed")]
    {
        let (tokenizer_data, weights_data) = extract_embedded_models()?;
        Ok((tokenizer_data, weights_data))
    }
}

#[cfg(feature = "embed")]
fn extract_embedded_models() -> Result<(Vec<u8>, Vec<u8>)> {
    use std::io::{Read, Seek, SeekFrom};

    let exe_path = std::env::current_exe().context("Failed to get current executable path")?;

    let mut file = fs::File::open(&exe_path).context("Failed to open executable")?;
    let file_len = file.metadata()?.len();

    if file_len < TRAILER_SIZE as u64 {
        anyhow::bail!("Binary too small to contain embedded models");
    }

    file.seek(SeekFrom::End(-(TRAILER_SIZE as i64)))?;
    let mut trailer = [0u8; TRAILER_SIZE];
    file.read_exact(&mut trailer)?;

    let mut magic = [0u8; MAGIC.len()];
    magic.copy_from_slice(&trailer[..MAGIC.len()]);

    if magic != MAGIC {
        anyhow::bail!("No embedded models found in binary. Run phile-inject first.");
    }

    let archive_len = u64::from_le_bytes(trailer[MAGIC.len()..].try_into().unwrap());
    let archive_start = file_len - TRAILER_SIZE as u64 - archive_len;

    file.seek(SeekFrom::Start(archive_start))?;
    let mut archive_data = vec![0u8; archive_len as usize];
    file.read_exact(&mut archive_data)?;

    let decompressed = decompress_archive(&archive_data)?;
    let mut archive = tar::Archive::new(Cursor::new(decompressed));

    let mut tokenizer_data = None;
    let mut weights_data = None;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.to_string_lossy().to_string();

        if path == "tokenizer.json" {
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            tokenizer_data = Some(data);
        } else if path == "weights.gguf" {
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            weights_data = Some(data);
        }
    }

    let tokenizer_data = tokenizer_data.context("tokenizer.json not found in embedded archive")?;
    let weights_data = weights_data.context("weights.gguf not found in embedded archive")?;

    Ok((tokenizer_data, weights_data))
}

#[cfg(feature = "embed")]
fn decompress_archive(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::GzDecoder;

    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

fn print_model_info(model_content: &gguf_file::Content, elapsed_secs: f32) {
    let total_size_in_bytes = model_content
        .tensor_infos
        .values()
        .map(|tensor| {
            let elem_count = tensor.shape.elem_count();
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size()
        })
        .sum();

    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        model_content.tensor_infos.len(),
        format_size(total_size_in_bytes),
        elapsed_secs,
    );
}

fn init_model(
    model_content: gguf_file::Content,
    model_file: &mut Cursor<Vec<u8>>,
    device: &Device,
) -> Result<ModelWeights> {
    #[cfg(feature = "gemma")]
    {
        Ok(ModelWeights::from_gguf(model_content, model_file, device)?)
    }

    #[cfg(feature = "phi")]
    {
        let use_flash_attn = cfg!(feature = "flash-attn");

        Ok(ModelWeights::from_gguf(
            use_flash_attn,
            model_content,
            model_file,
            device,
        )?)
    }
}

pub(crate) fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{size_in_bytes}B")
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}
