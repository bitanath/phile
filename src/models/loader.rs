#[allow(unused)]
use std::{fs, io::Cursor};

use anyhow::{Context, Result};
use candle_core::{
    Device,
    quantized::gguf_file::{self},
};
use tokenizers::Tokenizer;

#[cfg(feature = "gemma")]
use candle_transformers::models::quantized_gemma3::ModelWeights;

#[cfg(feature = "phi")]
use candle_transformers::models::quantized_phi3::ModelWeights;

#[cfg(feature = "embed")]
use crate::LlmModelAssets;

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
        let tokenizer_data = LlmModelAssets::get(tokenizers_file)
            .context("couldn't load tokenizer data")?
            .data
            .to_vec();

        let weights_data = LlmModelAssets::get(weights_file)
            .context("couldn't load weights data")?
            .data
            .to_vec();

        Ok((tokenizer_data, weights_data))
    }
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
