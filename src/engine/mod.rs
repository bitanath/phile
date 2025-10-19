//! General inference code for LLMs. Based on Candle Examples.
//!
//! See: <https://github.com/huggingface/candle/blob/main/candle-examples/examples/quantized-gemma/main.rs>

#[allow(unused_imports)]
use std::{
    fmt, fs,
    io::{Cursor, Write},
};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_utils::device::get_device;

use crate::models::loader::load_model;

pub(crate) mod token_output_stream;
use token_output_stream::TokenOutputStream;

#[cfg(feature = "gemma")]
use candle_transformers::models::quantized_gemma3::ModelWeights;

#[cfg(feature = "phi")]
use candle_transformers::models::quantized_phi3::ModelWeights;

pub(crate) type RenderTemplateFn =
    fn(messages: &[Message], add_generation_prompt: bool) -> Result<String>;

#[allow(dead_code)]
#[derive(Clone)]
pub(crate) enum MessageRole {
    Assistant,
    System,
    User,
}

#[derive(Clone)]
pub(crate) struct Message {
    pub(crate) content: String,
    pub(crate) role: MessageRole,
}

impl Message {
    pub(crate) fn user(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            role: MessageRole::User,
        }
    }
}

pub(crate) struct Engine {
    device: Device,
    model: ModelWeights,
    logits_processor: LogitsProcessor,
    tos: TokenOutputStream,
    eos_token: u32,
    max_context_len: usize,
    token_history: Vec<u32>,
    render_template: RenderTemplateFn,
}

impl Engine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: Device,
        model_dir_name: &str,
        tokenizers_file: &str,
        weights_file: &str,
        eos: &str,
        max_context_len: usize,
        seed: u64,
        sampling_params: Sampling,
        verbose: bool,
        render_template: RenderTemplateFn,
    ) -> Result<Self> {
        let (tokenizer, model) = load_model(
            &device,
            model_dir_name,
            tokenizers_file,
            weights_file,
            verbose,
        )?;

        let tos = TokenOutputStream::new(tokenizer);

        let eos_token = tos
            .get_token(eos)
            .context(format!("Couldn't load token: {eos:?}"))?;

        Ok(Self {
            device,
            model,
            logits_processor: LogitsProcessor::from_sampling(seed, sampling_params),
            tos,
            eos_token,
            token_history: Vec::new(),
            max_context_len,
            render_template,
        })
    }

    pub(crate) fn generate(&mut self, input: &str, to_sample: usize) -> Result<Vec<String>> {
        let user_message = Message::user(input);

        self.generate_from_messages(&[user_message], to_sample)
    }

    pub(crate) fn generate_from_messages(
        &mut self,
        messages: &[Message],
        to_sample: usize,
    ) -> Result<Vec<String>> {
        let rendered_input = (self.render_template)(messages, true)?;
        let starting_tokens = self.prepare_starting_tokens(&rendered_input, to_sample)?;

        self.generate_tokens(&starting_tokens, to_sample)
    }

    fn prepare_starting_tokens(&mut self, input: &str, to_sample: usize) -> Result<Vec<u32>> {
        let tokens = self
            .tos
            .tokenizer()
            .encode(input, true)
            .map_err(anyhow::Error::msg)?;

        let input_tokens: Vec<u32> = tokens.get_ids().to_vec();
        self.token_history.extend_from_slice(&input_tokens);

        self.fit_to_context(&self.token_history, to_sample)
    }

    fn fit_to_context(&self, input_tokens: &[u32], to_sample: usize) -> Result<Vec<u32>> {
        const MAX_LEN_PADDING: usize = 10;

        let starting_tokens_len = input_tokens.len();
        let total_needed = starting_tokens_len + to_sample;

        if total_needed > self.max_context_len - MAX_LEN_PADDING {
            let to_remove = total_needed + MAX_LEN_PADDING - self.max_context_len;
            let start_index = starting_tokens_len.saturating_sub(to_remove);

            Ok(input_tokens[start_index..].to_vec())
        } else {
            Ok(input_tokens.to_vec())
        }
    }

    fn generate_tokens(
        &mut self,
        starting_tokens: &[u32],
        to_sample: usize,
    ) -> Result<Vec<String>> {
        let starting_tokens_len = starting_tokens.len();
        let mut output = Vec::new();
        let mut last_token = None;

        for index in 0..=to_sample {
            let next_token = match last_token {
                None => self.generate_next_token(starting_tokens, 0)?,
                Some(token) => self.generate_next_token(&[token], starting_tokens_len + index)?,
            };

            last_token = Some(next_token);

            if let Some(t) = self.output_token(next_token)? {
                output.push(t);
            }

            if next_token == self.eos_token {
                break;
            }
        }

        // Clear out the token output stream between runs to prevent
        // decoding issues since special tokens are ignored.
        self.tos.clear();

        Ok(output)
    }

    fn generate_next_token(&mut self, input_tokens: &[u32], index_pos: usize) -> Result<u32> {
        let input = Tensor::new(input_tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        let next_token = self.logits_processor.sample(&logits)?;

        self.token_history.push(next_token);

        Ok(next_token)
    }

    fn output_token(&mut self, next_token: u32) -> Result<Option<String>> {
        if let Some(t) = self.tos.next_token(next_token)? {
            print!("{}", &t);
            std::io::stdout().flush()?;
            Ok(Some(t))
        } else {
            Ok(None)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_engine(
    model_dir_name: &str,
    tokenizers_file: &str,
    weights_file: &str,
    max_context_len: usize,
    eos: &str,
    verbose: bool,
    render_template: RenderTemplateFn,
) -> Result<Engine> {
    if verbose {
        println!("{}", hardware_info());
    }

    let device = get_device(false, !verbose)?;
    let seed = 123;
    let sampling_params = Sampling::ArgMax;

    Engine::new(
        device,
        model_dir_name,
        tokenizers_file,
        weights_file,
        eos,
        max_context_len,
        seed,
        sampling_params,
        verbose,
        render_template,
    )
}

pub(crate) fn hardware_info() -> String {
    format!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    )
}
