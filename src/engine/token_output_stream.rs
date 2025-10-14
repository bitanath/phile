//! This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
//! streaming way rather than having to wait for the full decoding.
//!
//! See: <https://github.com/huggingface/candle/blob/main/candle-examples/src/token_output_stream.rs>

use anyhow::Result;

pub(crate) struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub(crate) fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    pub(crate) fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => anyhow::bail!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub(crate) fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };

        self.tokens.push(token);

        let text = self.decode(&self.tokens[self.prev_index..])?;

        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };

        let text = self.decode(&self.tokens[self.prev_index..])?;

        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    #[allow(dead_code)]
    pub(crate) fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    #[allow(dead_code)]
    pub(crate) fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub(crate) fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    #[allow(dead_code)]
    pub(crate) fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}
