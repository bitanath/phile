//! Inference code for Gemma family.

use anyhow::Result;
use askama::Template;

use crate::{
    engine::{Engine, Message, MessageRole, build_engine},
    models::{ModelConfig, RoleMapper},
};

impl ModelConfig {
    pub const GEMMA: Self = Self {
        max_context_len: 8192,
        eos_token: "<end_of_turn>",
    };
}

pub(crate) struct GemmaRoles;
impl RoleMapper for GemmaRoles {
    fn map_role(role: &MessageRole) -> &'static str {
        match role {
            MessageRole::Assistant => "model",
            MessageRole::System | MessageRole::User => "user",
        }
    }
}

#[derive(Template)]
#[template(path = "gemma.j2")]
pub(crate) struct GemmaTemplate {
    messages: Vec<Message>,
    assistant_role: MessageRole,
    // https://huggingface.co/docs/transformers/en/chat_templating#addgenerationprompt
    add_generation_prompt: bool,
}

impl GemmaTemplate {
    pub(crate) fn get_role_string(&self, role: &MessageRole) -> &'static str {
        GemmaRoles::map_role(role)
    }
}

pub(crate) fn gemma_build(
    model_dir_name: &str,
    tokenizers_file: &str,
    weights_file: &str,
    verbose: bool,
) -> Result<Engine> {
    build_engine(
        model_dir_name,
        tokenizers_file,
        weights_file,
        ModelConfig::GEMMA.max_context_len,
        ModelConfig::GEMMA.eos_token,
        verbose,
        render_template,
    )
}

pub(crate) fn render_template(messages: &[Message], add_generation_prompt: bool) -> Result<String> {
    let template = GemmaTemplate {
        messages: messages.to_vec(),
        assistant_role: MessageRole::Assistant,
        add_generation_prompt,
    };

    let output = template.render()?.trim_start().to_string();

    Ok(output)
}
