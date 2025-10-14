use crate::engine::MessageRole;

pub(crate) mod gemma;
pub(crate) mod loader;

#[allow(dead_code)]
pub(crate) mod phi;

pub(crate) trait RoleMapper {
    fn map_role(role: &MessageRole) -> &'static str;
}

pub(crate) struct ModelConfig {
    pub max_context_len: usize,
    pub eos_token: &'static str,
}
