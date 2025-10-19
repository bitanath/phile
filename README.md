# phile

Single file llm, but in _rust_. phi + file = phile.

### Setup

1. Make sure you have access to the Gemma model family on Hugging Face.
   1. https://huggingface.co/docs/hub/en/models-gated
2. Make sure your Hugging Face auth token is set as it will be needed to fetch the Gemma model files (since Gemma is a gated model).
   1. https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command

### Notes

See: https://github.com/huggingface/candle/pull/3104

### Cross compile

https://github.com/cross-rs/cross

#### Linux x86 CPU

```sh
$ cross build --target x86_64-unknown-linux-gnu --features embed,openssl
```

### Resources

1. Template playground
   1. https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=google%2Fgemma-3-1b-it-qat-q4_0-gguf
   2. https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=microsoft%2FPhi-3-mini-4k-instruct-gguf
2. https://huggingface.co/docs/transformers/en/chat_templating#addgenerationprompt
3. https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-server/src/interactive_mode.rs
