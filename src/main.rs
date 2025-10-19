//! # phile: phi file  
//!
//! ## Example
//!
//! ```console
//! $ phile "hi"
//! ```

#![warn(unused_extern_crates)]

use anyhow::Result;
use clap::Parser;
use rustyline::{DefaultEditor, error::ReadlineError};

pub(crate) mod clap_utils;
use clap_utils::get_styled_terminal_output;

pub(crate) mod engine;
pub(crate) mod models;

#[cfg(feature = "gemma")]
use models::gemma::gemma_build as build_engine;

#[cfg(feature = "phi")]
use models::gemma::phi_build as build_engine;

#[derive(Parser)]
#[command(version, about, long_about = None, styles=get_styled_terminal_output())]
pub(crate) struct Cli {
    /// input string (positional / default arg)
    input: Option<String>,

    #[arg(
        short = 'i',
        default_value_t = false,
        help = "Interactive mode. NOTE: when in interactive mode the input arg will be ignored."
    )]
    interactive: bool,

    #[arg(short = 'v', default_value_t = false, help = "Verbose output")]
    verbose: bool,
}

#[cfg(all(feature = "gemma", feature = "phi"))]
compile_error!("Cannot enable both 'gemma' and 'phi' features simultaneously");

#[cfg(all(feature = "gemma", feature = "embed"))]
#[derive(rust_embed::Embed)]
#[folder = "models/bartowski--google_gemma-3-1b-it-qat-GGUF"]
pub(crate) struct LlmModelAssets;

#[cfg(all(feature = "phi", feature = "embed"))]
#[derive(rust_embed::Embed)]
#[folder = "models/microsoft--Phi-3-mini-4k-instruct-gguf"]
pub(crate) struct LlmModelAssets;

/// Where the magic happens.
fn main() -> Result<()> {
    let cli = Cli::parse();

    let tokenizers_file = "tokenizer.json";
    let model_dir_name: &'static str;
    let weights_file: &'static str;

    #[cfg(feature = "gemma")]
    {
        model_dir_name = "bartowski--google_gemma-3-1b-it-qat-GGUF";
        weights_file = "google_gemma-3-1b-it-qat-Q4_0.gguf";
    }

    #[cfg(feature = "phi")]
    {
        model_dir_name = "microsoft--Phi-3-mini-4k-instruct-gguf";
        weights_file = "Phi-3-mini-4k-instruct-q4.gguf";
    }

    let to_sample = 100;
    let mut engine = build_engine(model_dir_name, tokenizers_file, weights_file, cli.verbose)?;

    if cli.interactive {
        let mut rl = DefaultEditor::new()?;

        loop {
            let readline = rl.readline(">> ");

            match readline {
                Ok(line) => {
                    rl.add_history_entry(line.as_str())?;
                    let _ = engine.generate(&line, to_sample)?;

                    // Output empty line so Rustyline doesn't overwrite output
                    #[allow(clippy::println_empty_string)]
                    {
                        println!("");
                    }
                }
                Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                    exit_handler();
                }
                Err(err) => {
                    println!("Error: {:?}", err);
                    std::process::exit(1);
                }
            }
        }
    } else if let Some(input) = &cli.input {
        let _ = engine.generate(input, to_sample)?;
    } else {
        println!("No input provided");
    }

    Ok(())
}

fn exit_handler() {
    println!("Goodbye...");
    std::process::exit(0);
}
