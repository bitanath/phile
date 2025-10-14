//! Helper utils for CLAP / command line.

use clap::builder::styling::{AnsiColor, Styles};

/// Helper function to style terminal output
/// See: <https://stackoverflow.com/a/76916424>
pub fn get_styled_terminal_output() -> Styles {
    Styles::styled()
        .header(AnsiColor::Yellow.on_default())
        .usage(AnsiColor::Green.on_default())
        .literal(AnsiColor::Green.on_default())
        .placeholder(AnsiColor::Green.on_default())
}
