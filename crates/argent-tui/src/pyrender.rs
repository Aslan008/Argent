//! The production block renderer: ask the Python bridge.
//!
//! `ReplBridge.render_block(kind, json, width)` prints the block with the REPL's
//! own `ui.py` and hands back ANSI, so the TUI shows exactly what the terminal
//! would — including Pygments colours the Rust side could never reproduce.
//!
//! The call takes the GIL. It is made once per block and width (results are
//! cached by `view.rs`), never per frame, so a busy agent thread costs at most a
//! short wait the first time a block appears.

use pyo3::prelude::*;
use ratatui::text::Line;
use serde_json::Value;

use crate::ansi::parse_ansi;
use crate::renderer::BlockRenderer;

pub struct PyRenderer {
    bridge: PyObject,
}

impl PyRenderer {
    pub fn new(bridge: PyObject) -> Self {
        Self { bridge }
    }
}

impl BlockRenderer for PyRenderer {
    fn render(&self, kind: &str, data: &Value, width: u16) -> Result<Vec<Line<'static>>, String> {
        let json = serde_json::to_string(data).map_err(|e| e.to_string())?;
        Python::with_gil(|py| {
            self.bridge
                .call_method1(py, "render_block", (kind, json, width))
                .and_then(|out| out.extract::<String>(py))
        })
        .map(|ansi| parse_ansi(&ansi))
        .map_err(|e| e.to_string())
    }
}
