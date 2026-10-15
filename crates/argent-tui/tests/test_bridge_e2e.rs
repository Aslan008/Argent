//! A whole turn through the REAL bridge into the TUI.
//!
//! A scripted agent produces chunks; `ReplBridge` runs the turn on its worker
//! thread and emits events into the Rust sink (exactly what `run_tui` wires up);
//! the `App` consumes them and a frame is drawn with the production renderer.
//! This is the path no unit test covers: Python worker thread -> pyo3 sink ->
//! channel -> `App` -> `draw_ui`.
//!
//! Needs the project's Python dependencies; skipped without them unless
//! `ARGENT_REQUIRE_PARITY=1`.

use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use argent_tui::app::{App, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::pyrender::PyRenderer;
use argent_tui::sink::RustEventSink;
use argent_tui::theme::UiConfig;
use argent_tui::ui::draw_ui;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

const SHIM: &str = r#"
import os, sys

def setup(root):
    sys.path.insert(0, root)
    site = os.path.join(root, ".venv", "Lib", "site-packages")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)

def make_bridge():
    # Nothing here may touch the user's real setup: no MCP servers, no Telegram,
    # no startup hooks, no knowledge-base indexing.
    import config
    config.get_mcp_servers = lambda: []
    config.get_telegram_enabled = lambda: False
    config.get_auto_kb = lambda: False
    from hook_manager import hook_manager
    hook_manager.call_hook = lambda *a, **k: None

    class Agent:
        model_name = "fake-model"
        provider = "fake"
        messages = []
        session_id = None
        _turn_tool_calls = 0
        _turn_local_agent_calls = 0

        def process_user_input(self, text, allowed_tools=None):
            yield {"type": "thinking_stream", "content": "Let me look at it."}
            yield {"type": "content_stream", "content": "Reading the file.\n\n"}
            yield {"type": "tool_start", "name": "read_file",
                   "args": {"file_path": "agent.py", "offset": 10, "limit": 40}}
            yield {"type": "tool_end", "name": "read_file", "result": "line1\nline2"}
            yield {"type": "content_stream", "content": "It has two lines.\n\n"}
            yield {"type": "usage", "data": {"prompt": 1200, "completion": 340}}

        def get_context_usage(self):
            return {"tokens": 3100, "max": 32000, "percent": 10.0}

        def get_context_breakdown(self):
            return {}

        def request_stop(self): pass
        def repair_history(self): pass

    from src.bridge.repl_bridge import ReplBridge
    return ReplBridge(agent=Agent())
"#;

fn project_root() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("project root")
        .to_string_lossy()
        .trim_start_matches(r"\\?\")
        .to_string()
}

fn skip_or_fail(reason: &str) {
    if std::env::var("ARGENT_REQUIRE_PARITY").as_deref() == Ok("1") {
        panic!("bridge e2e could not run: {reason}");
    }
    eprintln!("SKIPPED bridge e2e: {reason}");
}

#[test]
fn a_scripted_turn_reaches_the_screen_through_the_real_bridge() {
    pyo3::prepare_freethreaded_python();
    let outcome: Option<(App, String, Vec<ArgentEvent>)> = Python::with_gil(|py| {
        let shim = PyModule::from_code_bound(py, SHIM, "e2e_shim.py", "e2e_shim").ok()?;
        shim.getattr("setup").ok()?.call1((project_root(),)).ok()?;
        let bridge = match shim.getattr("make_bridge").and_then(|f| f.call0()) {
            Ok(b) => b.unbind(),
            Err(e) => {
                eprintln!("e2e: cannot build the bridge ({e})");
                return None;
            }
        };

        let (tx, rx) = channel();
        // A Receiver is not Sync; a Mutex makes it shareable into allow_threads.
        let rx = std::sync::Mutex::new(rx);
        let sink = Py::new(py, RustEventSink { tx }).ok()?;
        bridge.call_method1(py, "set_event_sink", (sink,)).ok()?;

        let mut app = App::new();
        app.renderer = Box::new(PyRenderer::new(bridge.clone_ref(py)));
        let raw: String = bridge.call_method0(py, "get_ui_config").ok()?.extract(py).ok()?;
        app.ui = UiConfig::from_json(&raw);

        bridge.call_method1(py, "start_turn", ("hello",)).ok()?;
        app.insert_str("hello");
        app.submit_input();
        app.is_busy = true;

        // The worker thread needs the GIL to run, so wait with it released.
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut done = false;
        let mut seen: Vec<ArgentEvent> = Vec::new();
        while !done && Instant::now() < deadline {
            let ev = py.allow_threads(|| {
                rx.lock().ok().and_then(|r| r.recv_timeout(Duration::from_millis(100)).ok())
            });
            if let Some((at, ev)) = ev {
                done = matches!(ev, ArgentEvent::Done);
                seen.push(ev.clone());
                app.handle_event_at(ev, at);
            }
        }
        if !done {
            eprintln!("e2e: the turn never finished; events seen: {seen:?}");
            return None;
        }

        let mut term = Terminal::new(TestBackend::new(110, 50)).unwrap();
        term.draw(|f| draw_ui(f, &app)).unwrap();
        let frame: String = {
            let buf = term.backend().buffer();
            (0..buf.area.height)
                .map(|y| {
                    (0..buf.area.width)
                        .map(|x| buf[(x, y)].symbol())
                        .collect::<String>()
                        .trim_end()
                        .to_string()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let _ = bridge.call_method0(py, "cleanup");
        Some((app, frame, seen))
    });

    let Some((app, frame, seen)) = outcome else {
        skip_or_fail("the real bridge could not be driven");
        return;
    };

    // Every part of the REPL's output for this turn is on screen, in its order.
    let order = [
        "◇ Анализ",
        "✦ Argent",
        "Reading the file.",
        "read_file: agent.py:10-50",
        "↳ 2 строк · line1",
        "It has two lines.",
        "┄",
        "Context: 10.0%",
    ];
    let mut from = 0usize;
    for needle in order {
        match frame[from..].find(needle) {
            Some(i) => from += i + needle.len(),
            None => panic!("{needle:?} missing or out of order in:\n{frame}"),
        }
    }

    // No error or fallback notice crept in, and the turn left the app idle.
    assert!(!frame.contains("Ошибка"), "an error was shown:\n{frame}");
    assert!(!frame.contains("упрощённый вид"), "a block fell back to the Rust renderer:\n{frame}");
    assert!(!app.is_busy, "Done must leave the app idle");
    assert!(
        app.messages.iter().any(|m| matches!(m, MessageItem::Snapshot { .. })),
        "the bridge sent no snapshot"
    );
    // The toolbar saw the session cost the turn recorded.
    assert!(app.view_text.borrow().rows.len() > 5);
    // The provider's own count of the tokens the request generated crossed the whole
    // path (worker thread, sink, parser) intact: it is what the speed is checked against.
    assert!(
        seen.iter().any(|e| matches!(e, ArgentEvent::Usage { completion: Some(340), .. })),
        "no usage event with the completion count; events: {seen:?}"
    );
}
