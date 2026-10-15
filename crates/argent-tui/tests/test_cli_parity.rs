//! Visual parity with the REPL, checked against the REPL itself.
//!
//! The same scripted turn is run twice:
//!
//! 1. through the REPL's real stream renderer (`cli_ui.render_response_stream`,
//!    i.e. Rich + Pygments), captured as ANSI at a fixed width;
//! 2. through the TUI: the same agent chunks become bridge events, the `App`
//!    consumes them and `draw_ui` paints a frame, with the production Python
//!    block renderer behind it.
//!
//! The two screens are then compared cell by cell — symbol, foreground,
//! background and modifiers. Nothing is compared against a stored picture, so
//! a change to `ui.py` or `theme.yaml` moves both sides together, while a drift
//! in the TUI's own assembly (spacing, headers, ordering) fails here.
//!
//! The test needs the project's Python dependencies. Without them it says so and
//! passes, unless `ARGENT_REQUIRE_PARITY=1` (set in CI) makes that a failure.

use argent_tui::ansi::parse_ansi;
use argent_tui::app::App;
use argent_tui::events::{parse_py_event, ArgentEvent};
use argent_tui::pyrender::PyRenderer;
use argent_tui::theme::UiConfig;
use argent_tui::ui::draw_ui;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::Terminal;
use serde_json::{json, Value};

const WIDTH: u16 = 100;
const HEIGHT: u16 = 90;

const SHIM: &str = r#"
import io, json, sys, types

def setup(root):
    import os
    sys.path.insert(0, root)
    # The embedded interpreter is the base one PYO3_PYTHON points at; a project
    # virtualenv's packages are not on its path unless added here.
    site = os.path.join(root, ".venv", "Lib", "site-packages")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)

def _console(width):
    from rich.console import Console
    import ui
    return Console(file=io.StringIO(), theme=ui.custom_theme, safe_box=True, width=width,
                   height=40, force_terminal=False, color_system="truecolor",
                   legacy_windows=False)

class _Clock:
    # A frozen clock: every duration is 0, so "0ms" / "0.0s" are deterministic.
    def time(self): return 1000.0
    def monotonic(self): return 1000.0
    def sleep(self, s): pass

class _Agent:
    _turn_tool_calls = 0
    _turn_local_agent_calls = 0
    _esc_listener = None
    def get_context_usage(self): return {"tokens": 3100, "max": 32000, "percent": 10.0}
    def get_context_breakdown(self): return {}

def cli_transcript(chunks_json, width):
    import ui
    from src.cli import cli_ui
    chunks = json.loads(chunks_json)
    con = _console(width)
    saved = (ui.console, cli_ui.console, cli_ui.time)
    ui.console = con; cli_ui.console = con; cli_ui.time = _Clock()
    try:
        cli_ui.render_response_stream(_Agent(), iter(chunks))
    finally:
        ui.console, cli_ui.console, cli_ui.time = saved
    return con.file.getvalue()

def bridge_events(chunks_json):
    from src.bridge.events import to_event
    out = []
    for c in json.loads(chunks_json):
        ev = to_event(c)
        if ev is None:
            continue
        if ev["type"] == "tool_end":
            ev["duration_secs"] = 0.0
        out.append(ev)
    return out

class Bridge:
    def render_block(self, kind, payload_json, width):
        from src.bridge.render import render_block
        return render_block(kind, json.loads(payload_json), int(width))
    def get_ui_config(self):
        from src.bridge.render import ui_config
        # Parity is with the terminal REPL, which only has the classic look; the
        # premium look is a TUI choice (tests/test_premium_ui.rs covers it).
        return json.dumps(dict(ui_config(), style="classic"), ensure_ascii=False)

def make_bridge():
    return Bridge()

def cli_live(kind, payload_json, width):
    """One frame of a REPL live display, from the REPL's own classes."""
    import ui
    from src.cli import cli_ui
    p = json.loads(payload_json)
    con = _console(width)
    saved = (ui.console, cli_ui.console, cli_ui.time)
    ui.console = con; cli_ui.console = con; cli_ui.time = _Clock()
    try:
        turn_start = 1000.0 - p.get("total", 0)
        if kind == "waiting":
            r = cli_ui._WaitingDisplay(turn_start)
        elif kind in ("thinking", "rethink"):
            if kind == "thinking":
                r = cli_ui._ThinkingDisplay(label_style=ui.c.get("reasoning_label"),
                                            text_style=ui.c.get("reasoning_text"),
                                            start_turn_time=turn_start)
            else:
                r = cli_ui._ThinkingDisplay(label_style="bold #d18f52", text_style="italic #ffe0b2",
                                            start_turn_time=turn_start,
                                            title="Перепроверка (Rethink):",
                                            spinner_label="Стресс-тест решения...")
            r.thinking_text = p["text"]
        elif kind == "tool_wait":
            r = cli_ui._ToolWaitDisplay(p["name"], turn_start, p.get("args") or {})
        elif kind == "tool_gen":
            r = cli_ui._ToolGenDisplay(start_turn_time=turn_start)
            r.tool_name = p["name"]; r.total_bytes = p["bytes"]; r.tool_text = ""
        elif kind == "tail":
            r = cli_ui.content_tail_renderable(p["pending"], 1000.0, 1000.0, turn_start, 40, None)
        elif kind == "queue":
            r = cli_ui.queue_panel(p["items"])
        else:
            raise ValueError(kind)
        con.print(r)
    finally:
        ui.console, cli_ui.console, cli_ui.time = saved
    return con.file.getvalue()

def snapshot_case(width):
    """The post-response lines: the REPL's printer vs the bridge's event."""
    import ui
    from src.cli import cli_ui
    from src.bridge.session import AgentSession
    from usage_tracker import usage as su
    usage = {"prompt": 1200, "completion": 340, "cached": 900, "cost": 0.0021}
    con = _console(width)
    saved = (ui.console, cli_ui.console)
    ui.console = con; cli_ui.console = con
    try:
        su.reset()
        cli_ui._print_turn_snapshot(_Agent(), 8.2, timing={"first": 0.3, "tools": 1.0, "_ts": None},
                                    usage=dict(usage), tool_calls=3)
        ui.print_context_usage(3100, 32000, 10.0, out=con)
        cli = con.file.getvalue()
        su.reset()
        sess = AgentSession(agent=_Agent())
        snap = {"total_s": 8.2, "ttft_s": 0.3, "tool_s": 1.0, "tool_calls": 3}
        ev = {"type": "turn_snapshot", "data": snap}
        ev.update(sess._snapshot_markup(snap, dict(usage)))
        su.reset()
    finally:
        ui.console, cli_ui.console = saved
    return cli, ev

def toolbar_case(mode):
    """The REPL's bottom toolbar text for a fixed state, plus that state as the
    bridge would report it."""
    import os, time
    from prompt_toolkit.formatted_text import to_formatted_text
    from src.cli import cli_prompt as cp
    import config as cfg
    from usage_tracker import usage as su

    class A:
        provider = "openrouter"
        model_name = "qwen/qwen3-coder:free"
        strategy = type("CloudStrategy", (), {})()
        _last_context_usage = {"percent": 12, "tokens": 3100, "max": 32768}

    su.reset()
    su.add({"prompt": 1200, "completion": 340, "cost": 0.0021})
    cp._SESSION_START_TIME = time.time() - 303
    html = cp.build_bottom_toolbar(A(), {"mode": mode})()
    plain = "".join(frag[1] for frag in to_formatted_text(html))
    state = {
        "provider": A.provider, "model": A.model_name, "tier": cp._tier_label(A()),
        "mode": mode, "thinking_label": cfg.get_thinking_display_label() or "",
        "cwd": os.path.basename(os.getcwd()) or os.getcwd(),
        "ctx_tokens": "3100", "ctx_max": "32768", "ctx_percent": "12",
        "session_summary": su.format_session(),
    }
    su.reset()
    return plain, state
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

struct Env<'py> {
    shim: Bound<'py, PyModule>,
    bridge: PyObject,
}

fn load(py: Python<'_>) -> Option<Env<'_>> {
    let shim = PyModule::from_code_bound(py, SHIM, "parity_shim.py", "parity_shim").ok()?;
    shim.getattr("setup").ok()?.call1((project_root(),)).ok()?;
    // Prove the real renderer is importable before relying on it.
    if let Err(e) = py.import_bound("src.bridge.render") {
        eprintln!("parity: project Python is not importable ({e})");
        return None;
    }
    let bridge = shim.getattr("make_bridge").ok()?.call0().ok()?.unbind();
    Some(Env { shim, bridge })
}

fn skip_or_fail(reason: &str) {
    if std::env::var("ARGENT_REQUIRE_PARITY").as_deref() == Ok("1") {
        panic!("visual parity could not run: {reason}");
    }
    eprintln!("SKIPPED visual parity: {reason}");
}

/// What the REPL prints for `chunks`, minus the post-response snapshot (it is
/// timing-dependent and has its own comparison).
fn cli_rows(env: &Env<'_>, chunks: &Value) -> Vec<Line<'static>> {
    let ansi: String = env
        .shim
        .getattr("cli_transcript")
        .unwrap()
        .call1((chunks.to_string(), WIDTH))
        .unwrap()
        .extract()
        .unwrap();
    let mut rows = parse_ansi(&ansi);
    if let Some(cut) = rows
        .iter()
        .position(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>().starts_with("  ┄"))
    {
        rows.truncate(cut);
    }
    rows
}

/// The TUI's frame for the same chunks; returns (buffer, transcript row count).
fn tui_screen(env: &Env<'_>, py: Python<'_>, chunks: &Value) -> (Buffer, usize) {
    let events: Bound<'_, PyList> = env
        .shim
        .getattr("bridge_events")
        .unwrap()
        .call1((chunks.to_string(),))
        .unwrap()
        .downcast_into()
        .unwrap();

    let mut app = App::new();
    app.renderer = Box::new(PyRenderer::new(env.bridge.clone_ref(py)));
    let raw: String = env.bridge.call_method0(py, "get_ui_config").unwrap().extract(py).unwrap();
    app.ui = UiConfig::from_json(&raw);

    app.insert_str("prompt");
    app.submit_input(); // the user's own row sits above the reply
    for item in events.iter() {
        let dict = item.downcast::<PyDict>().unwrap();
        if let Some(ev) = parse_py_event(dict).unwrap() {
            app.handle_event(ev);
        }
    }
    app.handle_event(ArgentEvent::Done);

    let mut term = Terminal::new(TestBackend::new(WIDTH, HEIGHT)).unwrap();
    term.draw(|f| draw_ui(f, &app)).unwrap();
    let rows = app.view_text.borrow().rows.len();
    (term.backend().buffer().clone(), rows)
}

/// Paint `rows` into a buffer the way the TUI does, for cell-wise comparison.
fn paint(rows: &[Line<'static>]) -> Buffer {
    let mut buf = Buffer::empty(Rect::new(0, 0, WIDTH, rows.len().max(1) as u16));
    for (y, l) in rows.iter().enumerate() {
        buf.set_line(0, y as u16, l, WIDTH);
    }
    buf
}

fn row_text(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>().trim_end().to_string()
}

fn describe(buf: &Buffer, y: u16) -> String {
    let mut s = String::new();
    for x in 0..buf.area.width.min(60) {
        let c = &buf[(x, y)];
        if c.symbol() != " " {
            s.push_str(&format!("[{x}:{:?} fg={:?} bg={:?} {:?}]", c.symbol(), c.fg, c.bg, c.modifier));
        }
    }
    s
}

/// The embedded interpreter and the REPL's module-level console are global, so
/// the scenarios run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Assert that the TUI shows, below the user's row, exactly what the REPL printed.
fn assert_parity(name: &str, chunks: Value) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let Some(env) = load(py) else {
            skip_or_fail("src.bridge.render could not be imported");
            return;
        };
        let want_rows = cli_rows(&env, &chunks);
        let (got, got_rows) = tui_screen(&env, py, &chunks);
        let want = paint(&want_rows);

        // Row 0 of the TUI is the user's prompt echo; the REPL's own prompt line
        // comes from prompt_toolkit and is not part of the stream renderer.
        let mut text_want: Vec<String> = Vec::new();
        for y in 0..want.area.height.min(want_rows.len() as u16) {
            text_want.push(row_text(&want, y));
        }
        let mut text_got: Vec<String> = Vec::new();
        for y in 1..got_rows as u16 {
            text_got.push(row_text(&got, y));
        }
        assert_eq!(
            text_got, text_want,
            "{name}: the TUI's rows differ from what the REPL prints\n--- REPL ---\n{}\n--- TUI ---\n{}",
            text_want.join("\n"),
            text_got.join("\n")
        );

        for (i, y_want) in (0..want_rows.len() as u16).enumerate() {
            let y_got = (i + 1) as u16;
            for x in 0..WIDTH {
                let (a, b) = (&want[(x, y_want)], &got[(x, y_got)]);
                // Blank cells carry no visible style; trailing padding is not
                // part of the picture.
                if a.symbol() == " " && b.symbol() == " " && a.bg == b.bg {
                    continue;
                }
                assert!(
                    a.symbol() == b.symbol() && a.fg == b.fg && a.bg == b.bg && a.modifier == b.modifier,
                    "{name}: cell ({x},{y_want}) differs\nREPL: {:?} fg={:?} bg={:?} {:?}\nTUI:  {:?} fg={:?} bg={:?} {:?}\nrow REPL: {}\nrow TUI:  {}",
                    a.symbol(), a.fg, a.bg, a.modifier,
                    b.symbol(), b.fg, b.bg, b.modifier,
                    describe(&want, y_want),
                    describe(&got, y_got),
                );
            }
        }
    });
}

fn content(s: &str) -> Value {
    json!({"type": "content_stream", "content": s})
}

#[test]
fn a_plain_streamed_answer() {
    assert_parity(
        "plain answer",
        json!([
            content("Hello "),
            content("world\n\n"),
            content("A second paragraph with **bold** and `code`.\n\n"),
            content("last words"),
        ]),
    );
}

#[test]
fn rich_markdown_in_an_answer() {
    let md = "# Title\n\nPara with *italic* and a [link](https://example.com).\n\n\
## List\n\n- one\n- two\n  - nested\n1. first\n2. second\n\n> quoted\n\n\
| Parameter | Value |\n|---|---|\n| model | qwen |\n| ctx | 32k |\n\n\
```python\ndef f(x):\n    return x  # note\n```\n\n---\n\nThe end.\n\n";
    assert_parity("markdown", json!([content(md)]));
}

#[test]
fn reasoning_then_answer() {
    assert_parity(
        "reasoning",
        json!([
            {"type": "thinking_stream", "content": "First I read the file.\n"},
            {"type": "thinking_stream", "content": "Then I fix the function carefully."},
            content("Done.\n\n"),
        ]),
    );
}

#[test]
fn a_tool_call_between_two_answers() {
    assert_parity(
        "tool flow",
        json!([
            content("Looking at it.\n\n"),
            {"type": "tool_start", "name": "read_file",
             "args": {"file_path": "agent.py", "offset": 10, "limit": 40}},
            {"type": "tool_end", "name": "read_file", "result": "line1\nline2\nline3"},
            content("It has three lines.\n\n"),
        ]),
    );
}

#[test]
fn several_tools_in_a_row() {
    assert_parity(
        "tool run",
        json!([
            {"type": "tool_start", "name": "grep_search", "args": {"query": "needle", "path": "src/app.rs"}},
            {"type": "tool_end", "name": "grep_search", "result": "match"},
            {"type": "tool_start", "name": "run_command", "args": {"command": "pytest -q"}},
            {"type": "tool_end", "name": "run_command", "result": "collected 3 items\n=== 3 passed in 0.5s ==="},
            content("All green.\n\n"),
        ]),
    );
}

#[test]
fn a_failed_tool_is_expanded() {
    assert_parity(
        "failed tool",
        json!([
            {"type": "tool_start", "name": "read_file", "args": {"file_path": "missing.py"}},
            {"type": "tool_end", "name": "read_file", "result": "Error: file not found"},
        ]),
    );
}

#[test]
fn a_diff_block() {
    assert_parity(
        "diff",
        json!([
            {"type": "tool_start", "name": "replace_in_file", "args": {"file_path": "agent.py"}},
            {"type": "diff", "file": "agent.py",
             "diff": "--- a/agent.py\n+++ b/agent.py\n@@ -1,2 +1,2 @@\n-old line\n+new line\n context"},
            {"type": "tool_end", "name": "replace_in_file", "result": "Successfully replaced"},
        ]),
    );
}

#[test]
fn errors_and_system_notices() {
    assert_parity(
        "errors",
        json!([
            {"type": "error", "content": "[System: context trimmed to fit the window]"},
            {"type": "error", "content": "boom: the provider failed"},
        ]),
    );
}

#[test]
fn a_checkpoint_line() {
    assert_parity(
        "checkpoint",
        json!([
            {"type": "checkpoint", "sha": "abc1234", "label": "before edit"},
            content("Saved.\n\n"),
        ]),
    );
}

// ── live displays ───────────────────────────────────────────────────────────
//
// The frames the REPL refreshes while a turn runs (`cli_ui._WaitingDisplay`,
// `_ThinkingDisplay`, ...), compared with the TUI's live rows built from the
// same state. The clock is frozen on the Python side and the spinner frame is
// pinned on the Rust side, so a frame is reproducible.

use argent_tui::view::{item_view, live_rows};
use std::time::{Duration, Instant};

fn live_cli(env: &Env<'_>, kind: &str, payload: &Value) -> Vec<Line<'static>> {
    let ansi: String = env
        .shim
        .getattr("cli_live")
        .unwrap()
        .call1((kind, payload.to_string(), WIDTH))
        .unwrap()
        .extract()
        .unwrap();
    parse_ansi(&ansi)
}

/// A busy app one user turn in, whose turn began `total` seconds ago, with the
/// production block renderer behind it.
fn busy_app(py: Python<'_>, env: &Env<'_>, total: u64) -> App {
    let mut app = App::new();
    app.renderer = Box::new(PyRenderer::new(env.bridge.clone_ref(py)));
    app.spinner_frame.set(Some(0));
    app.insert_str("go");
    app.submit_input();
    app.is_busy = true;
    app.turn_started = Some(Instant::now() - Duration::from_secs(total));
    app
}

fn assert_rows_equal(name: &str, want: &[Line<'static>], got: &[Line<'static>]) {
    let (w, g) = (paint(want), paint(got));
    let text = |b: &Buffer, n: usize| (0..n as u16).map(|y| row_text(b, y)).collect::<Vec<_>>();
    assert_eq!(
        text(&g, got.len()),
        text(&w, want.len()),
        "{name}: live rows differ\n--- REPL ---\n{}\n--- TUI ---\n{}",
        text(&w, want.len()).join("\n"),
        text(&g, got.len()).join("\n")
    );
    for y in 0..want.len() as u16 {
        for x in 0..WIDTH {
            let (a, b) = (&w[(x, y)], &g[(x, y)]);
            if a.symbol() == " " && b.symbol() == " " && a.bg == b.bg {
                continue;
            }
            assert!(
                a.symbol() == b.symbol() && a.fg == b.fg && a.bg == b.bg && a.modifier == b.modifier,
                "{name}: live cell ({x},{y}) differs\nREPL: {:?} fg={:?} bg={:?} {:?}\nTUI:  {:?} fg={:?} bg={:?} {:?}\nrow REPL: {}\nrow TUI:  {}",
                a.symbol(), a.fg, a.bg, a.modifier, b.symbol(), b.fg, b.bg, b.modifier,
                describe(&w, y), describe(&g, y),
            );
        }
    }
}

fn with_env(f: impl FnOnce(Python<'_>, &Env<'_>)) {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| match load(py) {
        Some(env) => f(py, &env),
        None => skip_or_fail("src.bridge.render could not be imported"),
    });
}

#[test]
fn live_waiting_for_the_first_token() {
    with_env(|py, env| {
        let app = busy_app(py, env, 8);
        let want = live_cli(env, "waiting", &json!({"total": 8}));
        assert_rows_equal("waiting", &want, &live_rows(&app, WIDTH, 40));
    });
}

#[test]
fn live_reasoning_stream() {
    with_env(|py, env| {
        let mut app = busy_app(py, env, 8);
        app.handle_event(ArgentEvent::Thinking("line one\nline two of thinking".into()));
        let want = live_cli(env, "thinking", &json!({"total": 8, "text": "line one\nline two of thinking"}));
        assert_rows_equal("thinking", &want, &live_rows(&app, WIDTH, 40));
    });
}

#[test]
fn live_rethink_stream() {
    with_env(|py, env| {
        let mut app = busy_app(py, env, 8);
        app.handle_event(ArgentEvent::Rethink("checking the edge case".into()));
        let want = live_cli(env, "rethink", &json!({"total": 8, "text": "checking the edge case"}));
        assert_rows_equal("rethink", &want, &live_rows(&app, WIDTH, 40));
    });
}

#[test]
fn live_tool_wait_line() {
    with_env(|py, env| {
        let mut app = busy_app(py, env, 8);
        let mut args = std::collections::HashMap::new();
        args.insert("file_path".to_string(), json!("agent.py"));
        args.insert("offset".to_string(), json!(10));
        args.insert("limit".to_string(), json!(40));
        app.handle_event(ArgentEvent::ToolStart { name: "read_file".into(), args });
        let want = live_cli(
            env,
            "tool_wait",
            &json!({"total": 8, "name": "read_file",
                    "args": {"file_path": "agent.py", "offset": 10, "limit": 40}}),
        );
        // The running tool is a message of its own, drawn in place.
        let view = item_view(&app, app.messages.len() - 1, WIDTH);
        let got: Vec<Line<'static>> = view.lines();
        assert_rows_equal("tool wait", &want, &got);
    });
}

#[test]
fn live_tool_generation_box() {
    with_env(|py, env| {
        let mut app = busy_app(py, env, 8);
        app.handle_event(ArgentEvent::ToolGenerating { name: "write_file".into(), bytes: 1234, delta: String::new() });
        let want = live_cli(env, "tool_gen", &json!({"total": 8, "name": "write_file", "bytes": 1234}));
        let view = item_view(&app, app.messages.len() - 1, WIDTH);
        let got: Vec<Line<'static>> = view.lines();
        assert_rows_equal("tool generation", &want, &got);
    });
}

#[test]
fn live_answer_tail_and_spinner() {
    with_env(|py, env| {
        let mut app = busy_app(py, env, 8);
        app.handle_event(ArgentEvent::Content("first paragraph\n\nan unfinished second one".into()));
        let want = live_cli(env, "tail", &json!({"total": 8, "pending": "an unfinished second one"}));
        assert_rows_equal("answer tail", &want, &live_rows(&app, WIDTH, 40));
    });
}

#[test]
fn live_queue_box() {
    with_env(|py, env| {
        let mut app = busy_app(py, env, 8);
        app.message_queue = vec!["first queued".into(), "second queued message".into()];
        let want = live_cli(env, "queue", &json!({"items": ["first queued", "second queued message"]}));
        let live = live_rows(&app, WIDTH, 40);
        // The box sits under the waiting spinner.
        assert_rows_equal("queue box", &want, &live[1..]);
    });
}

// ── post-response lines ─────────────────────────────────────────────────────

#[test]
fn the_turn_snapshot_matches_the_repls_lines() {
    with_env(|py, env| {
        let case = env.shim.getattr("snapshot_case").unwrap().call1((WIDTH,)).unwrap();
        let (cli_ansi, ev): (String, Bound<'_, PyDict>) = case.extract().unwrap();
        let want = parse_ansi(&cli_ansi);

        let mut app = App::new();
        app.renderer = Box::new(PyRenderer::new(env.bridge.clone_ref(py)));
        let parsed = parse_py_event(&ev).unwrap().expect("turn_snapshot event");
        app.handle_event(parsed);
        let view = item_view(&app, app.messages.len() - 1, WIDTH);
        let got: Vec<Line<'static>> = view.lines();
        assert_rows_equal("turn snapshot", &want, &got);
    });
}

// ── bottom toolbar ──────────────────────────────────────────────────────────

#[test]
fn the_toolbar_text_is_the_repls_toolbar() {
    with_env(|_py, env| {
        for mode in ["CHAT", "VIBE"] {
            let case = env.shim.getattr("toolbar_case").unwrap().call1((mode,)).unwrap();
            let (want, state): (String, std::collections::HashMap<String, String>) =
                case.extract().unwrap();

            let mut app = App::new();
            app.update_status(state);
            app.status.start_time = Instant::now() - Duration::from_secs(303);
            let (line, _targets) = argent_tui::ui::toolbar(&app, 400);
            let got: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

            assert_eq!(got.trim_end(), want.trim_end(), "{mode}: toolbar text differs from the REPL's");
            // The whole bar is reverse video, as prompt_toolkit's `bottom-toolbar`.
            assert!(line.spans.iter().all(|s| s.style.add_modifier.contains(ratatui::style::Modifier::REVERSED)));
        }
    });
}
