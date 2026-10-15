//! Draw a frame of the TUI as an HTML page, colours and all — a design-review tool.
//!
//! The conversation is played through the REAL bridge (a scripted agent produces
//! the answer, `ReplBridge` emits the events, the production Python renderer draws
//! every block), so what you see is what the terminal shows, not a mock-up.
//!
//! ```text
//! ARGENT_FRAME_OUT=frame.html ARGENT_FRAME_STYLE=premium ARGENT_FRAME_SCENE=chat \
//!     cargo run --example frame_html
//! ```
//!
//! `ARGENT_FRAME_STYLE`: `premium` (default) or `classic`.
//! `ARGENT_FRAME_SCENE`: `chat` (an answer on screen), `menu` (suggestions open and
//! Tab pressed once), `typing` (text in the prompt), `scrolled` (back in the chat
//! with the "newest" chip), `select` (a mouse selection over the answer).
//! `ARGENT_FRAME_W` / `ARGENT_FRAME_H`: the terminal size (default 110 x 46).

use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use argent_tui::app::{App, CompletionItem};
use argent_tui::events::ArgentEvent;
use argent_tui::pyrender::PyRenderer;
use argent_tui::sink::RustEventSink;
use argent_tui::theme::UiConfig;
use argent_tui::ui::draw_ui;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};
use ratatui::Terminal;

const SHIM: &str = r##"
import os, sys

def setup(root):
    sys.path.insert(0, root)
    site = os.path.join(root, ".venv", "Lib", "site-packages")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)

ANSWER = """# Что изменилось

Я разобрал **меню инвентаря** и поправил три места. Главное — `InventoryMenu.cs`:

## Причина

Список строился до того, как загружались данные, поэтому он оставался пустым.

### Что сделано
- убрал преждевременный вызов `Build()`
- добавил подписку на событие `OnLoaded`
- [x] проверил в редакторе
- [ ] прогнать тесты

> Важно: порядок инициализации теперь задаёт `Bootstrap`, а не сцена.

```csharp
void OnLoaded()
{
    Build(items);
    Refresh();
}
```

| Файл | Строк | Статус |
|------|-------|--------|
| InventoryMenu.cs | 42 | изменён |
| Bootstrap.cs | 7 | новый |

---

Готово. Запустите сцену и проверьте вкладку «Предметы».
"""

def make_bridge(startup=False):
    import config
    config.get_mcp_servers = lambda: [{"name": "kern", "command": "x"}] if startup else []
    config.get_telegram_enabled = lambda: False
    config.get_auto_kb = lambda: False
    from hook_manager import hook_manager
    hook_manager.call_hook = lambda *a, **k: None
    if startup:
        # A previous session to offer, and a config file so this is not a first run.
        import src.startup as st
        meta = {"id": "s1", "saved_at": "2026-10-06T22:00:00", "preview": "почини меню инвентаря", "cwd": os.getcwd()}
        saved = {"id": "s1", "saved_at": meta["saved_at"], "model": "", "cwd": meta["cwd"],
                 "messages": [{"role": "system", "content": "s"},
                              {"role": "user", "content": "почини меню инвентаря"},
                              {"role": "assistant", "content": "Готово, меню починено."}]}
        st.get_last_session = lambda: meta
        st.load_session = lambda sid: saved
        st.carried_over_memory = lambda: ["цель: починить меню"]
        if not config.CONFIG_FILE.exists():
            config.CONFIG_FILE.write_text("{}", encoding="utf-8")
        from mcp_client import mcp_client
        mcp_client.register_and_start = lambda name, cfg: "Started " + name

    class Agent:
        model_name = "glm-5.1"
        provider = "zai"
        messages = []
        session_id = None
        _turn_tool_calls = 0
        _turn_local_agent_calls = 0

        def process_user_input(self, text, allowed_tools=None):
            yield {"type": "thinking_stream", "content": "Нужно понять, почему список пуст."}
            yield {"type": "content_stream", "content": "Сначала посмотрю, как строится список, и найду вызовы `Build()`.\n\n"}
            yield {"type": "tool_start", "name": "read_file",
                   "args": {"file_path": "Assets/InventoryMenu.cs", "offset": 1, "limit": 60}}
            yield {"type": "tool_end", "name": "read_file", "result": "using UnityEngine;\nclass InventoryMenu {}",
                   "duration_secs": 0.028}
            yield {"type": "tool_start", "name": "grep_search", "args": {"query": "Build(", "path": "Assets"}}
            yield {"type": "tool_end", "name": "grep_search",
                   "result": "Assets/InventoryMenu.cs:12: Build(items);\nAssets/Bootstrap.cs:7: Build(items);",
                   "duration_secs": 0.31}
            yield {"type": "tool_start", "name": "run_command", "args": {"command": "dotnet test --no-build"}}
            yield {"type": "tool_end", "name": "run_command",
                   "result": "Error: exit code 1\nInventoryMenuTests.Build_EmptyList failed\n  Expected: 3\n  Actual: 0",
                   "duration_secs": 4.2}
            yield {"type": "thinking_stream", "content": "Тест падает из-за порядка инициализации. Исправлю подписку."}
            yield {"type": "tool_start", "name": "replace_in_file",
                   "args": {"file_path": "Assets/InventoryMenu.cs"}}
            yield {"type": "tool_end", "name": "replace_in_file", "result": "Successfully applied", "duration_secs": 0.04}
            yield {"type": "diff", "file": "Assets/InventoryMenu.cs",
                   "diff": "--- a/Assets/InventoryMenu.cs\n+++ b/Assets/InventoryMenu.cs\n@@ -10,7 +10,8 @@\n void Start()\n {\n-    Build(items);\n+    Bootstrap.OnLoaded += OnLoaded;\n+    Refresh();\n }\n"}
            for i in range(0, len(ANSWER), 24):
                yield {"type": "content_stream", "content": ANSWER[i:i + 24]}
            yield {"type": "usage", "data": {"prompt": 5200, "completion": 640}}

        def get_context_usage(self):
            return {"tokens": 18400, "max": 128000, "percent": 14.4}

        def get_context_breakdown(self):
            return {}

        def request_stop(self): pass
        def repair_history(self): pass

    from src.bridge.repl_bridge import ReplBridge
    return ReplBridge(agent=Agent())
"##;

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

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

// ── colours ─────────────────────────────────────────────────────────────────

const ANSI16: [&str; 16] = [
    "#484f58", "#ff7b72", "#3fb950", "#d29922", "#58a6ff", "#bc8cff", "#39c5cf", "#b1bac4",
    "#6e7681", "#ffa198", "#56d364", "#e3b341", "#79c0ff", "#d2a8ff", "#56d4dd", "#f0f6fc",
];
const PAGE_FG: &str = "#c9d1d9";
const PAGE_BG: &str = "#0d1117";

fn hex(c: Color) -> Option<String> {
    Some(match c {
        Color::Reset => return None,
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        Color::Black => ANSI16[0].into(),
        Color::Red => ANSI16[1].into(),
        Color::Green => ANSI16[2].into(),
        Color::Yellow => ANSI16[3].into(),
        Color::Blue => ANSI16[4].into(),
        Color::Magenta => ANSI16[5].into(),
        Color::Cyan => ANSI16[6].into(),
        Color::Gray => ANSI16[7].into(),
        Color::DarkGray => ANSI16[8].into(),
        Color::LightRed => ANSI16[9].into(),
        Color::LightGreen => ANSI16[10].into(),
        Color::LightYellow => ANSI16[11].into(),
        Color::LightBlue => ANSI16[12].into(),
        Color::LightMagenta => ANSI16[13].into(),
        Color::LightCyan => ANSI16[14].into(),
        Color::White => ANSI16[15].into(),
        Color::Indexed(n) if n < 16 => ANSI16[n as usize].into(),
        Color::Indexed(n) if n >= 232 => {
            let v = 8 + (n - 232) * 10;
            format!("#{v:02x}{v:02x}{v:02x}")
        }
        Color::Indexed(n) => {
            let n = n - 16;
            let level = |i: u8| if i == 0 { 0 } else { 55 + i * 40 };
            format!("#{:02x}{:02x}{:02x}", level(n / 36), level((n / 6) % 6), level(n % 6))
        }
    })
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn to_html(buf: &Buffer, cursor: Option<(u16, u16)>) -> String {
    let mut out = String::new();
    for y in 0..buf.area.height {
        let mut run = String::new();
        let mut cur: Option<(String, bool)> = None;
        let flush = |run: &mut String, style: &Option<(String, bool)>, out: &mut String| {
            if run.is_empty() {
                return;
            }
            match style {
                Some((css, _)) if !css.is_empty() => {
                    out.push_str(&format!("<span style=\"{css}\">{}</span>", escape(run)));
                }
                _ => out.push_str(&escape(run)),
            }
            run.clear();
        };
        for x in 0..buf.area.width {
            let cell = &buf[(x, y)];
            let mut fg = hex(cell.fg);
            let mut bg = hex(cell.bg);
            if cell.modifier.contains(Modifier::REVERSED) {
                let f = fg.clone().unwrap_or_else(|| PAGE_FG.to_string());
                let b = bg.clone().unwrap_or_else(|| PAGE_BG.to_string());
                fg = Some(b);
                bg = Some(f);
            }
            let mut css = String::new();
            if let Some(f) = &fg {
                css.push_str(&format!("color:{f};"));
            }
            if let Some(b) = &bg {
                css.push_str(&format!("background:{b};"));
            }
            if cell.modifier.contains(Modifier::BOLD) {
                css.push_str("font-weight:700;");
            }
            if cell.modifier.contains(Modifier::ITALIC) {
                css.push_str("font-style:italic;");
            }
            if cell.modifier.contains(Modifier::UNDERLINED) {
                css.push_str("text-decoration:underline;");
            }
            if cell.modifier.contains(Modifier::DIM) {
                css.push_str("opacity:.62;");
            }
            if cursor == Some((x, y)) {
                css.push_str("outline:1px solid #f0f6fc;");
            }
            let key = Some((css, false));
            if cur != key {
                flush(&mut run, &cur, &mut out);
                cur = key;
            }
            run.push_str(cell.symbol());
        }
        flush(&mut run, &cur, &mut out);
        out.push('\n');
    }
    out
}

fn page(title: &str, body: &str) -> String {
    format!(
        "<!doctype html><meta charset=\"utf-8\"><title>{title}</title>\
<style>body{{margin:0;background:#010409;padding:18px}}\
.term{{display:inline-block;background:{PAGE_BG};color:{PAGE_FG};padding:6px 0;\
border:1px solid #30363d;border-radius:8px;overflow:hidden}}\
pre{{margin:0;font:14px/1.22 'Cascadia Mono','JetBrains Mono',Consolas,monospace;white-space:pre}}\
h1{{font:600 13px system-ui;color:#8b949e;margin:0 0 10px}}</style>\
<h1>{title}</h1><div class=\"term\"><pre>{body}</pre></div>"
    )
}

fn main() {
    let style = env_or("ARGENT_FRAME_STYLE", "premium");
    let scene = env_or("ARGENT_FRAME_SCENE", "chat");
    let width: u16 = env_or("ARGENT_FRAME_W", "110").parse().unwrap_or(110);
    let height: u16 = env_or("ARGENT_FRAME_H", "46").parse().unwrap_or(46);
    let out_path = env_or("ARGENT_FRAME_OUT", "frame.html");
    let startup = scene == "startup";
    std::env::set_var("ARGENT_UI_STYLE", &style);
    if !startup {
        std::env::set_var("ARGENT_NO_STARTUP_PROMPTS", "1");
    }
    std::env::set_var("ARGENT_DISABLE_AUTOMATIONS", "1");
    // Nothing here may touch the real home: settings, sessions and logs all live there.
    let home = std::env::temp_dir().join("argent-frame-home");
    std::fs::create_dir_all(&home).expect("a scratch home");
    std::env::set_var("USERPROFILE", &home);
    std::env::set_var("HOME", &home);

    pyo3::prepare_freethreaded_python();
    let app = Python::with_gil(|py| -> Result<App, String> {
        let shim = PyModule::from_code_bound(py, SHIM, "frame_shim.py", "frame_shim").map_err(|e| e.to_string())?;
        shim.getattr("setup").and_then(|f| f.call1((project_root(),))).map_err(|e| e.to_string())?;
        let bridge = shim.getattr("make_bridge").and_then(|f| f.call1((startup,))).map_err(|e| e.to_string())?.unbind();

        let (tx, rx) = channel();
        let rx = std::sync::Mutex::new(rx);
        let sink = Py::new(py, RustEventSink { tx }).map_err(|e| e.to_string())?;
        bridge.call_method1(py, "set_event_sink", (sink,)).map_err(|e| e.to_string())?;

        let mut app = App::new();
        app.renderer = Box::new(PyRenderer::new(bridge.clone_ref(py)));
        let raw: String = bridge
            .call_method0(py, "get_ui_config")
            .and_then(|v| v.extract(py))
            .map_err(|e| e.to_string())?;
        app.ui = UiConfig::from_json(&raw);
        app.spinner_frame.set(Some(0));

        let comps: Vec<std::collections::HashMap<String, String>> = bridge
            .call_method0(py, "get_command_completions")
            .and_then(|v| v.extract(py))
            .unwrap_or_default();
        for item in comps {
            let name = item.get("name").cloned().unwrap_or_default();
            let help = item.get("help").cloned().unwrap_or_default();
            if !name.is_empty() {
                app.completions.push(CompletionItem { name, help });
            }
        }
        if let Ok(state) = bridge
            .call_method0(py, "get_state")
            .and_then(|v| v.extract::<std::collections::HashMap<String, String>>(py))
        {
            app.update_status(state);
        }

        if startup {
            // The TUI opens with the question; stop while it is on screen. Nothing else
            // (the MCP servers least of all) may have been announced by then.
            let deadline = Instant::now() + Duration::from_secs(20);
            while app.active_select.is_none() && Instant::now() < deadline {
                let ev = py.allow_threads(|| rx.lock().ok().and_then(|r| r.recv_timeout(Duration::from_millis(100)).ok()));
                if let Some((at, ev)) = ev {
                    app.handle_event_at(ev, at);
                }
            }
            let Some(select) = app.active_select.as_ref() else {
                return Err("the start-up question never came".into());
            };
            eprintln!("question: {} {:?}", select.title, select.options);
            return Ok(app);
        }

        let prompt = "почини меню инвентаря: список пустой";
        bridge.call_method1(py, "start_turn", (prompt,)).map_err(|e| e.to_string())?;
        app.insert_str(prompt);
        app.submit_input();
        app.is_busy = true;
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut done = false;
        while !done && Instant::now() < deadline {
            let ev = py.allow_threads(|| rx.lock().ok().and_then(|r| r.recv_timeout(Duration::from_millis(100)).ok()));
            if let Some((at, ev)) = ev {
                done = matches!(ev, ArgentEvent::Done);
                app.handle_event_at(ev, at);
            }
        }
        if !done {
            return Err("the scripted turn never finished".into());
        }
        let _ = bridge.call_method0(py, "cleanup");
        Ok(app)
    });
    let mut app = match app {
        Ok(a) => a,
        Err(e) => {
            eprintln!("frame_html: {e}");
            std::process::exit(2);
        }
    };

    // The scripted agent has no real provider; show what a session would.
    app.status.provider = "zai".into();
    app.status.model = "glm-5.1".into();
    app.status.tier = "cloud".into();

    let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
    let draw = |app: &App, term: &mut Terminal<TestBackend>| {
        term.draw(|f| draw_ui(f, app)).unwrap();
    };
    draw(&app, &mut term); // the first frame teaches the view where its rows are

    match scene.as_str() {
        "menu" => {
            app.insert_str("/");
            app.insert_char('s');
            app.cycle_completion(true, false);
            app.cycle_completion(true, false);
        }
        "typing" => app.insert_str("а теперь добавь тест на пустой инвентарь\nи проверь вкладку «Предметы»"),
        "scrolled" => app.scroll_up(14),
        "select" => {
            let rows = app.view_text.borrow().rows.clone();
            let first = rows.iter().position(|r| r.contains("убрал")).unwrap_or(4) as u16;
            app.select_cells((4, first), (30, first + 2));
        }
        _ => {}
    }
    draw(&app, &mut term);
    draw(&app, &mut term);

    let buf = term.backend().buffer().clone();
    let cursor = term.get_cursor_position().ok().map(|p| (p.x, p.y));
    let title = format!("Argent TUI · {style} · {scene} · {width}x{height}");
    let html = page(&title, &to_html(&buf, cursor));
    std::fs::write(&out_path, html).expect("write the page");
    println!("{out_path}");
}
