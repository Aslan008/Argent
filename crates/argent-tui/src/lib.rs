// `useless_conversion` fires on the `PyResult` -> `PyResult` rewrite that the
// `#[pyfunction]` / `#[pymethods]` macros generate, not on any real code here.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods};
use std::collections::HashMap;
use std::io::stdout;
use std::sync::mpsc::channel;
use std::time::Duration;

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableFocusChange, DisableMouseCapture,
        EnableBracketedPaste, EnableFocusChange, EnableMouseCapture, Event, KeyCode,
        KeyEvent, KeyEventKind, KeyModifiers,
        KeyboardEnhancementFlags, MouseButton, MouseEventKind, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

pub mod ansi;
pub mod app;
pub mod cards;
pub mod events;
pub mod markdown;
pub mod premium;
pub mod pyrender;
pub mod renderer;
pub mod selection;
pub mod sink;
pub mod speed;
pub mod stream;
pub mod theme;
pub mod toolrender;
pub mod ui;
pub mod view;

/// Where diagnostic lines go: `.argent/tui-diagnostics.log`, or the file named by
/// `ARGENT_TUI_DIAG` (used by tests so they never write into the project).
fn diag_path() -> std::path::PathBuf {
    match std::env::var("ARGENT_TUI_DIAG") {
        Ok(custom) => std::path::PathBuf::from(custom),
        Err(_) => std::path::Path::new(".argent").join("tui-diagnostics.log"),
    }
}

/// Append one line. Returns the error instead of hiding it, so a test can tell a
/// working writer from a silent no-op.
pub fn diag_to(path: &std::path::Path, text: &str) -> std::io::Result<()> {
    use std::io::Write;

    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.len() > 512 * 1024 {
            let _ = std::fs::remove_file(path);
        }
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    writeln!(file, "t={secs} {text}")
}

fn diag(text: &str) {
    // Best effort: a diagnostics line must never be able to break a turn.
    let _ = diag_to(&diag_path(), text);
}

#[cfg(test)]
mod diag_tests {
    use super::diag_to;

    /// `target/` belongs to this build and is always writable; `%TEMP%` is not, and a
    /// diagnostics test that cannot create its own file proves nothing.
    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("diag-tests");
        std::fs::create_dir_all(&dir).expect("target доступен для записи");
        dir.join(name)
    }

    #[test]
    fn a_line_is_written_and_can_be_read_back() {
        let path = scratch("unit.log");
        let _ = std::fs::remove_file(&path);

        diag_to(&path, "alive tick=7 busy=false").expect("строка пишется");
        diag_to(&path, "resize 120x40").expect("и вторая тоже");

        let text = std::fs::read_to_string(&path).expect("файл читается");
        let _ = std::fs::remove_file(&path);

        assert!(text.contains("alive tick=7 busy=false"), "{text}");
        assert!(text.contains("resize 120x40"), "дописывается, а не перезаписывается: {text}");
        assert!(text.contains("t="), "у каждой строки есть метка времени: {text}");
    }

    #[test]
    fn a_file_that_cannot_be_written_reports_instead_of_panicking() {
        // A directory where a file should be: the writer must say so, quietly.
        let dir = scratch("blocked");
        std::fs::create_dir_all(&dir).expect("каталог создан");
        assert!(diag_to(&dir, "x").is_err());
        let _ = std::fs::remove_dir(&dir);
    }
}

use app::{App, ClickAction, CompletionItem, MessageItem, QueueTrigger, Submit};
use events::ArgentEvent;

/// How long a burst of keys may pause, in ms, and still be one paste (a long paste reaches the
/// console in chunks).
const PASTE_GAP_MS: u64 = 15;

/// The text a key stands for when it is part of a pasted burst: plain characters, Enter and Tab.
fn paste_char(key: &KeyEvent) -> Option<char> {
    if key.kind == KeyEventKind::Release {
        return None;
    }
    if !(key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT) {
        return None;
    }
    match key.code {
        KeyCode::Enter => Some('\n'),
        KeyCode::Tab => Some('\t'),
        KeyCode::Char(c) if c == '\r' || c == '\n' || c == '\t' || !c.is_control() => Some(c),
        _ => None,
    }
}

/// Three keys or more, one right after another, are a paste; two (a letter and Enter) are a person.
fn is_paste_burst(burst: &[KeyEvent]) -> bool {
    burst.len() >= 3
}

#[cfg(test)]
mod paste_tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn enter_in_a_burst_is_a_newline_not_a_submit() {
        let burst = [key(KeyCode::Char('a')), key(KeyCode::Enter), key(KeyCode::Char('b'))];
        assert!(is_paste_burst(&burst));
        let text: String = burst.iter().filter_map(paste_char).collect();
        assert_eq!(text, "a\nb");
    }

    #[test]
    fn a_letter_and_enter_are_typing() {
        assert!(!is_paste_burst(&[key(KeyCode::Char('a')), key(KeyCode::Enter)]));
    }

    #[test]
    fn shortcuts_are_not_paste_keys() {
        let ctrl = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(paste_char(&ctrl), None);
        let shift = KeyEvent::new(KeyCode::Char('Я'), KeyModifiers::SHIFT);
        assert_eq!(paste_char(&shift), Some('Я'));
        assert_eq!(paste_char(&key(KeyCode::Up)), None);
    }
}
use sink::RustEventSink;

/// The Latin key a Cyrillic letter sits on, so shortcuts work in a Russian layout.
///
/// A keyboard sends the character the layout produces, not the key's position:
/// Ctrl+C in ЙЦУКЕН arrives as `с`, and a `KeyCode::Char('c')` arm never matches.
/// Anything that is not Cyrillic is returned unchanged.
fn latin_of(ch: char) -> char {
    match ch.to_lowercase().next().unwrap_or(ch) {
        'й' => 'q', 'ц' => 'w', 'у' => 'e', 'к' => 'r', 'е' => 't', 'н' => 'y',
        'г' => 'u', 'ш' => 'i', 'щ' => 'o', 'з' => 'p', 'х' => '[', 'ъ' => ']',
        'ф' => 'a', 'ы' => 's', 'в' => 'd', 'а' => 'f', 'п' => 'g', 'р' => 'h',
        'о' => 'j', 'л' => 'k', 'д' => 'l', 'ж' => ';', 'э' => '\'',
        'я' => 'z', 'ч' => 'x', 'с' => 'c', 'м' => 'v', 'и' => 'b', 'т' => 'n',
        'ь' => 'm', 'б' => ',', 'ю' => '.', 'ё' => '`',
        other => other,
    }
}

/// Put the clipboard into the prompt.
///
/// A picture on the clipboard becomes an attachment: the pixels go to Python,
/// which encodes them the way `view_image` does and queues them for the next turn
/// (`vision.queue_image`). Anything else is pasted as text, exactly as Ctrl+V
/// behaved before images were supported.
fn paste_clipboard(app: &mut App, clipboard: &mut Option<arboard::Clipboard>, bridge: &PyObject) {
    let Some(clip) = clipboard.as_mut() else {
        return;
    };

    if let Ok(img) = clip.get_image() {
        if img.width > 0 && img.height > 0 {
            let (w, h) = (img.width, img.height);
            let rgba = img.bytes.to_vec();
            let reply = Python::with_gil(|py| -> Option<String> {
                let bytes = PyBytes::new_bound(py, &rgba);
                match bridge.call_method1(py, "queue_clipboard_image", (w, h, bytes)) {
                    Ok(value) => value.extract::<String>(py).ok(),
                    Err(err) => {
                        err.print(py);
                        None
                    }
                }
            });
            // The bridge answers with the size on success, with the reason on
            // failure; either way the user must see it, or a paste looks ignored.
            if let Some(reply) = reply {
                let note = if reply.contains('×') {
                    format!("Скриншот прикреплён к следующему сообщению: {reply}")
                } else {
                    format!("Не удалось вставить изображение: {reply}")
                };
                app.messages.push(MessageItem::Notice(note));
                // The strip above the prompt draws what the bridge holds, so the
                // list is read now instead of at the next 2-second poll.
                pull_state(app, bridge);
                return;
            }
        }
    }

    if let Ok(text) = clip.get_text() {
        if !text.is_empty() {
            app.insert_str(&text.replace("\r\n", "\n"));
        }
    }
}

/// Hand a path to the system's default application, and say so if it refuses.
///
/// One place for both the picture card and the artifact card: the bridge answers
/// with an empty string on success and with the reason on failure, and a silent
/// failure would look like the click did nothing.
fn open_path_in_system(app: &mut crate::app::App, bridge: &PyObject, path: String) {
    if path.is_empty() {
        return;
    }
    let answer = Python::with_gil(|py| {
        bridge
            .call_method1(py, "open_path_externally", (path,))
            .ok()
            .and_then(|v| v.extract::<String>(py).ok())
    });
    if let Some(message) = answer.filter(|m| !m.is_empty()) {
        app.messages.push(crate::app::MessageItem::Notice(message));
    }
}

/// Read the bridge's state once: the toolbar's numbers, the attachments and the
/// queue all come from it, and a click must show its result at once rather than
/// at the next two-second poll.
fn pull_state(app: &mut App, bridge: &PyObject) {
    Python::with_gil(|py| {
        let Ok(state_obj) = bridge.call_method0(py, "get_state") else {
            return;
        };
        let Ok(dict) = state_obj.extract::<Bound<'_, PyDict>>(py) else {
            return;
        };
        let mut state_map = HashMap::new();
        for (k, v) in dict.iter() {
            let k_str: String = k.extract().unwrap_or_default();
            let v_str = v.extract::<String>().unwrap_or_else(|_| v.to_string());
            state_map.insert(k_str, v_str);
        }
        app.update_status(state_map);
    });
}

#[pyfunction]
fn run_tui(py: Python, bridge: PyObject) -> PyResult<()> {
    // 1. Setup in-memory event channel and sink
    let (tx, rx) = channel();
    let sink = Py::new(py, RustEventSink { tx: tx.clone() })?;
    bridge.call_method1(py, "set_event_sink", (sink,))?;

    // Toolbar state is pulled on its own thread. Doing it here, on the drawing
    // thread, meant taking the GIL — and the GIL belongs to the agent's worker
    // while it thinks, streams and parses. The interface therefore froze solid
    // during long turns: no keys, no wheel, nothing, until the worker let go.
    // Now the worker's Python work can take as long as it likes; this thread only
    // reads whatever the poller last managed to fetch.
    let poll_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poll_stop_thread = poll_stop.clone();
    let poll_tx = tx.clone();
    let poll_bridge = bridge.clone_ref(py);
    let poller = std::thread::spawn(move || {
        use std::sync::atomic::Ordering::Relaxed;
        while !poll_stop_thread.load(Relaxed) {
            std::thread::sleep(Duration::from_millis(1000));
            if poll_stop_thread.load(Relaxed) {
                break;
            }
            let map = Python::with_gil(|py| -> Option<HashMap<String, String>> {
                let state_obj = poll_bridge.call_method0(py, "get_state").ok()?;
                let dict = state_obj.extract::<Bound<'_, PyDict>>(py).ok()?;
                let mut state_map = HashMap::new();
                for (k, v) in dict.iter() {
                    let k_str: String = k.extract().unwrap_or_default();
                    let v_str = v.extract::<String>().unwrap_or_else(|_| v.to_string());
                    state_map.insert(k_str, v_str);
                }
                Some(state_map)
            });
            match map {
                Some(state_map) => {
                    if poll_tx
                        .send((std::time::Instant::now(), ArgentEvent::State(state_map)))
                        .is_err()
                    {
                        break;      // the UI is gone
                    }
                }
                None => {
                    if poll_tx.send((std::time::Instant::now(), ArgentEvent::State(HashMap::new()))).is_err() {
                        break;
                    }
                }
            }
        }
    });

    struct PollerGuard {
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }
    impl Drop for PollerGuard {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }
    let _poller_guard = PollerGuard { stop: poll_stop.clone(), handle: Some(poller) };

    // 2. Load initial state, completions, and persistent history
    let mut app = App::new();

    // Static blocks are rendered by the bridge with the REPL's own code, and the
    // colours the native widgets use come from the same theme.yaml.
    app.renderer = Box::new(pyrender::PyRenderer::new(bridge.clone_ref(py)));
    if let Ok(raw) = bridge
        .call_method0(py, "get_ui_config")
        .and_then(|v| v.extract::<String>(py))
    {
        app.ui = theme::UiConfig::from_json(&raw);
    }

    if let Ok(state_obj) = bridge.call_method0(py, "get_state") {
        if let Ok(dict) = state_obj.extract::<Bound<'_, PyDict>>(py) {
            let mut state_map = HashMap::new();
            for (k, v) in dict.iter() {
                let k_str: String = k.extract().unwrap_or_default();
                let v_str = v.to_string();
                state_map.insert(k_str, v_str);
            }
            app.update_status(state_map);
        }
    }

    if let Ok(comps_obj) = bridge.call_method0(py, "get_command_completions") {
        if let Ok(list) = comps_obj.extract::<Vec<HashMap<String, String>>>(py) {
            for item in list {
                let name = item.get("name").cloned().unwrap_or_default();
                let help = item.get("help").cloned().unwrap_or_default();
                if !name.is_empty() {
                    app.completions.push(CompletionItem { name, help });
                }
            }
        }
    }

    // `/queue` is answered by this frontend (the queue is Rust state), so the
    // bridge's registry cannot list it — and a command nobody can find is not a
    // command. It rides in the completion list with the rest.
    if !app.completions.iter().any(|c| c.name == "/queue") {
        app.completions.push(CompletionItem {
            name: "/queue".to_string(),
            help: "Очередь: list | now <№> — отправить сейчас | rm <№> | edit <№> <текст> | add <текст> | in 10m <текст> | on error[:текст] <текст> | on success <текст> | on stuck <текст> | clear".to_string(),
        });
    }

    if let Ok(hist_obj) = bridge.call_method0(py, "get_history") {
        if let Ok(list) = hist_obj.extract::<Vec<String>>(py) {
            app.input_history = list;
        }
    }

    // Learn the level cycle the bridge supports (defaults are used if this fails).
    if let Ok(levels_obj) = bridge.call_method0(py, "get_thinking_levels") {
        if let Ok(levels) = levels_obj.extract::<Vec<String>>(py) {
            app.set_thinking_levels(levels);
        }
    }

    // 3. Build the terminal BEFORE mutating global terminal state. If this
    //    fails there is nothing to restore; if anything after it fails the
    //    TerminalGuard below unwinds the terminal back to a usable state.
    // Without this the OS tab/window keeps the command line that launched it —
    // `C:\Windows\system32\cmd.exe` — which is the first thing anyone sees and
    // says nothing about what runs inside.
    let _ = execute!(std::io::stdout(), crossterm::terminal::SetTitle("Argent"));

    let stdout = stdout();
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    struct TerminalGuard {
        terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
        /// Whether DISAMBIGUATE_ESCAPE_CODES was actually pushed. Only pop it
        /// if it was, otherwise Drop errors on consoles that never supported it.
        keyboard_enhancement: bool,
        mouse_capture: bool,
    }

    impl TerminalGuard {
        /// Re-assert mouse reporting without touching the flag.
        ///
        /// Restoring a minimized window (or switching away and back) can leave the
        /// terminal with mouse reporting switched off while this process still
        /// believes it is on: the wheel then arrives as arrow keys and clicks never
        /// arrive at all. Writing the mode again is idempotent and costs one escape
        /// sequence, so it is safe to do on every resize and every regain of focus.
        fn ensure_mouse_capture(&mut self) -> Result<(), String> {
            if self.mouse_capture {
                execute!(self.terminal.backend_mut(), EnableMouseCapture).map_err(|e| e.to_string())?;
            }
            Ok(())
        }

        fn toggle_mouse_capture(&mut self) -> Result<bool, String> {
            if self.mouse_capture {
                execute!(self.terminal.backend_mut(), DisableMouseCapture).map_err(|e| e.to_string())?;
                self.mouse_capture = false;
                Ok(false)
            } else {
                execute!(self.terminal.backend_mut(), EnableMouseCapture).map_err(|e| e.to_string())?;
                self.mouse_capture = true;
                Ok(true)
            }
        }
    }

    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            let _ = disable_raw_mode();
            if self.keyboard_enhancement {
                let _ = execute!(
                    self.terminal.backend_mut(),
                    PopKeyboardEnhancementFlags
                );
            }
            let _ = execute!(
                self.terminal.backend_mut(),
                LeaveAlternateScreen,
                DisableBracketedPaste,
                DisableFocusChange,
                DisableMouseCapture
            );
            let _ = self.terminal.show_cursor();
        }
    }

    // 4. Run main loop with GIL released
    let res = py.allow_threads(move || -> Result<(), String> {
        let mut guard = TerminalGuard { terminal, keyboard_enhancement: false, mouse_capture: false };

        enable_raw_mode().map_err(|e| e.to_string())?;
        execute!(
            guard.terminal.backend_mut(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            EnableFocusChange
        )
        .map_err(|e| e.to_string())?;

        // Keyboard enhancement is OPTIONAL and unsupported on the legacy Windows
        // console API ("Keyboard progressive enhancement not implemented for the
        // legacy Windows API."). Without it Ctrl+arrow and friends arrive in
        // their legacy form, which is degraded but usable — so never let it abort
        // startup, and never try to pop a flag we failed to push.
        guard.keyboard_enhancement = execute!(
            guard.terminal.backend_mut(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok();

        let mut clipboard = arboard::Clipboard::new().ok();
        let mut last_ctrl_c: Option<std::time::Instant> = None;
    // Throttle for the presence heartbeat: one call a second is enough to prove a
    // human is here, and a keystroke storm must not cross the GIL per character.
    let mut last_presence: Option<std::time::Instant> = None;
        let mut last_draw = std::time::Instant::now() - Duration::from_secs(1);

        // The mouse wheel scrolls the transcript and a click can open a block, so
        // capture is on from the start; Shift+drag still selects with the
        // terminal's own selection, and Alt+M / F2 hands the mouse back.
        match guard.toggle_mouse_capture() {
            Ok(_) => app.mouse_capture_enabled = true,
            Err(e) => {
                // Without capture the console turns a wheel notch into Up/Down, and
                // those walk the input history: the transcript then sits still while
                // the prompt fills with old messages. Say so instead of letting it
                // look as if the wheel were bound to the prompt.
                app.messages.push(MessageItem::Notice(format!(
                    "Захват мыши не включился ({e}): колесо будет листать историю ввода, а не чат. Alt+M — попробовать снова."
                )));
            }
        }

        // Events read ahead while looking for a pasted burst and not consumed yet.
        let mut backlog: std::collections::VecDeque<Event> = std::collections::VecDeque::new();

        // The terminal can drop mouse reporting without telling us — minimizing and
        // restoring the window is enough. The fingerprint is unmistakable: the console
        // turns a wheel notch into Up/Down, so the wheel scrolls the prompt's history
        // while the chat sits still, and no click ever arrives. Re-asserting the mode
        // is idempotent, so doing it on a timer costs one escape sequence and makes the
        // loss self-healing instead of needing a restart.
        let mut last_capture_check = std::time::Instant::now();
        // One line a minute: it says whether the loop itself was alive when a report
        // says "everything froze". Both are cheap, and both are the only evidence.
        let mut last_heartbeat = std::time::Instant::now();

        loop {
            app.tick = app.tick.wrapping_add(1);

            // The toolbar's numbers arrive from the poller thread as State events; this
            // loop no longer calls Python for them, so it cannot be blocked by the worker.

            // Drain all pending events from Python
            while let Ok((at, event)) = rx.try_recv() {
                let is_done = matches!(event, ArgentEvent::Done);
                // A queued message can be waiting for exactly this: a failure to
                // recover from, or a successful end to report on. Both are read
                // before the event is consumed, because the text is the trigger.
                let mut queue_moved = false;
                if let ArgentEvent::Error { text, .. } = &event {
                    app.turn_had_error = true;
                    queue_moved = app.fire_error_conditions(text) > 0;
                }
                app.handle_event_at(event, at);
                if is_done {
                    // "On success" means the turn really succeeded: a turn that
                    // reported an error on its way out is not one.
                    if !app.turn_had_error && app.release_queued(QueueTrigger::Success, "") > 0 {
                        queue_moved = true;
                    }
                    app.turn_had_error = false;
                    app.dirty.set(true);
                    // The toolbar catches up on the next State event (≤ 1 s); fetching it
                    // here meant a GIL wait of up to hundreds of milliseconds on the very
                    // thread that reads the keyboard.
                }
                if queue_moved {
                    app.dirty.set(true);
                }
            }

            // Time-based queue conditions, and the stuck check: a message queued
            // "in 10m" or "when this step hangs" has nothing else to wake it.
            if app.tick_queue() > 0 {
                app.dirty.set(true);
            }

            app.drop_stale_selection();

            // A command handler asked us to quit (/exit). Run the bridge's
            // shutdown path (session save, MCP/Telegram/browser teardown) exactly
            // once, then leave the loop.
            if app.should_quit {
                Python::with_gil(|py| {
                    let _ = bridge.call_method0(py, "cleanup");
                });
                break;
            }
            // A queued message was asked to go out immediately (⏎ on its row, or
            // `/queue now <n>`). The turn in the way has to end first, and this is the
            // only side holding the bridge to say so. `is_cancelling` keeps the
            // request from cancelling a turn that is already unwinding.
            if app.start_queued_now {
                app.start_queued_now = false;
                if app.is_busy && !app.is_cancelling {
                    app.is_cancelling = true;
                    Python::with_gil(|py| {
                        let _ = bridge.call_method0(py, "cancel_turn");
                    });
                    app.dirty.set(true);
                }
            }

            // Check if agent finished and there are queued messages to run
            if !app.is_busy {
                if let Some(next_prompt) = app.pop_next_queued_message() {
                    app.dirty.set(true);
                    let mut rejected = false;
                    Python::with_gil(|py| {
                        let _ = bridge.call_method1(py, "start_turn", (next_prompt.clone(),));
                        rejected = bridge
                            .call_method0(py, "turn_was_rejected")
                            .and_then(|f| f.extract::<bool>(py))
                            .unwrap_or(false);
                    });
                    if rejected {
                        app.requeue_rejected_message(next_prompt);
                    }
                }
            }

            // The background window follows its process: a copy of the output tail
            // every 600 ms while it is open. A copy — `read_background_command`
            // drains the queue, and draining it here would take the output away from
            // the agent that may be waiting for it.
            if app.bg_view.is_some() {
                let now = std::time::Instant::now();
                let due = match app.bg_view_refresh.get() {
                    None => true,
                    Some(at) => now.duration_since(at) >= Duration::from_millis(600),
                };
                if due {
                    let pid = app.bg_view.as_ref().map(|v| v.pid.clone()).unwrap_or_default();
                    let fresh = Python::with_gil(|py| {
                        bridge
                            .call_method1(py, "background_view", (pid,))
                            .ok()
                            .and_then(|r| r.extract::<String>(py).ok())
                    });
                    if let Some(raw) = fresh {
                        app.apply_bg_view(&raw);
                    }
                    app.bg_view_refresh.set(Some(now));
                    app.dirty.set(true);
                }
            }
            // The sub-agent window follows its worker the same way: a fresh copy of
            // the log once a second while it is open.
            if app.sub_view.is_some() {
                let now = std::time::Instant::now();
                let due = match app.sub_view_refresh.get() {
                    None => true,
                    Some(at) => now.duration_since(at) >= Duration::from_millis(900),
                };
                if due {
                    let id = app.sub_view.as_ref().map(|v| v.id).unwrap_or(0);
                    let fresh = Python::with_gil(|py| {
                        bridge
                            .call_method1(py, "subagent_detail", (id,))
                            .ok()
                            .and_then(|r| r.extract::<String>(py).ok())
                    });
                    if let Some(raw) = fresh {
                        app.apply_sub_view(&raw);
                    }
                    app.sub_view_refresh.set(Some(now));
                }
            }

            // Draw only when something changed, or to advance a spinner while a
            // turn runs. An idle TUI therefore does no work at all.
            if app.dirty.get() || (app.is_busy && last_draw.elapsed() >= Duration::from_millis(80)) {
                app.dirty.set(false);
                last_draw = std::time::Instant::now();
                if let Err(e) = guard.terminal.draw(|f| ui::draw_ui(f, &app)) {
                    return Err(e.to_string());
                }
                // The frame knows how far back the chat really goes: take the offset in.
                app.sync_scroll();
                // A block that fell back to the Rust renderer is reported once,
                // so a broken bridge render is noticed instead of silently drab.
                let failed = app.render_error.borrow_mut().take();
                if let Some(why) = failed {
                    app.messages.push(MessageItem::Notice(format!(
                        "[System: рендер блока через Python не удался, показан упрощённый вид — {why}]"
                    )));
                    app.dirty.set(true);
                }
            }

            // Poll input events; short while a turn runs (spinners), longer idle.
            let poll_for = if app.is_busy { 30 } else { 60 };
            // A slow heartbeat for the mouse mode: cheap, silent, and the only thing that
            // heals a terminal which forgot it while the app was minimized.
            if last_capture_check.elapsed() >= Duration::from_millis(2000) {
                let _ = guard.ensure_mouse_capture();
                last_capture_check = std::time::Instant::now();
            }
            if last_heartbeat.elapsed() >= Duration::from_secs(60) {
                diag(&format!(
                    "alive tick={} busy={} msgs={} mouse={:?} capture={} ask={} approval={} regions={}",
                    app.tick,
                    app.is_busy,
                    app.messages.len(),
                    app.mouse_pos,
                    app.mouse_capture_enabled,
                    app.active_select.is_some(),
                    app.active_approval.is_some(),
                    app.clickable_regions.borrow().len(),
                ));
                last_heartbeat = std::time::Instant::now();
            }
            let mut from_backlog = false;
            let next_event = if let Some(queued) = backlog.pop_front() {
                from_backlog = true;
                Some(queued)
            } else if event::poll(Duration::from_millis(poll_for)).map_err(|e| e.to_string())? {
                Some(event::read().map_err(|e| e.to_string())?)
            } else {
                None
            };
            if let Some(next_event) = next_event {
                app.dirty.set(true);
                // Presence: a real keystroke, paste or click re-arms the window in
                // which vibe ASKS before refusing. Only these count — a moving
                // pointer is not presence, and nothing from the agent passes here.
                let human = match &next_event {
                    Event::Key(_) | Event::Paste(_) => true,
                    Event::Mouse(m) => !matches!(m.kind, MouseEventKind::Moved),
                    _ => false,
                };
                if human
                    && match last_presence {
                        None => true,
                        Some(at) => at.elapsed() >= Duration::from_millis(1000),
                    }
                {
                    last_presence = Some(std::time::Instant::now());
                    Python::with_gil(|py| {
                        let _ = bridge.call_method0(py, "touch_presence");
                    });
                }
                // An input event can only reach this process when the terminal window
                // has keyboard focus, so the event itself is proof of focus. Trusting
                // `FocusGained` alone is what broke the UI: a minimize or a window
                // switch can lose the regain, `focused` stayed false, and every
                // keystroke was dropped — no typing, no Enter, no fold toggles — while
                // the last frame (tooltip included) stayed on screen. It looked dead
                // until a restart.
                if !app.focused
                    && matches!(&next_event, Event::Key(_) | Event::Mouse(_) | Event::Paste(_))
                {
                    app.note_input();
                }
                match next_event {
                    // Mouse Wheel Scroll support & interactive button clicks
                    Event::Mouse(mouse) => match mouse.kind {
                        // A pointer that moves is how hover hints and the
                        // context breakdown appear: the TUI behaves like a
                        // desktop window, not like a terminal that ignores it.
                        MouseEventKind::Moved => {
                            app.set_mouse_pos(mouse.column, mouse.row);
                        }
                        MouseEventKind::ScrollUp => {
                            // The wheel event carries the position it happened at, and
                            // that is the only honest answer to "what is under the
                            // pointer": the last `Moved` can be minutes old, so a wheel
                            // over the chat scrolled the prompt (or the checklist) that
                            // the pointer had once rested on.
                            app.set_mouse_pos(mouse.column, mouse.row);
                            app.wheel(true, std::time::Instant::now());
                        }
                        MouseEventKind::ScrollDown => {
                            app.set_mouse_pos(mouse.column, mouse.row);
                            app.wheel(false, std::time::Instant::now());
                        }
                        // As in Windows terminals: a right click copies what is selected,
                        // and with nothing selected pastes into the prompt.
                        MouseEventKind::Down(MouseButton::Right) => {
                            let picked = if app.mouse_sel.is_some() {
                                app.selected_transcript_text()
                            } else {
                                app.selected_text()
                            };
                            if let Some(text) = picked {
                                if let Some(ref mut clip) = clipboard {
                                    let _ = clip.set_text(text);
                                }
                                app.mouse_sel = None;
                                app.selecting = false;
                                app.clear_selection();
                            } else if app.active_select.is_none() && app.active_approval.is_none() {
                                paste_clipboard(&mut app, &mut clipboard, &bridge);
                            }
                        }
                        MouseEventKind::Down(MouseButton::Left) => {
                            if let Some(action) = app.get_clicked_action(mouse.column, mouse.row) {
                                match action {
                                    ClickAction::CycleThinkingLevel => {
                                        if let Some(next) = app.next_thinking_level() {
                                            Python::with_gil(|py| {
                                                match bridge.call_method1(py, "set_thinking_level", (next.clone(),)) {
                                                    Ok(applied) => {
                                                        let lvl = applied.extract::<String>(py).unwrap_or(next);
                                                        app.apply_thinking_level(&lvl);
                                                    }
                                                    Err(e) => {
                                                        app.messages.push(MessageItem::Error {
                                                            text: format!("Не удалось изменить уровень мышления: {e}"),
                                                            hint: None,
                                                        });
                                                    }
                                                }
                                            });
                                        }
                                    }
                                    ClickAction::OpenModelPicker => {
                                        if app.is_busy {
                                            app.messages.push(MessageItem::Notice(
                                                "Дождитесь завершения хода перед выбором модели".to_string(),
                                            ));
                                        } else {
                                            app.is_command_in_flight = true;
                                            app.is_busy = true;
                                            Python::with_gil(|py| {
                                                let _ = bridge.call_method1(py, "start_turn", ("/model".to_string(),));
                                            });
                                        }
                                    }
                                    ClickAction::OpenProviderPicker => {
                                        if app.is_busy {
                                            app.messages.push(MessageItem::Notice(
                                                "Дождитесь завершения хода перед выбором провайдера".to_string(),
                                            ));
                                        } else {
                                            app.is_command_in_flight = true;
                                            app.is_busy = true;
                                            Python::with_gil(|py| {
                                                let _ = bridge.call_method1(py, "start_turn", ("/provider".to_string(),));
                                            });
                                        }
                                    }
                                    ClickAction::RewindLast => {
                                        if app.is_busy {
                                            app.messages.push(MessageItem::Notice(
                                                "Дождитесь завершения хода (или остановите его) перед возвратом".to_string(),
                                            ));
                                        } else {
                                            // The bridge confirms, restores the files and the conversation,
                                            // and answers with a `rewound` event.
                                            app.is_command_in_flight = true;
                                            app.is_busy = true;
                                            Python::with_gil(|py| {
                                                let _ = bridge.call_method1(py, "start_turn", ("/back".to_string(),));
                                            });
                                        }
                                    }
                                    ClickAction::ToggleThinking(idx) => {
                                        app.toggle_block_by_idx(idx);
                                    }
                                    // The turn summary is an accordion: one row of numbers,
                                    // the REPL's full breakdown on demand.
                                    ClickAction::ToggleSnapshot(idx) => {
                                        app.toggle_block_by_idx(idx);
                                    }
                                    ClickAction::ToggleSubagents => {
                                        app.sub_folded = !app.sub_folded;
                                        app.dirty.set(true);
                                    }
                                    // One worker, looked at closely: its own log is asked for
                                    // on demand and never polled for every agent at once.
                                    ClickAction::OpenSubagent(id) => {
                                        app.open_sub_view(id);
                                        let raw = Python::with_gil(|py| {
                                            bridge
                                                .call_method1(py, "subagent_detail", (id,))
                                                .ok()
                                                .and_then(|v| v.extract::<String>(py).ok())
                                        });
                                        if let Some(raw) = raw {
                                            app.apply_sub_view(&raw);
                                        }
                                    }
                                    // Stopping a worker takes two clicks, like stopping a
                                    // process: the panel is scanned with the mouse.
                                    ClickAction::StopSubagent(id) => {
                                        if app.pending_sub_stop == Some(id) {
                                            app.pending_sub_stop = None;
                                            let answer = Python::with_gil(|py| {
                                                bridge
                                                    .call_method1(py, "stop_subagent", (id,))
                                                    .ok()
                                                    .and_then(|v| v.extract::<String>(py).ok())
                                            });
                                            match answer.as_deref() {
                                                Some("") => app.messages.push(MessageItem::Notice(format!(
                                                    "Субагент #{id} остановлен"
                                                ))),
                                                Some(msg) if !msg.is_empty() => {
                                                    app.messages.push(MessageItem::Notice(msg.to_string()))
                                                }
                                                _ => {}
                                            }
                                        } else {
                                            app.pending_sub_stop = Some(id);
                                            app.dirty.set(true);
                                        }
                                    }
                                    ClickAction::ToggleTool(idx) => {
                                        app.toggle_block_by_idx(idx);
                                        // An opened picture is meant to be looked at, and the
                                        // frame in the poll is 96 columns wide: ask for the
                                        // screen-size grid once, then the rows cache is
                                        // invalidated by the new `seq` and the card redraws
                                        // from it. A click that folds it back costs nothing.
                                        let path = match app.messages.get(idx) {
                                            Some(MessageItem::Tool { expanded: true, .. }) => app
                                                .previews
                                                .get(&idx)
                                                .filter(|p| p.full.cols < 200)
                                                .map(|p| p.path.clone()),
                                            _ => None,
                                        };
                                        if let Some(path) = path {
                                            let old_seq = app.previews.get(&idx).map(|p| p.seq).unwrap_or(0);
                                            // Build the grid at the width it will be DRAWN, so
                                            // PIL resamples once (LANCZOS) instead of the
                                            // frontend averaging a 320-column grid down again.
                                            let win_w = crossterm::terminal::size()
                                                .map(|(w, _)| w as i64)
                                                .unwrap_or(0);
                                            let raw = Python::with_gil(|py| {
                                                bridge
                                                    .call_method1(py, "preview_full", (path, win_w))
                                                    .ok()
                                                    .and_then(|v| v.extract::<String>(py).ok())
                                            });
                                            if let Some(mut big) = raw
                                                .as_deref()
                                                .and_then(crate::app::parse_image_preview)
                                            {
                                                // A fresh `seq` is what makes the rows cache
                                                // rebuild; the poll never sends it again.
                                                big.seq = old_seq.saturating_add(1);
                                                app.previews.insert(idx, big);
                                            }
                                        }
                                    }
                                    ClickAction::ShowContext => {
                                        // The gauge says `ctx 41%`; this says 41% of WHAT.
                                        let text = {
                                            let st = &app.status;
                                            let pct = st.ctx_percent.round() as i64;
                                            if st.ctx_max > 0 {
                                                format!(
                                                    "Контекст: {} из {} токенов ({pct}%) · {}:{} · tier {}",
                                                    st.ctx_tokens, st.ctx_max, st.provider, st.model, st.tier
                                                )
                                            } else {
                                                format!(
                                                    "Контекст ещё не измерен · {}:{} · tier {}",
                                                    st.provider, st.model, st.tier
                                                )
                                            }
                                        };
                                        app.messages.push(MessageItem::Notice(text));
                                    }
                                    ClickAction::ShowSpend => {
                                        let text = {
                                            let st = &app.status;
                                            let mut out = String::new();
                                            if !st.session_summary.is_empty() {
                                                out.push_str(&st.session_summary);
                                            }
                                            if !st.spend_breakdown.is_empty() {
                                                if !out.is_empty() {
                                                    out.push_str(" · ");
                                                }
                                                out.push_str(&st.spend_breakdown);
                                            }
                                            if out.is_empty() {
                                                out.push_str("Расход в этой сессии пока нулевой");
                                            }
                                            out
                                        };
                                        app.messages.push(MessageItem::Notice(text));
                                    }
                                    ClickAction::CopyResponse(idx) => {
                                        // The whole run, not one segment: a re-sent
                                        // answer is two adjacent items and the button
                                        // lives on the first.
                                        let text_opt = app
                                            .assistant_run_content(idx)
                                            .or_else(|| app.assistant_content_at(idx))
                                            .or_else(|| app.last_assistant_content());
                                        if let Some(text) = text_opt {
                                            let count = text.chars().count();
                                            if let Some(ref mut clip) = clipboard {
                                                if clip.set_text(text).is_ok() {
                                                    app.messages.push(MessageItem::Notice(format!(
                                                        "Ответ ассистента скопирован в буфер обмена ({count} симв.)"
                                                    )));
                                                }
                                            }
                                        } else {
                                            app.messages.push(MessageItem::Notice(
                                                "Нет ответа ассистента для копирования".to_string(),
                                            ));
                                        }
                                    }
                                    ClickAction::CopyLastResponse => {
                                        if let Some(text) = app.last_assistant_content() {
                                            let count = text.chars().count();
                                            if let Some(ref mut clip) = clipboard {
                                                if clip.set_text(text).is_ok() {
                                                    app.messages.push(MessageItem::Notice(format!(
                                                        "Ответ ассистента скопирован в буфер обмена ({count} симв.)"
                                                    )));
                                                }
                                            }
                                        } else {
                                            app.messages.push(MessageItem::Notice(
                                                "Нет ответа ассистента для копирования".to_string(),
                                            ));
                                        }
                                    }
                                    ClickAction::CopyLastCode => {
                                        if let Some(code) = app.last_assistant_code() {
                                            let count = code.chars().count();
                                            if let Some(ref mut clip) = clipboard {
                                                if clip.set_text(code).is_ok() {
                                                    app.messages.push(MessageItem::Notice(format!(
                                                        "Блоки кода скопированы в буфер обмена ({count} симв.)"
                                                    )));
                                                }
                                            }
                                        } else {
                                            app.messages.push(MessageItem::Notice(
                                                "В последнем ответе нет блоков кода ```...```".to_string(),
                                            ));
                                        }
                                    }
                                    ClickAction::CopyCode(code) => {
                                        let count = code.chars().count();
                                        if let Some(ref mut clip) = clipboard {
                                            if clip.set_text(code).is_ok() {
                                                app.messages.push(MessageItem::Notice(format!(
                                                    "Блок кода скопирован в буфер обмена ({count} симв.)"
                                                )));
                                            }
                                        }
                                    }
                                    ClickAction::ScrollToBottom => {
                                        app.scroll_down(usize::MAX);
                                    }
                                    ClickAction::ToggleTodos => {
                                        app.todos_folded = !app.todos_folded;
                                    }
                                    ClickAction::ToggleTodosDone => {
                                        app.todos_done_expanded = !app.todos_done_expanded;
                                    }
                                    ClickAction::ToggleSubTodos(idx) => {
                                        if let Some(open) = app.todos_sub_open.get_mut(idx) {
                                            *open = !*open;
                                        }
                                    }
                                    ClickAction::ToggleBackground => {
                                        app.bg_folded = !app.bg_folded;
                                    }
                                    ClickAction::ToggleBackgroundDone => {
                                        app.bg_done_folded = !app.bg_done_folded;
                                    }
                                    ClickAction::ToggleFold(idx) => {
                            app.toggle_fold(idx);
                        }
                        ClickAction::OpenBackground(pid) => {
                            app.open_bg_view(&pid);
                        }
                        // Stopping a process takes two clicks on the same row:
                        // the first arms it and says so, the second sends
                        // `/stop <pid>` — the same command the prompt takes, so
                        // it goes through whatever approval the session has.
                        ClickAction::StopBackground(pid) => {
                            if app.pending_stop.as_deref() != Some(pid.as_str()) {
                                app.pending_stop = Some(pid.clone());
                                app.messages.push(MessageItem::Notice(format!(
                                    "Ещё раз на ⏹ у pid {pid} — процесс будет остановлен"
                                )));
                            } else {
                                app.pending_stop = None;
                                if app.is_busy {
                                    app.messages.push(MessageItem::Notice(
                                        "Ход ещё идёт — остановите процесс после него".to_string(),
                                    ));
                                } else {
                                    app.is_command_in_flight = true;
                                    app.is_busy = true;
                                    Python::with_gil(|py| {
                                        let _ = bridge.call_method1(
                                            py, "start_turn", (format!("/stop {pid}"),));
                                    });
                                }
                            }
                        }
                        ClickAction::ShowBackgroundOutput(pid) => {
                                        // The output of a process runs to hundreds of lines and
                                        // would bury the chat if drawn here: the command prints it
                                        // in the transcript, where it can be scrolled and copied.
                                        if app.is_busy {
                                            app.messages.push(MessageItem::Notice(
                                                "Дождитесь завершения хода — вывод откроется в чате".to_string(),
                                            ));
                                        } else {
                                            app.is_command_in_flight = true;
                                            app.is_busy = true;
                                            Python::with_gil(|py| {
                                                let _ = bridge.call_method1(
                                                    py, "start_turn", (format!("/jobs {pid}"),));
                                            });
                                        }
                                    }
                                    ClickAction::ShowImage(idx) => {
                                        // Ask the bridge for a screen-sized grid: the one in
                                        // the poll is 96 columns wide, and a picture blown up
                                        // from it is exactly the blur people complain about.
                                        // The path is passed so an OLDER card opens its own
                                        // picture, not whichever frame was published last.
                                        // Falls back to the cached frame if the file is gone.
                                        let path = app
                                            .previews
                                            .get(&idx)
                                            .map(|p| p.path.clone())
                                            .unwrap_or_default();
                                        let raw = Python::with_gil(|py| {
                                            let win_w = crossterm::terminal::size()
                                                .map(|(w, _)| w as i64)
                                                .unwrap_or(0);
                                            bridge
                                                .call_method1(py, "preview_full", (path.clone(), win_w))
                                                .ok()
                                                .and_then(|v| v.extract::<String>(py).ok())
                                        });
                                        let big = raw
                                            .as_deref()
                                            .and_then(crate::app::parse_image_preview);
                                        if let Some(prev) = big {
                                            app.image_view = Some(prev);
                                        } else if let Some(prev) = app.previews.get(&idx).cloned() {
                                            app.image_view = Some(prev);
                                        }
                                    }
                                    // A terminal draws a picture as blocks of colour; the
                                    // system viewer draws it as pixels. This is the honest
                                    // way to examine a screenshot.
                                    ClickAction::OpenImageExternally(idx) => {
                                        let path = app
                                            .previews
                                            .get(&idx)
                                            .map(|p| p.path.clone())
                                            .unwrap_or_default();
                                        open_path_in_system(&mut app, &bridge, path);
                                    }
                                    // An artifact card: the same door, for the markdown
                                    // the run just wrote.
                                    ClickAction::OpenArtifactExternally(path) => {
                                        open_path_in_system(&mut app, &bridge, path);
                                    }
                                    // A queued message: take it into the draft. The slot
                                    // stays put, so pressing Enter replaces it in place
                                    // instead of moving it to the end of the queue.
                                    ClickAction::EditQueue(idx) => {
                                        if app.edit_queued(idx) {
                                            app.messages.push(MessageItem::Notice(format!(
                                                "Правка сообщения {} в очереди: Enter заменит его, Esc отменит",
                                                idx + 1
                                            )));
                                        }
                                    }
                                    ClickAction::DropQueue(idx) => {
                                        if let Some(removed) = app.drop_queued(idx) {
                                            let short: String = removed.chars().take(60).collect();
                                            app.messages.push(MessageItem::Notice(format!(
                                                "Убрано из очереди: {short}"
                                            )));
                                        }
                                    }
                                    // ⏎: send this message now. It moves to the front and
                                    // the loop stops the turn in the way; the text itself
                                    // is a prompt, not a stopped process, so nothing is
                                    // lost either way.
                                    ClickAction::SendQueueNow(idx) => {
                                        if let Some(text) = app.promote_queued(idx) {
                                            let short: String = text.chars().take(60).collect();
                                            app.messages.push(MessageItem::Notice(format!(
                                                "Отправлю сейчас: {short}"
                                            )));
                                        }
                                    }
                                    // A picture attached to the next message.
                                    ClickAction::ShowAttachment(idx) => {
                                        let raw = Python::with_gil(|py| {
                                            bridge
                                                .call_method1(py, "pending_image_full", (idx,))
                                                .ok()
                                                .and_then(|v| v.extract::<String>(py).ok())
                                        });
                                        if let Some(p) = raw.as_deref().and_then(crate::app::parse_image_preview) {
                                            app.image_view = Some(p);
                                        }
                                    }
                                    ClickAction::DropAttachment(idx) => {
                                        Python::with_gil(|py| {
                                            let _ = bridge.call_method1(py, "drop_pending_image", (idx,));
                                        });
                                        app.attachments.clear();
                                        app.messages.push(MessageItem::Notice(
                                            "Вложение убрано из сообщения".to_string(),
                                        ));
                                        pull_state(&mut app, &bridge);
                                    }
                                    ClickAction::ToggleMouseCapture => {
                                        match guard.toggle_mouse_capture() {
                                            Ok(enabled) => {
                                                app.mouse_capture_enabled = enabled;
                                                let msg = if enabled {
                                                    "Захват мыши: ВКЛ (скролл колесом активен, выделение текста терминалом: зажмите Shift)"
                                                } else {
                                                    "Захват мыши: ВЫКЛ (свободное выделение текста без Shift)"
                                                };
                                                app.messages.push(MessageItem::Notice(msg.to_string()));
                                            }
                                            Err(e) => {
                                                app.messages.push(MessageItem::Notice(format!("Ошибка переключения мыши: {e}")));
                                            }
                                        }
                                    }
                                }
                            } else {
                                let clicks = app.click_count(mouse.column, mouse.row, std::time::Instant::now());
                                // The prompt is an editor, not a chat area: a press in
                                // it puts the caret under the pointer.
                                if app.begin_input_selection(mouse.column, mouse.row, clicks) {
                                    // handled by the draft
                                } else if app.in_transcript(mouse.row) {
                                    // Not a click target: select transcript text. A second
                                    // press selects the word, a third the row.
                                    app.begin_selection(mouse.column, mouse.row, clicks);
                                } else {
                                    app.mouse_sel = None;
                                }
                            }
                        }
                        MouseEventKind::Drag(MouseButton::Left) => {
                            if app.input_selecting {
                                app.drag_input_selection(mouse.column, mouse.row);
                            } else {
                                app.drag_selection(mouse.column, mouse.row);
                            }
                        }
                        MouseEventKind::Up(MouseButton::Left) => {
                            if app.input_selecting {
                                // A selection in the draft is not copied on release:
                                // it is there to be cut, replaced or copied by hand.
                                app.finish_input_selection();
                            } else if let Some(text) = app.finish_selection() {
                                if let Some(ref mut clip) = clipboard {
                                    let _ = clip.set_text(text);
                                }
                            }
                        }
                        _ => {}
                    },

                    Event::Resize(w, h) => {
                        app.mouse_sel = None;
                        app.selecting = false;
                        diag(&format!("resize {w}x{h} mouse={:?}", app.mouse_pos));
                        // A resize invalidates the pointer's coordinates: they were
                        // measured against the old layout, and a hover hint left on
                        // screen would point at nothing.
                        app.forget_pointer();
                        // A terminal resize is also the recovery path from a
                        // stale/corrupted alternate-screen frame: repaint the
                        // whole surface, exactly like the fullscreen toggle
                        // users already use to clear the glitch.
                        let _ = guard.terminal.clear();
                        // Restoring a minimized window is exactly when mouse
                        // reporting goes missing. Put it back, quietly.
                        let _ = guard.ensure_mouse_capture();
                    }

                    // Bracketed paste: arrives as one event instead of a keystroke storm, which is
// what makes multi-line pastes usable. Guarded so a paste can never dismiss the
// approval or select modal.
Event::FocusGained => {
    app.focused = true;
    diag("focus gained");
    let _ = guard.ensure_mouse_capture();
}
Event::FocusLost => {
    app.focused = false;
    diag("focus lost");
    app.clear_selection();    // A tooltip is drawn from the pointer's last position.
    // Left behind, it hangs over the chat forever once the
    // window comes back — a hint attached to nothing.
    app.forget_pointer();
}                        Event::Paste(text)
                            if app.focused
                                && app.active_approval.is_none()
                                && app.active_select.is_none() =>
                        {
                            app.insert_str(&text);
                        }

                    // Keyboard events. The legacy CLI compared window handles to
                    // refuse input while unfocused; crossterm reports focus changes
                    // directly, which is both cheaper and not Win32-specific.
                    Event::Key(_key) if !app.focused => {
                        app.clear_selection();
                    }
                    Event::Key(key) => {
                        // CRITICAL: Filter out Release events to prevent duplicate characters on Windows!
                        if key.kind == KeyEventKind::Release {
                            continue;
                        }
                        // A wheel notch that arrives as Up/Down is the console telling us
                        // it is not reporting the mouse: capture was dropped (a minimize
                        // and restore does it). The pointer was forgotten on that same
                        // event, so this is the exact moment to put the mode back.
                        if app.mouse_pos.is_none()
                            && matches!(key.code, KeyCode::Up | KeyCode::Down)
                        {
                            // The console is not reporting the mouse: the wheel is
                            // arriving as keys. Say so, and put the mode back.
                            diag(&format!("wheel-as-keys {:?} capture={}", key.code, app.mouse_capture_enabled));
                            let _ = guard.ensure_mouse_capture();
                        }
                        // Any key returns the prompt to the caret: a peek back at
                        // earlier lines is a look, and typing ends it.
                        app.input_scroll = 0;
                        // The window on a background task is a window: Esc closes it
                        // and the scroll keys move its own text, not the chat behind.
                        if app.sub_view.is_some() {
                            match key.code {
                                KeyCode::Esc | KeyCode::Char('q') => {
                                    app.close_sub_view();
                                    continue;
                                }
                                KeyCode::PageUp => {
                                    app.sub_view_page(true);
                                    continue;
                                }
                                KeyCode::PageDown => {
                                    app.sub_view_page(false);
                                    continue;
                                }
                                KeyCode::Home => {
                                    app.sub_view_scroll(-1000);
                                    continue;
                                }
                                KeyCode::End => {
                                    app.sub_view_scroll(1000);
                                    continue;
                                }
                                KeyCode::Up => {
                                    app.sub_view_scroll(-1);
                                    continue;
                                }
                                KeyCode::Down => {
                                    app.sub_view_scroll(1);
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        if app.bg_view.is_some() {
                            match key.code {
                                KeyCode::Esc | KeyCode::Char('q') => {
                                    app.close_bg_view();
                                    continue;
                                }
                                KeyCode::PageUp => {
                                    app.bg_view_page(true);
                                    continue;
                                }
                                KeyCode::PageDown => {
                                    app.bg_view_page(false);
                                    continue;
                                }
                                KeyCode::Home => {
                                    app.bg_view_scroll(-1000);
                                    continue;
                                }
                                KeyCode::End => {
                                    app.bg_view_scroll(1000);
                                    continue;
                                }
                                KeyCode::Up => {
                                    app.bg_view_scroll(-1);
                                    continue;
                                }
                                KeyCode::Down => {
                                    app.bg_view_scroll(1);
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        // The full-screen picture is a look, not a mode: any key
                        // closes it and nothing of it reaches the prompt. One key
                        // is the exception — O hands the real file to the system
                        // viewer, because a character grid cannot make a
                        // 1920-pixel screenshot readable.
                        if app.image_view.is_some() {
                            let opened_with = key.code;
                            let path = app.image_view.as_ref().map(|p| p.path.clone()).unwrap_or_default();
                            app.image_view = None;
                            if matches!(opened_with, KeyCode::Char('o') | KeyCode::Char('O')) {
                                let answer = Python::with_gil(|py| {
                                    bridge
                                        .call_method1(py, "open_image", (path,))
                                        .ok()
                                        .and_then(|v| v.extract::<String>(py).ok())
                                });
                                match answer.as_deref() {
                                    Some("") => app.messages.push(MessageItem::Notice(
                                        "Изображение открыто в системном просмотрщике".to_string(),
                                    )),
                                    Some(msg) if !msg.is_empty() => app.messages.push(MessageItem::Notice(msg.to_string())),
                                    _ => {}
                                }
                            }
                            continue;
                        }
                        // A Windows console does not deliver a paste as Event::Paste: the text
                        // arrives as a storm of key presses, and every Enter in it would submit
                        // the line. Keys that arrive back to back are therefore taken as one paste.
                        if !from_backlog
                            && app.active_approval.is_none()
                            && app.active_select.is_none()
                            && paste_char(&key).is_some()
                        {
                            let mut burst = vec![key];
                            let mut after: Option<Event> = None;
                            loop {
                                // Inside a burst wait a moment: a long paste reaches the console in chunks.
                                let wait = if burst.len() > 1 { PASTE_GAP_MS } else { 0 };
                                if !event::poll(Duration::from_millis(wait)).map_err(|e| e.to_string())? {
                                    break;
                                }
                                match event::read().map_err(|e| e.to_string())? {
                                    Event::Key(k) if k.kind == KeyEventKind::Release => {}
                                    Event::Key(k) if paste_char(&k).is_some() => burst.push(k),
                                    other => {
                                        after = Some(other);
                                        break;
                                    }
                                }
                            }
                            if is_paste_burst(&burst) {
                                let text: String = burst.iter().filter_map(paste_char).collect();
                                app.insert_str(&text);
                                if let Some(other) = after {
                                    backlog.push_back(other);
                                }
                                continue;
                            }
                            // Just quick typing: the keys are handled one by one, in their order.
                            for k in burst.drain(1..).rev() {
                                backlog.push_front(Event::Key(k));
                            }
                            if let Some(other) = after {
                                backlog.push_back(other);
                            }
                        }
                        // Ctrl+C copies a mouse selection; scrolling keeps it (it is
                        // anchored to the text, not to the screen); Esc only dismisses
                        // it; any other key dismisses it and then does its own work.
                        if app.mouse_sel.is_some() {
                            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                            if ctrl && matches!(key.code, KeyCode::Char('c' | 'C' | 'с' | 'С')) {
                                if let Some(text) = app.selected_transcript_text() {
                                    if let Some(ref mut clip) = clipboard {
                                        let _ = clip.set_text(text);
                                    }
                                }
                                app.mouse_sel = None;
                                app.selecting = false;
                                continue;
                            }
                            if key.code == KeyCode::Esc {
                                app.mouse_sel = None;
                                app.selecting = false;
                                continue;
                            }
                            let moves_view = matches!(key.code, KeyCode::PageUp | KeyCode::PageDown)
                                || (matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Home | KeyCode::End)
                                    && key.modifiers.intersects(
                                        KeyModifiers::CONTROL | KeyModifiers::SHIFT | KeyModifiers::ALT,
                                    ));
                            if !moves_view {
                                app.mouse_sel = None;
                                app.selecting = false;
                            }
                        }

                        // 0. Select modal interception
                        if let Some(ref mut select) = app.active_select {
                            match key.code {
                                KeyCode::Up => {
                                    select.move_up();
                                }
                                KeyCode::Down => {
                                    select.move_down();
                                }
                                KeyCode::Char(' ') if select.multi => {
                                    select.toggle_current();
                                }
                                KeyCode::Enter => {
                                    if let Some(s) = app.active_select.take() {
                                        let chosen = s.selected_value().unwrap_or_default();
                                        // Leaving a trace: without this the modal closed
                                        // and the chat showed neither the question nor
                                        // the answer, so a person could not check what
                                        // they had picked.
                                        app.note_answer(&chosen);
                                        Python::with_gil(|py| {
                                            let _ = bridge.call_method1(py, "reply_select", (&s.id, &chosen));
                                        });
                                    }
                                }
                                KeyCode::Esc => {
                                    if let Some(s) = app.active_select.take() {
                                        app.note_answer("");
                                        Python::with_gil(|py| {
                                            let _ = bridge.call_method1(py, "reply_select", (&s.id, ""));
                                        });
                                    }
                                }
                                // ЙЦУКЕН: в русской раскладке приходит `с`, и сырое
                                // сравнение с 'c' не срабатывало — Ctrl+C в модалке
                                // не делал ничего. В промпте клавиша нормализуется
                                // через latin_of, а сюда этот путь не доходит.
                                KeyCode::Char('c') | KeyCode::Char('с') | KeyCode::Char('С')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    if let Some(s) = app.active_select.take() {
                                        app.note_answer("");
                                        Python::with_gil(|py| {
                                            let _ = bridge.call_method1(py, "reply_select", (&s.id, ""));
                                        });
                                    }
                                }
                                // The typed answer is editable: the caret walks with
                                // the arrows, Home/End, and deletes on either side.
                                KeyCode::Backspace => select.filter_backspace(),
                                KeyCode::Delete => select.filter_delete(),
                                KeyCode::Left => select.filter_left(),
                                KeyCode::Right => select.filter_right(),
                                KeyCode::Home => select.filter_home(),
                                KeyCode::End => select.filter_end(),
                                KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    select.filter_home();
                                }
                                KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    select.filter_end();
                                }
                                KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    select.filter_clear();
                                }
                                KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    select.filter_insert(c);
                                }
                                _ => {}
                            }
                            continue;
                        }

                        // 1. Approval modal interception
                        // The question is drawn like questionary draws it: arrow keys
                        // and Enter, as in the REPL. The letter shortcuts stay as a
                        // bonus; "always" only answers when the list offers it.
                        if app.active_approval.is_some() {
                            let (aid, prompt, picked) = {
                                let a = app.active_approval.as_ref().expect("checked above");
                                (a.id, a.resolved_prompt(), a.selected_decision())
                            };
                            let count = prompt.choices.len().max(1);
                            let mut decision: Option<&str> = None;
                            match key.code {
                                KeyCode::Up => {
                                    if let Some(a) = app.active_approval.as_mut() {
                                        a.selected = if a.selected == 0 { count - 1 } else { a.selected - 1 };
                                    }
                                }
                                KeyCode::Down => {
                                    if let Some(a) = app.active_approval.as_mut() {
                                        a.selected = (a.selected + 1) % count;
                                    }
                                }
                                // A y/N question defaults to No, like questionary's.
                                KeyCode::Enter => {
                                    decision = Some(if prompt.confirm { "deny" } else { "pick" });
                                }
                                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Char('н') | KeyCode::Char('Н') => {
                                    decision = Some("once");
                                }
                                KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Char('ф') | KeyCode::Char('Ф')
                                    if prompt.decisions.iter().any(|d| d == "always") =>
                                {
                                    decision = Some("always");
                                }
                                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Char('т') | KeyCode::Char('Т') | KeyCode::Esc => {
                                    decision = Some("deny");
                                }
                                // Тот же ЙЦУКЕН-случай: без `с` отказ по Ctrl+C не работал.
                                KeyCode::Char('c') | KeyCode::Char('с') | KeyCode::Char('С')
                                    if key.modifiers.contains(KeyModifiers::CONTROL) => {
                                    decision = Some("deny");
                                }
                                _ => {}
                            }
                            if let Some(d) = decision {
                                let chosen = if d == "pick" { picked } else { d.to_string() };
                                app.active_approval = None;
                                // The modal is taller than the normal prompt.
                                // Repaint the whole screen when it leaves so a
                                // stale terminal row cannot survive under the
                                // toolbar (the "toolbar disappears" glitch).
                                let _ = guard.terminal.clear();
                                Python::with_gil(|py| {
                                    let _ = bridge.call_method1(py, "reply_approval", (aid, chosen));
                                });
                            }
                            continue;
                        }

                        // 2. Normal key processing

                        // F1 — the command palette. It owns the keyboard while it is
                        // open: typing filters by name or by what the command does,
                        // arrows move, Enter drops the command into the prompt. All
                        // of it is handled here so none of the bindings below can
                        // hijack a keystroke meant for the search box.
                        if key.code == KeyCode::F(1) {
                            if app.palette_is_open() {
                                app.palette_close();
                            } else {
                                app.open_palette();
                            }
                            continue;
                        }
                        // Ctrl+F — search the transcript. Ctrl+R searches the words
                        // you TYPED; this finds where the agent said something.
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && matches!(key.code, KeyCode::Char('f') | KeyCode::Char('F') | KeyCode::Char('а') | KeyCode::Char('А'))
                        {
                            if app.search_is_open() {
                                app.search_close();
                            } else {
                                app.open_search();
                            }
                            continue;
                        }
                        if app.search_is_open() {
                            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                            match key.code {
                                KeyCode::Esc => app.search_close(),
                                KeyCode::Enter => {
                                    app.search_accept();
                                }
                                KeyCode::Up => app.search_move(-1),
                                KeyCode::Down => app.search_move(1),
                                KeyCode::Backspace => app.search_backspace(),
                                KeyCode::Char(c) if !ctrl => app.search_push(c),
                                _ => {}
                            }
                            continue;
                        }
                        if app.palette_is_open() {
                            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                            match key.code {
                                KeyCode::Esc => app.palette_close(),
                                KeyCode::Enter => {
                                    app.palette_accept();
                                }
                                KeyCode::Up => app.palette_move(-1),
                                KeyCode::Down => app.palette_move(1),
                                KeyCode::Backspace => app.palette_backspace(),
                                KeyCode::Char(c) if !ctrl => app.palette_push(c),
                                _ => {}
                            }
                            continue;
                        }

                        // Check for Ctrl+Backspace / ASCII DEL (0x7F) / ASCII BS (0x08) / Ctrl+W / Ctrl+H
                        let is_ctrl_backspace = (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Backspace)
                            || key.code == KeyCode::Char('\u{7f}')
                            || (key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('\u{8}'))
                            || (key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('w') | KeyCode::Char('W') | KeyCode::Char('ц') | KeyCode::Char('Ц')))
                            || (key.modifiers.contains(KeyModifiers::CONTROL) && matches!(key.code, KeyCode::Char('h') | KeyCode::Char('H') | KeyCode::Char('р') | KeyCode::Char('Р')));

                        if is_ctrl_backspace {
                            app.delete_word_left();
                            continue;
                        }

                        // Check for Ctrl+Delete
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Delete {
                            app.delete_word_right();
                            continue;
                        }

                        // Ctrl+Left / Ctrl+Right jump by word; with Shift they select by word.
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Left {
                            if key.modifiers.contains(KeyModifiers::SHIFT) {
                                app.select_word_left();
                            } else {
                                app.clear_selection();
                                app.move_word_left();
                                app.check_completions();
                            }
                            continue;
                        }
                        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Right {
                            if key.modifiers.contains(KeyModifiers::SHIFT) {
                                app.select_word_right();
                            } else {
                                app.clear_selection();
                                app.move_word_right();
                                app.check_completions();
                            }
                            continue;
                        }

                        // Check for Shift+Enter / Alt+Enter / Ctrl+Enter / Ctrl+J -> Insert newline
                        let is_newline = (key.modifiers.contains(KeyModifiers::SHIFT) && (key.code == KeyCode::Enter || key.code == KeyCode::Char('\r')))
                            || (key.modifiers.contains(KeyModifiers::ALT) && (key.code == KeyCode::Enter || key.code == KeyCode::Char('\r')))
                            || (key.modifiers.contains(KeyModifiers::CONTROL) && (key.code == KeyCode::Enter || key.code == KeyCode::Char('\r') || key.code == KeyCode::Char('j') || key.code == KeyCode::Char('J') || key.code == KeyCode::Char('о') || key.code == KeyCode::Char('О')))
                            || key.code == KeyCode::Char('\n');

                        if is_newline {
                            app.insert_char('\n');
                            continue;
                        }

                        // A Cyrillic layout sends the Russian letter for the same key
                        // (Ctrl+C arrives as 'с'), so every shortcut silently stopped
                        // working for anyone typing in Russian. Normalise ONLY when a
                        // modifier is held: plain typing must still insert Cyrillic.
                        let code = if key.modifiers.contains(KeyModifiers::CONTROL)
                            || key.modifiers.contains(KeyModifiers::ALT)
                        {
                            match key.code {
                                KeyCode::Char(ch) => KeyCode::Char(latin_of(ch)),
                                other => other,
                            }
                        } else {
                            key.code
                        };
                        match (key.modifiers, code) {
                            (KeyModifiers::CONTROL, KeyCode::Char('x')) if app.has_selection() => {
                                if let Some(text) = app.selected_text() {
                                    if let Some(ref mut clip) = clipboard {
                                        let _ = clip.set_text(text);
                                    }
                                }
                                app.delete_selection();
                            }
                            (KeyModifiers::NONE, KeyCode::Backspace) if app.has_selection() => {
                                app.delete_selection();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('c')) => {
                                // A turn in flight is the priority: Ctrl+C must
                                // stop the work, even when a draft is sitting in
                                // the prompt or a selection exists. Clearing the
                                // input first was why a busy agent looked like it
                                // ignored Ctrl+C.
                                if app.is_busy {
                                    // First press stops the turn, a second one exits —
                                    // so a busy agent can never trap the user in the TUI.
                                    //
                                    // The second press counts whenever a stop is STILL in
                                    // flight, not only inside a 1.5 s window: a stop that
                                    // has been asked for and has not landed means a tool is
                                    // holding the turn, and someone pressing again two
                                    // seconds later was re-arming the timer forever — the
                                    // one case where the way out mattered most.
                                    let stop_was_in_flight = app.is_cancelling;
                                    app.is_cancelling = true;
                                    Python::with_gil(|py| {
                                        let _ = bridge.call_method0(py, "cancel_turn");
                                    });
                                    let now = std::time::Instant::now();
                                    let twice = last_ctrl_c
                                        .map(|t| now.duration_since(t) < std::time::Duration::from_millis(1500))
                                        .unwrap_or(false);
                                    if twice || stop_was_in_flight {
                                        Python::with_gil(|py| {
                                            let _ = bridge.call_method0(py, "cleanup");
                                        });
                                        break;
                                    }
                                    last_ctrl_c = Some(now);
                                } else if app.has_selection() {
                                    if let Some(text) = app.selected_text() {
                                        if let Some(ref mut clip) = clipboard {
                                            let _ = clip.set_text(text);
                                        }
                                    }
                                    app.clear_selection();
                                } else if !app.input.is_empty() {
                                    app.input.clear();
                                    app.cursor_pos = 0;
                                    last_ctrl_c = None;
                                } else {
                                    // Idle: like the REPL, ask for a second Ctrl+C
                                    // within the window before leaving, so a stray
                                    // press never ends the session.
                                    let now = std::time::Instant::now();
                                    let twice = last_ctrl_c
                                        .map(|t| now.duration_since(t) < std::time::Duration::from_millis(1500))
                                        .unwrap_or(false);
                                    if twice {
                                        Python::with_gil(|py| {
                                            let _ = bridge.call_method0(py, "cleanup");
                                        });
                                        break;
                                    }
                                    last_ctrl_c = Some(now);
                                    app.messages.push(MessageItem::System(
                                        "[dim](Нажмите Ctrl+C ещё раз для выхода)[/dim]".to_string(),
                                    ));
                                }
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('d')) => {
                                if app.input.is_empty() {
                                    // EOF on an empty line, like the legacy prompt.
                                    Python::with_gil(|py| {
                                        let _ = bridge.call_method0(py, "cleanup");
                                    });
                                    break;
                                }
                                app.delete_char();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('v')) => {
                                paste_clipboard(&mut app, &mut clipboard, &bridge);
                            }
                            // Same paste on keys the terminal does not swallow: some
                            // hosts turn Ctrl+V into a text-only paste, which silently
                            // drops a picture on the clipboard.
                            (KeyModifiers::ALT, KeyCode::Char('v')) => {
                                paste_clipboard(&mut app, &mut clipboard, &bridge);
                            }
                            (m, KeyCode::Char('v'))
                                if m.contains(KeyModifiers::CONTROL) && m.contains(KeyModifiers::SHIFT) =>
                            {
                                paste_clipboard(&mut app, &mut clipboard, &bridge);
                            }
                            // Collapsible-block navigation. Space used to double as the reasoning
                            // toggle, which made it impossible to tell an indent
                            // from an expand; Space is now always a space.
                            (KeyModifiers::CONTROL, KeyCode::Char('n') | KeyCode::Char('т')) => {
                                app.focus_next_block();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('p') | KeyCode::Char('з')) => {
                                app.focus_prev_block();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('t')) => {                                if let Some(next) = app.next_thinking_level() {
                                    let result = Python::with_gil(|py| {
                                        bridge
                                            .call_method1(py, "set_thinking_level", (next.clone(),))
                                            .and_then(|v| v.extract::<String>(py))
                                    });
                                    match result {
                                        Ok(applied) => app.apply_thinking_level(&applied),
                                        Err(e) => {
                                            let msg = format!("Не удалось сменить уровень мышления: {e}");
                                            app.set_thinking_error(msg);
                                        }
                                    }
                                }
                            }
                            (KeyModifiers::ALT, KeyCode::Char('t')) => {
                                app.transpose_words();
                            }
                            (KeyModifiers::ALT | KeyModifiers::SHIFT, KeyCode::Char('T')) => {
                                app.transpose_chars();
                            }
                            (KeyModifiers::ALT, KeyCode::Char('r')) => {
                                app.show_tool_results = !app.show_tool_results;
                                app.messages.push(MessageItem::Notice(format!(
                                    "Показ результатов инструментов: {}",
                                    if app.show_tool_results { "всегда" } else { "свёрнуто" }
                                )));
                            }
                            // Alt+C / Alt+С: Copy clean markdown of last assistant response to clipboard
                            (KeyModifiers::ALT, KeyCode::Char('c') | KeyCode::Char('C') | KeyCode::Char('с') | KeyCode::Char('С')) => {
                                if let Some(text) = app.last_assistant_content() {
                                    let count = text.chars().count();
                                    if let Some(ref mut clip) = clipboard {
                                        match clip.set_text(text) {
                                            Ok(_) => {
                                                app.messages.push(MessageItem::Notice(format!(
                                                    "Ответ ассистента скопирован в буфер обмена ({count} симв.)"
                                                )));
                                            }
                                            Err(e) => {
                                                app.messages.push(MessageItem::Notice(format!(
                                                    "Ошибка копирования в буфер: {e}"
                                                )));
                                            }
                                        }
                                    }
                                } else {
                                    app.messages.push(MessageItem::Notice(
                                        "Нет ответа ассистента для копирования".to_string(),
                                    ));
                                }
                            }
                            // Alt+K: Copy only code blocks from last assistant response
                            (m, KeyCode::Char('k') | KeyCode::Char('K') | KeyCode::Char('л') | KeyCode::Char('Л'))
                                if m.contains(KeyModifiers::ALT) =>
                            {
                                if let Some(code) = app.last_assistant_code() {
                                    let count = code.chars().count();
                                    if let Some(ref mut clip) = clipboard {
                                        match clip.set_text(code) {
                                            Ok(_) => {
                                                app.messages.push(MessageItem::Notice(format!(
                                                    "Блоки кода скопированы в буфер обмена ({count} симв.)"
                                                )));
                                            }
                                            Err(e) => {
                                                app.messages.push(MessageItem::Notice(format!(
                                                    "Ошибка копирования кода: {e}"
                                                )));
                                            }
                                        }
                                    }
                                } else {
                                    app.messages.push(MessageItem::Notice(
                                        "В последнем ответе нет блоков кода ```...```".to_string(),
                                    ));
                                }
                            }
                            // Alt+O / F3: Open model selection popup (/model)
                            (KeyModifiers::ALT, KeyCode::Char('o') | KeyCode::Char('O') | KeyCode::Char('щ') | KeyCode::Char('Щ'))
                                | (KeyModifiers::NONE, KeyCode::F(3)) =>
                            {
                                if app.is_busy {
                                    app.messages.push(MessageItem::Notice(
                                        "Дождитесь завершения хода перед выбором модели".to_string(),
                                    ));
                                } else {
                                    app.is_command_in_flight = true;
                                    app.is_busy = true;
                                    Python::with_gil(|py| {
                                        let _ = bridge.call_method1(py, "start_turn", ("/model".to_string(),));
                                    });
                                }
                            }
                            // Alt+P / F4: Open provider selection popup (/provider)
                            (KeyModifiers::ALT, KeyCode::Char('p') | KeyCode::Char('P') | KeyCode::Char('з') | KeyCode::Char('З'))
                                | (KeyModifiers::NONE, KeyCode::F(4)) =>
                            {
                                if app.is_busy {
                                    app.messages.push(MessageItem::Notice(
                                        "Дождитесь завершения хода перед выбором провайдера".to_string(),
                                    ));
                                } else {
                                    app.is_command_in_flight = true;
                                    app.is_busy = true;
                                    Python::with_gil(|py| {
                                        let _ = bridge.call_method1(py, "start_turn", ("/provider".to_string(),));
                                    });
                                }
                            }
                            // Alt+M / F2: Toggle mouse capture mode (wheel scroll vs free text selection without Shift)
                            (KeyModifiers::ALT, KeyCode::Char('m') | KeyCode::Char('M') | KeyCode::Char('ь') | KeyCode::Char('Ь'))
                                | (KeyModifiers::NONE, KeyCode::F(2)) =>
                            {
                                match guard.toggle_mouse_capture() {
                                    Ok(enabled) => {
                                        app.mouse_capture_enabled = enabled;
                                        if enabled {
                                            app.messages.push(MessageItem::Notice(
                                                "Захват мыши: ВКЛ (скролл колесом активен, выделение текста терминалом: зажмите Shift)".to_string(),
                                            ));
                                        } else {
                                            app.messages.push(MessageItem::Notice(
                                                "Захват мыши: ВЫКЛ (свободное выделение текста без Shift)".to_string(),
                                            ));
                                        }
                                    }
                                    Err(e) => {
                                        app.messages.push(MessageItem::Notice(format!("Ошибка переключения мыши: {e}")));
                                    }
                                }
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('u')) => {
                                app.delete_to_start();
                            }
                            // Shift+arrows extend the selection; plain arrows clear it.
                            // `A | B` inside a pattern is an or-pattern, not a bitwise
                            // OR, so "Shift plus optional Alt" needs a guard.
                            (m, KeyCode::Left)
                                if m.contains(KeyModifiers::SHIFT) && m.contains(KeyModifiers::ALT) =>
                            {
                                app.extend_selection(-1);
                            }
                            (m, KeyCode::Left) if m.contains(KeyModifiers::SHIFT) => {
                                app.set_cursor(app.cursor_pos.saturating_sub(1), true);
                            }
                            (m, KeyCode::Right)
                                if m.contains(KeyModifiers::SHIFT) && m.contains(KeyModifiers::ALT) =>
                            {
                                app.extend_selection(1);
                            }
                            (m, KeyCode::Right) if m.contains(KeyModifiers::SHIFT) => {
                                app.set_cursor(app.cursor_pos + 1, true);
                            }
                            (m, KeyCode::Home) if m.contains(KeyModifiers::SHIFT) => {
                                app.set_cursor(0, true);
                            }
                            (m, KeyCode::End) if m.contains(KeyModifiers::SHIFT) => {
                                app.set_cursor(app.input.chars().count(), true);
                            }
                            // With a selection: Ctrl+C copies, Ctrl+X cuts, Backspace
                            // deletes. Without one they keep their normal meaning.
                            (KeyModifiers::CONTROL, KeyCode::Char('y')) => {
                                app.yank();
                            }
                            (KeyModifiers::ALT, KeyCode::Char('y')) => {
                                app.yank_pop();
                            }
                            (KeyModifiers::ALT, KeyCode::Char('d')) => {
                                app.delete_word_right();
                            }
                            (KeyModifiers::ALT, KeyCode::Backspace) => {
                                app.delete_word_left();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('z'))
                            | (KeyModifiers::CONTROL, KeyCode::Char('_')) => {
                                app.undo();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('r')) => {
                                if app.search_active {
                                    app.history_search_prev();
                                } else {
                                    app.history_search_start();
                                }
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('g')) if app.search_active => {
                                app.history_search_cancel();
                            }
                            (KeyModifiers::NONE, KeyCode::Enter) if app.search_active => {
                                app.history_search_accept();
                            }
                            (KeyModifiers::NONE, KeyCode::Esc) if app.search_active => {
                                app.history_search_cancel();
                            }
                            (KeyModifiers::NONE, KeyCode::Backspace) if app.search_active => {
                                app.history_search_backspace();
                            }
                            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c))
                                if app.search_active
                                    && !key.modifiers.contains(KeyModifiers::CONTROL)
                                    && !key.modifiers.contains(KeyModifiers::ALT) =>
                            {
                                app.history_search_push(c);
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('k')) => {
                                app.delete_to_end();
                            }
                            // Ctrl+A selects everything: the whole input, or, with
                            // nothing typed, the whole conversation (Ctrl+C copies it).
                            (
                                KeyModifiers::CONTROL,
                                KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Char('ф') | KeyCode::Char('Ф'),
                            ) => {
                                app.select_all();
                            }
                            (KeyModifiers::CONTROL, KeyCode::Char('e')) => {
                                app.move_cursor_end();
                            }
                            // Chat scroll via Shift+Up/Down, Ctrl+Up/Down, Alt+Up/Down
                            (m, KeyCode::Up)
                                if m.contains(KeyModifiers::SHIFT)
                                    || m.contains(KeyModifiers::CONTROL)
                                    || m.contains(KeyModifiers::ALT) =>
                            {
                                app.scroll_up(3);
                            }
                            (m, KeyCode::Down)
                                if m.contains(KeyModifiers::SHIFT)
                                    || m.contains(KeyModifiers::CONTROL)
                                    || m.contains(KeyModifiers::ALT) =>
                            {
                                app.lower(3);
                            }
                            // PageUp / PageDown with any modifier. A tool card with the
                            // focus pages its own output first: the card is a window, and
                            // the keys should move what the person is looking at.
                            (_, KeyCode::PageUp) if !app.page_focused_card(true) => {
                                app.scroll_up(10);
                            }
                            (_, KeyCode::PageUp) => {}
                            (_, KeyCode::PageDown) if !app.page_focused_card(false) => {
                                app.lower(10);
                            }
                            (_, KeyCode::PageDown) => {}
                            // Ctrl+Home: scroll to very top of chat history
                            (m, KeyCode::Home) if m.contains(KeyModifiers::CONTROL) => {
                                app.scroll_up(usize::MAX);
                            }
                            // Ctrl+End: scroll to newest messages (bottom of chat)
                            (m, KeyCode::End) if m.contains(KeyModifiers::CONTROL) => {
                                app.scroll_down(usize::MAX);
                            }
                            (KeyModifiers::NONE, KeyCode::Esc) => {
                                if app.cancel_completion() {
                                    // Closed the suggestions (and undid a Tab preview).
                                } else if app.queue_edit.is_some() {
                                    // Leave the edit: the message stays in the queue as
                                    // it was, and the draft that was put aside returns.
                                    app.cancel_queue_edit();
                                } else if !app.input.is_empty() {
                                    app.input.clear();
                                    app.cursor_pos = 0;
                                } else if app.is_busy {
                                    Python::with_gil(|py| {
                                        let _ = bridge.call_method0(py, "cancel_turn");
                                    });
                                }
                                // Esc on an idle, empty prompt does nothing. It used to
                                // leave the TUI without saving the session or stopping
                                // MCP servers and background commands.
                            }
                            (KeyModifiers::NONE, KeyCode::Enter) => {
                                // Accepting a completion must NOT also submit the turn:
                                // consume Enter, a second Enter submits (shell behaviour).
                                if (app.show_completions || app.show_file_completions) && app.accept_completion() {
                                    continue;
                                }

                                if let Some(prompt) = app.submit_input() {
                                    // Only TUI-owned commands are handled here; /exit,
                                    // /clear and the rest go to the bridge so their real
                                    // handlers run (chat export, agent history reset).
                                    let prompt = match app.handle_local_command(&prompt) {
                                        Submit::Handled => continue,
                                        Submit::Dispatch => prompt,
                                    };
                                    app.is_command_in_flight = prompt.starts_with('/') || prompt.starts_with('!');
                                    let mut rejected = false;
                                    let turn_ok = Python::with_gil(|py| {
                                        let _ = bridge.call_method1(py, "append_history", (&prompt,));
                                        match bridge.call_method1(py, "start_turn", (prompt.clone(),)) {
                                            Ok(_) => {
                                                rejected = bridge
                                                    .call_method0(py, "turn_was_rejected")
                                                    .and_then(|f| f.extract::<bool>(py))
                                                    .unwrap_or(false);
                                                true
                                            }
                                            Err(e) => {
                                                app.messages.push(MessageItem::Error {
                                                    text: format!("Ошибка запуска: {e}"),
                                                    hint: None,
                                                });
                                                false
                                            }
                                        }
                                    });
                                    if turn_ok {
                                        if rejected {
                                            app.is_command_in_flight = false;
                                            app.requeue_rejected_message(prompt);
                                        } else {
                                            app.is_busy = true;
                                        }
                                    } else {
                                        app.is_command_in_flight = false;
                                    }
                                }
                            }
                            // With a suggestion popup open Tab cycles through the
                            // suggestions (Shift+Tab goes back) and the popup stays;
                            // otherwise Tab acts on the focused collapsible block (a
                            // reasoning accordion or a tool card), as does Shift+Tab.
                            // Shift+Tab arrives as BackTab on most terminals.
                            (KeyModifiers::NONE, KeyCode::Tab) => {
                                app.tab(true);
                            }
                            (_, KeyCode::BackTab) | (KeyModifiers::SHIFT, KeyCode::Tab) => {
                                app.tab(false);
                            }
                            (KeyModifiers::NONE, KeyCode::Up) => {
                                if app.cycle_completion(false, true) {
                                    // moved to the previous suggestion
                                } else if !app.move_cursor_up()
                                    && !app.history_prev()
                                {
                                    // Nothing left going back: pull the newest
                                    // queued message back for editing.
                                    app.recall_last_queued();
                                }
                            }
                            (KeyModifiers::NONE, KeyCode::Down) => {
                                if app.cycle_completion(true, true) {
                                    // moved to the next suggestion
                                } else if !app.move_cursor_down() {
                                    app.history_next();
                                }
                            }
                            (KeyModifiers::NONE, KeyCode::Left) => {
                                app.move_cursor_left();
                            }
                            (KeyModifiers::NONE, KeyCode::Right) => {
                                app.move_cursor_right();
                            }
                            (KeyModifiers::NONE, KeyCode::Home) => {
                                app.move_cursor_home();
                            }
                            (KeyModifiers::NONE, KeyCode::End) => {
                                app.move_cursor_end();
                            }
                            (KeyModifiers::NONE, KeyCode::Backspace) => {
                                app.backspace();
                            }
                            (KeyModifiers::NONE, KeyCode::Delete) => {
                                app.delete_char();
                            }
                            // No special case for Space: it falls through to the Char arm below and is
                            // always inserted as a space.
                            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => {
                                app.insert_char(c);
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            } else if app.selection_tick(std::time::Instant::now()) {
                app.dirty.set(true);
            }
        }

        // The loop is over: _poller_guard stops the poller before the terminal is handed back,
        // so it cannot keep calling into Python while the interpreter shuts down.

        Ok(())
    });

    res.map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
}

#[pymodule]
fn argent_tui(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_tui, m)?)?;
    m.add_class::<RustEventSink>()?;
    Ok(())
}
