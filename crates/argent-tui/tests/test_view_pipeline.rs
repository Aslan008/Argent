//! The transcript pipeline: block cache, visible-window rendering, scrolling,
//! the prompt/menu/toolbar stack, clicks, and event ordering.
//!
//! These use a counting renderer so "rendered once per width" and "only the
//! visible tail is rendered" are measured, not assumed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use argent_tui::app::{App, ClickAction, CompletionItem, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::renderer::{BlockRenderer, FallbackRenderer};
use argent_tui::ui::{draw_ui, toolbar};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};
use ratatui::text::Line;
use ratatui::Terminal;
use serde_json::Value;

/// Delegates to the Rust fallback and counts how often it is asked.
struct Counting {
    calls: Arc<AtomicUsize>,
}

impl BlockRenderer for Counting {
    fn render(&self, kind: &str, data: &Value, width: u16) -> Result<Vec<Line<'static>>, String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        FallbackRenderer.render(kind, data, width)
    }
}

/// Always fails, to exercise the degraded path.
struct Broken;

impl BlockRenderer for Broken {
    fn render(&self, kind: &str, _d: &Value, _w: u16) -> Result<Vec<Line<'static>>, String> {
        Err(format!("boom in {kind}"))
    }
}

fn counted_app() -> (App, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut app = App::new();
    app.renderer = Box::new(Counting { calls: calls.clone() });
    (app, calls)
}

fn draw(app: &App, w: u16, h: u16) -> Buffer {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    term.backend().buffer().clone()
}

fn rows(buf: &Buffer) -> Vec<String> {
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol())
                .collect::<String>()
                .trim_end()
                .to_string()
        })
        .collect()
}

fn frame(app: &App, w: u16, h: u16) -> String {
    rows(&draw(app, w, h)).join("\n")
}

fn tool(name: &str, result: &str) -> MessageItem {
    MessageItem::Tool {
        name: name.into(),
        args: HashMap::new(),
        result: Some(result.into()),
        duration_secs: Some(0.01),
        expanded: false,
        is_running: false,
    }
}

// ── the block cache ─────────────────────────────────────────────────────────

#[test]
fn a_block_is_rendered_once_per_width() {
    let (mut app, calls) = counted_app();
    app.messages.push(MessageItem::User("hi".into()));
    app.messages.push(MessageItem::assistant("answer\n\n", "reasoning text"));
    app.messages.push(tool("read_file", "a\nb"));
    app.messages.push(MessageItem::Diff { file: "f.py".into(), diff: "+x".into() });

    draw(&app, 100, 50);
    let first = calls.load(Ordering::SeqCst);
    assert!(first >= 4, "the blocks were never rendered ({first})");

    draw(&app, 100, 50);
    assert_eq!(calls.load(Ordering::SeqCst), first, "an unchanged frame must not re-render anything");

    draw(&app, 80, 50);
    let after_resize = calls.load(Ordering::SeqCst);
    assert!(after_resize > first, "a new width needs new rows");

    draw(&app, 80, 50);
    assert_eq!(calls.load(Ordering::SeqCst), after_resize);
}

#[test]
fn streaming_renders_only_the_chunk_that_just_completed() {
    let (mut app, calls) = counted_app();
    app.handle_event(ArgentEvent::Content("first\n\n".into()));
    draw(&app, 100, 40);
    let c1 = calls.load(Ordering::SeqCst);

    app.handle_event(ArgentEvent::Content("second\n\n".into()));
    draw(&app, 100, 40);
    assert_eq!(calls.load(Ordering::SeqCst) - c1, 1, "only the new chunk is rendered");

    // The unfinished tail is live text, never sent to the block renderer.
    app.handle_event(ArgentEvent::Content("typing".into()));
    draw(&app, 100, 40);
    assert_eq!(calls.load(Ordering::SeqCst) - c1, 1, "an incomplete block must not be rendered");
    assert!(frame(&app, 100, 40).contains("typing"), "the live tail is on screen");
}

#[test]
fn a_changed_message_invalidates_only_itself() {
    let (mut app, calls) = counted_app();
    app.messages.push(tool("read_file", "one"));
    app.messages.push(tool("grep_search", "two"));
    draw(&app, 100, 40);
    let c1 = calls.load(Ordering::SeqCst);

    if let Some(MessageItem::Tool { expanded, .. }) = app.messages.get_mut(1) {
        *expanded = true;
    }
    draw(&app, 100, 40);
    assert_eq!(calls.load(Ordering::SeqCst) - c1, 1, "toggling one card re-renders that card only");
}

#[test]
fn stale_cache_entries_are_pruned() {
    let (mut app, _calls) = counted_app();
    for i in 0..400 {
        app.messages.push(tool("read_file", &format!("r{i}")));
    }
    // Scroll to the top so every message gets rendered and cached.
    app.scroll_offset = usize::MAX / 4;
    draw(&app, 100, 30);
    assert!(app.view_cache.borrow().len() > 100);

    app.messages.truncate(5);
    app.scroll_offset = 0;
    draw(&app, 100, 30);
    assert!(app.view_cache.borrow().len() <= 100, "entries of removed messages linger");
}

#[test]
fn a_render_failure_degrades_one_block_and_is_reported_once() {
    let mut app = App::new();
    app.renderer = Box::new(Broken);
    app.messages.push(MessageItem::Diff { file: "f.py".into(), diff: "+KEEP_ME".into() });
    app.messages.push(tool("read_file", "TOOL_RESULT_SURVIVES"));

    let f = frame(&app, 100, 30);
    assert!(f.contains("KEEP_ME"), "the fallback must still show the diff:\n{f}");
    assert!(f.contains("read_file"), "the fallback must still show the tool:\n{f}");

    let err = app.render_error.borrow().clone();
    assert!(err.is_some_and(|e| e.contains("boom in")), "the failure was not recorded");
    // Taking it clears it; further failures do not pile up a list.
    app.render_error.borrow_mut().take();
    assert!(app.render_error.borrow().is_none());
}

// ── only what is visible is built ───────────────────────────────────────────

#[test]
fn a_long_session_renders_only_the_visible_tail() {
    let (mut app, calls) = counted_app();
    for i in 0..600 {
        app.messages.push(tool("read_file", &format!("result {i}")));
    }
    draw(&app, 100, 20);
    let n = calls.load(Ordering::SeqCst);
    assert!(n < 40, "{n} blocks rendered for a 20-row screen; the whole history was walked");
    assert!(frame(&app, 100, 20).contains("result 599"), "the newest message must be on screen");
}

#[test]
fn scrolling_up_reveals_older_messages_progressively() {
    let (mut app, calls) = counted_app();
    for i in 0..300 {
        app.messages.push(tool("read_file", &format!("entry {i:03}")));
    }
    draw(&app, 100, 20);
    let tail_only = calls.load(Ordering::SeqCst);
    assert!(app.max_scroll.get() > 0, "there is more above than what was built");

    app.scroll_up(200);
    let f = frame(&app, 100, 20);
    assert!(calls.load(Ordering::SeqCst) > tail_only, "scrolling must build older rows");
    assert!(!f.contains("entry 299"), "the newest rows should be below the window:\n{f}");
}

#[test]
fn scrolling_past_the_top_clamps_to_the_first_message() {
    let (mut app, _c) = counted_app();
    for i in 0..60 {
        app.messages.push(MessageItem::User(format!("message number {i}")));
    }
    app.scroll_offset = 1_000_000;
    let f = frame(&app, 100, 20);
    assert!(f.contains("message number 0"), "the very first message must be reachable:\n{f}");
    // The window is full, not a half-empty one hanging from the top.
    let filled = f.lines().filter(|l| l.contains("message number")).count();
    assert!(filled >= 15, "only {filled} rows filled after clamping:\n{f}");
    assert!(app.max_scroll.get() > 0);
}

#[test]
fn a_short_transcript_sits_at_the_top_with_the_prompt_under_it() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Banner("1.0".into()));
    app.handle_event(ArgentEvent::System("first line".into()));
    let r = rows(&draw(&app, 100, 30));
    assert!(r[0].contains("Argent Coder"), "{r:?}");
    assert!(r[1].contains("first line"), "{r:?}");
    assert!(r[2].starts_with('\u{276f}'), "the prompt must follow the transcript directly: {r:?}");
    assert!(r[29].contains("CHAT"), "the toolbar sits on the last row: {:?}", r[29]);
}

// ── prompt ──────────────────────────────────────────────────────────────────

fn cursor(app: &App, w: u16, h: u16) -> (u16, u16) {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    let p = term.get_cursor_position().unwrap();
    (p.x, p.y)
}

#[test]
fn the_cursor_sits_after_the_typed_text() {
    let mut app = App::new();
    app.insert_str("ab");
    let (x, y) = cursor(&app, 100, 20);
    assert_eq!(x, 4, "two cells of prompt plus two characters");
    assert_eq!(y, 0, "no transcript: the prompt is the first row");
}

#[test]
fn the_cursor_follows_wrapped_and_wide_text() {
    let mut app = App::new();
    app.insert_str(&"x".repeat(25));
    let (x, y) = cursor(&app, 12, 20); // 10 usable columns per row
    assert_eq!((x, y), (2 + 5, 2), "25 chars at 10 per row end on the third row, column 5");

    let mut wide = App::new();
    wide.insert_str("\u{4e2d}\u{6587}"); // two double-width glyphs
    let (x, _) = cursor(&wide, 100, 20);
    assert_eq!(x, 2 + 4, "wide glyphs take two columns each");
}

#[test]
fn multiline_input_indents_continuation_rows() {
    let mut app = App::new();
    app.insert_str("one");
    app.insert_char('\n');
    app.insert_str("two");
    let r = rows(&draw(&app, 100, 20));
    assert!(r[0].starts_with("\u{276f} one"), "{r:?}");
    assert!(r[1].starts_with("  two"), "{r:?}");
}

#[test]
fn while_a_turn_runs_typing_is_echoed_with_the_repls_label() {
    let mut app = App::new();
    app.is_busy = true;
    app.insert_str("next idea");
    let f = frame(&app, 120, 20);
    assert!(f.contains("\u{276f} [Ввод на лету]: next idea"), "{f}");
    assert!(f.contains("(Enter: в очередь"), "the key hint is missing:\n{f}");
}

#[test]
fn while_a_turn_runs_an_empty_prompt_shows_no_prompt_row() {
    let mut app = App::new();
    app.is_busy = true;
    app.turn_started = Some(std::time::Instant::now());
    let f = frame(&app, 120, 20);
    assert!(!f.contains("[Ввод на лету]"), "{f}");
    assert!(f.contains("Ожидание ответа"), "the waiting spinner is the only live row:\n{f}");
}

#[test]
fn a_selection_in_the_prompt_is_drawn_reversed() {
    let mut app = App::new();
    app.insert_str("hello");
    app.set_cursor(0, false);
    app.extend_selection(3);
    let buf = draw(&app, 100, 20);
    assert!(buf[(2, 0)].modifier.contains(Modifier::REVERSED), "the selected glyph is not highlighted");
    assert!(!buf[(6, 0)].modifier.contains(Modifier::REVERSED), "unselected text must stay plain");
}

// ── completion menu (prompt_toolkit's) ──────────────────────────────────────

fn app_with_commands(typed: &str) -> App {
    let mut app = App::new();
    for (n, h) in [("/model", "choose a model"), ("/mode", "switch mode"), ("/help", "show help")] {
        app.completions.push(CompletionItem { name: n.into(), help: h.into() });
    }
    app.insert_str(typed);
    app
}

#[test]
fn the_completion_menu_opens_below_the_prompt() {
    let app = app_with_commands("/mo");
    let buf = draw(&app, 100, 20);
    let r = rows(&buf);
    assert!(r[0].contains("/mo"), "{r:?}");
    assert!(r[1].contains("/model") && r[1].contains("choose a model"), "{r:?}");
    assert!(r[2].contains("/mode"), "{r:?}");
    assert!(!r[3].contains("/help"), "a non-matching command is listed: {r:?}");
}

#[test]
fn the_menu_uses_prompt_toolkits_colours() {
    let mut app = app_with_commands("/mo");
    app.completion_idx = 1;
    let buf = draw(&app, 100, 20);
    // Row 1 is a plain entry (grey bar), row 2 is the current one (reversed).
    assert_eq!(buf[(3, 1)].bg, Color::Rgb(0xbb, 0xbb, 0xbb));
    assert_eq!(buf[(3, 1)].fg, Color::Black);
    assert!(buf[(3, 2)].modifier.contains(Modifier::REVERSED), "the current entry must be reversed");
}

#[test]
fn the_menu_shows_at_most_eight_rows() {
    let mut app = App::new();
    for i in 0..20 {
        app.completions.push(CompletionItem { name: format!("/cmd{i:02}"), help: String::new() });
    }
    app.insert_str("/cmd");
    let f = frame(&app, 100, 30);
    let shown = f.lines().filter(|l| l.contains("/cmd")).count();
    // One of those rows is the prompt itself.
    assert_eq!(shown - 1, 8, "{f}");
}

// ── toolbar ─────────────────────────────────────────────────────────────────

#[test]
fn toolbar_segments_carry_hidden_click_targets() {
    let mut app = App::new();
    app.status.provider = "openrouter".into();
    app.status.model = "qwen3".into();
    app.status.thinking_label = "AUTO".into();
    let (line, targets) = toolbar(&app, 200);
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();

    let model = targets.iter().find(|(_, _, a)| *a == ClickAction::OpenModelPicker).expect("model target");
    let think = targets.iter().find(|(_, _, a)| *a == ClickAction::CycleThinkingLevel).expect("think target");
    let slice = |a: u16, b: u16| text.chars().skip(a as usize).take((b - a) as usize).collect::<String>();
    assert_eq!(slice(model.0, model.1), "openrouter:qwen3");
    assert_eq!(slice(think.0, think.1), "think:AUTO");
}

#[test]
fn the_toolbar_is_clipped_to_the_width_without_breaking() {
    let mut app = App::new();
    app.status.provider = "p".repeat(40);
    app.status.model = "m".repeat(40);
    for w in [10u16, 20, 40, 60] {
        let (line, targets) = toolbar(&app, w);
        assert_eq!(argent_tui::ansi::line_width(&line), w as usize, "width {w}");
        assert!(targets.iter().all(|(a, _, _)| *a < w));
    }
}

#[test]
fn context_percentage_is_colour_coded_like_the_repl() {
    let mut app = App::new();
    app.status.ctx_tokens = 10;
    for (pct, want) in [(10.0, Color::Rgb(0x4c, 0xaf, 0x50)), (60.0, Color::Rgb(0xff, 0xb3, 0x00)), (90.0, Color::Rgb(0xe5, 0x39, 0x35))] {
        app.status.ctx_percent = pct;
        let (line, _) = toolbar(&app, 200);
        let colours: Vec<Option<Color>> = line.spans.iter().filter(|s| s.content.ends_with('%')).map(|s| s.style.fg).collect();
        assert_eq!(colours, vec![Some(want)], "at {pct}%");
    }
}

// ── clicks and selection ────────────────────────────────────────────────────

#[test]
fn tool_and_reasoning_rows_are_click_targets() {
    let mut app = App::new();
    app.messages.push(MessageItem::assistant("answer\n\n", "some reasoning"));
    app.messages.push(tool("read_file", "x"));
    draw(&app, 100, 30);

    let r = rows(&draw(&app, 100, 30));
    let tool_row = r.iter().position(|l| l.contains("read_file")).unwrap() as u16;
    let reasoning_row = r.iter().position(|l| l.contains('\u{25c7}')).unwrap() as u16;

    assert_eq!(app.get_clicked_action(5, tool_row), Some(ClickAction::ToggleTool(1)));
    assert_eq!(app.get_clicked_action(5, reasoning_row), Some(ClickAction::ToggleThinking(0)));
    let answer_row = r.iter().position(|l| l.contains("answer")).unwrap() as u16;
    assert_eq!(app.get_clicked_action(5, answer_row), None, "plain answer text is not a button");
}

#[test]
fn a_mouse_selection_copies_the_rows_under_it() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::System("alpha beta gamma".into()));
    app.handle_event(ArgentEvent::System("second row here".into()));
    draw(&app, 100, 20);
    assert!(app.in_transcript(0) && app.in_transcript(1));
    assert!(!app.in_transcript(5), "below the transcript is not selectable text");

    assert!(app.select_cells((6, 0), (9, 0)));
    assert_eq!(app.selected_transcript_text().as_deref(), Some("beta"));

    assert!(app.select_cells((6, 0), (5, 1)));
    assert_eq!(app.selected_transcript_text().as_deref(), Some("beta gamma\nsecond"));

    // Dragging the other way gives the same text.
    assert!(app.select_cells((5, 1), (6, 0)));
    assert_eq!(app.selected_transcript_text().as_deref(), Some("beta gamma\nsecond"));
}

#[test]
fn the_selection_is_painted_reversed() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::System("highlight me".into()));
    draw(&app, 100, 20);
    assert!(app.select_cells((0, 0), (3, 0)));
    let buf = draw(&app, 100, 20);
    assert!(buf[(0, 0)].modifier.contains(Modifier::REVERSED));
    assert!(buf[(3, 0)].modifier.contains(Modifier::REVERSED));
    assert!(!buf[(8, 0)].modifier.contains(Modifier::REVERSED));
}

// ── event ordering (what the REPL prints, in the order it prints it) ───────

fn kinds(app: &App) -> Vec<&'static str> {
    app.messages
        .iter()
        .map(|m| match m {
            MessageItem::User(_) => "user",
            MessageItem::Assistant { .. } => "assistant",
            MessageItem::Rethink { .. } => "rethink",
            MessageItem::Tool { .. } => "tool",
            MessageItem::Diff { .. } => "diff",
            MessageItem::Snapshot { .. } => "snapshot",
            MessageItem::ToolGenerating { .. } => "toolgen",
            MessageItem::Notice(_) => "notice",
            MessageItem::System(_) => "system",
            MessageItem::Banner(_) => "banner",
            MessageItem::Error { .. } => "error",
            MessageItem::Checkpoint { .. } => "checkpoint",
        })
        .collect()
}

#[test]
fn output_printed_while_a_tool_runs_lands_above_its_result() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ToolStart { name: "replace_in_file".into(), args: HashMap::new() });
    app.handle_event(ArgentEvent::Diff { file: "f.py".into(), diff: "+x".into() });
    app.handle_event(ArgentEvent::Notice("note".into()));
    app.handle_event(ArgentEvent::Checkpoint { sha: "abc".into(), label: "l".into() });
    assert_eq!(kinds(&app), vec!["diff", "notice", "checkpoint", "tool"]);

    app.handle_event(ArgentEvent::ToolEnd {
        name: "replace_in_file".into(),
        result: "ok".into(),
        duration_secs: Some(0.1),
    });
    match app.messages.last() {
        Some(MessageItem::Tool { is_running: false, result: Some(r), .. }) => assert_eq!(r, "ok"),
        other => panic!("the tool result went to the wrong item: {other:?}"),
    }
}

#[test]
fn reasoning_that_resumes_after_the_answer_is_a_new_block() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("first thoughts".into()));
    app.handle_event(ArgentEvent::Content("answer".into()));
    app.handle_event(ArgentEvent::Thinking("second thoughts".into()));
    assert_eq!(kinds(&app), vec!["assistant", "assistant"]);
    match (&app.messages[0], &app.messages[1]) {
        (MessageItem::Assistant { thinking: a, content: c, .. }, MessageItem::Assistant { thinking: b, .. }) => {
            assert_eq!(a, "first thoughts");
            assert_eq!(c, "answer");
            assert_eq!(b, "second thoughts");
        }
        _ => unreachable!(),
    }
}

#[test]
fn a_rethink_is_a_block_of_its_own() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("first pass".into()));
    app.handle_event(ArgentEvent::Rethink("checking it".into()));
    app.handle_event(ArgentEvent::Rethink(" again".into()));
    app.handle_event(ArgentEvent::Done);
    assert_eq!(kinds(&app), vec!["assistant", "rethink"]);
    let f = frame(&app, 120, 30);
    assert!(f.contains("◇ Анализ") && f.contains("◈ Перепроверка"), "{f}");
    match &app.messages[1] {
        MessageItem::Rethink { text, is_streaming, .. } => {
            assert_eq!(text, "checking it again");
            assert!(!is_streaming, "Done must close the stream");
        }
        _ => unreachable!(),
    }
}

#[test]
fn full_thinking_mode_opens_new_reasoning_by_default() {
    let mut app = App::new();
    app.ui.thinking_mode = "full".into();
    app.handle_event(ArgentEvent::Thinking("deep thoughts".into()));
    app.handle_event(ArgentEvent::Done);
    match &app.messages[0] {
        MessageItem::Assistant { thinking_expanded, .. } => assert!(*thinking_expanded),
        _ => unreachable!(),
    }
    assert!(frame(&app, 100, 30).contains("└── "), "the full reasoning tree is not drawn");
}

#[test]
fn the_queue_box_replaces_the_old_queued_notice() {
    let mut app = App::new();
    app.is_busy = true;
    app.insert_str("later");
    assert!(app.submit_input().is_none(), "a busy app queues instead of starting a turn");
    assert_eq!(app.message_queue, vec!["later".to_string()]);
    assert!(
        !app.messages.iter().any(|m| matches!(m, MessageItem::Notice(_))),
        "the REPL shows a box, not a notice"
    );
    assert!(frame(&app, 100, 30).contains("Очередь сообщений (1)"));
}

#[test]
fn streamed_tool_arguments_are_shown_in_the_generation_box() {
    let mut app = App::new();
    app.is_busy = true;
    app.handle_event(ArgentEvent::ToolGenerating {
        name: "write_file".into(),
        bytes: 20,
        delta: "{\"file_path\": \"x.py\"".into(),
    });
    app.handle_event(ArgentEvent::ToolGenerating {
        name: "write_file".into(),
        bytes: 40,
        delta: ", \"content\": \"hi\"}".into(),
    });
    let f = frame(&app, 100, 30);
    assert!(f.contains("Генерация: write_file (40 B"), "{f}");
    assert!(f.contains("{\"file_path\": \"x.py\", \"content\": \"hi\"}"), "the streamed text is missing:\n{f}");
}

// ── approval prompt data ────────────────────────────────────────────────────

#[test]
fn approval_wording_and_decisions_follow_the_repl() {
    use argent_tui::app::ApprovalData;
    let granted = ApprovalData {
        id: 1,
        action: "git status".into(),
        destructive: false,
        grant_key: Some("git".into()),
        prompt: None,
        selected: 1,
    };
    let p = granted.resolved_prompt();
    assert_eq!(p.question, "Разрешить это действие?");
    assert!(!p.confirm);
    assert_eq!(p.choices.len(), 3);
    assert!(p.choices[1].contains("'git'"));
    assert_eq!(granted.selected_decision(), "always");

    let destructive = ApprovalData { destructive: true, selected: 0, ..granted.clone() };
    let p = destructive.resolved_prompt();
    assert!(p.confirm, "a destructive action gets a y/N question");
    assert!(!p.decisions.iter().any(|d| d == "always"));

    // A pointer past the end never panics and falls back to the last choice.
    let wild = ApprovalData { selected: 99, ..granted };
    assert_eq!(wild.selected_decision(), "deny");
}

// ── robustness: no state may panic at any terminal size ─────────────────────

fn states() -> Vec<(&'static str, App)> {
    let mut v: Vec<(&'static str, App)> = Vec::new();

    v.push(("idle", App::new()));

    let mut menu = app_with_commands("/mo");
    for i in 0..12 {
        menu.completions.push(CompletionItem { name: format!("/mo{i}"), help: "help text".into() });
    }
    v.push(("menu", menu));

    let mut busy = App::new();
    busy.is_busy = true;
    busy.insert_str("typing while busy\nwith two lines");
    busy.message_queue = vec!["queued one".into(), "queued two".into()];
    v.push(("busy", busy));

    let mut long = App::new();
    long.insert_str(&"wrap me ".repeat(60));
    v.push(("long input", long));

    let mut approval = App::new();
    approval.handle_event(ArgentEvent::ApprovalRequest {
        id: 1,
        action: "rm -rf build".into(),
        destructive: true,
        grant_key: None,
        prompt: None,
    });
    v.push(("approval", approval));

    let mut select = App::new();
    select.active_select = Some(argent_tui::app::SelectModalData::new(
        "s".into(),
        "Pick one".into(),
        (0..40).map(|i| format!("option {i}")).collect(),
        Some("option 20".into()),
    ));
    v.push(("select", select));

    let mut rich = App::new();
    rich.handle_event(ArgentEvent::Banner("1.0".into()));
    rich.handle_event(ArgentEvent::Thinking("deep thought\n".repeat(30)));
    rich.handle_event(ArgentEvent::Content("# H\n\n```rust\nlet x = 1;\n```\n\npartial".into()));
    rich.handle_event(ArgentEvent::ToolStart { name: "run_command".into(), args: HashMap::new() });
    rich.handle_event(ArgentEvent::ToolStart { name: "read_file".into(), args: HashMap::new() });
    rich.messages.push(tool("grep_search", &"line\n".repeat(80)));
    rich.is_busy = true;
    v.push(("rich transcript", rich));

    v
}

#[test]
fn no_state_panics_at_any_terminal_size() {
    for (name, app) in states() {
        for w in (1u16..=40).chain([80, 120, 300]) {
            for h in (1u16..=24).chain([60, 200]) {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = draw(&app, w, h);
                }));
                assert!(result.is_ok(), "{name}: draw panicked at {w}x{h}");
            }
        }
    }
}

#[test]
fn nothing_is_drawn_outside_the_frame_and_rows_never_overflow() {
    for (name, app) in states() {
        for (w, h) in [(20u16, 6u16), (30, 8), (40, 10), (80, 24)] {
            let buf = draw(&app, w, h);
            // The buffer is exactly the frame; every row is exactly `w` wide.
            assert_eq!(buf.area.width, w);
            assert_eq!(buf.area.height, h);
            let toolbar_row = row_text_of(&buf, h - 1);
            assert!(
                toolbar_row.contains("CHAT") || w < 30,
                "{name} at {w}x{h}: the toolbar is not on the last row: {toolbar_row:?}"
            );
        }
    }
}

fn row_text_of(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>()
}
