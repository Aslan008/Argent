//! LMTrust L6 (Cross-System) x L9 (Negative Space) for the render pipeline.
//!
//! Layer decision tree: this is a cross-layer system (events -> App -> ui.rs ->
//! markdown.rs / toolrender.rs -> ratatui buffer), so the high-ROI layers are
//! L6 (No-Mock E2E + Cross-Layer Consistency) and L9 (what must NOT happen).
//!
//! The pre-existing `test_ui_render_all_message_types` only asserts that
//! `draw_ui` returns `Ok`. That is an L0 smoke assertion: it cannot tell a
//! rendered frame from an empty one. These tests assert on the actual terminal
//! buffer, which is the only place a rendering regression is observable.

use std::collections::HashMap;

use argent_tui::app::{App, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

/// Render one frame and flatten the buffer to text, one string per row.
fn render(app: &App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let area = buf.area;
    let mut out = String::new();
    for y in 0..area.height {
        for x in 0..area.width {
            let cell = &buf.content[(y * area.width + x) as usize];
            out.push_str(cell.symbol());
        }
        out.push('\n');
    }
    out
}

/// A terminal wide and tall enough that nothing is scrolled out of view.
fn render_tall(app: &App) -> String {
    render(app, 160, 60)
}

// ── L6: No-Mock E2E — events in, pixels out ────────────────────────────────

/// L6 x D3. A streamed assistant answer that never had a thinking phase must
/// still reach the screen.
///
/// The obvious blind spot: `test_ui_render_all_message_types` builds an
/// Assistant carrying BOTH `content` and `thinking`, so it can never observe a
/// content-only message. In `ui.rs` the content branch is nested inside
/// `if !thinking.is_empty() || is_streaming_thinking`, so this is exactly the
/// shape that disappears.
#[test]
fn l6_assistant_content_without_thinking_is_rendered() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("UNIQUE_ANSWER_TOKEN".into()));

    let frame = render_tall(&app);
    assert!(
        frame.contains("UNIQUE_ANSWER_TOKEN"),
        "assistant content with no thinking phase never reached the screen.\n--- frame ---\n{frame}"
    );
}

/// L6 x D3. `content_replace` is how a non-streaming (full-text) answer arrives.
/// It creates an Assistant with an empty `thinking` field by construction.
#[test]
fn l6_content_replace_answer_is_rendered() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ContentReplace("REPLACED_FULL_TEXT".into()));

    let frame = render_tall(&app);
    assert!(
        frame.contains("REPLACED_FULL_TEXT"),
        "content_replace answer never reached the screen.\n--- frame ---\n{frame}"
    );
}

/// L6 x D3. Control case: with a thinking phase present the content does render.
/// This pins the difference to the thinking field, not to the renderer.
#[test]
fn l6_assistant_content_with_thinking_is_rendered() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("reasoning first".into()));
    app.handle_event(ArgentEvent::Content("CONTENT_TOKEN".into()));

    let frame = render_tall(&app);
    assert!(
        frame.contains("CONTENT_TOKEN"),
        "control case failed: content with thinking is also invisible.\n--- frame ---\n{frame}"
    );
}

/// L6 x D3. A user turn must be visible without any agent activity at all.
#[test]
fn l6_user_turn_is_rendered() {
    let mut app = App::new();
    app.insert_str("ASK_THE_USER_TOKEN");
    app.submit_input();

    let frame = render_tall(&app);
    assert!(
        frame.contains("ASK_THE_USER_TOKEN"),
        "the user's own prompt is not on screen.\n--- frame ---\n{frame}"
    );
}

/// L6 x D3. Notices and errors carry the feedback the user acts on.
#[test]
fn l6_notice_and_error_are_rendered() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Notice("NOTICE_TOKEN".into()));
    app.handle_event(ArgentEvent::Error {
        text: "ERROR_TOKEN".into(),
        hint: Some("HINT_TOKEN".into()),
    });

    let frame = render_tall(&app);
    for token in ["NOTICE_TOKEN", "ERROR_TOKEN", "HINT_TOKEN"] {
        assert!(frame.contains(token), "{token} is not on screen.\n--- frame ---\n{frame}");
    }
}

/// L6 x Cross-Layer Consistency. The tool header is built in `toolrender.rs`
/// and painted in `ui.rs`; a rename on either side must be caught here, not by
/// a user staring at a blank transcript.
#[test]
fn l6_tool_header_survives_the_layer_boundary() {
    let mut app = App::new();
    let mut args = HashMap::new();
    args.insert("file_path".to_string(), serde_json::json!("CROSS_LAYER_FILE.rs"));
    args.insert("offset".to_string(), serde_json::json!(10));
    args.insert("limit".to_string(), serde_json::json!(5));
    app.handle_event(ArgentEvent::ToolStart {
        name: "read_file".into(),
        args,
    });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "read_file".into(),
        result: "line one\nline two".into(),
        duration_secs: Some(0.2),
    });

    let frame = render_tall(&app);
    assert!(
        // The REPL prints `offset-(offset+limit)`.
        frame.contains("CROSS_LAYER_FILE.rs:10-15"),
        "tool header did not make it across toolrender -> ui.\n--- frame ---\n{frame}"
    );
}

/// L6 x Cross-Layer Consistency. A diff travels events -> App -> `diff_stats`
/// -> ui. The `+++`/`---` file headers must not be counted as changes.
#[test]
fn l6_diff_stats_exclude_file_headers() {
    let mut app = App::new();
    let diff = "--- a/x.rs\n+++ b/x.rs\n@@ -1 +1 @@\n-old\n+new";
    app.handle_event(ArgentEvent::Diff {
        file: "src/x.rs".into(),
        diff: diff.into(),
    });

    let frame = render_tall(&app);
    assert!(
        frame.contains("(+1, -1)"),
        "diff stats are wrong: headers leaked into the counts.\n--- frame ---\n{frame}"
    );
}

/// L6 x D3. Markdown in an assistant answer must be rendered, not shown raw.
#[test]
fn l6_assistant_markdown_reaches_the_screen() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("t".into()));
    app.handle_event(ArgentEvent::Content(
        "# MARKDOWN_HEADING\n\n- MARKDOWN_ITEM\n".into(),
    ));

    let frame = render_tall(&app);
    assert!(
        frame.contains("MARKDOWN_HEADING"),
        "markdown heading missing.\n--- frame ---\n{frame}"
    );
    assert!(
        frame.contains("MARKDOWN_ITEM"),
        "markdown list item missing.\n--- frame ---\n{frame}"
    );
}

/// L6 x D3. Markdown must also survive when there is no thinking phase — the
/// same nesting that hides plain content would hide formatted content.
#[test]
fn l6_assistant_markdown_without_thinking_reaches_the_screen() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("NO_THINK_MD_HEADING".into()));

    let frame = render_tall(&app);
    assert!(
        frame.contains("NO_THINK_MD_HEADING"),
        "markdown answer without a thinking phase is invisible.\n--- frame ---\n{frame}"
    );
}

// ── L9: Negative Space — what must NOT happen ───────────────────────────────

/// L9 x D3. The "expand reasoning" affordance belongs to a thinking block. A
/// message that never had one must not advertise it.
#[test]
fn l9_no_expand_affordance_without_a_thinking_phase() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("PLAIN_ANSWER".into()));

    let frame = render_tall(&app);
    assert!(
        !frame.contains("[+] \u{0440}\u{0430}\u{0437}\u{0432}\u{0435}\u{0440}\u{043d}\u{0443}\u{0442}\u{044c}"),
        "a reasoning expander is offered for a message that has no reasoning.\n--- frame ---\n{frame}"
    );
    assert!(
        !frame.contains("[-] \u{0441}\u{043a}\u{0440}\u{044b}\u{0442}\u{044c}"),
        "a reasoning collapse affordance is offered for a message that has no reasoning.\n--- frame ---\n{frame}"
    );
}

/// L9 x D3. No reasoning block, therefore no `N.Ns` timing footer either.
#[test]
fn l9_no_reasoning_footer_without_a_thinking_phase() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("PLAIN_ANSWER_2".into()));

    let frame = render_tall(&app);
    assert!(
        !frame.contains("\u{0410}\u{043d}\u{0430}\u{043b}\u{0438}\u{0437}"),
        "a reasoning footer is rendered for a message with no reasoning.\n--- frame ---\n{frame}"
    );
}

/// L9 x D3. `MessageItem::Error` already prefixes with "Ошибка:". Re-prefixing
/// a message that already says "Error:" produces "Ошибка: Error: ...".
#[test]
fn l9_error_prefix_is_not_doubled() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Error {
        text: "Error: already prefixed".into(),
        hint: None,
    });

    let frame = render_tall(&app);
    assert!(
        !frame.contains("\u{041e}\u{0448}\u{0438}\u{0431}\u{043a}\u{0430}: Error:"),
        "the error prefix was applied twice.\n--- frame ---\n{frame}"
    );
}

/// L9 x D4. A turn that finished must not leave a spinner running forever.
#[test]
fn l9_finished_turn_does_not_render_a_spinner() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("done talking".into()));
    app.handle_event(ArgentEvent::Done);
    app.tick = 0;

    let frame = render_tall(&app);
    assert!(
        !frame.contains("\u{0a0b}"),
        "a braille spinner is still painted after Done.\n--- frame ---\n{frame}"
    );
}

/// L9 x D3. An empty transcript is the prompt and the toolbar, like a fresh
/// terminal: there is no welcome strip of its own (the bridge sends the REPL's
/// banner as an event).
#[test]
fn l9_empty_transcript_is_just_the_prompt_and_toolbar() {
    let app = App::new();
    let frame = render_tall(&app);
    assert!(frame.contains('\u{276f}'), "the prompt glyph is missing:\n{frame}");
    assert!(frame.contains("CHAT"), "the toolbar is missing:\n{frame}");
    assert!(!frame.contains("ARGENT CODER"), "the old header strip is back:\n{frame}");
}

/// L6. The bridge's start screen (a banner event, then system lines) reaches
/// the screen the way the REPL prints it.
#[test]
fn l6_start_screen_events_are_rendered() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Banner("9.9".into()));
    app.handle_event(ArgentEvent::System("Провайдер: TEST".into()));
    let frame = render_tall(&app);
    assert!(frame.contains("Argent Coder v9.9"), "{frame}");
    assert!(frame.contains("Провайдер: TEST"), "{frame}");
}

/// L9 x D4. `Done` backfills a result for tools still marked running. It must
/// not invent one for tools that already reported.
#[test]
fn l9_done_does_not_overwrite_a_finished_tool_result() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ToolStart {
        name: "read_file".into(),
        args: HashMap::new(),
    });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "read_file".into(),
        result: "ORIGINAL_RESULT".into(),
        duration_secs: Some(0.1),
    });
    app.handle_event(ArgentEvent::Done);

    match app.messages.iter().find(|m| matches!(m, MessageItem::Tool { .. })) {
        Some(MessageItem::Tool { result, .. }) => assert_eq!(
            result.as_deref(),
            Some("ORIGINAL_RESULT"),
            "Done clobbered a result the tool had already reported"
        ),
        other => panic!("no tool card: {other:?}"),
    }
}

/// L9 x D5. Nothing in a rendered frame may be a lone surrogate or otherwise
/// unprintable; a broken glyph means a lossy slice somewhere in the pipeline.
#[test]
fn l9_render_never_panics_across_degenerate_terminal_sizes() {
    let mut app = App::new();
    app.insert_str("wide \u{65e5}\u{672c}\u{8a9e} tail");
    app.submit_input();
    app.handle_event(ArgentEvent::Thinking("reasoning".into()));
    app.handle_event(ArgentEvent::Content("# heading\n\n- item\n\n```rust\nlet x = 1;\n```".into()));
    app.handle_event(ArgentEvent::ToolStart {
        name: "run_command".into(),
        args: HashMap::new(),
    });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "ok".into(),
        duration_secs: Some(0.3),
    });

    // 0x0 up to a size that forces wrapping, list re-flow and modal centring.
    for (w, h) in [(0u16, 0u16), (1, 1), (4, 4), (10, 6), (20, 10), (80, 24), (200, 60)] {
        let frame = render(&app, w, h);
        assert!(!frame.contains('\u{fffd}'), "replacement char at {w}x{h}");
    }
}
