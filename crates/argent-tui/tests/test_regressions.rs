//! Regression tests for bugs found in the 2026-10-03 audit of `argent-tui`.
//!
//! Each test here fails against the pre-fix code and passes after.

use argent_tui::app::{App, CompletionItem, MessageItem, SelectModalData};
use argent_tui::events::ArgentEvent;

fn eight_options() -> Vec<String> {
    ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

// ── Bug 1: selected_idx lived in the filtered index space ────────────────────

#[test]
fn regression_select_filter_returns_option_not_typed_text() {
    // default "hotel" sits at index 7 of the FULL list.
    let mut sel = SelectModalData::new("id".into(), "t".into(), eight_options(), Some("hotel".into()));
    assert_eq!(sel.selected_idx, 7);

    // Narrowing the filter must not make the answer collapse to "h".
    sel.filter = "h".into();
    let value = sel.selected_value().unwrap_or_default();
    assert!(
        eight_options().contains(&value),
        "answer must be one of the options, got {:?}",
        value
    );
    assert_eq!(value, "hotel", "tracked selection should survive filtering");
}

#[test]
fn regression_select_never_returns_filter_text_while_options_match() {
    for ch in ["a", "b", "c", "d", "e", "f", "g", "h", "o", "l", "t"] {
        let mut sel =
            SelectModalData::new("id".into(), "t".into(), eight_options(), Some("hotel".into()));
        sel.filter = ch.to_string();
        let value = sel.selected_value().unwrap_or_default();
        assert!(
            eight_options().contains(&value),
            "filter {:?} produced non-option answer {:?}",
            ch,
            value
        );
    }
}

#[test]
fn regression_select_freetext_only_when_nothing_matches() {
    let mut sel =
        SelectModalData::new("id".into(), "t".into(), eight_options(), Some("hotel".into()));
    // No option contains "zzz" -> the typed text IS the intended answer.
    sel.filter = "zzz".into();
    assert_eq!(sel.selected_value(), Some("zzz".to_string()));
}

// ── Bug 2: move_up/move_down corrupted selected_idx under a filter ───────────

#[test]
fn regression_select_navigation_keeps_index_in_range() {
    for default_idx in 0..8 {
        let mut sel = SelectModalData::new(
            "id".into(),
            "t".into(),
            eight_options(),
            Some(eight_options()[default_idx].clone()),
        );
        sel.filter = "o".into(); // narrows to a handful of options
        let len = sel.filtered_options().len();
        assert!(len > 0);

        for _ in 0..12 {
            sel.move_up();
            assert!(
                sel.selected_idx < eight_options().len(),
                "selected_idx {} escaped the options list",
                sel.selected_idx
            );
            assert!(
                eight_options()
                    .get(sel.selected_idx)
                    .map(|o| o.to_lowercase().contains("o"))
                    .unwrap_or(false),
                "selected_idx {} does not point at a filtered option",
                sel.selected_idx
            );
            sel.move_down();
            assert!(
                eight_options()
                    .get(sel.selected_idx)
                    .map(|o| o.to_lowercase().contains("o"))
                    .unwrap_or(false),
                "selected_idx {} does not point at a filtered option",
                sel.selected_idx
            );
        }
    }
}

#[test]
fn regression_select_navigation_never_returns_filter_text() {
    let mut sel =
        SelectModalData::new("id".into(), "t".into(), eight_options(), Some("hotel".into()));
    sel.filter = "o".into();
    for _ in 0..12 {
        sel.move_up();
        let v = sel.selected_value().unwrap_or_default();
        assert!(eight_options().contains(&v), "navigation produced {:?}", v);
    }
}

#[test]
fn regression_select_empty_option_list_is_safe() {
    let mut sel = SelectModalData::new("id".into(), "t".into(), Vec::new(), Some("fallback".into()));
    assert_eq!(sel.filter, "fallback");
    assert_eq!(sel.selected_value(), Some("fallback".to_string()));
    sel.move_up();
    sel.move_down();
    assert_eq!(sel.selected_value(), Some("fallback".to_string()));
}

// ── Bug 6: cancel used to emit "done", so a queued message could start a
//          second concurrent turn ─────────────────────────────────────────────

#[test]
fn regression_cancelling_keeps_turn_busy() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("partial".into()));
    assert!(app.is_busy);

    // Cancel acknowledged, turn still unwinding: must stay busy so the main
    // loop does not dispatch the next queued message.
    app.handle_event(ArgentEvent::Cancelling);
    assert!(app.is_busy, "cancelling must not look like a finished turn");
    assert!(app.is_cancelling);

    // Only the worker's terminal Done clears it.
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_busy);
    assert!(!app.is_cancelling);
}

#[test]
fn regression_queue_waits_while_cancelling() {
    let mut app = App::new();
    app.insert_str("first");
    assert_eq!(app.submit_input(), Some("first".to_string()));
    // The turn only becomes busy once the agent starts streaming.
    app.handle_event(ArgentEvent::Content("working".into()));
    assert!(app.is_busy);

    app.insert_str("second");
    assert!(app.submit_input().is_none(), "while busy the prompt is queued");
    assert_eq!(app.message_queue.len(), 1);

    app.handle_event(ArgentEvent::Cancelling);
    assert!(
        app.pop_next_queued_message().is_none(),
        "queued message must not start while the cancelled turn unwinds"
    );

    app.handle_event(ArgentEvent::Done);
    assert_eq!(
        app.pop_next_queued_message(),
        Some("second".to_string()),
        "queue must drain only after the real Done"
    );
}

#[test]
fn regression_rejected_turn_is_requeued_at_front() {
    let mut app = App::new();
    app.insert_str("older");
    assert_eq!(app.submit_input(), Some("older".to_string()));
    app.handle_event(ArgentEvent::Content("working".into()));
    assert!(app.is_busy);

    app.insert_str("newest");
    assert!(app.submit_input().is_none());
    assert_eq!(app.message_queue.len(), 1);

    // The turn ends, so the queue is allowed to dispatch its head.
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_busy);

    // Newest prompt is dispatched optimistically, then the bridge refuses.
    let prompt = app.pop_next_queued_message().unwrap();
    assert_eq!(prompt, "newest");
    assert!(app.is_busy);

    app.requeue_rejected_message(prompt);

    assert_eq!(app.message_queue.first().map(|s| s.as_str()), Some("newest"));
    assert_eq!(app.message_queue.len(), 1);
    assert!(!app.is_busy, "a refused turn must not leave the UI busy");
    // The optimistic bubble was rolled back, so no duplicate is shown.
    assert!(
        !matches!(app.messages.last(), Some(MessageItem::User(u)) if *u == "newest"),
        "rejected prompt must not leave a User bubble behind"
    );
}

// ── Bug 8: completions were not recomputed when the cursor moved ────────────

fn app_with_commands() -> App {
    let mut app = App::new();
    for name in ["/help", "/model", "/clear"] {
        app.completions.push(CompletionItem { name: name.into(), help: String::new() });
    }
    app
}

#[test]
fn regression_cursor_move_refreshes_completion_popup() {
    let mut app = app_with_commands();
    app.insert_str("/hel"); // cursor at 4, prefix "/hel"
    assert!(app.show_completions);

    // Stepping left narrows the prefix to "/he" -> still one match.
    app.move_cursor_left();
    assert!(app.show_completions, "popup must follow the cursor");
    assert_eq!(app.get_matched_completions().len(), 1);
    assert_eq!(app.get_matched_completions()[0].name, "/help");
    assert!(app.completion_idx < app.get_matched_completions().len());

    // Stepping left again to "/h" -> still only /help matches.
    app.move_cursor_left();
    assert!(app.show_completions);
    assert_eq!(app.get_matched_completions().len(), 1);

    // One more step reaches the bare "/" prefix -> every command matches.
    app.move_cursor_left();
    assert!(app.show_completions);
    assert_eq!(app.get_matched_completions().len(), 3);
    assert!(app.completion_idx < app.get_matched_completions().len());
}

#[test]
fn regression_cursor_move_closes_popup_when_prefix_stops_matching() {
    let mut app = app_with_commands();
    app.insert_str("/hel");
    assert!(app.show_completions);

    // Cursor to position 0: nothing precedes it, so there is no prefix and no
    // popup — this is what previously left a stale popup with a stale index.
    app.move_cursor_home();
    assert!(!app.show_completions, "no prefix before cursor means no popup");

    // Typing a non-command prefix must not open one either.
    app.move_cursor_end();
    app.insert_char('x');
    assert!(!app.show_completions, "non-command text must not open the popup");

    // Back to a valid prefix -> popup returns.
    app.move_cursor_end();
    app.backspace();
    assert!(app.show_completions);
}

// ── Bug 9: commands containing a space were unreachable ─────────────────────

#[test]
fn regression_multiword_command_is_completable() {
    let mut app = App::new();
    app.completions.push(CompletionItem { name: "/skill import".into(), help: String::new() });
    app.completions.push(CompletionItem { name: "/skills".into(), help: String::new() });

    app.insert_str("/skill im");
    assert!(
        app.show_completions,
        "a command with a space must stay completable"
    );
    let matched = app.get_matched_completions();
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].name, "/skill import");
    assert!(app.completion_idx < matched.len());
}

// ── Bug 3: Enter accepted a completion and then also submitted the turn ──────
// The guard lives in the event loop in lib.rs; what is asserted here is the
// state contract that guard depends on: accepting a completion leaves the
// command text in `input` so an explicit second submit is what sends it.

#[test]
fn regression_completion_acceptance_leaves_input_submittable() {
    let mut app = app_with_commands();
    app.insert_str("/hel");
    let matched = app.get_matched_completions();
    let choice = matched.get(app.completion_idx).expect("valid completion index");
    app.input = choice.name.clone();
    app.cursor_pos = app.input.chars().count();
    app.show_completions = false;

    assert_eq!(app.input, "/help");
    // Submitting here is the SECOND Enter, which is the intended send path.
    assert_eq!(app.submit_input(), Some("/help".to_string()));
}

// ── Regression guard: no panic on degenerate widths (clippy div_ceil change) ─

#[test]
fn regression_visual_lines_never_divide_by_zero() {
    let mut app = App::new();
    app.input = "a\nbb\nccc".into();
    for w in [0usize, 1, 2, 9, 10, 11, 1000] {
        assert!(app.input_visual_lines_count(w) >= 1, "width {} produced 0", w);
    }
    app.input = String::new();
    for w in [0usize, 1, 10] {
        assert_eq!(app.input_visual_lines_count(w), 1);
    }
}

#[test]
fn regression_on_filter_changed_retains_valid_selection_or_picks_first() {
    let mut sel = SelectModalData::new(
        "id".into(),
        "title".into(),
        vec!["apple".into(), "banana".into(), "cherry".into()],
        Some("cherry".into()),
    );
    assert_eq!(sel.selected_idx, 2);

    // Filter "ch": cherry still matches, so selected_idx remains 2
    sel.filter = "ch".into();
    sel.on_filter_changed();
    assert_eq!(sel.selected_idx, 2);

    // Filter "ban": cherry does NOT match, banana matches at orig_idx 1
    sel.filter = "ban".into();
    sel.on_filter_changed();
    assert_eq!(sel.selected_idx, 1);
}

// ── P0-P2 Solution Plan Regressions ──────────────────────────────────────────

#[test]
fn regression_p0_multi_select_and_checkbox_serialization() {
    let opts = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
    let mut sel = SelectModalData::new_full(
        "sel_test".into(),
        "Choose items".into(),
        opts,
        None,
        true,
        Some(vec![true, false, false]),
    );

    assert!(sel.multi);
    assert_eq!(sel.checked, vec![true, false, false]);

    // Initial selected value is JSON array containing ["alpha"]
    let val_json = sel.selected_value().unwrap();
    let parsed: Vec<String> = serde_json::from_str(&val_json).unwrap();
    assert_eq!(parsed, vec!["alpha"]);

    // Move to index 1 ("beta") and toggle it
    sel.move_down();
    assert_eq!(sel.selected_idx, 1);
    sel.toggle_current();
    assert_eq!(sel.checked, vec![true, true, false]);

    let val_json2 = sel.selected_value().unwrap();
    let parsed2: Vec<String> = serde_json::from_str(&val_json2).unwrap();
    assert_eq!(parsed2, vec!["alpha", "beta"]);

    // Toggle index 0 ("alpha") off
    sel.move_up();
    assert_eq!(sel.selected_idx, 0);
    sel.toggle_current();
    assert_eq!(sel.checked, vec![false, true, false]);

    let val_json3 = sel.selected_value().unwrap();
    let parsed3: Vec<String> = serde_json::from_str(&val_json3).unwrap();
    assert_eq!(parsed3, vec!["beta"]);
}

#[test]
fn regression_p1_usage_max_zero_does_not_clobber_ctx_max() {
    let mut app = App::new();
    app.status.ctx_tokens = 500;
    app.status.ctx_max = 128_000;
    app.status.ctx_percent = 45.0;

    // Provider sends token update without max limit
    app.handle_event(ArgentEvent::Usage {
        tokens: 1500,
        max: 0,
        percent: 0.0,
        completion: None,
    });

    assert_eq!(app.status.ctx_tokens, 1500);
    assert_eq!(app.status.ctx_max, 128_000, "ctx_max must not be overwritten with 0");
    assert_eq!(app.status.ctx_percent, 45.0, "ctx_percent must not be overwritten when max is 0");
}

#[test]
fn regression_p1_thinking_duration_captured_on_content_transition() {
    let mut app = App::new();

    app.handle_event(ArgentEvent::Thinking("Solving complex theorem...".into()));
    assert_eq!(app.messages.len(), 1);
    if let MessageItem::Assistant { is_streaming_thinking, thinking_duration, .. } = &app.messages[0] {
        assert!(*is_streaming_thinking);
        assert!(thinking_duration.is_none());
    } else {
        panic!("Expected Assistant message");
    }

    // Small sleep to ensure non-zero duration
    std::thread::sleep(std::time::Duration::from_millis(5));

    // Transition directly to Content
    app.handle_event(ArgentEvent::Content("Here is the answer".into()));
    if let MessageItem::Assistant { is_streaming_thinking, is_streaming_content, thinking_duration, .. } = &app.messages[0] {
        assert!(!*is_streaming_thinking);
        assert!(*is_streaming_content);
        assert!(thinking_duration.is_some(), "thinking_duration must be recorded upon transition to content");
        assert!(thinking_duration.unwrap() > 0.0);
    } else {
        panic!("Expected Assistant message");
    }
}

#[test]
fn regression_p1_tool_generating_cleaned_up_on_tool_start_and_done() {
    let mut app = App::new();

    app.handle_event(ArgentEvent::ToolGenerating {
        name: "test_tool".into(),
        bytes: 120,
        delta: String::new(),
    });
    assert!(app.messages.iter().any(|m| matches!(m, MessageItem::ToolGenerating { name, .. } if name == "test_tool")));

    // When tool starts, the generating card for that tool must be removed
    app.handle_event(ArgentEvent::ToolStart {
        name: "test_tool".into(),
        args: std::collections::HashMap::new(),
    });
    assert!(!app.messages.iter().any(|m| matches!(m, MessageItem::ToolGenerating { name, .. } if name == "test_tool")));

    // If an orphan ToolGenerating was left and Done arrives, it must be cleared
    app.handle_event(ArgentEvent::ToolGenerating {
        name: "orphan_tool".into(),
        bytes: 50,
        delta: String::new(),
    });
    assert!(app.messages.iter().any(|m| matches!(m, MessageItem::ToolGenerating { .. })));

    app.handle_event(ArgentEvent::Done);
    assert!(!app.messages.iter().any(|m| matches!(m, MessageItem::ToolGenerating { .. })));
}

#[test]
fn regression_p2_at_mention_only_at_word_boundary() {
    let mut app = App::new();

    // Email address: @ is preceded by non-whitespace -> must NOT trigger completions
    app.input = "Send to user@domain.com".into();
    app.cursor_pos = app.input.chars().count();
    app.check_completions();
    assert!(!app.show_file_completions, "email should not trigger file completions");

    // Leading @ -> must trigger
    app.input = "@Cargo".into();
    app.cursor_pos = app.input.chars().count();
    app.check_completions();
    assert!(app.show_file_completions, "leading @ should trigger file completions");

    // Space before @ -> must trigger
    app.input = "look at @src".into();
    app.cursor_pos = app.input.chars().count();
    app.check_completions();
    assert!(app.show_file_completions, "whitespace before @ should trigger file completions");
}

#[test]
fn regression_chat_scrolling_changes_viewport() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use argent_tui::ui::draw_ui;

    let mut app = App::new();
    for i in 0..30 {
        app.messages.push(MessageItem::User(format!("Message number {i}")));
    }

    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| draw_ui(f, &app)).unwrap();
    let initial_frame = format!("{:?}", terminal.backend().buffer());

    // Scroll up
    app.scroll_up(5);
    terminal.draw(|f| draw_ui(f, &app)).unwrap();
    let scrolled_frame = format!("{:?}", terminal.backend().buffer());

    assert_ne!(initial_frame, scrolled_frame, "Scrolling up must change the rendered buffer");
}

#[test]
fn regression_action_bar_no_mouse_button() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use argent_tui::ui::draw_ui;

    let app = App::new();
    let backend = TestBackend::new(100, 20);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|f| draw_ui(f, &app)).unwrap();

    // Verify that ToggleMouseCapture is NOT among registered clickables in action bar
    let action_bar_y = 20 - 2; // Action bar is at height - 2 (3 chunks + status)
    for x in 0..100 {
        if let Some(action) = app.get_clicked_action(x, action_bar_y) {
            assert_ne!(
                action,
                argent_tui::app::ClickAction::ToggleMouseCapture,
                "ToggleMouseCapture button should not be present in action bar"
            );
        }
    }
}

#[test]
fn regression_chat_scrolling_scroll_down_and_bounds() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use argent_tui::ui::draw_ui;

    let mut app = App::new();
    for i in 0..40 {
        app.messages.push(MessageItem::User(format!("Line item {i}")));
    }

    let backend = TestBackend::new(80, 20);
    let mut terminal = Terminal::new(backend).unwrap();

    // Initial draw establishes max_scroll
    terminal.draw(|f| draw_ui(f, &app)).unwrap();
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);

    // Scroll up
    app.scroll_up(10);
    assert_eq!(app.scroll_offset, 10);
    assert!(!app.auto_scroll);

    terminal.draw(|f| draw_ui(f, &app)).unwrap();

    // Scroll down back to bottom
    app.scroll_down(10);
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);
}

#[test]
fn regression_action_bar_no_copy_button_and_no_emojis() {
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use argent_tui::ui::draw_ui;

    let app = App::new();
    assert!(!app.mouse_capture_enabled, "mouse_capture_enabled must be false by default for native selection");

    let backend = TestBackend::new(120, 30);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| draw_ui(f, &app)).unwrap();

    let buf = terminal.backend().buffer().clone();
    let text: String = buf.content.iter().map(|c| c.symbol()).collect();
    for emoji in &["🌐", "⚡", "🧠", "📋", "📜", "⏱", "📁", "⏳", "🔍", "📄"] {
        assert!(!text.contains(emoji), "Buffer contains cheap emoji {emoji}");
    }
}

#[test]
fn regression_copying_an_answer_needs_no_visible_button() {
    // The REPL draws no "[ скопировать ]" strips under an answer, and neither
    // does the TUI. Copying is still one key away: the App can hand over the
    // clean markdown of the answer and of its code blocks.
    use argent_tui::ui::draw_ui;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    let mut app = App::new();
    app.messages.push(MessageItem::assistant(
        "Here is code:\n```rust\nlet x = 42;\n```",
        "",
    ));

    let mut terminal = Terminal::new(TestBackend::new(120, 30)).unwrap();
    terminal.draw(|f| draw_ui(f, &app)).unwrap();
    let text: String = terminal.backend().buffer().content.iter().map(|c| c.symbol()).collect();
    assert!(!text.contains("скопировать"), "a copy button is painted:\n{text}");

    assert_eq!(
        app.last_assistant_content().as_deref(),
        Some("Here is code:\n```rust\nlet x = 42;\n```")
    );
    assert_eq!(app.last_assistant_code().as_deref(), Some("let x = 42;"));
}
