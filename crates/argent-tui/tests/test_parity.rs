//! Parity features ported from the legacy CLI: selection, transpose, focus
//! gate, snapshot verbosity and the interactive-tool set.

use argent_tui::app::{App, MessageItem};

fn with_text(s: &str) -> App {
    let mut app = App::new();
    app.insert_str(s);
    app
}

// ── selection ─────────────────────────────────────────────────────────────

#[test]
fn shift_arrows_build_a_selection() {
    let mut app = with_text("hello world");
    app.set_cursor(0, false);
    app.extend_selection(5);
    assert!(app.has_selection());
    assert_eq!(app.selected_text().as_deref(), Some("hello"));
}

#[test]
fn selection_can_run_backwards() {
    let mut app = with_text("hello world");
    app.set_cursor(11, false);
    app.set_cursor(6, true);
    assert_eq!(app.selected_text().as_deref(), Some("world"));
}

#[test]
fn plain_movement_clears_the_selection() {
    let mut app = with_text("abc");
    app.set_cursor(0, true);
    app.extend_selection(2);
    assert!(app.has_selection());
    app.set_cursor(3, false);
    assert!(!app.has_selection());
    assert_eq!(app.selected_text(), None);
}

#[test]
fn deleting_a_selection_removes_it_and_moves_the_cursor() {
    let mut app = with_text("hello world");
    app.set_cursor(0, false);
    app.extend_selection(6);
    assert_eq!(app.delete_selection().as_deref(), Some("hello "));
    assert_eq!(app.input, "world");
    assert_eq!(app.cursor_pos, 0);
}

#[test]
fn deleting_an_empty_selection_returns_none() {
    let mut app = with_text("abc");
    app.set_cursor(1, false);
    assert_eq!(app.delete_selection(), None);
    assert_eq!(app.input, "abc");
}

#[test]
fn selection_bounds_are_clamped_to_the_buffer() {
    let mut app = with_text("ab");
    app.set_cursor(0, false);
    app.extend_selection(50);
    assert_eq!(app.selected_text().as_deref(), Some("ab"));
    // Extending from the end cannot grow a selection further.
    let before = app.selected_text();
    app.extend_selection(10);
    assert_eq!(app.selected_text(), before);
}

// ── transpose ─────────────────────────────────────────────────────────────

#[test]
fn transpose_chars_swaps_around_the_cursor() {
    let mut app = with_text("abcd");
    app.set_cursor(2, false);
    app.transpose_chars();
    assert_eq!(app.input, "acbd");
}

#[test]
fn transpose_chars_is_a_noop_on_a_single_character() {
    let mut app = with_text("a");
    app.transpose_chars();
    assert_eq!(app.input, "a");
}

#[test]
fn transpose_words_moves_the_word_before_the_cursor_after_it() {
    let mut app = with_text("one two");
    app.set_cursor(3, false); // end of "one"
    app.transpose_words();
    assert_eq!(app.input, "two one");
}

#[test]
fn transpose_words_needs_two_words() {
    let mut app = with_text("solo");
    app.set_cursor(4, false);
    app.transpose_words();
    assert_eq!(app.input, "solo");
}

// ── focus gate ────────────────────────────────────────────────────────────

#[test]
fn app_starts_focused() {
    assert!(App::new().focused);
}

#[test]
fn unfocus_clears_the_selection() {
    let mut app = with_text("abcdef");
    app.set_cursor(0, false);
    app.extend_selection(3);
    assert!(app.has_selection());
    app.clear_selection();
    assert!(!app.has_selection());
}

// ── interactive tools ─────────────────────────────────────────────────────

#[test]
fn interactive_tools_match_the_cli_set() {
    for t in [
        "ask_user_questions",
        "request_user_approval",
        "run_command",
        "run_admin_command",
        "start_background_command",
        "delete_file",
        "git_rollback",
    ] {
        assert!(App::is_interactive_tool(t), "{t} should be interactive");
    }
    for t in ["read_file", "grep_search", "write_file", "list_directory"] {
        assert!(!App::is_interactive_tool(t), "{t} is not interactive");
    }
}

// ── turn snapshot ─────────────────────────────────────────────────────────
//
// The snapshot text is built by the REPL's own `cli_ui.turn_snapshot_lines` and
// arrives in the event; the TUI shows it as it is. Only an older bridge, which
// sends the raw numbers alone, makes the TUI format the compact line itself.

fn snapshot_event(lines: Vec<String>, context_line: Option<String>) -> argent_tui::events::ArgentEvent {
    argent_tui::events::ArgentEvent::TurnSnapshot {
        total_s: 2.0,
        ttft_s: 0.0,
        tool_s: 0.0,
        tool_calls: 0,
        prompt_tokens: Some(10.0),
        completion_tokens: Some(5.0),
        cached_tokens: None,
        cost: None,
        ctx_tokens: None,
        ctx_max: None,
        ctx_percent: None,
        lines,
        context_line,
    }
}

#[test]
fn snapshot_mode_comes_from_the_bridge() {
    let mut app = App::new();
    assert_eq!(app.status.snapshot_mode, "compact");
    let mut state = std::collections::HashMap::new();
    state.insert("snapshot_mode".to_string(), "verbose".to_string());
    app.update_status(state);
    assert_eq!(app.status.snapshot_mode, "verbose");
}

#[test]
fn bridge_built_lines_are_kept_verbatim() {
    let mut app = App::new();
    let lines = vec![
        "  [dim]┄ 8.2s · 1.2k→340 tok[/dim]".to_string(),
        "  [dim]┄ ctx 3.1k/32k (10%)[/dim]".to_string(),
    ];
    app.handle_event(snapshot_event(lines.clone(), Some("[dim]Context: 9.5%[/dim]".into())));
    match app.messages.last() {
        Some(MessageItem::Snapshot { lines: got, context_line }) => {
            assert_eq!(*got, lines);
            assert_eq!(context_line.as_deref(), Some("[dim]Context: 9.5%[/dim]"));
        }
        other => panic!("expected a snapshot, got {other:?}"),
    }
}

#[test]
fn an_older_bridge_gets_the_compact_line_in_the_repl_format() {
    let mut app = App::new();
    app.handle_event(snapshot_event(Vec::new(), None));
    match app.messages.last() {
        Some(MessageItem::Snapshot { lines, .. }) => {
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0], "  [dim]┄ 2.0s · 10→5 tok[/dim]");
        }
        other => panic!("expected a snapshot, got {other:?}"),
    }
}

#[test]
fn the_fallback_line_computes_the_cache_share_from_the_prompt() {
    use argent_tui::events::ArgentEvent;
    let mut app = App::new();
    app.handle_event(ArgentEvent::TurnSnapshot {
        total_s: 8.2,
        ttft_s: 0.3,
        tool_s: 1.0,
        tool_calls: 3,
        prompt_tokens: Some(1200.0),
        completion_tokens: Some(340.0),
        cached_tokens: Some(900.0),
        cost: Some(0.0021),
        ctx_tokens: Some(3100.0),
        ctx_max: Some(32000.0),
        ctx_percent: Some(10.0),
        lines: Vec::new(),
        context_line: None,
    });
    match app.messages.last() {
        Some(MessageItem::Snapshot { lines, .. }) => assert_eq!(
            lines[0],
            "  [dim]┄ 8.2s · 1.2k→340 tok · cached 900 (75%) · 3 tools · ctx 3.1k/32k (10%) · $0.0021[/dim]"
        ),
        other => panic!("expected a snapshot, got {other:?}"),
    }
}

#[test]
fn off_mode_without_bridge_text_shows_nothing() {
    let mut app = App::new();
    app.status.snapshot_mode = "off".to_string();
    app.handle_event(snapshot_event(Vec::new(), None));
    assert!(!app.messages.iter().any(|m| matches!(m, MessageItem::Snapshot { .. })));
}

#[test]
fn session_totals_are_parsed_from_state() {
    let mut app = App::new();
    let mut state = std::collections::HashMap::new();
    state.insert("session_prompt".to_string(), "2400".to_string());
    state.insert("session_completion".to_string(), "1100".to_string());
    state.insert("session_cost".to_string(), "0.25".to_string());
    state.insert("session_requests".to_string(), "4".to_string());
    app.update_status(state);
    assert_eq!(app.status.session_prompt, 2400.0);
    assert_eq!(app.status.session_completion, 1100.0);
    assert_eq!(app.status.session_cost, 0.25);
    assert_eq!(app.status.session_requests, 4);
}

// ── tool-generating card ──────────────────────────────────────────────────

#[test]
fn tool_generating_updates_in_place_instead_of_stacking() {
    let mut app = App::new();
    use argent_tui::events::ArgentEvent;
    for bytes in [10usize, 20, 30] {
        app.handle_event(ArgentEvent::ToolGenerating { name: "write_file".into(), bytes, delta: String::new() });
    }
    let cards = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::ToolGenerating { .. }))
        .count();
    assert_eq!(cards, 1, "expected one card, got {cards}");
    match app.messages.last() {
        Some(MessageItem::ToolGenerating { bytes, .. }) => assert_eq!(*bytes, 30),
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn tool_generating_starts_a_new_card_for_a_new_tool() {
    let mut app = App::new();
    use argent_tui::events::ArgentEvent;
    app.handle_event(ArgentEvent::ToolGenerating { name: "a".into(), bytes: 5, delta: String::new() });
    app.handle_event(ArgentEvent::ToolGenerating { name: "b".into(), bytes: 7, delta: String::new() });
    let names: Vec<String> = app
        .messages
        .iter()
        .filter_map(|m| match m {
            MessageItem::ToolGenerating { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn interactive_tool_is_tracked_while_running() {
    let mut app = App::new();
    use argent_tui::events::ArgentEvent;
    app.handle_event(ArgentEvent::ToolStart {
        name: "run_command".into(),
        args: Default::default(),
    });
    assert_eq!(app.interactive_running.as_deref(), Some("run_command"));
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "ok".into(),
        duration_secs: Some(0.1),
    });
    assert_eq!(app.interactive_running, None);
}

#[test]
fn non_interactive_tool_does_not_set_the_flag() {
    let mut app = App::new();
    use argent_tui::events::ArgentEvent;
    app.handle_event(ArgentEvent::ToolStart {
        name: "read_file".into(),
        args: Default::default(),
    });
    assert_eq!(app.interactive_running, None);
}