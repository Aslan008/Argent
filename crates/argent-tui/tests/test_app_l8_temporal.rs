//! LMTrust L8 (Temporal) — does the code still work when things happen in a
//! different order?
//!
//! Layer decision tree: `App` is driven by two independent streams that interleave
//! unpredictably — Python events (`ArgentEvent`) and keystrokes — so ordering is
//! the dominant failure mode. L8 x D6 (concurrency/atomicity of the two streams)
//! and L8 x D2 (state transitions) are the high-ROI layers here.
//!
//! The pre-existing suite tests each transition in isolation with the "happy"
//! order. None of them reorders anything.

use std::collections::HashMap;

use argent_tui::app::{App, CompletionItem, MessageItem, Submit};
use argent_tui::events::ArgentEvent;

// ── L8: incremental reverse history search (Ctrl+R) ─────────────────────────

/// L8 x D2. Repeated Ctrl+R must walk backwards one entry at a time.
///
/// `history_search_prev` returns early into `history_search_start` whenever the
/// query is empty, which is exactly the state of a bare Ctrl+R. So the second
/// press restarts the search instead of advancing it.
#[test]
fn l8_repeated_ctrl_r_walks_back_one_entry_at_a_time() {
    let mut app = App::new();
    app.input_history = vec!["first".into(), "second".into(), "third".into()];

    assert!(app.history_search_start(), "search must start on a non-empty history");
    assert_eq!(app.input, "third", "the first Ctrl+R shows the newest entry");

    app.history_search_prev();
    assert_eq!(app.input, "second", "a second Ctrl+R must step back exactly one entry");

    app.history_search_prev();
    assert_eq!(app.input, "first", "a third Ctrl+R reaches the oldest entry");
}

/// L8 x D2. Esc must restore the line exactly as it was before the search
/// began. `search_saved` is snapshotted once, at search start.
#[test]
fn l8_escape_restores_the_draft_from_before_the_search() {
    let mut app = App::new();
    app.input_history = vec!["first".into(), "second".into()];
    app.insert_str("my draft");

    assert!(app.history_search_start());
    assert_eq!(app.input, "second", "the search replaced the line");
    app.history_search_prev();
    app.history_search_cancel();

    assert_eq!(
        app.input, "my draft",
        "Esc restored a history entry instead of the pre-search draft"
    );
}

/// L8 x D2. With a query typed in, Ctrl+R must visit every match in order.
///
/// `history_search_prev` decrements `search_idx` and then `apply_search`
/// decrements it again (`start.checked_sub(1)`), so each press moves two slots.
#[test]
fn l8_ctrl_r_with_a_query_visits_every_match_in_order() {
    let mut app = App::new();
    app.input_history = vec!["aaa".into(), "aab".into(), "aac".into()];

    assert!(app.history_search_start());
    app.history_search_push('a');
    assert_eq!(app.input, "aac", "the newest match comes first");

    app.history_search_prev();
    assert_eq!(app.input, "aab", "search must step one match at a time");

    app.history_search_prev();
    assert_eq!(app.input, "aaa", "search must reach the oldest match");
}

/// L8 x D2. Backspacing the query away must fall back to showing the newest
/// entry, not leave the last match stranded.
#[test]
fn l8_clearing_the_query_by_backspace_restores_a_valid_match() {
    let mut app = App::new();
    app.input_history = vec!["one".into(), "two".into(), "three".into()];

    assert!(app.history_search_start());
    app.history_search_push('t');
    app.history_search_push('h');
    assert_eq!(app.input, "three");

    app.history_search_backspace();
    app.history_search_backspace();
    assert_eq!(app.search_query, "");
    assert_eq!(
        app.input, "three",
        "with an empty query the newest entry must be shown"
    );
}

/// L8 x D2. A query that matches nothing must not corrupt the buffer, and Esc
/// must still restore the pre-search draft.
#[test]
fn l8_a_query_with_no_match_leaves_the_line_consistent() {
    let mut app = App::new();
    app.input_history = vec!["alpha".into()];
    app.insert_str("draft");

    assert!(app.history_search_start());
    assert_eq!(app.input, "alpha", "starting the search shows the newest match");

    app.history_search_push('z');
    app.history_search_push('z');
    assert_eq!(
        app.input, "alpha",
        "a fruitless query must leave the last valid match in place"
    );
    assert!(app.search_idx.is_none(), "no match means no tracked index");

    app.history_search_cancel();
    assert_eq!(app.input, "draft", "Esc must still restore the original draft");
}

/// L8 x D2. The typed character must not be swallowed when a search is started
/// implicitly. NOTE: unreachable from the current key handler, which only calls
/// `history_search_push` while `search_active` — kept as an API contract.
#[test]
fn l8_history_search_push_without_an_active_search_keeps_the_character() {
    let mut app = App::new();
    app.input_history = vec!["needle".into()];

    app.history_search_push('n');

    assert_eq!(app.search_query, "n", "the typed character was dropped");
}

// ── L8: history walk and the saved draft ───────────────────────────────────

/// L8 x D2. Up past the oldest entry, then Down again: the draft comes back and
/// the walk can be repeated.
#[test]
fn l8_history_walk_round_trips_to_the_draft() {
    let mut app = App::new();
    app.input_history = vec!["old".into(), "new".into()];
    app.insert_str("draft");

    assert!(app.history_prev());
    assert_eq!(app.input, "new");
    assert!(app.history_prev());
    assert_eq!(app.input, "old");
    assert!(!app.history_prev(), "already at the oldest entry");

    app.history_next();
    assert_eq!(app.input, "new");
    app.history_next();
    assert_eq!(app.input, "draft", "walking past the newest entry restores the draft");
    assert!(app.history_idx.is_none());
}

/// L8 x D2. Submitting clears the history cursor, so the next Up starts from
/// the newest entry rather than resuming a stale position.
#[test]
fn l8_submitting_resets_the_history_cursor() {
    let mut app = App::new();
    app.input_history = vec!["a".into(), "b".into()];
    app.history_prev();
    assert!(app.history_idx.is_some());

    app.insert_str("b");
    app.submit_input();

    assert!(app.history_idx.is_none(), "a submitted turn must reset the history cursor");
    // Submitting pushed the new text, so it is now the newest entry.
    assert_eq!(app.input_history.last().map(|s| s.as_str()), Some("bb"));
    assert!(app.history_prev());
    assert_eq!(app.input, "bb", "Up after a submit starts from the newest entry");
}

// ── L8: agent event ordering ───────────────────────────────────────────────

/// L8 x D2. `Done` arriving before any content must leave a consistent idle
/// state rather than half-open flags.
#[test]
fn l8_done_before_any_content_is_a_clean_noop() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_busy);
    assert!(!app.is_cancelling);
    assert!(app.active_approval.is_none());
}

/// L8 x D2. Content that arrives after `Done` belongs to a new turn and must
/// re-open the busy flag, or the queue would start a second concurrent turn.
#[test]
fn l8_content_after_done_reopens_the_busy_flag() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("first turn".into()));
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_busy);

    app.handle_event(ArgentEvent::Content("second turn".into()));
    assert!(app.is_busy, "a new content delta means a turn is in flight");
}

/// L8 x D2. Cancelling after `Done` is a stale event; it must not put the UI
/// back into a permanently "busy" state with no worker left to clear it.
#[test]
fn l8_cancelling_after_done_does_not_strand_the_queue() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("x".into()));
    app.handle_event(ArgentEvent::Done);
    app.handle_event(ArgentEvent::Cancelling);

    assert!(!app.is_busy, "a late Cancelling must not re-busy an idle UI");
}

/// L8 x D4. A `tool_end` for a tool that never started is dropped silently. It
/// must at least not fabricate a card or corrupt an unrelated one.
#[test]
fn l8_tool_end_without_a_matching_start_is_harmless() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ToolEnd {
        name: "ghost_tool".into(),
        result: "done".into(),
        duration_secs: Some(0.1),
    });

    let tools = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Tool { .. }))
        .count();
    assert_eq!(tools, 0, "a tool_end with no start must not create a card");
}

/// L8 x D4. Two concurrent tools with the same name: the first `tool_end` must
/// close the newest running card, leaving the older one spinning.
#[test]
fn l8_concurrent_same_named_tools_close_newest_first() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ToolStart { name: "run".into(), args: Default::default() });
    app.handle_event(ArgentEvent::ToolStart { name: "run".into(), args: Default::default() });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run".into(),
        result: "second result".into(),
        duration_secs: Some(0.1),
    });

    let running = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Tool { is_running: true, .. }))
        .count();
    assert_eq!(running, 1, "exactly one card should still be running");
}

/// L8 x D2. Content split by a tool call belongs to two separate assistant
/// bubbles, not one merged message.
#[test]
fn l8_content_around_a_tool_call_forms_separate_bubbles() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("before".into()));
    app.handle_event(ArgentEvent::ToolStart { name: "read_file".into(), args: Default::default() });
    app.handle_event(ArgentEvent::Content("after".into()));

    let assistants = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Assistant { .. }))
        .count();
    assert_eq!(assistants, 2, "content after a tool call must start a new bubble");
}

/// L8 x D2. Within one bubble, consecutive deltas must accumulate in order.
#[test]
fn l8_consecutive_deltas_accumulate_in_order() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content("one ".into()));
    app.handle_event(ArgentEvent::Content("two ".into()));
    app.handle_event(ArgentEvent::Content("three".into()));

    match app.messages.last() {
        Some(MessageItem::Assistant { content, .. }) => {
            assert_eq!(content, "one two three", "deltas were reordered or dropped")
        }
        other => panic!("expected an assistant bubble, got {other:?}"),
    }
}

/// L8 x D2. Reasoning emitted after a tool call opens a NEW assistant bubble,
/// exactly like content does — but nothing may be lost.
#[test]
fn l8_thinking_interrupted_by_a_tool_keeps_all_of_its_text() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("part one ".into()));
    app.handle_event(ArgentEvent::ToolStart { name: "grep".into(), args: Default::default() });
    app.handle_event(ArgentEvent::Thinking("part two".into()));

    let thoughts: Vec<String> = app
        .messages
        .iter()
        .filter_map(|m| match m {
            MessageItem::Assistant { thinking, .. } => Some(thinking.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(thoughts, vec!["part one ".to_string(), "part two".to_string()]);
    assert_eq!(
        thoughts.concat(),
        "part one part two",
        "reasoning text was lost across the tool call"
    );
}

/// L8 x D4. `Done` must stop every running spinner, including several.
#[test]
fn l8_done_stops_every_running_tool() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ToolStart { name: "a".into(), args: Default::default() });
    app.handle_event(ArgentEvent::ToolStart { name: "b".into(), args: Default::default() });
    app.handle_event(ArgentEvent::ToolStart { name: "c".into(), args: Default::default() });
    app.handle_event(ArgentEvent::Done);

    let running = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Tool { is_running: true, .. }))
        .count();
    assert_eq!(running, 0, "Done left a tool spinning forever");
}

// ── L8: the queue against a rejecting bridge ───────────────────────────────

/// L8 x D6. A prompt the bridge persistently refuses is pushed back to the
/// FRONT of the queue, so it is dispatched again on the very next tick — with
/// no backoff and no attempt counter. Every other queued message is starved and
/// the loop spins.
///
/// The existing `regression_rejected_turn_is_requeued_at_front` pins the
/// front-insertion as intended; what it does not check is what happens when the
/// rejection is permanent.
#[test]
fn l8_a_permanently_rejected_prompt_does_not_livelock_the_queue() {
    let mut app = App::new();
    app.message_queue = vec!["healthy".into(), "blocked".into()];

    let mut attempts: HashMap<String, usize> = HashMap::new();
    // 50 main-loop ticks; the bridge rejects every single prompt.
    for _ in 0..50 {
        if let Some(prompt) = app.pop_next_queued_message() {
            *attempts.entry(prompt.clone()).or_default() += 1;
            app.requeue_rejected_message(prompt);
        }
    }

    assert_eq!(
        app.message_queue.len(),
        0,
        "the queue must drain instead of spinning forever"
    );
    assert!(
        attempts.values().all(|&n| n <= 3),
        "a prompt was retried past the cap: {attempts:?}"
    );
    assert!(
        attempts.contains_key("blocked"),
        "the second prompt must still get its turn: {attempts:?}"
    );
}

/// L8 x D4. Giving up on a prompt has to be visible, not a silent stall.
#[test]
fn l8_exhausted_retry_budget_surfaces_an_error() {
    let mut app = App::new();
    app.message_queue = vec!["blocked".into()];
    for _ in 0..5 {
        if let Some(prompt) = app.pop_next_queued_message() {
            app.requeue_rejected_message(prompt);
        }
    }
    assert!(
        app.messages.iter().any(|m| matches!(m, MessageItem::Error { text, .. } if text.contains("blocked"))),
        "a dropped prompt must be reported to the user"
    );
}

/// L8 x D4. Any real turn output proves the bridge accepted, so the retry
/// budget starts over instead of being consumed by unrelated failures.
#[test]
fn l8_turn_output_resets_the_retry_budget() {
    let mut app = App::new();
    app.message_queue = vec!["p".into()];
    for _ in 0..2 {
        if let Some(prompt) = app.pop_next_queued_message() {
            app.requeue_rejected_message(prompt);
        }
    }
    assert_eq!(app.reject_streak, 2);

    app.handle_event(ArgentEvent::Content("the bridge accepted".into()));
    assert_eq!(app.reject_streak, 0, "turn output must clear the reject streak");
    app.handle_event(ArgentEvent::Done);

    let prompt = app.pop_next_queued_message().expect("still queued");
    app.requeue_rejected_message(prompt);
    assert_eq!(
        app.reject_streak, 1,
        "a fresh turn must not inherit an old reject streak"
    );
}

/// L8 x D2. A rejection must not roll back a User bubble that belongs to an
/// earlier, unrelated turn.
#[test]
fn l8_requeue_rollback_only_touches_its_own_bubble() {
    let mut app = App::new();
    app.insert_str("older");
    app.submit_input();
    app.handle_event(ArgentEvent::Content("working".into()));
    assert!(app.is_busy);

    // Queued behind the busy turn, so it is dispatched from the queue and the
    // rollback path is the one under test.
    app.insert_str("newest");
    assert!(app.submit_input().is_none());
    app.handle_event(ArgentEvent::Done);

    let prompt = app.pop_next_queued_message().expect("queued prompt dispatches after Done");
    app.handle_event(ArgentEvent::Notice("noise".into()));
    app.requeue_rejected_message(prompt);

    assert!(
        app.messages.iter().any(|m| matches!(m, MessageItem::User(u) if u == "older")),
        "the rollback ate an unrelated user message"
    );
}

// ── L8: completions against a mutating registry ────────────────────────────

/// L8 x D2. The popup index is bounded by the CURRENT match count. Narrowing
/// the prefix must never leave the index pointing past the end.
#[test]
fn l8_completion_index_stays_valid_as_the_prefix_narrows() {
    let mut app = App::new();
    for name in ["/help", "/history", "/hush"] {
        app.completions.push(CompletionItem { name: name.into(), help: String::new() });
    }
    app.insert_str("/h");
    assert_eq!(app.get_matched_completions().len(), 3);

    for _ in 0..5 {
        app.completion_next();
        let n = app.get_matched_completions().len();
        assert!(app.completion_idx < n, "completion_idx {} escaped {n} matches", app.completion_idx);
    }

    // Retype a narrower prefix from scratch: `insert_str` appends at the cursor.
    app.move_cursor_home();
    app.delete_to_start();
    app.insert_str("/his");
    let n = app.get_matched_completions().len();
    assert_eq!(n, 1, "narrowing the prefix must leave exactly one match");
    assert!(app.completion_idx < n, "narrowing the prefix left a stale index");
}

/// L8 x D2. Enter must not both accept a completion and submit the turn. The
/// state contract the event loop relies on: accepting leaves the text in place.
#[test]
fn l8_accepting_a_completion_does_not_clear_the_input() {
    let mut app = App::new();
    app.completions.push(CompletionItem { name: "/help".into(), help: String::new() });
    app.insert_str("/hel");
    assert!(app.show_completions);

    let matched = app.get_matched_completions();
    let choice = matched.get(app.completion_idx).expect("valid index");
    app.input = choice.name.clone();
    app.cursor_pos = app.input.chars().count();
    app.show_completions = false;

    assert_eq!(app.input, "/help");
    assert!(!app.show_completions);
}

/// L8 x D2. A command dispatched to the bridge must not be swallowed by the
/// TUI's local handler, in any capitalisation or with extra whitespace.
#[test]
fn l8_non_queue_commands_are_always_dispatched() {
    let mut app = App::new();
    for cmd in [
        "/exit",
        "/clear",
        "/  ",
        "/QUEUE",
        "/queuex",
        "/model",
        "",
        "   ",
        "not a command",
    ] {
        assert_eq!(
            app.handle_local_command(cmd),
            Submit::Dispatch,
            "{cmd:?} must reach the bridge"
        );
    }
}
