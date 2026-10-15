use argent_tui::app::{App, MessageItem};
use std::collections::HashMap;

#[test]
fn test_adversarial_scroll_overflow() {
    let mut app = App::new();
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);

    // Scroll down on 0 offset should not underflow
    app.scroll_down(100);
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);

    // Scroll up large amount
    app.scroll_up(usize::MAX - 5);
    assert_eq!(app.scroll_offset, usize::MAX - 5);
    assert!(!app.auto_scroll);

    // Further scroll up saturates, does not panic
    app.scroll_up(100);
    assert_eq!(app.scroll_offset, usize::MAX);

    // Scroll down back to 0
    app.scroll_down(usize::MAX);
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);
}

#[test]
fn test_adversarial_history_draft_retention() {
    let mut app = App::new();
    app.input_history.push("commit -m 'first'".to_string());
    app.input_history.push("git status".to_string());

    // User is typing a new prompt
    app.insert_str("draft in progress");
    assert_eq!(app.input, "draft in progress");

    // User presses Up to recall history
    app.history_prev();
    assert_eq!(app.input, "git status");
    assert_eq!(app.saved_draft.as_deref(), Some("draft in progress"));

    app.history_prev();
    assert_eq!(app.input, "commit -m 'first'");

    // User presses Up again beyond history start
    app.history_prev();
    assert_eq!(app.input, "commit -m 'first'");

    // User navigates back Down
    app.history_next();
    assert_eq!(app.input, "git status");

    // User exits history at the bottom -> Draft MUST be restored!
    app.history_next();
    assert_eq!(app.input, "draft in progress");
    assert_eq!(app.cursor_pos, "draft in progress".chars().count());
    assert!(app.history_idx.is_none());
    assert!(app.saved_draft.is_none());
}

#[test]
fn test_adversarial_accordion_toggles() {
    let mut app = App::new();

    // 1. Toggling on completely empty message list
    app.toggle_last_reasoning();
    app.toggle_last_tool();
    assert!(app.messages.is_empty());

    // 2. User message only
    app.messages.push(MessageItem::User("hello".to_string()));
    app.toggle_last_reasoning();
    app.toggle_last_tool();

    // 3. Assistant with empty thinking
    app.messages.push(MessageItem::Assistant {
        content: "hello".to_string(),
        thinking: String::new(),
        thinking_expanded: false,
        is_streaming_thinking: false,
        is_streaming_content: false,
        thinking_duration: None,
        thinking_start: None,
        stream: Default::default(),
    });
    app.toggle_last_reasoning(); // Should NOT expand because thinking is empty
    if let Some(MessageItem::Assistant { thinking_expanded, .. }) = app.messages.last() {
        assert!(!thinking_expanded);
    }

    // 4. Assistant with valid thinking
    app.messages.push(MessageItem::Assistant {
        content: "done".to_string(),
        thinking: "Let me ponder this...".to_string(),
        thinking_expanded: false,
        is_streaming_thinking: false,
        is_streaming_content: false,
        thinking_duration: None,
        thinking_start: None,
        stream: Default::default(),
    });
    app.toggle_last_reasoning();
    if let Some(MessageItem::Assistant { thinking_expanded, .. }) = app.messages.last() {
        assert!(thinking_expanded);
    }
    app.toggle_last_reasoning();
    if let Some(MessageItem::Assistant { thinking_expanded, .. }) = app.messages.last() {
        assert!(!thinking_expanded);
    }

    // 5. Tool toggle
    app.messages.push(MessageItem::Tool {
        name: "search".to_string(),
        args: HashMap::new(),
        result: Some("found 3 items".to_string()),
        duration_secs: Some(0.4),
        expanded: false,
        is_running: false,
    });
    app.toggle_last_tool();
    if let Some(MessageItem::Tool { expanded, .. }) = app.messages.last() {
        assert!(expanded);
    }
    app.toggle_last_tool();
    if let Some(MessageItem::Tool { expanded, .. }) = app.messages.last() {
        assert!(!expanded);
    }
}
