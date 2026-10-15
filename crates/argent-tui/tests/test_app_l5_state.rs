use std::collections::HashMap;
use argent_tui::app::{App, MessageItem};
use argent_tui::events::ArgentEvent;

#[test]
fn test_state_streaming_and_turn_lifecycle() {
    let mut app = App::new();

    // 1. Thinking delta arrives
    app.handle_event(ArgentEvent::Thinking("Analyzing problem...".to_string()));
    assert!(app.is_busy);
    assert_eq!(app.messages.len(), 1);
    if let MessageItem::Assistant { thinking, is_streaming_thinking, .. } = &app.messages[0] {
        assert_eq!(thinking, "Analyzing problem...");
        assert!(is_streaming_thinking);
    } else {
        panic!("Expected Assistant message item");
    }

    // Thinking continuation
    app.handle_event(ArgentEvent::Thinking(" Formulating solution.".to_string()));
    if let MessageItem::Assistant { thinking, is_streaming_thinking, .. } = &app.messages[0] {
        assert_eq!(thinking, "Analyzing problem... Formulating solution.");
        assert!(is_streaming_thinking);
    }

    // 2. Content delta arrives
    app.handle_event(ArgentEvent::Content("Here is the solution:".to_string()));
    if let MessageItem::Assistant { content, thinking, is_streaming_thinking, is_streaming_content, .. } = &app.messages[0] {
        assert_eq!(content, "Here is the solution:");
        assert_eq!(thinking, "Analyzing problem... Formulating solution.");
        assert!(!is_streaming_thinking);
        assert!(is_streaming_content);
    }

    // Content continuation
    app.handle_event(ArgentEvent::Content(" 42".to_string()));
    if let MessageItem::Assistant { content, .. } = &app.messages[0] {
        assert_eq!(content, "Here is the solution: 42");
    }

    // 3. Tool execution
    let mut args = HashMap::new();
    args.insert("cmd".to_string(), serde_json::Value::String("ls".to_string()));
    app.handle_event(ArgentEvent::ToolStart {
        name: "bash".to_string(),
        args,
    });
    assert_eq!(app.messages.len(), 2);
    if let MessageItem::Tool { name, is_running, result, .. } = &app.messages[1] {
        assert_eq!(name, "bash");
        assert!(is_running);
        assert!(result.is_none());
    }

    // Tool finishes
    app.handle_event(ArgentEvent::ToolEnd {
        name: "bash".to_string(),
        result: "file1.txt\nfile2.txt".to_string(),
        duration_secs: Some(0.15),
    });
    if let MessageItem::Tool { is_running, result, duration_secs, .. } = &app.messages[1] {
        assert!(!is_running);
        assert_eq!(result.as_deref(), Some("file1.txt\nfile2.txt"));
        assert_eq!(*duration_secs, Some(0.15));
    }

    // 4. Turn Done
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_busy);
    if let MessageItem::Assistant { is_streaming_content, is_streaming_thinking, .. } = &app.messages[0] {
        assert!(!is_streaming_content);
        assert!(!is_streaming_thinking);
    }
}

#[test]
fn test_state_message_queue_fifo() {
    let mut app = App::new();

    // Turn 1 starts directly
    app.insert_str("First prompt");
    let submitted1 = app.submit_input();
    assert_eq!(submitted1, Some("First prompt".to_string()));
    assert_eq!(app.messages.len(), 1); // User message added

    // Agent becomes busy
    app.is_busy = true;

    // While busy, user submits prompt 2 and prompt 3
    app.insert_str("Queued prompt 2");
    let submitted2 = app.submit_input();
    assert!(submitted2.is_none(), "When busy, prompt must be enqueued");
    assert_eq!(app.message_queue.len(), 1);

    app.insert_str("Queued prompt 3");
    let submitted3 = app.submit_input();
    assert!(submitted3.is_none());
    assert_eq!(app.message_queue.len(), 2);

    // Agent finishes current turn
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_busy);

    // Pop queued message 1
    let next1 = app.pop_next_queued_message();
    assert_eq!(next1, Some("Queued prompt 2".to_string()));
    assert!(app.is_busy);
    assert_eq!(app.message_queue.len(), 1);

    // Pop queued message 2
    app.is_busy = false;
    let next2 = app.pop_next_queued_message();
    assert_eq!(next2, Some("Queued prompt 3".to_string()));
    assert!(app.is_busy);
    assert_eq!(app.message_queue.len(), 0);

    // Queue is empty
    app.is_busy = false;
    let next3 = app.pop_next_queued_message();
    assert!(next3.is_none());
}

#[test]
fn test_state_approval_request() {
    let mut app = App::new();
    assert!(app.active_approval.is_none());

    app.handle_event(ArgentEvent::ApprovalRequest {
        id: 42,
        action: "delete database".to_string(),
        destructive: true,
        grant_key: Some("db:write".to_string()),
        prompt: None,
    });

    assert!(app.active_approval.is_some());
    let approval = app.active_approval.as_ref().unwrap();
    assert_eq!(approval.id, 42);
    assert_eq!(approval.action, "delete database");
    assert!(approval.destructive);
    assert_eq!(approval.grant_key.as_deref(), Some("db:write"));
}

#[test]
fn test_state_select_modal_and_filtering() {
    let mut app = App::new();
    assert!(app.active_select.is_none());

    app.handle_event(ArgentEvent::SelectRequest {
        id: "sel_provider".to_string(),
        title: "Select Provider".to_string(),
        options: vec![
            "ollama".to_string(),
            "zai".to_string(),
            "openrouter".to_string(),
            "xkiro".to_string(),
        ],
        default_value: Some("zai".to_string()),
        multi: false,
        initial_checked: None,
    });

    assert!(app.active_select.is_some());
    let select = app.active_select.as_mut().unwrap();
    assert_eq!(select.id, "sel_provider");
    assert_eq!(select.title, "Select Provider");
    // "zai" is at index 1
    assert_eq!(select.selected_idx, 1);
    assert_eq!(select.selected_value(), Some("zai".to_string()));

    // Move down
    select.move_down();
    assert_eq!(select.selected_idx, 2);
    assert_eq!(select.selected_value(), Some("openrouter".to_string()));

    // Move up
    select.move_up();
    assert_eq!(select.selected_idx, 1);
    assert_eq!(select.selected_value(), Some("zai".to_string()));

    // Filtering
    select.filter = "open".to_string();
    assert_eq!(select.filtered_options().len(), 1);
    select.selected_idx = 0;
    assert_eq!(select.selected_value(), Some("openrouter".to_string()));
}
