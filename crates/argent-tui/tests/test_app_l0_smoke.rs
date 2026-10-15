use argent_tui::app::App;

#[test]
fn test_app_initialization_defaults() {
    let app = App::new();
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);
    assert_eq!(app.messages.len(), 0);
    assert_eq!(app.message_queue.len(), 0);
    assert_eq!(app.input_history.len(), 0);
    assert!(app.history_idx.is_none());
    assert!(app.saved_draft.is_none());
    assert!(!app.is_busy);
    assert!(!app.is_cancelling);
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);
    assert_eq!(app.tick, 0);
}

#[test]
fn test_app_spinner_frames_cycle() {
    let mut app = App::new();
    let initial_frame = app.current_spinner();
    assert!(!initial_frame.is_empty());

    // Advance ticks and verify cycling without panic
    for _ in 0..50 {
        app.tick = app.tick.wrapping_add(1);
        let frame = app.current_spinner();
        assert!(!frame.is_empty());
    }
}
