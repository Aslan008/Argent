use std::collections::HashMap;
use argent_tui::app::App;

#[test]
fn test_insert_and_cursor_tracking() {
    let mut app = App::new();
    app.insert_char('H');
    app.insert_char('i');
    assert_eq!(app.input, "Hi");
    assert_eq!(app.cursor_pos, 2);

    app.move_cursor_left();
    assert_eq!(app.cursor_pos, 1);

    app.insert_char('e');
    app.insert_char('y');
    assert_eq!(app.input, "Heyi");
    assert_eq!(app.cursor_pos, 3);

    app.move_cursor_home();
    assert_eq!(app.cursor_pos, 0);

    app.move_cursor_end();
    assert_eq!(app.cursor_pos, 4);
}

#[test]
fn test_backspace_and_delete() {
    let mut app = App::new();
    app.insert_str("Hello");
    assert_eq!(app.input, "Hello");
    assert_eq!(app.cursor_pos, 5);

    app.backspace();
    assert_eq!(app.input, "Hell");
    assert_eq!(app.cursor_pos, 4);

    app.move_cursor_home();
    app.move_cursor_right(); // At 'e'
    app.delete_char(); // deletes 'e'
    assert_eq!(app.input, "Hll");
    assert_eq!(app.cursor_pos, 1);
}

#[test]
fn test_delete_to_start_and_end() {
    let mut app = App::new();
    app.insert_str("First Middle Last");
    // Move to 'Middle'
    app.cursor_pos = 6; // starts at 'M'
    app.delete_to_start();
    assert_eq!(app.input, "Middle Last");
    assert_eq!(app.cursor_pos, 0);

    app.cursor_pos = 6; // before 'Last'
    app.delete_to_end();
    assert_eq!(app.input, "Middle");
    assert_eq!(app.cursor_pos, 6);
}

#[test]
fn test_status_update_contract() {
    let mut app = App::new();
    let mut state = HashMap::new();
    state.insert("model".to_string(), "claude-3-7-sonnet".to_string());
    state.insert("provider".to_string(), "anthropic".to_string());
    state.insert("tier".to_string(), "fast".to_string());
    state.insert("ctx_tokens".to_string(), "4500".to_string());
    state.insert("ctx_max".to_string(), "200000".to_string());
    state.insert("ctx_percent".to_string(), "2.25".to_string());
    state.insert("vibe".to_string(), "true".to_string());
    state.insert("turn_counter".to_string(), "7".to_string());

    app.update_status(state);

    assert_eq!(app.status.model, "claude-3-7-sonnet");
    assert_eq!(app.status.provider, "anthropic");
    assert_eq!(app.status.tier, "fast");
    assert_eq!(app.status.ctx_tokens, 4500);
    assert_eq!(app.status.ctx_max, 200000);
    assert_eq!(app.status.ctx_percent, 2.25);
    assert!(app.status.vibe);
    assert_eq!(app.status.turn_counter, 7);
}
