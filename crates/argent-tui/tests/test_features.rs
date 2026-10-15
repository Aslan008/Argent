use argent_tui::app::{App, ClickAction};
use argent_tui::events::ArgentEvent;
use std::collections::HashMap;

#[test]
fn test_app_command_in_flight_lifecycle() {
    let mut app = App::new();
    assert!(!app.is_command_in_flight);

    // Typing and submitting a slash command while busy queues it
    app.is_busy = true;
    app.input = "/provider".to_string();
    assert!(app.submit_input().is_none());
    assert_eq!(app.message_queue.len(), 1);

    // Popping queued command sets is_command_in_flight = true
    app.is_busy = false;
    let next = app.pop_next_queued_message();
    assert_eq!(next, Some("/provider".to_string()));
    assert!(app.is_command_in_flight);
    assert!(app.is_busy);

    // Rejection resets is_command_in_flight
    app.requeue_rejected_message("/provider".to_string());
    assert!(!app.is_command_in_flight);
    assert!(!app.is_busy);

    // Popping again and receiving Done clears is_command_in_flight
    let _ = app.pop_next_queued_message();
    assert!(app.is_command_in_flight);
    app.handle_event(ArgentEvent::Done);
    assert!(!app.is_command_in_flight);
    assert!(!app.is_busy);
}

#[test]
fn test_app_update_status_context_and_provider() {
    let mut app = App::new();
    assert_eq!(app.status.ctx_max, 32768);

    let mut state = HashMap::new();
    state.insert("model".to_string(), "glm-5.1".to_string());
    state.insert("provider".to_string(), "zai".to_string());
    state.insert("ctx_tokens".to_string(), "1234".to_string());
    state.insert("ctx_max".to_string(), "131072".to_string());
    state.insert("ctx_percent".to_string(), "0.9".to_string());
    state.insert("vibe".to_string(), "True".to_string());

    app.update_status(state);

    assert_eq!(app.status.model, "glm-5.1");
    assert_eq!(app.status.provider, "zai");
    assert_eq!(app.status.ctx_tokens, 1234);
    assert_eq!(app.status.ctx_max, 131072);
    assert!((app.status.ctx_percent - 0.9).abs() < 1e-6);
    assert!(app.status.vibe);
}

#[test]
fn test_click_action_open_provider_picker() {
    let app = App::new();
    app.register_clickable(10, 20, 15, 1, ClickAction::OpenProviderPicker);
    let action = app.get_clicked_action(12, 20);
    assert_eq!(action, Some(ClickAction::OpenProviderPicker));
}
