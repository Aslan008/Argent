//! The thinking-level badge under the input box must reflect, and be able to
//! change, the reasoning effort the agent actually sends to the model.

use argent_tui::app::{App, MessageItem};

#[test]
fn default_level_is_auto() {
    let app = App::new();
    assert_eq!(app.status.thinking_level, "auto");
}

#[test]
fn level_is_parsed_from_bridge_state() {
    let mut app = App::new();
    let mut state = std::collections::HashMap::new();
    state.insert("thinking_level".to_string(), "xhigh".to_string());
    app.update_status(state);
    assert_eq!(app.status.thinking_level, "xhigh");
}

#[test]
fn empty_level_from_bridge_does_not_blank_the_badge() {
    let mut app = App::new();
    app.status.thinking_level = "high".to_string();
    let mut state = std::collections::HashMap::new();
    state.insert("thinking_level".to_string(), "   ".to_string());
    app.update_status(state);
    assert_eq!(app.status.thinking_level, "high");
}

#[test]
fn cycle_walks_the_documented_order() {
    let app = App::new();
    assert_eq!(app.thinking_levels, vec!["off", "auto", "low", "medium", "high", "xhigh"]);

    let mut seen = Vec::new();
    let mut level = "off".to_string();
    for _ in 0..6 {
        seen.push(level.clone());
        let mut probe = App::new();
        probe.status.thinking_level = level.clone();
        level = probe.next_thinking_level().expect("cycle must always produce a level");
    }
    assert_eq!(seen, vec!["off", "auto", "low", "medium", "high", "xhigh"]);
    assert_eq!(level, "off", "cycle must wrap back to the start");
}

#[test]
fn cycle_from_unknown_level_falls_back_to_first() {
    let mut app = App::new();
    app.status.thinking_level = "nonsense".to_string();
    assert_eq!(app.next_thinking_level().as_deref(), Some("off"));
}

#[test]
fn bridge_can_override_the_cycle() {
    let mut app = App::new();
    app.set_thinking_levels(vec!["off".into(), "max".into()]);
    assert_eq!(app.thinking_levels, vec!["off", "max"]);
    app.status.thinking_level = "off".to_string();
    assert_eq!(app.next_thinking_level().as_deref(), Some("max"));
}

#[test]
fn empty_override_does_not_clear_the_cycle() {
    let mut app = App::new();
    app.set_thinking_levels(Vec::new());
    assert_eq!(app.thinking_levels.len(), 6, "an empty list must be ignored");
}

#[test]
fn applying_a_confirmed_level_clears_any_error() {
    let mut app = App::new();
    app.set_thinking_error("boom".into());
    assert!(app.thinking_error.is_some());
    app.apply_thinking_level("medium");
    assert!(app.thinking_error.is_none());
    assert_eq!(app.status.thinking_level, "medium");
}

#[test]
fn a_rejected_change_surfaces_an_error_and_keeps_the_old_level() {
    let mut app = App::new();
    app.status.thinking_level = "low".to_string();
    app.set_thinking_error("Unknown level 'ultra'".into());
    assert_eq!(
        app.status.thinking_level, "low",
        "a failed write must not change the displayed level"
    );
    assert!(app.messages.iter().any(|m| matches!(m, MessageItem::Error { text, .. } if text.contains("ultra"))));
}