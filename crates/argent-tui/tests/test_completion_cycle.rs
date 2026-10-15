//! Tab cycles through the suggestions instead of taking one and closing the popup.
//!
//! Tab used to accept the highlighted command and dismiss the list, so a second
//! look at the alternatives meant typing the prefix again. Now Tab moves to the
//! next suggestion (Shift+Tab to the previous), shows it in the input, and the list
//! stays — still filtered by what was typed before the first Tab.

use argent_tui::app::{App, CompletionItem};

fn app_with(names: &[&str]) -> App {
    let mut app = App::new();
    for n in names {
        app.completions.push(CompletionItem { name: (*n).to_string(), help: format!("{n} help") });
    }
    app
}

fn type_str(app: &mut App, s: &str) {
    for c in s.chars() {
        app.insert_char(c);
    }
}

fn matches(app: &App) -> Vec<String> {
    app.get_matched_completions().iter().map(|c| c.name.clone()).collect()
}

const COMMANDS: [&str; 6] = ["/help", "/history", "/model", "/mcp", "/memory", "/skill import"];

#[test]
fn the_popup_opens_on_a_slash_and_lists_what_matches() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    assert!(app.show_completions);
    assert_eq!(matches(&app), vec!["/help", "/history"]);
}

#[test]
fn tab_shows_the_highlighted_suggestion_and_keeps_the_popup() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    assert!(app.cycle_completion(true, false));
    assert_eq!(app.input, "/help");
    assert_eq!(app.cursor_pos, 5);
    assert!(app.show_completions, "the popup stays open");
}

#[test]
fn tab_again_moves_on_and_the_list_does_not_collapse() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    app.cycle_completion(true, false);
    assert_eq!(matches(&app), vec!["/help", "/history"], "still the matches for what was typed");

    app.cycle_completion(true, false);
    assert_eq!(app.input, "/history");
    assert_eq!(matches(&app), vec!["/help", "/history"]);
    assert_eq!(app.completion_idx, 1);

    app.cycle_completion(true, false);
    assert_eq!(app.input, "/help", "past the last one it wraps to the first");
    assert!(app.show_completions);
}

#[test]
fn shift_tab_goes_back_and_wraps() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    assert!(app.cycle_completion(false, false));
    assert_eq!(app.input, "/history", "back from the first lands on the last");
    app.cycle_completion(false, false);
    assert_eq!(app.input, "/help");
}

#[test]
fn the_arrow_keys_step_on_their_first_press() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    assert!(app.cycle_completion(true, true));
    assert_eq!(app.input, "/history", "Down moves to the second suggestion at once");
    assert!(app.cycle_completion(false, true));
    assert_eq!(app.input, "/help");
}

#[test]
fn enter_accepts_what_is_shown_and_closes_the_popup() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    app.cycle_completion(true, false);
    app.cycle_completion(true, false);
    assert!(app.accept_completion(), "the caller must swallow this Enter");
    assert_eq!(app.input, "/history");
    assert!(!app.show_completions);
    assert!(app.completion_cycle.is_none());
}

#[test]
fn enter_without_cycling_accepts_the_highlighted_suggestion() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/m");
    app.completion_next();
    assert!(app.accept_completion());
    assert_eq!(app.input, "/mcp");
}

#[test]
fn escape_undoes_the_preview_and_closes_the_popup() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    app.cycle_completion(true, false);
    app.cycle_completion(true, false);
    assert!(app.cancel_completion());
    assert_eq!(app.input, "/h", "back to what was typed");
    assert_eq!(app.cursor_pos, 2);
    assert!(!app.show_completions);
}

#[test]
fn escape_with_no_popup_is_not_consumed() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "hello");
    assert!(!app.cancel_completion());
    assert_eq!(app.input, "hello");
}

#[test]
fn typing_after_a_tab_ends_the_cycle_and_follows_the_text_again() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    app.cycle_completion(true, false);
    app.cycle_completion(true, false);
    assert_eq!(app.input, "/history");
    app.backspace();
    app.backspace();
    assert!(app.completion_cycle.is_none());
    assert_eq!(app.input, "/histo");
    assert_eq!(matches(&app), vec!["/history"], "the list follows the edited text");
}

#[test]
fn tab_without_a_popup_is_left_for_other_uses() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "plain text");
    assert!(!app.cycle_completion(true, false));
    assert_eq!(app.input, "plain text");
}

#[test]
fn every_command_is_reachable_not_just_the_first_ten() {
    let names: Vec<String> = (0..25).map(|i| format!("/cmd{i:02}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut app = app_with(&refs);
    type_str(&mut app, "/");
    assert_eq!(matches(&app).len(), 25, "the list is not cut at ten");

    let mut seen = Vec::new();
    for _ in 0..25 {
        app.cycle_completion(true, false);
        seen.push(app.input.clone());
    }
    assert_eq!(seen, names, "Tab visits all of them, in order");
    app.cycle_completion(true, false);
    assert_eq!(app.input, "/cmd00", "and wraps");
}

#[test]
fn a_tab_preview_is_one_undo_step() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    app.cycle_completion(true, false);
    app.cycle_completion(true, false);
    app.cycle_completion(true, false);
    assert!(app.undo());
    assert_eq!(app.input, "/h");
}

#[test]
fn multi_word_commands_are_reachable_by_tab() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/sk");
    app.cycle_completion(true, false);
    assert_eq!(app.input, "/skill import");
}

// ── @file mentions ───────────────────────────────────────────────────────────

#[test]
fn tab_cycles_through_file_mentions_and_keeps_the_text_around_them() {
    let mut app = App::new();
    type_str(&mut app, "look at @Cargo");
    assert!(app.show_file_completions, "the popup lists files");
    let files = app.file_completions.clone();
    assert!(files.len() >= 2, "Cargo.toml and Cargo.lock: {files:?}");

    assert!(app.cycle_completion(true, false));
    assert_eq!(app.input, format!("look at @{}", files[0]));
    assert!(app.show_file_completions, "the popup stays open");
    assert_eq!(app.file_completions, files, "and keeps its list");

    app.cycle_completion(true, false);
    assert_eq!(app.input, format!("look at @{}", files[1]));
}

#[test]
fn a_file_mention_in_the_middle_of_a_line_keeps_what_follows_the_cursor() {
    let mut app = App::new();
    type_str(&mut app, "see @Cargo now");
    // Put the cursor right after "@Cargo".
    app.set_cursor(10, false);
    assert!(app.show_file_completions);
    let first = app.file_completions[0].clone();
    assert!(app.cycle_completion(true, false));
    assert_eq!(app.input, format!("see @{first} now"));
    assert_eq!(app.cursor_pos, format!("see @{first}").chars().count(), "the cursor stays at the end of the mention");
}

#[test]
fn enter_on_a_file_mention_adds_the_space_and_closes_the_popup() {
    let mut app = App::new();
    type_str(&mut app, "see @Cargo");
    let first = app.file_completions[0].clone();
    assert!(app.accept_completion());
    assert_eq!(app.input, format!("see @{first} "));
    assert!(!app.show_file_completions);
}

// ── the key handlers' entry points ───────────────────────────────────────────

#[test]
fn tab_cycles_when_a_popup_is_open_and_otherwise_does_nothing_harmful() {
    let mut app = app_with(&COMMANDS);
    type_str(&mut app, "/h");
    app.tab(true);
    assert_eq!(app.input, "/help");
    app.tab(true);
    assert_eq!(app.input, "/history");
    app.tab(false);
    assert_eq!(app.input, "/help");

    let mut plain = App::new();
    type_str(&mut plain, "text");
    plain.tab(true); // no popup, no block to open: nothing happens
    assert_eq!(plain.input, "text");
}

#[test]
fn ctrl_a_selects_the_input_and_with_nothing_typed_the_conversation() {
    let mut app = App::new();
    type_str(&mut app, "abc");
    app.select_all();
    assert_eq!(app.selected_text().as_deref(), Some("abc"));
    assert!(app.mouse_sel.is_none(), "typed text wins over the transcript");

    let mut empty = App::new();
    empty.select_all();
    assert!(empty.mouse_sel.is_none(), "an empty conversation has nothing to select");
}
