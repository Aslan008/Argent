//! Selecting and replacing text in the prompt: Ctrl+A, typing over a selection,
//! selecting by word.
//!
//! Typing with a selection active used to drop the selection and insert next to the
//! old text, and Delete / paste ignored it, so "select all, then type" — the reason
//! to have Ctrl+A at all — left the old text in place.

use argent_tui::app::App;

fn app_with(text: &str) -> App {
    let mut app = App::new();
    app.insert_str(text);
    app
}

#[test]
fn select_all_selects_the_whole_input() {
    let mut app = app_with("one two three");
    assert!(app.select_all_input());
    assert!(app.has_selection());
    assert_eq!(app.selected_text().as_deref(), Some("one two three"));
    assert_eq!(app.cursor_pos, 13);
}

#[test]
fn select_all_spans_several_lines() {
    let mut app = app_with("first\nsecond\nthird");
    assert!(app.select_all_input());
    assert_eq!(app.selected_text().as_deref(), Some("first\nsecond\nthird"));
}

#[test]
fn select_all_on_an_empty_prompt_selects_nothing() {
    let mut app = App::new();
    assert!(!app.select_all_input());
    assert!(!app.has_selection());
}

#[test]
fn typing_over_a_full_selection_replaces_everything() {
    let mut app = app_with("old text");
    app.select_all_input();
    app.insert_char('x');
    assert_eq!(app.input, "x");
    assert_eq!(app.cursor_pos, 1);
    assert!(!app.has_selection());
}

#[test]
fn typing_over_a_partial_selection_replaces_only_that_part() {
    let mut app = app_with("hello world");
    app.set_cursor(6, false);
    app.set_cursor(11, true); // "world"
    assert_eq!(app.selected_text().as_deref(), Some("world"));
    app.insert_char('X');
    assert_eq!(app.input, "hello X");
}

#[test]
fn pasting_over_a_selection_replaces_it() {
    let mut app = app_with("abc");
    app.select_all_input();
    app.insert_str("pasted\ntext");
    assert_eq!(app.input, "pasted\ntext");
}

#[test]
fn backspace_and_delete_remove_the_selection_and_nothing_more() {
    let mut app = app_with("keep drop keep");
    app.set_cursor(5, false);
    app.set_cursor(9, true);
    assert_eq!(app.selected_text().as_deref(), Some("drop"));
    app.backspace();
    assert_eq!(app.input, "keep  keep");

    let mut app = app_with("keep drop keep");
    app.set_cursor(5, false);
    app.set_cursor(9, true);
    app.delete_char();
    assert_eq!(app.input, "keep  keep");
}

#[test]
fn deleting_a_word_with_a_selection_removes_the_selection() {
    let mut app = app_with("alpha beta gamma");
    app.set_cursor(6, false);
    app.set_cursor(10, true);
    app.delete_word_left();
    assert_eq!(app.input, "alpha  gamma");
}

#[test]
fn replacing_a_selection_is_a_single_undo_step() {
    let mut app = app_with("original");
    app.select_all_input();
    app.insert_char('z');
    assert_eq!(app.input, "z");
    assert!(app.undo());
    assert_eq!(app.input, "original", "one undo brings everything back, not a half-edit");
}

#[test]
fn select_all_then_cut_leaves_an_empty_prompt() {
    let mut app = app_with("everything");
    app.select_all_input();
    assert_eq!(app.delete_selection().as_deref(), Some("everything"));
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);
}

// ── by word ──────────────────────────────────────────────────────────────────

#[test]
fn ctrl_shift_left_grows_the_selection_a_word_at_a_time() {
    let mut app = app_with("one two three");
    app.select_word_left();
    assert_eq!(app.selected_text().as_deref(), Some("three"));
    app.select_word_left();
    assert_eq!(app.selected_text().as_deref(), Some("two three"));
    app.select_word_left();
    assert_eq!(app.selected_text().as_deref(), Some("one two three"));
}

#[test]
fn ctrl_shift_right_selects_forward() {
    let mut app = app_with("one two three");
    app.set_cursor(0, false);
    app.select_word_right();
    assert_eq!(app.selected_text().as_deref(), Some("one"));
    app.select_word_right();
    assert_eq!(app.selected_text().as_deref(), Some("one two"));
}

#[test]
fn selecting_back_over_the_anchor_shrinks_then_flips_the_selection() {
    let mut app = app_with("one two three");
    app.set_cursor(4, false);
    app.select_word_right(); // "two"
    assert_eq!(app.selected_text().as_deref(), Some("two"));
    app.select_word_left(); // back to the anchor
    assert!(!app.has_selection());
}

#[test]
fn moving_by_word_keeps_working_without_shift() {
    let mut app = app_with("one two");
    app.move_word_left();
    assert_eq!(app.cursor_pos, 4);
    app.move_word_left();
    assert_eq!(app.cursor_pos, 0);
    app.move_word_right();
    assert_eq!(app.cursor_pos, 3);
}

#[test]
fn select_all_closes_an_open_command_popup() {
    let mut app = App::new();
    app.completions.push(argent_tui::app::CompletionItem { name: "/help".into(), help: String::new() });
    app.insert_char('/');
    assert!(app.show_completions);
    app.select_all_input();
    assert!(!app.show_completions, "no popup over a selection");
}
