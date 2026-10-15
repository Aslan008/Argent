//! Selecting text in the transcript with the mouse while the chat scrolls.
//!
//! The selection used to be two screen cells. Scrolling slid other text under them,
//! so what was highlighted and what got copied stopped agreeing. These tests scroll
//! in the middle of a selection and check that the text, the highlight and the copy
//! all stay on the words that were chosen.

use std::time::{Duration, Instant};

use argent_tui::app::{App, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::Modifier;
use ratatui::Terminal;

const W: u16 = 40;
const H: u16 = 10;

fn draw(app: &App) -> Buffer {
    let mut term = Terminal::new(TestBackend::new(W, H)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    term.backend().buffer().clone()
}

fn row_of(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>().trim_end().to_string()
}

/// The screen row showing `needle`.
fn find_row(buf: &Buffer, needle: &str) -> u16 {
    (0..buf.area.height)
        .find(|&y| row_of(buf, y).contains(needle))
        .unwrap_or_else(|| panic!("`{needle}` is not on screen"))
}

fn reversed(buf: &Buffer, x: u16, y: u16) -> bool {
    buf[(x, y)].modifier.contains(Modifier::REVERSED)
}

/// `line 00` .. `line NN`, one message each, drawn once so the view knows its rows.
fn numbered(n: usize) -> App {
    let mut app = App::new();
    for i in 0..n {
        app.handle_event(ArgentEvent::System(format!("line {i:02}")));
    }
    draw(&app);
    app
}

// ── the bug itself ───────────────────────────────────────────────────────────

#[test]
fn scrolling_does_not_change_what_is_selected() {
    let mut app = numbered(40);
    let buf = draw(&app);
    let y = find_row(&buf, "line 35");
    assert!(app.select_cells((0, y), (6, y + 1)));
    assert_eq!(app.selected_transcript_text().as_deref(), Some("line 35\nline 36"));

    app.scroll_up(3);
    draw(&app);
    assert_eq!(
        app.selected_transcript_text().as_deref(),
        Some("line 35\nline 36"),
        "the same words after the chat moved"
    );

    app.scroll_down(3);
    draw(&app);
    assert_eq!(app.selected_transcript_text().as_deref(), Some("line 35\nline 36"));
}

#[test]
fn the_highlight_moves_with_the_text() {
    let mut app = numbered(40);
    let before = draw(&app);
    let y = find_row(&before, "line 35");
    assert!(app.select_cells((0, y), (6, y)));
    assert!(reversed(&draw(&app), 0, y), "painted where the words are");

    app.scroll_up(3);
    let after = draw(&app);
    let moved = find_row(&after, "line 35");
    assert_eq!(moved, y + 3, "the words slid down three rows");
    assert!(reversed(&after, 0, moved), "and the highlight went with them");
    assert!(reversed(&after, 6, moved));
    assert!(!reversed(&after, 0, y), "nothing is painted on the text that took their place");
}

#[test]
fn a_selection_that_scrolled_out_of_view_still_copies_everything() {
    let mut app = numbered(40);
    let buf = draw(&app);
    let top = find_row(&buf, "line 32");
    let bottom = find_row(&buf, "line 39");
    assert!(app.select_cells((0, top), (6, bottom)));

    app.scroll_up(20);
    draw(&app);
    let text = app.selected_transcript_text().expect("still selected");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.first(), Some(&"line 32"));
    assert_eq!(lines.last(), Some(&"line 39"));
    assert_eq!(lines.len(), 8, "every row between, none lost to scrolling: {text}");
}

#[test]
fn a_selection_can_start_above_what_is_on_screen_by_scrolling_while_dragging() {
    let mut app = numbered(60);
    let buf = draw(&app);
    let bottom = find_row(&buf, "line 59");
    // Press at the end of the last line, then take the pointer to the first row of
    // the chat and hold it there.
    assert!(app.begin_selection(6, bottom, 1));
    app.drag_selection(0, 0);
    assert_eq!(app.drag_edge, Some(-1), "held at the top edge: the chat scrolls on");

    let mut now = Instant::now();
    for _ in 0..12 {
        now += Duration::from_millis(60);
        app.selection_tick(now);
        draw(&app);
    }
    assert!(app.scroll_offset >= 10, "the chat scrolled up by itself ({})", app.scroll_offset);

    let copied = app.finish_selection().expect("a drag copies what it covered");
    assert!(copied.contains("line 59"), "the start of the drag is kept: {copied}");
    let first = copied.lines().next().unwrap();
    let n: usize = first.trim_start_matches("line ").parse().unwrap();
    assert!(n <= 49, "the selection grew upward past what was on screen: {first}");
}

#[test]
fn scrolling_with_the_wheel_mid_drag_keeps_the_end_under_the_pointer() {
    let mut app = numbered(60);
    let buf = draw(&app);
    let y = find_row(&buf, "line 55");
    assert!(app.begin_selection(0, y, 1));
    app.drag_selection(6, y);
    app.scroll_up(4);
    draw(&app);
    app.selection_tick(Instant::now());
    // The pointer did not move, but the row under it now is four rows earlier.
    let text = app.finish_selection().unwrap();
    // The press stayed on "line 55"; the pointer, not having moved, is now over the
    // row that slid under it, four rows up.
    assert!(text.contains("line 53"), "the end followed the pointer onto the new row: {text}");
    assert!(text.lines().count() >= 4, "{text}");
    assert!(text.lines().last().unwrap().starts_with('l'), "the press is still where it was: {text}");
}

// ── clicks ───────────────────────────────────────────────────────────────────

#[test]
fn a_plain_click_selects_nothing() {
    let mut app = numbered(5);
    assert!(app.begin_selection(2, 1, 1));
    assert_eq!(app.finish_selection(), None);
    assert!(app.mouse_sel.is_none());
}

#[test]
fn double_click_selects_the_word_and_triple_click_the_row() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::System("alpha beta_gamma delta".into()));
    draw(&app);

    assert!(app.begin_selection(8, 0, 2));
    assert_eq!(app.finish_selection().as_deref(), Some("beta_gamma"));

    assert!(app.begin_selection(8, 0, 3));
    assert_eq!(app.finish_selection().as_deref(), Some("alpha beta_gamma delta"));
}

#[test]
fn a_one_letter_word_is_still_a_selection() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::System("a b c".into()));
    draw(&app);
    assert!(app.begin_selection(2, 0, 2));
    assert_eq!(app.finish_selection().as_deref(), Some("b"));
}

#[test]
fn clicks_in_quick_succession_on_one_spot_count_up() {
    let mut app = numbered(3);
    let t = Instant::now();
    assert_eq!(app.click_count(3, 1, t), 1);
    assert_eq!(app.click_count(3, 1, t + Duration::from_millis(200)), 2);
    assert_eq!(app.click_count(4, 1, t + Duration::from_millis(350)), 3);
    assert_eq!(app.click_count(4, 1, t + Duration::from_millis(500)), 1, "after a triple it starts over");
    assert_eq!(app.click_count(4, 1, t + Duration::from_millis(2000)), 1, "too slow");
    assert_eq!(app.click_count(30, 2, t + Duration::from_millis(2100)), 1, "another row");
}

// ── Ctrl+A on the conversation ───────────────────────────────────────────────

#[test]
fn select_all_takes_the_whole_conversation_not_just_what_is_on_screen() {
    let mut app = numbered(40);
    assert!(app.select_all_transcript());
    let text = app.selected_transcript_text().unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.first(), Some(&"line 00"));
    assert_eq!(lines.last(), Some(&"line 39"));
    assert_eq!(lines.len(), 40);
}

#[test]
fn select_all_on_an_empty_conversation_does_nothing() {
    let mut app = App::new();
    draw(&app);
    assert!(!app.select_all_transcript());
    assert!(app.mouse_sel.is_none());
}

#[test]
fn select_all_is_painted_on_every_visible_row() {
    let mut app = numbered(40);
    assert!(app.select_all_transcript());
    let buf = draw(&app);
    let y = find_row(&buf, "line 36");
    assert!(reversed(&buf, 0, y) && reversed(&buf, 6, y));
}

// ── housekeeping ─────────────────────────────────────────────────────────────

#[test]
fn a_selection_in_a_cleared_conversation_is_dropped() {
    let mut app = numbered(10);
    assert!(app.select_all_transcript());
    app.messages.clear();
    app.drop_stale_selection();
    assert!(app.mouse_sel.is_none());
}

#[test]
fn new_messages_do_not_disturb_a_selection_further_up() {
    let mut app = numbered(30);
    let buf = draw(&app);
    let y = find_row(&buf, "line 29");
    assert!(app.select_cells((0, y), (6, y)));
    app.handle_event(ArgentEvent::System("a late arrival".into()));
    draw(&app);
    assert_eq!(app.selected_transcript_text().as_deref(), Some("line 29"));
}

#[test]
fn a_selection_over_a_tool_card_copies_the_card_text() {
    let mut app = App::new();
    app.messages.push(MessageItem::System("before".into()));
    app.messages.push(MessageItem::Notice("after".into()));
    draw(&app);
    assert!(app.select_all_transcript());
    let text = app.selected_transcript_text().unwrap();
    assert!(text.contains("before") && text.contains("after"), "{text}");
}
