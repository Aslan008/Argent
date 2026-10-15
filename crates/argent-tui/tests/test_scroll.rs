//! Scrolling the chat: the mouse wheel, its acceleration, and reaching both ends.
//!
//! The event loop draws a frame after every event and then lets the app take in the
//! offset the frame measured (`sync_scroll`); `turn` does the same, so these tests
//! scroll the way the real loop does.

use std::time::{Duration, Instant};

use argent_tui::app::App;
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn long_chat() -> App {
    let mut app = App::new();
    for i in 0..400 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    draw(&app);
    app
}

fn draw(app: &App) -> String {
    let mut term = Terminal::new(TestBackend::new(60, 20)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    let buf = term.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// One wheel notch followed by the frame it causes.
fn turn(app: &mut App, up: bool, at: Instant) {
    app.wheel(up, at);
    draw(app);
    app.sync_scroll();
}

#[test]
fn one_notch_is_three_rows() {
    let mut app = long_chat();
    turn(&mut app, true, Instant::now());
    assert_eq!(app.scroll_offset, 3);
    turn(&mut app, false, Instant::now() + Duration::from_secs(1));
    assert_eq!(app.scroll_offset, 0);
}

#[test]
fn a_wheel_that_keeps_turning_speeds_up() {
    let mut app = long_chat();
    let t = Instant::now();
    let mut steps = Vec::new();
    for i in 0..12u64 {
        let before = app.scroll_offset;
        turn(&mut app, true, t + Duration::from_millis(30 * i));
        steps.push(app.scroll_offset - before);
    }
    assert_eq!(steps[0], 3, "the first notch is plain: {steps:?}");
    assert!(steps[1] > steps[0] && steps[2] > steps[1], "then it builds up: {steps:?}");
    assert_eq!(*steps.last().unwrap(), 12, "and stops at twelve: {steps:?}");
    assert!(steps.iter().all(|&s| s <= 12));
}

#[test]
fn a_pause_or_a_change_of_direction_starts_over() {
    let mut app = long_chat();
    let t = Instant::now();
    for i in 0..6u64 {
        turn(&mut app, true, t + Duration::from_millis(20 * i));
    }
    let before = app.scroll_offset;
    turn(&mut app, true, t + Duration::from_millis(1000)); // a pause
    assert_eq!(app.scroll_offset - before, 3);

    let before = app.scroll_offset;
    turn(&mut app, false, t + Duration::from_millis(1030)); // the other way
    assert_eq!(before - app.scroll_offset, 3);
}

#[test]
fn the_wheel_cannot_scroll_past_either_end() {
    let mut app = long_chat();
    let t = Instant::now();
    for i in 0..600u64 {
        turn(&mut app, true, t + Duration::from_millis(10 * i));
    }
    assert!(draw(&app).contains("line 000"), "the first line is reachable by wheel alone");
    let top = app.scroll_offset;
    turn(&mut app, true, t + Duration::from_secs(60));
    assert_eq!(app.scroll_offset, top, "no further than the first row");

    for i in 0..600u64 {
        turn(&mut app, false, t + Duration::from_secs(120) + Duration::from_millis(10 * i));
    }
    assert_eq!(app.scroll_offset, 0, "back at the newest message");
    assert!(draw(&app).contains("line 399"));
}

#[test]
fn ctrl_home_goes_all_the_way_to_the_first_message() {
    let mut app = long_chat();
    app.scroll_up(usize::MAX);
    let screen = draw(&app);
    assert!(screen.contains("line 000"), "the very first line, in one press:\n{screen}");
    app.sync_scroll();
    assert!(app.scroll_offset < usize::MAX, "the real distance is taken in: {}", app.scroll_offset);
    assert!(app.scroll_offset > 300);
}

#[test]
fn ctrl_end_comes_back_to_the_last() {
    let mut app = long_chat();
    app.scroll_up(usize::MAX);
    draw(&app);
    app.sync_scroll();
    app.scroll_down(usize::MAX);
    let screen = draw(&app);
    assert!(screen.contains("line 399"), "{screen}");
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll, "following new output again");
}

#[test]
fn one_notch_after_ctrl_home_moves_from_the_top_not_from_infinity() {
    let mut app = long_chat();
    app.scroll_up(usize::MAX);
    draw(&app);
    app.sync_scroll();
    let top = app.scroll_offset;
    turn(&mut app, false, Instant::now());
    assert_eq!(app.scroll_offset, top - 3);
}

#[test]
fn the_offset_never_stays_beyond_what_the_chat_has() {
    let mut app = App::new();
    for i in 0..5 {
        app.handle_event(ArgentEvent::System(format!("line {i}")));
    }
    app.scroll_up(usize::MAX);
    draw(&app);
    app.sync_scroll();
    assert_eq!(app.scroll_offset, 0, "a chat that fits the screen has nowhere to scroll");
}

// ── scrolling past the end ───────────────────────────────────────────────────
//
// A console lets you scroll below the last line, leaving empty space under the
// newest output. The chat does the same on the scroll KEYS (Shift/Ctrl/Alt+Down,
// PageDown): at the newest message the space opens; anything new, or scrolling up,
// closes it again. The wheel does not open it (see the tests at the end).

/// Scroll down one step with a key, as Shift+Down does, then draw.
fn key_down(app: &mut App) {
    app.lower(3);
    draw(app);
    app.sync_scroll();
}

fn gap_below_the_last_line(app: &App) -> usize {
    let screen = draw(app);
    let lines: Vec<&str> = screen.lines().collect();
    let last = lines.iter().rposition(|l| l.contains("line 399") || l.contains("line 004")).expect("the newest line");
    let prompt = lines.iter().position(|l| l.starts_with("❯")).expect("the prompt");
    prompt - last - 1
}

#[test]
fn scrolling_down_by_key_at_the_newest_message_opens_empty_space_under_it() {
    let mut app = long_chat();
    assert_eq!(gap_below_the_last_line(&app), 0, "to begin with the prompt is right under the chat");
    key_down(&mut app);
    assert_eq!(app.overscroll, 3);
    assert_eq!(gap_below_the_last_line(&app), 3);
    key_down(&mut app);
    assert_eq!(gap_below_the_last_line(&app), 6);
}

#[test]
fn the_space_has_a_limit_and_the_chat_keeps_at_least_half_its_rows() {
    let mut app = long_chat();
    for _ in 0..40u64 {
        key_down(&mut app);
    }
    assert!(app.overscroll <= argent_tui::app::OVERSCROLL_MAX);
    let screen = draw(&app);
    let rows_of_chat = screen.lines().position(|l| l.starts_with("❯")).unwrap() - gap_below_the_last_line(&app);
    assert!(rows_of_chat >= 8, "a 20-row screen keeps half its chat rows: {rows_of_chat}");
}

#[test]
fn scrolling_up_closes_the_space_before_it_moves_the_chat() {
    let mut app = long_chat();
    key_down(&mut app);
    key_down(&mut app);
    assert_eq!(app.overscroll, 6);
    turn(&mut app, true, Instant::now() + Duration::from_secs(2));
    assert_eq!((app.overscroll, app.scroll_offset), (3, 0), "the gap shrinks, the chat stays put");
    turn(&mut app, true, Instant::now() + Duration::from_secs(3));
    assert_eq!((app.overscroll, app.scroll_offset), (0, 0));
    turn(&mut app, true, Instant::now() + Duration::from_secs(4));
    assert_eq!((app.overscroll, app.scroll_offset), (0, 3), "only now does it scroll back");
}

#[test]
fn new_output_closes_the_space() {
    let mut app = long_chat();
    key_down(&mut app);
    assert_eq!(app.overscroll, 3);
    app.handle_event(ArgentEvent::System("a late line".into()));
    assert_eq!(app.overscroll, 0);
}

#[test]
fn sending_a_message_closes_the_space() {
    let mut app = long_chat();
    key_down(&mut app);
    app.insert_str("hello");
    app.submit_input();
    assert_eq!(app.overscroll, 0);
}

#[test]
fn jumping_to_the_ends_closes_the_space() {
    let mut app = long_chat();
    key_down(&mut app);
    app.scroll_down(usize::MAX);
    assert_eq!(app.overscroll, 0, "Ctrl+End");
    key_down(&mut app);
    app.scroll_up(usize::MAX);
    assert_eq!(app.overscroll, 0, "Ctrl+Home");
}

#[test]
fn no_space_opens_by_key_while_the_chat_is_scrolled_back() {
    let mut app = long_chat();
    turn(&mut app, true, Instant::now());
    turn(&mut app, true, Instant::now() + Duration::from_secs(1));
    assert!(app.scroll_offset > 0);
    key_down(&mut app);
    assert_eq!(app.overscroll, 0, "going down first just comes back toward the newest");
}

#[test]
fn the_cursor_follows_the_prompt_down() {
    let mut app = long_chat();
    app.insert_str("typing");
    key_down(&mut app);
    let mut term = Terminal::new(TestBackend::new(60, 20)).unwrap();
    term.draw(|f| draw_ui(f, &app)).unwrap();
    let cursor = term.get_cursor_position().unwrap();
    let screen = draw(&app);
    let prompt = screen.lines().position(|l| l.starts_with("❯ typing")).unwrap() as u16;
    assert_eq!(cursor.y, prompt);
}

#[test]
fn a_short_conversation_gets_the_space_from_the_keys_too() {
    let mut app = App::new();
    for i in 0..5 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    draw(&app);
    key_down(&mut app);
    assert_eq!(gap_below_the_last_line(&app), 3, "the prompt moves down, the lines stay where they are");
}

#[test]
fn the_premium_chip_offers_the_way_back_from_the_space() {
    use argent_tui::app::ClickAction;
    use argent_tui::theme::UiStyle;
    let mut app = long_chat();
    app.ui.set_style(UiStyle::Premium);
    key_down(&mut app);
    let screen = draw(&app);
    let y = screen.lines().position(|l| l.contains("К последнему")).expect("the chip") as u16;
    let x = screen.lines().nth(y as usize).unwrap().chars().position(|c| c == '↓').unwrap() as u16;
    assert_eq!(app.get_clicked_action(x, y), Some(ClickAction::ScrollToBottom));
}

// ── the wheel never moves the input field ────────────────────────────────────

/// The screen row the prompt is on.
fn prompt_row(app: &App) -> usize {
    draw(app).lines().position(|l| l.starts_with("❯")).expect("the prompt")
}

#[test]
fn the_wheel_never_pushes_the_prompt_up_the_screen() {
    let mut app = long_chat();
    let before = prompt_row(&app);
    let t = Instant::now();
    for i in 0..40u64 {
        turn(&mut app, false, t + Duration::from_millis(15 * i));
    }
    assert_eq!(app.overscroll, 0, "no empty space opened under the newest message");
    assert_eq!(prompt_row(&app), before, "the input field stayed where it was");
    assert_eq!(app.scroll_offset, 0);
    assert!(draw(&app).contains("line 399"), "the newest message is still on screen");
}

#[test]
fn the_wheel_never_moves_the_prompt_of_a_short_conversation_either() {
    let mut app = App::new();
    for i in 0..5 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    let before = prompt_row(&app);
    for i in 0..10u64 {
        turn(&mut app, false, Instant::now() + Duration::from_secs(i));
    }
    assert_eq!(prompt_row(&app), before);
    assert_eq!(gap_below_the_last_line(&app), 0);
}

#[test]
fn the_wheel_still_closes_a_gap_a_key_opened() {
    let mut app = long_chat();
    key_down(&mut app);
    key_down(&mut app);
    assert_eq!(app.overscroll, 6);
    turn(&mut app, false, Instant::now());
    assert_eq!(app.overscroll, 6, "turning down does not widen it");
    turn(&mut app, true, Instant::now() + Duration::from_secs(1));
    assert_eq!(app.overscroll, 3, "turning up closes it first, as before");
}
