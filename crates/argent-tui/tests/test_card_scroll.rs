//! A tool card is a window on its whole output.
//!
//! The card used to stop at fifteen lines and say "… ещё N строк", with no way to read
//! the rest. Now the whole result is reachable: the wheel over the card moves the card
//! (not the chat behind it), PgUp/PgDn page the focused card, and the last line of a
//! thousand-line run is one flick away.

use std::collections::HashMap;
use std::time::Instant;

use argent_tui::app::{App, MessageItem};
use argent_tui::theme::UiStyle;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf.content[(y * buf.area.width + x) as usize].symbol())
                .collect::<String>()
        })
        .collect()
}

fn found(screen: &[String], needle: &str) -> Option<usize> {
    screen.iter().position(|r| r.contains(needle))
}

/// The premium look is the one that owns the tool card layout.
fn app_with_output(lines: usize, expanded: bool) -> App {
    let mut app = App::new();
    app.ui.set_style(UiStyle::Premium);
    let result: String = (0..lines).map(|i| format!("строка {i}\n")).collect();
    app.messages.push(MessageItem::Tool {
        name: "run_command".into(),
        args: HashMap::new(),
        result: Some(result),
        duration_secs: Some(1.0),
        expanded,
        is_running: false,
    });
    app
}

#[test]
fn the_card_knows_how_far_it_can_scroll() {
    let app = app_with_output(60, true);
    render(&app, 100, 40);

    // 60 lines, a 15-row window: 45 rows of travel.
    assert_eq!(app.card_max.borrow().get(&0).copied(), Some(45));
}

#[test]
fn the_wheel_over_the_card_scrolls_the_card_and_not_the_chat() {
    let mut app = app_with_output(60, true);
    render(&app, 100, 40);
    let (x, y, .., idx) = app.card_rects.borrow()[0];
    app.mouse_pos = Some((x + 1, y + 1));

    let chat_before = app.scroll_offset;
    app.wheel(false, Instant::now());

    assert_eq!(idx, 0);
    assert_eq!(app.card_scroll.get(&0).copied(), Some(3), "карточка листнулась");
    assert_eq!(app.scroll_offset, chat_before, "чат за карточкой не поехал");
}

#[test]
fn the_window_stops_at_the_last_line_and_reaches_it() {
    let mut app = app_with_output(60, true);
    render(&app, 100, 40);
    let (x, y, .., _) = app.card_rects.borrow()[0];
    app.mouse_pos = Some((x + 1, y + 1));

    for _ in 0..50 {
        app.wheel(false, Instant::now());
    }

    assert_eq!(app.card_scroll.get(&0).copied(), Some(45), "дальше конца не уходит");
    let screen = render(&app, 100, 80);
    assert!(
        found(&screen, "строка 59").is_some(),
        "последняя строка достижима: {screen:?}"
    );
}

#[test]
fn a_folded_card_takes_no_wheel() {
    let app = app_with_output(60, false);
    render(&app, 100, 40);

    assert_eq!(app.card_max.borrow().get(&0).copied(), Some(0));
    assert!(app.card_rects.borrow().is_empty(), "свёрнутая карточка колесо не забирает");
}

#[test]
fn the_footer_says_where_the_window_is() {
    let app = app_with_output(60, true);
    let screen = render(&app, 100, 40);

    assert!(found(&screen, "из 60").is_some(), "{screen:?}");
    assert!(found(&screen, "колесо").is_some(), "{screen:?}");
}

#[test]
fn paging_moves_the_focused_card() {
    let mut app = app_with_output(60, true);
    render(&app, 100, 40);
    app.focus_idx = Some(0);
    app.focus_pinned = true;

    assert!(app.page_focused_card(false), "PgDn принадлежит карточке");
    assert_eq!(app.card_scroll.get(&0).copied(), Some(10));

    assert!(app.page_focused_card(true));
    assert_eq!(app.card_scroll.get(&0).copied(), Some(0));
}

#[test]
fn a_card_that_fits_keeps_the_chat_keys() {
    let mut app = app_with_output(5, true);
    render(&app, 100, 40);
    app.focus_idx = Some(0);

    assert!(!app.page_focused_card(false), "листать нечего — ключи остаются чату");
}
