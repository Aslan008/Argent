//! Мышь в поле ввода: клик ставит каретку, протяжка выделяет, двойной клик берёт
//! слово, тройной — весь черновик.
//!
//! Проверяется через настоящую отрисовку: геометрию промпта записывает кадр, и
//! именно её читает обработчик клика.

use argent_tui::app::{App, ClickAction, MessageItem};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, w: u16, h: u16) {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
}

fn app_with(input: &str) -> App {
    let mut app = App::new();
    app.messages.push(MessageItem::User("запусти сборку".into()));
    app.messages.push(MessageItem::assistant("запустил", ""));
    app.input = input.to_string();
    app.cursor_pos = 0;
    app
}

#[test]
fn a_click_in_the_prompt_puts_the_caret_where_the_pointer_is() {
    let mut app = app_with("hello world");
    render(&app, 80, 24);
    let g = app.input_geom.get();
    assert!(g.rows >= 1, "промпт отрисован");

    let clicked = app.begin_input_selection(g.left + g.prefix + 6, g.top, 1);

    assert!(clicked, "клик попал в поле ввода");
    assert_eq!(app.cursor_pos, 6, "каретка встала под указатель");
    assert!(app.sel_anchor.is_none(), "одиночный клик ничего не выделяет");
}

#[test]
fn dragging_through_the_draft_selects_it() {
    let mut app = app_with("hello world");
    render(&app, 80, 24);
    let g = app.input_geom.get();

    app.begin_input_selection(g.left + g.prefix, g.top, 1);
    app.drag_input_selection(g.left + g.prefix + 5, g.top);

    assert_eq!(app.selected_text().as_deref(), Some("hello"));
    app.finish_input_selection();
    assert_eq!(app.selected_text().as_deref(), Some("hello"), "выделение остаётся после отпускания");
}

#[test]
fn a_double_click_takes_the_word_under_the_pointer() {
    let mut app = app_with("собери проект заново");
    render(&app, 80, 24);
    let g = app.input_geom.get();

    let clicked = app.begin_input_selection(g.left + g.prefix + 2, g.top, 2);

    assert!(clicked);
    assert_eq!(app.selected_text().as_deref(), Some("собери"));
}

#[test]
fn a_triple_click_takes_the_whole_draft() {
    let mut app = app_with("первая строка\nвторая строка");
    render(&app, 80, 24);
    let g = app.input_geom.get();

    app.begin_input_selection(g.left + g.prefix + 3, g.top, 3);

    assert_eq!(app.selected_text().as_deref(), Some("первая строка\nвторая строка"));
}

#[test]
fn a_click_on_a_wrapped_row_maps_into_that_row() {
    // Каждая строка ровно по ширине переноса, поэтому индекс считается прямо.
    let app = app_with(&"x".repeat(400));
    render(&app, 80, 24);
    let g = app.input_geom.get();
    assert!(g.rows >= 2, "черновик переносится: {} строк", g.rows);

    let second_row = app.input_char_at(g.left + g.prefix, g.top + 1);
    assert_eq!(second_row, Some(g.avail), "начало второй строки — это её первый символ");

    let inside = app.input_char_at(g.left + g.prefix + 5, g.top + 1);
    assert_eq!(inside, Some(g.avail + 5));
}

#[test]
fn a_click_outside_the_prompt_keeps_its_old_meaning() {
    let mut app = app_with("черновик");
    render(&app, 80, 24);
    let g = app.input_geom.get();
    app.cursor_pos = 3;

    // Строка чата, а не ввода.
    let in_chat = app.begin_input_selection(g.left, g.top.saturating_sub(1), 1);
    assert!(!in_chat);
    assert_eq!(app.cursor_pos, 3, "каретка не двинулась");

    // И правее текста на последней строке уже никого нет.
    assert!(app.input_char_at(g.left + g.prefix, g.top + g.rows as u16).is_none());
}

#[test]
fn a_question_or_a_modal_takes_the_prompt_away_from_the_mouse() {
    let mut app = app_with("черновик");
    app.active_select = Some(argent_tui::app::SelectModalData::new(
        "sel".into(),
        "Выбор".into(),
        vec!["Да".into(), "Нет".into()],
        None,
    ));
    render(&app, 80, 24);

    assert_eq!(app.input_geom.get().rows, 0, "в модальном опросе поля ввода нет");
    assert!(!app.begin_input_selection(10, 20, 1));
}

#[test]
fn typing_over_a_selection_replaces_it() {
    let mut app = app_with("hello world");
    render(&app, 80, 24);
    let g = app.input_geom.get();
    app.begin_input_selection(g.left + g.prefix, g.top, 1);
    app.drag_input_selection(g.left + g.prefix + 5, g.top);

    app.insert_char('!');

    assert_eq!(app.input, "! world", "символ заменил выделенное");
    assert_eq!(app.cursor_pos, 1);
}

#[test]
fn a_click_in_the_prompt_does_not_disturb_a_transcript_selection() {
    let mut app = app_with("черновик");
    render(&app, 80, 24);
    let g = app.input_geom.get();
    assert_eq!(ClickAction::ShowImage(0).hint(), Some("Клик — открыть изображение на весь экран · Esc закроет"));

    app.begin_input_selection(g.left + g.prefix + 2, g.top, 1);
    assert!(app.mouse_sel.is_none(), "выделение чата не начинается в поле ввода");

    app.finish_input_selection();
    assert!(!app.input_selecting, "протяжка закончена");
}
