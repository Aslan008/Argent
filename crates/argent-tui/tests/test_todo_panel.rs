//! The agent's checklist: parsed from the bridge state, drawn between the chat and the prompt.

use std::collections::HashMap;

use argent_tui::app::{parse_todos, App, MessageItem, TodoItem};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| (0..buf.area.width).map(|x| buf.content[(y * buf.area.width + x) as usize].symbol()).collect::<String>())
        .collect()
}

fn item(content: &str, status: &str) -> TodoItem {
    TodoItem { content: content.into(), status: status.into() }
}

fn app_with(items: Vec<TodoItem>) -> App {
    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай рефакторинг".into()));
    app.messages.push(MessageItem::assistant("берусь", ""));
    app.status.todos = items;
    app
}

fn row_of(screen: &[String], needle: &str) -> Option<usize> {
    screen.iter().position(|r| r.contains(needle))
}

#[test]
fn parses_the_bridge_json() {
    let todos = parse_todos(r#"[{"content":"read","status":"completed"},{"content":"edit","status":"in_progress"},{"content":"test"}]"#);
    assert_eq!(todos.len(), 3);
    assert!(todos[0].is_done() && todos[1].is_active() && !todos[2].is_done() && !todos[2].is_active());
    assert_eq!(todos[2].status, "pending");
}

#[test]
fn unreadable_payloads_give_an_empty_list() {
    for bad in ["", "[]", "not json", "{\"a\":1}", "[1,2]", "[{\"status\":\"pending\"}]", "[{\"content\":\"  \"}]"] {
        assert!(parse_todos(bad).is_empty(), "{bad:?}");
    }
}

#[test]
fn the_state_refresh_carries_the_list_and_clears_it() {
    let mut app = App::new();
    let mut state = HashMap::new();
    state.insert("todos".to_string(), r#"[{"content":"one","status":"in_progress"}]"#.to_string());
    app.update_status(state);
    assert_eq!(app.status.todos, vec![item("one", "in_progress")]);
    let mut cleared = HashMap::new();
    cleared.insert("todos".to_string(), "[]".to_string());
    app.update_status(cleared);
    assert!(app.status.todos.is_empty());
}

#[test]
fn draws_the_steps_with_their_marks_between_the_chat_and_the_prompt() {
    let app = app_with(vec![item("read the code", "completed"), item("write the fix", "in_progress"), item("run tests", "pending")]);
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Задачи").expect("header");
    assert!(screen[header].contains("1/3"), "{}", screen[header]);
    let done = row_of(&screen, "read the code").unwrap();
    let active = row_of(&screen, "write the fix").unwrap();
    let pending = row_of(&screen, "run tests").unwrap();
    assert!(screen[done].contains('✓') && screen[active].contains('▶') && screen[pending].contains('○'));
    assert_eq!((done, active, pending), (header + 1, header + 2, header + 3));
    let chat = row_of(&screen, "берусь").unwrap();
    assert!(chat < header, "the checklist is below the conversation");
    let prompt = screen.iter().rposition(|r| r.contains('❯')).expect("prompt");     // your message also starts with ❯
    assert!(prompt > pending, "and above the prompt:\n{}", screen.join("\n"));
}

#[test]
fn nothing_is_drawn_without_a_list() {
    let screen = render(&app_with(vec![]), 80, 24);
    assert!(row_of(&screen, "Задачи").is_none());
}

#[test]
fn a_question_takes_the_place_of_the_checklist() {
    let mut app = app_with(vec![item("a step", "in_progress")]);
    app.active_select = Some(argent_tui::app::SelectModalData::new(
        "sel".into(), "Выбор".into(), vec!["Да".into(), "Нет".into()], None));
    let screen = render(&app, 80, 24);
    assert!(row_of(&screen, "a step").is_none());
}

#[test]
fn a_long_list_keeps_the_current_step_in_view_and_counts_the_rest() {
    let mut items: Vec<TodoItem> = (1..=20).map(|i| item(&format!("step {i}"), if i <= 12 { "completed" } else { "pending" })).collect();
    items[12].status = "in_progress".into();
    let screen = render(&app_with(items), 80, 30);
    assert!(row_of(&screen, "step 13").is_some(), "the active step is visible:\n{}", screen.join("\n"));
    assert!(row_of(&screen, "step 1 ").is_none() || row_of(&screen, "step 20").is_none());
    let more = row_of(&screen, "ещё").expect("a 'more' row");
    assert!(screen[more].contains("… ещё"), "{}", screen[more]);
    assert!(row_of(&screen, "20/").is_none() && row_of(&screen, "12/20").is_some());
}

#[test]
fn it_gives_way_on_a_short_terminal_and_never_panics() {
    let app = app_with((1..=6).map(|i| item(&format!("step {i}"), "pending")).collect());
    for (w, h) in [(8, 4), (20, 6), (30, 8), (40, 10), (80, 12), (200, 60)] {
        let screen = render(&app, w, h);
        assert!(screen.iter().any(|r| r.contains('❯')) || h < 6, "the prompt survives at {w}x{h}");
    }
}

#[test]
fn long_steps_are_clipped_to_the_width() {
    let app = app_with(vec![item(&"очень длинный шаг ".repeat(20), "in_progress")]);
    for row in render(&app, 40, 20) {
        assert!(row.chars().count() <= 40);
    }
}

#[test]
fn the_premium_look_colours_the_marks_and_strikes_out_what_is_done() {
    use argent_tui::theme::UiStyle;
    use ratatui::style::Modifier;

    let mut app = app_with(vec![item("read the code", "completed"), item("write the fix", "in_progress"), item("run tests", "pending")]);
    app.ui.set_style(UiStyle::Premium);
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
    terminal.draw(|f| draw_ui(f, &app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let p = app.ui.palette;

    let active = buf.content.iter().find(|c| c.symbol() == "▶").expect("active mark");
    assert_eq!(active.fg, p.accent);
    let done_at = buf.content.iter().position(|c| c.symbol() == "✓").expect("done mark");
    assert_eq!(buf.content[done_at].fg, p.ok);
    let first_letter = &buf.content[done_at + 2];
    assert_eq!(first_letter.symbol(), "r");
    assert!(first_letter.modifier.contains(Modifier::CROSSED_OUT), "finished steps are struck out");
    let pending = buf.content.iter().find(|c| c.symbol() == "○").expect("pending mark");
    assert_eq!(pending.fg, p.muted);
}

#[test]
fn the_header_is_a_button_that_folds_the_list_to_one_row() {
    use argent_tui::app::ClickAction;

    let mut app = app_with(vec![item("read the code", "completed"), item("write the fix", "in_progress"), item("run tests", "pending")]);
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Задачи").unwrap();
    assert!(screen[header].contains('▾'), "open: {}", screen[header]);
    assert_eq!(app.get_clicked_action(3, header as u16), Some(ClickAction::ToggleTodos));

    app.todos_folded = true;
    let folded = render(&app, 80, 24);
    let header = row_of(&folded, "Задачи").unwrap();
    assert!(folded[header].contains('▸') && folded[header].contains("1/3"), "{}", folded[header]);
    assert!(folded[header].contains("write the fix"), "folded, it still says what is going on: {}", folded[header]);
    assert!(row_of(&folded, "run tests").is_none() && row_of(&folded, "read the code").is_none());
    assert_eq!(app.get_clicked_action(3, header as u16), Some(ClickAction::ToggleTodos), "and it unfolds again");
}

#[test]
fn the_list_is_a_glance_not_a_screen() {
    let items: Vec<TodoItem> = (1..=30).map(|i| item(&format!("step {i}"), "pending")).collect();
    let screen = render(&app_with(items), 80, 50);
    let header = row_of(&screen, "Задачи").unwrap();
    let prompt = screen.iter().rposition(|r| r.contains('❯')).unwrap();
    assert!(prompt - header <= 8, "at most the header, five steps and the 'more' row: {}", prompt - header);
}
