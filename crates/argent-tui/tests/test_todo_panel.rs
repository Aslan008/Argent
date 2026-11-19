//! The agent's checklist: parsed from the bridge state, drawn between the chat and the prompt.
//!
//! Finished steps fold into one «✓ N готово» row (opened by a click) so the
//! screen goes to what is left; the step being worked on carries how long it has
//! been running, because that is what tells a stuck run from a moving one.

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
    TodoItem { content: content.into(), status: status.into(), since: None }
}

fn running(content: &str, seconds_ago: f64) -> TodoItem {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    TodoItem { content: content.into(), status: "in_progress".into(), since: Some(now - seconds_ago) }
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
    let todos = parse_todos(r#"[{"content":"read","status":"completed"},{"content":"edit","status":"in_progress","since":1700000000.5},{"content":"test"}]"#);
    assert_eq!(todos.len(), 3);
    assert!(todos[0].is_done() && todos[1].is_active() && !todos[2].is_done() && !todos[2].is_active());
    assert_eq!(todos[2].status, "pending");
    assert_eq!(todos[1].since, Some(1700000000.5));
    assert_eq!(todos[2].since, None, "у шага вне работы времени нет");
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
fn finished_steps_fold_into_one_row_and_the_work_left_is_listed() {
    let app = app_with(vec![
        item("read the code", "completed"),
        item("write the fix", "in_progress"),
        item("run tests", "pending"),
    ]);
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Задачи").expect("header");
    assert!(screen[header].contains("1/3"), "{}", screen[header]);

    let done_row = row_of(&screen, "готово").expect("свёрнутая строка выполненного");
    assert_eq!(done_row, header + 1);
    assert!(screen[done_row].contains("✓ 1 готово"), "{}", screen[done_row]);
    assert!(screen[done_row].contains("клик"), "видно, что строку можно открыть");
    assert!(
        row_of(&screen, "read the code").is_none(),
        "свёрнутое не занимает места:\n{}",
        screen.join("\n")
    );

    let active = row_of(&screen, "write the fix").expect("текущий шаг виден");
    let pending = row_of(&screen, "run tests").expect("следующий виден");
    assert!(screen[active].contains('▶') && screen[pending].contains('○'));
    assert_eq!((active, pending), (done_row + 1, done_row + 2));

    let chat = row_of(&screen, "берусь").unwrap();
    assert!(chat < header, "the checklist is below the conversation");
    let prompt = screen.iter().rposition(|r| r.contains('❯')).expect("prompt");
    assert!(prompt > pending, "and above the prompt:\n{}", screen.join("\n"));
}

#[test]
fn a_click_on_that_row_opens_the_finished_steps() {
    use argent_tui::app::ClickAction;

    let mut app = app_with(vec![
        item("read the code", "completed"),
        item("write the fix", "in_progress"),
    ]);
    let screen = render(&app, 80, 24);
    let done_row = row_of(&screen, "готово").unwrap();
    assert_eq!(app.get_clicked_action(3, done_row as u16), Some(ClickAction::ToggleTodosDone));

    app.todos_done_expanded = true;
    let open = render(&app, 80, 24);
    let step = row_of(&open, "read the code").expect("раскрытое видно");
    assert!(open[step].contains('✓'), "{}", open[step]);
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
fn a_long_list_keeps_the_current_step_in_view_and_says_what_is_hidden() {
    let mut items: Vec<TodoItem> =
        (1..=30).map(|i| item(&format!("step {i}"), if i <= 12 { "completed" } else { "pending" })).collect();
    items[12].status = "in_progress".into();
    let screen = render(&app_with(items), 80, 20);
    let text = screen.join("\n");
    assert!(row_of(&screen, "step 13").is_some(), "the active step is visible:\n{text}");
    assert!(row_of(&screen, "12/30").is_some(), "прогресс виден:\n{text}");
    let more = screen
        .iter()
        .position(|r| r.contains("ниже") || r.contains("выше"))
        .unwrap_or_else(|| panic!("скрытое названо числом:\n{text}"));
    assert!(screen[more].contains('…'), "{}", screen[more]);
    assert!(row_of(&screen, "step 30").is_none(), "хвост не влезает:\n{text}");
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
fn the_current_step_says_how_long_it_has_been_running() {
    let app = app_with(vec![running("write the fix", 125.0)]);
    let screen = render(&app, 80, 24);
    let active = row_of(&screen, "write the fix").expect("шаг виден");
    assert!(screen[active].contains("2м 05с"), "{}", screen[active]);
}

#[test]
fn a_step_that_hangs_is_flagged() {
    let app = app_with(vec![running("write the fix", 700.0)]);
    let screen = render(&app, 80, 24);
    let active = row_of(&screen, "write the fix").expect("шаг виден");
    assert!(
        screen[active].contains("долго"),
        "через 10+ минут это уже не «нормальная работа»:\n{}",
        screen[active]
    );
}

#[test]
fn a_fresh_step_is_not_flagged() {
    let app = app_with(vec![running("write the fix", 5.0)]);
    let screen = render(&app, 80, 24);
    let active = row_of(&screen, "write the fix").expect("шаг виден");
    assert!(!screen[active].contains("долго"), "{}", screen[active]);
    assert!(screen[active].contains("5с"), "{}", screen[active]);
}

#[test]
fn the_folded_header_also_carries_the_time() {
    let mut app = app_with(vec![running("write the fix", 125.0)]);
    app.todos_folded = true;
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Задачи").unwrap();
    assert!(
        screen[header].contains("write the fix") && screen[header].contains("2м 05с"),
        "{}",
        screen[header]
    );
}

#[test]
fn the_premium_look_colours_the_marks_and_strikes_out_what_is_done() {
    use argent_tui::theme::UiStyle;
    use ratatui::style::Modifier;

    let mut app = app_with(vec![
        item("read the code", "completed"),
        item("write the fix", "in_progress"),
        item("run tests", "pending"),
    ]);
    app.ui.set_style(UiStyle::Premium);
    app.todos_done_expanded = true;
    let mut terminal = Terminal::new(TestBackend::new(80, 24)).expect("terminal");
    terminal.draw(|f| draw_ui(f, &app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let p = app.ui.palette;

    let active = buf.content.iter().find(|c| c.symbol() == "▶").expect("active mark");
    assert_eq!(active.fg, p.accent);
    let done_at = buf.content.iter().position(|c| c.symbol() == "✓").expect("done mark");
    assert_eq!(buf.content[done_at].fg, p.ok);
    // «✓ 1 готово» — счётчик; зачёркнутый текст шага идёт ниже.
    let struck = buf
        .content
        .iter()
        .find(|c| c.symbol() == "r" && c.modifier.contains(Modifier::CROSSED_OUT));
    assert!(struck.is_some(), "finished steps are struck out");
    let pending = buf.content.iter().find(|c| c.symbol() == "○").expect("pending mark");
    assert_eq!(pending.fg, p.muted);
}

#[test]
fn the_header_is_a_button_that_folds_the_list_to_one_row() {
    use argent_tui::app::ClickAction;

    let mut app = app_with(vec![
        item("read the code", "completed"),
        item("write the fix", "in_progress"),
        item("run tests", "pending"),
    ]);
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
    assert!(row_of(&folded, "готово").is_none(), "свёрнутый список — одна строка");
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
