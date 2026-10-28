//! Background commands: parsed from the bridge state and drawn under the checklist.

use std::collections::HashMap;

use argent_tui::app::{parse_background, parse_background_done, App, BgDone, BgTask, ClickAction, MessageItem};
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

fn task(pid: &str, command: &str, elapsed: u64) -> BgTask {
    BgTask { pid: pid.into(), command: command.into(), elapsed, pty: false }
}

fn done(pid: &str, command: &str, code: i64, secs: u64) -> BgDone {
    BgDone { pid: pid.into(), command: command.into(), code: Some(code), secs }
}

fn app_with(tasks: Vec<BgTask>) -> App {
    let mut app = App::new();
    app.messages.push(MessageItem::User("запусти сборку".into()));
    app.messages.push(MessageItem::assistant("запустил", ""));
    app.status.background = tasks;
    app
}

fn row_of(screen: &[String], needle: &str) -> Option<usize> {
    screen.iter().position(|r| r.contains(needle))
}

#[test]
fn parses_the_bridge_json() {
    let tasks = parse_background(
        r#"[{"pid":"42","command":"npm run dev","elapsed":95,"pty":false},
            {"pid":"43","command":"pytest","elapsed":7,"pty":true}]"#,
    );
    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].pid, "42");
    assert_eq!(tasks[0].elapsed, 95);
    assert!(tasks[1].pty);
}

#[test]
fn unreadable_payloads_give_an_empty_list() {
    for bad in ["", "[]", "not json", "{\"pid\":\"1\"}", "[1,2]", "[{\"pid\":\"1\"}]", "[{\"command\":\"x\"}]"] {
        assert!(parse_background(bad).is_empty(), "{bad:?}");
    }
}

#[test]
fn parses_the_finished_json() {
    let items = parse_background_done(
        r#"[{"pid":"5","command":"npm run build","code":0,"secs":131},
            {"pid":"6","command":"pytest","code":1,"secs":122},
            {"pid":"7","command":"probe","code":null,"secs":4}]"#,
    );
    assert_eq!(items.len(), 3);
    assert!(items[0].succeeded() && !items[1].succeeded());
    assert_eq!(items[1].code, Some(1));
    assert_eq!(items[2].code, None, "неизвестный код не выдаётся за ноль");
    for bad in ["", "[]", "nope", "{}", r#"[{"pid":"1"}]"#] {
        assert!(parse_background_done(bad).is_empty(), "{bad:?}");
    }
}

#[test]
fn the_state_refresh_carries_the_running_tasks() {
    let mut app = App::new();
    let mut state = HashMap::new();
    state.insert("background".to_string(), r#"[{"pid":"7","command":"npm run dev","elapsed":12,"pty":false}]"#.to_string());
    state.insert("background_done".to_string(), "3".to_string());
    state.insert(
        "background_finished".to_string(),
        r#"[{"pid":"5","command":"npm run build","code":0,"secs":131}]"#.to_string(),
    );
    app.update_status(state);
    assert_eq!(app.status.background, vec![task("7", "npm run dev", 12)]);
    assert_eq!(app.status.background_done, 3);
    assert_eq!(app.status.bg_done, vec![done("5", "npm run build", 0, 131)]);

    let mut cleared = HashMap::new();
    cleared.insert("background".to_string(), "[]".to_string());
    app.update_status(cleared);
    assert!(app.status.background.is_empty());
}

#[test]
fn nothing_is_drawn_while_nothing_runs() {
    let screen = render(&app_with(vec![]), 80, 24);
    assert!(row_of(&screen, "Фоновые").is_none());
    assert!(row_of(&screen, "Завершённые").is_none());
}

#[test]
fn folded_by_default_and_the_header_still_says_what_runs() {
    let app = app_with(vec![task("42", "npm run dev", 95), task("43", "pytest", 7)]);
    assert!(app.bg_folded, "свёрнута по умолчанию");
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Фоновые задачи").expect("заголовок");
    assert!(screen[header].contains('▸'), "свёрнутая: {}", screen[header]);
    assert!(screen[header].contains('2'), "счётчик: {}", screen[header]);
    assert!(screen[header].contains("npm run dev"), "что крутится: {}", screen[header]);
    assert!(row_of(&screen, "pytest").is_none(), "второго процесса в свёрнутой панели нет");
}

#[test]
fn expanded_rows_carry_the_pid_and_the_elapsed_time() {
    let mut app = app_with(vec![task("42", "npm run dev", 95), task("43", "pytest", 7)]);
    app.bg_folded = false;
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Фоновые задачи").unwrap();
    assert!(screen[header].contains('▾'), "развёрнутая: {}", screen[header]);
    let first = row_of(&screen, "npm run dev").unwrap();
    let second = row_of(&screen, "pytest").unwrap();
    assert_eq!((first, second), (header + 1, header + 2));
    assert!(screen[first].contains("pid 42") && screen[first].contains("1:35"), "{}", screen[first]);
    assert!(screen[second].contains("7s"), "{}", screen[second]);
    assert!(!screen[first].contains('▸'), "у свёрнутого заголовка своя шапка");
}

#[test]
fn the_finished_fold_is_offered_only_when_something_finished() {
    let mut app = app_with(vec![task("42", "npm run dev", 5)]);
    app.bg_folded = false;
    assert!(row_of(&render(&app, 80, 24), "Завершённые").is_none(), "нечего открывать");

    app.status.background_done = 3;
    app.status.bg_done = vec![done("5", "npm run build", 0, 131)];
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Завершённые").expect("вложенный заголовок");
    assert!(screen[header].contains('▸'), "свёрнут по умолчанию: {}", screen[header]);
    assert!(screen[header].contains('3'), "счётчик: {}", screen[header]);
    assert!(row_of(&screen, "npm run build").is_none(), "и строки не видны");
    assert_eq!(app.get_clicked_action(3, header as u16), Some(ClickAction::ToggleBackgroundDone));
}

#[test]
fn unfolding_the_finished_fold_shows_them_and_each_row_opens_its_output() {
    let mut app = app_with(vec![task("42", "npm run dev", 5)]);
    app.bg_folded = false;
    app.bg_done_folded = false;
    app.status.background_done = 2;
    app.status.bg_done = vec![done("5", "npm run build", 0, 131), done("6", "pytest", 1, 122)];

    let screen = render(&app, 80, 30);
    let header = row_of(&screen, "Завершённые").expect("заголовок");
    assert!(screen[header].contains('▾'), "развёрнут: {}", screen[header]);
    let ok_row = row_of(&screen, "npm run build").expect("успешная задача");
    let bad_row = row_of(&screen, "pytest").expect("упавшая задача");
    assert!(screen[ok_row].contains('✓') && screen[ok_row].contains("2:11"), "{}", screen[ok_row]);
    assert!(screen[bad_row].contains('✗') && screen[bad_row].contains("2:02"), "{}", screen[bad_row]);
    // Newest first: pytest finished after the build, so it sits above it.
    assert!(bad_row < ok_row, "свежие сверху:\n{}", screen.join("\n"));
    assert_eq!(
        app.get_clicked_action(3, ok_row as u16),
        Some(ClickAction::OpenBackground("5".into())),
        "клик по строке открывает окно этого процесса"
    );
}

#[test]
fn the_history_stays_reachable_after_the_last_process_exits() {
    let mut app = App::new();
    app.status.background = vec![];
    app.status.background_done = 1;
    app.status.bg_done = vec![done("9", "npm run build", 0, 30)];

    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Фоновые задачи").expect("панель не исчезает вместе с процессом");
    assert!(screen[header].contains('0'), "{}=", screen[header]);
    assert!(screen[header].contains("завершено 1"), "{}", screen[header]);
}

#[test]
fn the_header_folds_and_unfolds_the_panel() {
    let mut app = app_with(vec![task("42", "npm run dev", 5)]);
    app.bg_folded = false;
    let screen = render(&app, 80, 24);
    let header = row_of(&screen, "Фоновые задачи").unwrap();
    assert_eq!(app.get_clicked_action(3, header as u16), Some(ClickAction::ToggleBackground));

    app.bg_folded = true;
    let folded = render(&app, 80, 24);
    let header = row_of(&folded, "Фоновые задачи").unwrap();
    assert!(folded[header].contains('▸'));
    assert_eq!(app.get_clicked_action(3, header as u16), Some(ClickAction::ToggleBackground));
}

#[test]
fn a_question_takes_the_place_of_the_panel() {
    let mut app = app_with(vec![task("42", "npm run dev", 5)]);
    app.active_select = Some(argent_tui::app::SelectModalData::new(
        "sel".into(), "Выбор".into(), vec!["Да".into(), "Нет".into()], None));
    let screen = render(&app, 80, 24);
    assert!(row_of(&screen, "Фоновые задачи").is_none());
}

#[test]
fn a_long_list_counts_the_rest_and_stays_short() {
    let tasks: Vec<BgTask> = (1..=12).map(|i| task(&i.to_string(), &format!("job {i}"), i as u64)).collect();
    let mut app = app_with(tasks);
    app.bg_folded = false;
    app.status.background_done = 0;
    let screen = render(&app, 80, 40);
    let header = row_of(&screen, "Фоновые задачи").unwrap();
    let prompt = screen.iter().rposition(|r| r.contains('❯')).unwrap();
    assert!(prompt - header <= 8, "не больше восьми строк: {}", prompt - header);
    assert!(row_of(&screen, "… ещё").is_some());
}

#[test]
fn long_commands_are_clipped_to_the_width() {
    let app = app_with(vec![task("42", &"очень длинная команда ".repeat(20), 5)]);
    for row in render(&app, 40, 20) {
        assert!(row.chars().count() <= 40);
    }
}

#[test]
fn a_wrapped_draft_keeps_the_header_on_screen() {
    // Three draft lines in a ten-row window is an ordinary long message. The panel
    // used to vanish exactly then — while the person was writing.
    let mut app = app_with(vec![task("42", "ping -n 300 127.0.0.1", 9)]);
    app.input = "первая строка\nвторая строка\nтретья".into();
    let screen = render(&app, 80, 10);

    let header = row_of(&screen, "Фоновые задачи").expect("шапка панели осталась");
    assert!(screen.iter().any(|r| r.contains('❯')), "промпт тоже на месте");
    let chat = row_of(&screen, "запусти сборку").expect("чат не съеден панелью");
    assert!(chat < header, "панель ниже чата");
}

#[test]
fn it_gives_way_on_a_short_terminal_and_never_panics() {
    let mut app = app_with(vec![task("42", "npm run dev", 5), task("43", "pytest", 7)]);
    app.bg_folded = false;
    app.status.background_done = 2;
    for (w, h) in [(8, 4), (20, 6), (30, 8), (40, 10), (80, 12), (200, 60)] {
        let screen = render(&app, w, h);
        assert!(screen.iter().any(|r| r.contains('❯')) || h < 6, "промпт живёт при {w}x{h}");
    }
}
