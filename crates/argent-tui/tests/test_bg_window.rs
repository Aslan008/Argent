//! The window on a background task: parsed from the bridge, scrolled, drawn, and
//! opened by a click on its row.
//!
//! The panel under the checklist holds one line per job — a build prints hundreds.
//! This window is the way to look inside a job that is still running, and it reads
//! the same output tail the agent reads: it copies it, never drains it.

use std::time::Instant;

use argent_tui::app::{parse_bg_view, App, BgTask, ClickAction, MessageItem};
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

fn row_of(screen: &[String], needle: &str) -> Option<usize> {
    screen.iter().position(|r| r.contains(needle))
}

fn payload(status: &str, output: &str) -> String {
    format!(
        r#"{{"pid":"42","status":"{status}","command":"npm run dev","elapsed":95,"code":null,"output":"{}","error":""}}"#,
        output.replace('\\', "\\\\").replace('\n', "\\n").replace('"', "\\\"")
    )
}

fn lines(count: usize) -> String {
    (0..count).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n")
}

fn app_with_task() -> App {
    let mut app = App::new();
    app.messages.push(MessageItem::User("запусти сборку".into()));
    app.status.background = vec![BgTask {
        pid: "42".into(),
        command: "npm run dev".into(),
        elapsed: 95,
        pty: false,
    }];
    app
}

// ── what the bridge sends ────────────────────────────────────────────────────

#[test]
fn parses_the_payload() {
    let view = parse_bg_view(&payload("running", "vite ready\nwatching")).expect("window");
    assert_eq!(view.pid, "42");
    assert_eq!(view.command, "npm run dev");
    assert_eq!(view.status, "running");
    assert_eq!(view.elapsed, 95);
    assert_eq!(view.code, None);
    assert_eq!(view.output, "vite ready\nwatching");
    assert!(view.running());
}

#[test]
fn a_broken_payload_is_no_window() {
    for bad in ["", "nope", "[]", "{"] {
        assert!(parse_bg_view(bad).is_none(), "{bad:?}");
    }
}

#[test]
fn a_finished_task_carries_its_exit_code() {
    let raw = r#"{"pid":"7","status":"done","command":"pytest","elapsed":41,"code":1,"output":"failed"}"#;
    let view = parse_bg_view(raw).expect("window");
    assert!(!view.running());
    assert_eq!(view.code, Some(1));
}

// ── what the window shows ────────────────────────────────────────────────────

#[test]
fn it_follows_the_newest_line_by_default() {
    let view = parse_bg_view(&payload("running", &lines(100))).expect("window");
    let rows = view.window(40, 10);
    assert_eq!(rows.len(), 10);
    assert!(rows[9].contains("line 99"), "{:?}", rows[9]);
}

#[test]
fn scrolling_back_reveals_earlier_lines() {
    let mut view = parse_bg_view(&payload("done", &lines(100))).expect("window");
    view.scroll = 5;
    let rows = view.window(40, 10);
    assert!(rows[9].contains("line 94"), "{:?}", rows[9]);
}

#[test]
fn it_never_scrolls_past_the_first_line() {
    let mut view = parse_bg_view(&payload("done", &lines(100))).expect("window");
    view.scroll = 10_000;
    let rows = view.window(40, 10);
    assert!(rows[0].contains("line 0"), "{:?}", rows[0]);
}

#[test]
fn long_lines_are_wrapped_not_lost() {
    let view = parse_bg_view(&payload("running", &"x".repeat(120))).expect("window");
    let rows = view.window(40, 10);
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.chars().count() <= 40));
}

#[test]
fn a_refresh_keeps_where_the_person_scrolled() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&payload("running", &lines(50)));
    app.bg_view.as_mut().unwrap().scroll = 7;
    app.apply_bg_view(&payload("running", &lines(60)));
    assert_eq!(app.bg_view.as_ref().unwrap().scroll, 7);
    assert!(app.bg_view.as_ref().unwrap().output.contains("line 59"));
}

#[test]
fn an_answer_for_another_task_does_not_replace_this_window() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&payload("running", "mine"));
    app.apply_bg_view(r#"{"pid":"99","status":"done","output":"someone else"}"#);
    assert_eq!(app.bg_view.as_ref().unwrap().output, "mine");
}

// ── how it opens and moves ───────────────────────────────────────────────────

#[test]
fn a_click_on_a_running_row_opens_its_window() {
    let mut app = app_with_task();
    app.bg_folded = false;
    let screen = render(&app, 90, 28);
    let row = row_of(&screen, "npm run dev").expect("строка задачи");
    assert_eq!(
        app.get_clicked_action(3, row as u16),
        Some(ClickAction::OpenBackground("42".into()))
    );

    app.open_bg_view("42");
    let view = app.bg_view.as_ref().expect("окно открыто");
    assert_eq!(view.pid, "42");
    assert_eq!(view.command, "npm run dev");
}

#[test]
fn opening_a_finished_row_keeps_its_code() {
    let mut app = app_with_task();
    app.status.bg_done = vec![argent_tui::app::BgDone {
        pid: "7".into(),
        command: "cargo build".into(),
        code: Some(1),
        secs: 12,
    }];
    app.open_bg_view("7");
    let view = app.bg_view.as_ref().unwrap();
    assert_eq!(view.code, Some(1));
    assert_eq!(view.status, "done");
}

#[test]
fn the_wheel_scrolls_the_window_not_the_chat() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&payload("running", &lines(200)));
    render(&app, 100, 30); // the frame reports the window's inner size
    app.set_mouse_pos(50, 15);

    app.wheel(true, Instant::now());

    assert_eq!(app.bg_view.as_ref().unwrap().scroll, 1);
    assert_eq!(app.scroll_offset, 0, "чат под окном не двигается");
}

#[test]
fn the_scroll_keys_move_the_window_a_page_at_a_time() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&payload("running", &lines(200)));
    render(&app, 100, 30);

    app.bg_view_page(true);
    let page = app.bg_view.as_ref().unwrap().scroll;
    assert!(page > 5, "страница больше одного ряда: {page}");

    app.bg_view_page(false);
    assert_eq!(app.bg_view.as_ref().unwrap().scroll, 0);

    app.close_bg_view();
    assert!(app.bg_view.is_none());
}

#[test]
fn the_wheel_over_the_chat_returns_the_prompt_to_its_caret() {
    let mut app = app_with_task();
    app.input = "много\nстрок\nв\nполе".into();
    app.input_scroll = 3;
    app.set_mouse_pos(40, 5); // над чатом, не над полем ввода

    app.wheel(false, Instant::now());

    assert_eq!(app.input_scroll, 0, "поле возвращается к каретке");
}

// ── how it looks ─────────────────────────────────────────────────────────────

#[test]
fn the_window_is_drawn_over_the_chat() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&payload("running", "vite ready\nwatching for changes"));
    let screen = render(&app, 100, 30);

    let title = row_of(&screen, "npm run dev").expect("заголовок с командой");
    assert!(screen[title].contains("42"), "{:?}", screen[title]);
    assert!(row_of(&screen, "выполняется").is_some(), "состояние");
    assert!(row_of(&screen, "vite ready").is_some(), "вывод процесса");
    assert!(row_of(&screen, "закрыть").is_some(), "подсказка, как выйти");
}

#[test]
fn a_silent_process_says_so_instead_of_an_empty_box() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&payload("running", ""));
    let screen = render(&app, 100, 30);
    assert!(row_of(&screen, "пока ничего не выведено").is_some(), "{screen:?}");
}

#[test]
fn a_task_that_vanished_is_reported_not_blank() {
    let mut app = app_with_task();
    app.open_bg_view("42");
    app.apply_bg_view(&format!(
        r#"{{"pid":"42","status":"unknown","command":"","elapsed":0,"code":null,"output":"","error":"задача 42 не найдена"}}"#
    ));
    let screen = render(&app, 100, 30);
    assert!(row_of(&screen, "не найдена").is_some(), "{screen:?}");
}
