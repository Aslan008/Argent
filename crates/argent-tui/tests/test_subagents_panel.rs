//! Субагенты: список над вводом и отдельное окно на одного.
//!
//! Раньше каждая строка воркера печаталась в чат (`_stream_worker` → `print_system`),
//! и рой из четырёх агентов заливал разговор. Теперь строки живут в своём окне:
//! список отвечает «кто есть», клик по агенту — «что он делает».

use std::collections::HashMap;

use argent_tui::app::{parse_subagents, App, ClickAction, SubRun, SubView};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn frame(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let area = buf.area;
    let mut out = Vec::new();
    for y in 0..area.height {
        let mut row = String::new();
        for x in 0..area.width {
            row.push_str(buf.content[(y * area.width + x) as usize].symbol());
        }
        out.push(row);
    }
    out
}

fn payload() -> String {
    serde_json::json!([
        {"id": 3, "role": "coder", "task": "чинит парсер", "status": "running",
         "elapsed": 130.0, "model": "qwen", "files": 0, "last": "пишет src/app.rs", "patch": false},
        {"id": 2, "role": "reviewer", "task": "смотрит тесты", "status": "done",
         "elapsed": 64.0, "model": "glm", "files": 3, "last": "готово", "patch": true},
        {"id": 1, "role": "critic", "task": "ищет дыры", "status": "failed",
         "elapsed": 9.0, "model": "glm", "files": 0, "last": "", "patch": false}
    ])
    .to_string()
}

fn app_with_agents() -> App {
    let mut app = App::new();
    app.update_status(HashMap::from([("subagents".to_string(), payload())]));
    app
}

#[test]
fn the_payload_is_read_and_garbage_is_ignored() {
    let runs = parse_subagents(&payload());
    assert_eq!(runs.len(), 3);
    assert_eq!(runs[0].id, 3);
    assert_eq!(runs[0].role, "coder");
    assert!(runs[0].running());
    assert_eq!(runs[0].elapsed, 130);
    assert_eq!(runs[1].files, 3);
    assert!(runs[1].patch);

    assert!(parse_subagents("").is_empty());
    assert!(parse_subagents("не json").is_empty());
    assert!(parse_subagents(r#"[{"role": "coder"}]"#).is_empty(), "без id агента нет");
}

#[test]
fn the_list_is_one_row_until_it_is_opened() {
    let app = app_with_agents();
    let rows = frame(&app, 100, 30);
    let head = rows
        .iter()
        .position(|r| r.contains("Субагенты"))
        .unwrap_or_else(|| panic!("нет панели субагентов:\n{}", rows.join("\n")));

    assert!(rows[head].contains("1 работают"), "{}", rows[head]);
    assert!(rows[head].contains("2 готово"), "{}", rows[head]);
    // The folded row still names what is going on: the running worker and its line.
    assert!(rows[head].contains("coder #3"), "{}", rows[head]);
    assert!(rows[head].contains("пишет src/app.rs"), "{}", rows[head]);
    // The finished ones stay out of sight.
    assert!(
        !rows.iter().any(|r| r.contains("reviewer #2")),
        "свёрнутый список не должен перечислять всех:\n{}",
        rows.join("\n")
    );
}

#[test]
fn opened_it_lists_every_agent_and_a_row_opens_one() {
    let mut app = app_with_agents();
    app.sub_folded = false;
    let rows = frame(&app, 100, 30);
    let head = rows
        .iter()
        .position(|r| r.contains("Субагенты"))
        .unwrap_or_else(|| panic!("нет панели:\n{}", rows.join("\n")));

    for needle in ["coder #3", "reviewer #2", "critic #1"] {
        assert!(
            rows.iter().any(|r| r.contains(needle)),
            "в раскрытом списке нет {needle}:\n{}",
            rows.join("\n")
        );
    }
    assert!(rows[head].contains('▾'), "раскрытая панель помечена: {}", rows[head]);

    // Клик по строке агента открывает его окно (id — в действии).
    let row = rows.iter().position(|r| r.contains("coder #3")).expect("строка coder");
    assert_eq!(
        app.get_clicked_action(5, row as u16),
        Some(ClickAction::OpenSubagent(3))
    );
    // Заголовок сворачивает список.
    assert_eq!(
        app.get_clicked_action(5, head as u16),
        Some(ClickAction::ToggleSubagents)
    );
}

#[test]
fn only_a_running_agent_has_a_stop_button() {
    let mut app = app_with_agents();
    app.sub_folded = false;
    let rows = frame(&app, 100, 30);
    let running = rows.iter().position(|r| r.contains("coder #3")).expect("coder");
    let done = rows.iter().position(|r| r.contains("reviewer #2")).expect("reviewer");

    assert_eq!(
        app.get_clicked_action(99, running as u16),
        Some(ClickAction::StopSubagent(3))
    );
    // On a finished row the far right is still the row itself, not a stop button.
    assert_eq!(
        app.get_clicked_action(99, done as u16),
        Some(ClickAction::OpenSubagent(2))
    );
}

#[test]
fn the_window_shows_the_workers_own_dialogue() {
    let mut app = app_with_agents();
    app.open_sub_view(3);
    app.apply_sub_view(
        &serde_json::json!({
            "id": 3, "role": "coder", "task": "чинит парсер", "status": "running",
            "elapsed": 130.0, "model": "qwen", "isolation": "worktree",
            "files": ["src/parser.py"], "patch": "/tmp/3-coder.patch",
            "report": "", "error": "",
            "lines": ["читаю src/parser.py", "запускаю pytest", "3 теста падают"]
        })
        .to_string(),
    );

    let rows = frame(&app, 120, 40);
    let text = rows.join("\n");
    assert!(text.contains("coder #3"), "заголовок окна:\n{text}");
    assert!(text.contains("работает"), "статус в заголовке:\n{text}");
    assert!(text.contains("Задача: чинит парсер"), "задача:\n{text}");
    assert!(text.contains("запускаю pytest"), "диалог воркера:\n{text}");
    assert!(text.contains("Esc — закрыть"), "путь назад:\n{text}");
}

#[test]
fn the_window_can_be_closed_and_does_not_take_another_agents_answer() {
    let mut app = app_with_agents();
    app.open_sub_view(3);
    // An answer for a different agent is not for this window.
    app.apply_sub_view(r#"{"id": 2, "role": "reviewer", "lines": ["чужой лог"]}"#);
    assert_eq!(app.sub_view.as_ref().map(|v| v.role.clone()), Some("coder".to_string()));

    app.apply_sub_view(r#"{"id": 3, "role": "coder", "status": "done", "lines": ["готово"]}"#);
    assert!(app.sub_view.as_ref().is_some_and(|v| !v.running()));

    app.close_sub_view();
    let rows = frame(&app, 120, 40);
    assert!(
        !rows.iter().any(|r| r.contains("Esc — закрыть")),
        "окно закрыто:\n{}",
        rows.join("\n")
    );
}

#[test]
fn the_window_keeps_the_scroll_position_on_a_refresh() {
    let mut view = SubView { id: 3, role: "coder".into(), ..Default::default() };
    view.lines = (0..100).map(|i| format!("строка {i}")).collect();
    assert_eq!(view.scroll, 0);

    // A refresh must not throw the reader back to the tail.
    let mut app = app_with_agents();
    app.open_sub_view(3);
    if let Some(v) = app.sub_view.as_mut() {
        v.scroll = 7;
    }
    app.apply_sub_view(&serde_json::json!({"id": 3, "role": "coder", "lines": ["x"]}).to_string());
    assert_eq!(app.sub_view.as_ref().map(|v| v.scroll), Some(7));
}

#[test]
fn a_run_without_lines_still_renders() {
    // A fresh worker has printed nothing yet; the window is not an error.
    let run = SubRun {
        id: 9,
        role: "critic".into(),
        task: String::new(),
        status: "running".into(),
        elapsed: 0,
        model: String::new(),
        files: 0,
        last: String::new(),
        patch: false,
    };
    assert_eq!(run.mark(), "●");
    assert_eq!(run.label(), "critic #9");
}
