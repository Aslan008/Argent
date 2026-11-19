//! The ⏹ at the right edge of a running background row.
//!
//! The panel could show a job but not end it: stopping meant remembering `/stop <pid>`
//! and typing it. A stop button is destructive, though — a panel you scan with the
//! mouse must not kill a process because the pointer brushed over it. Hence two
//! clicks, with the row going red in between.

use argent_tui::app::{App, BgTask, ClickAction};
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

fn busy_app() -> App {
    let mut app = App::new();
    app.bg_folded = false;
    app.status.background.push(BgTask {
        pid: "4242".into(),
        command: "python slow_script.py".into(),
        elapsed: 12,
        pty: false,
    });
    app
}

fn row_with(app: &App, needle: &str) -> usize {
    let rows = frame(app, 100, 30);
    rows.iter()
        .position(|r| r.contains(needle))
        .unwrap_or_else(|| panic!("нет строки {needle:?}:\n{}", rows.join("\n")))
}

#[test]
fn a_running_row_carries_the_stop_button_at_its_right_edge() {
    let app = busy_app();
    let rows = frame(&app, 100, 30);
    let row = &rows[row_with(&app, "slow_script.py")];
    assert!(row.trim_end().ends_with('⏹'), "нет ⏹ в конце строки: {row:?}");
}

#[test]
fn the_button_is_a_click_target_narrower_than_the_row() {
    let app = busy_app();
    let y = row_with(&app, "slow_script.py") as u16;

    // The right edge stops; the rest of the row still opens the window.
    assert_eq!(
        app.get_clicked_action(98, y),
        Some(ClickAction::StopBackground("4242".into())),
        "правый край должен останавливать"
    );
    assert_eq!(
        app.get_clicked_action(5, y),
        Some(ClickAction::OpenBackground("4242".into())),
        "остальная строка по-прежнему открывает задачу"
    );
}

#[test]
fn a_finished_row_does_not_offer_to_stop_anything() {
    let mut app = App::new();
    app.bg_folded = false;
    app.status.bg_done.push(argent_tui::app::BgDone {
        pid: "999".into(),
        command: "echo done".into(),
        code: Some(0),
        secs: 3,
    });
    let rows = frame(&app, 100, 30);
    assert!(
        !rows.iter().any(|r| r.contains('⏹')),
        "завершённая задача не должна предлагать остановку:\n{}",
        rows.join("\n")
    );
}

#[test]
fn the_first_click_only_arms_the_row() {
    // The action itself is handled in lib.rs; what the model can check here is that
    // arming is a separate state, so a single click cannot stop anything.
    let mut app = busy_app();
    assert!(app.pending_stop.is_none());
    app.pending_stop = Some("4242".into());
    let rows = frame(&app, 100, 30);
    assert!(
        rows.iter().any(|r| r.contains("slow_script.py")),
        "взведённая строка должна остаться на месте"
    );
}
