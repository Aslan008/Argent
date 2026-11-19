//! The bottom toolbar is the one row that must survive both an approval modal
//! and a tall stack of tool cards. A regression here is invisible to a smoke
//! test that only asserts `draw_ui` returned Ok.

use std::collections::HashMap;

use argent_tui::app::App;
use argent_tui::events::{ApprovalPrompt, ArgentEvent};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let area = buf.area;
    let mut out = String::new();
    for y in 0..area.height {
        for x in 0..area.width {
            out.push_str(buf.content[(y * area.width + x) as usize].symbol());
        }
        out.push('\n');
    }
    out
}

fn add_tool(app: &mut App) {
    let mut args = HashMap::new();
    args.insert("command".to_string(), serde_json::json!("python -c 'print(1)'"));
    app.handle_event(ArgentEvent::ToolStart {
        name: "run_command".to_string(),
        args,
    });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".to_string(),
        result: "Exit code: 0\nOUTPUT:\n1".to_string(),
        duration_secs: Some(1.0),
    });
}

fn approval(app: &mut App, destructive: bool) {
    app.handle_event(ArgentEvent::ApprovalRequest {
        id: 7,
        action: "run_command: python -c 'print(1)'".to_string(),
        destructive,
        grant_key: Some("run_command".to_string()),
        prompt: Some(ApprovalPrompt {
            question: "Разрешить выполнение команды?".to_string(),
            choices: vec!["Да".to_string(), "Всегда".to_string(), "Нет".to_string()],
            decisions: vec!["once".to_string(), "always".to_string(), "deny".to_string()],
            confirm: false,
        }),
    });
}

fn last_row(frame: &str) -> &str {
    frame.lines().last().unwrap_or("")
}

#[test]
fn approval_modal_never_hides_the_toolbar() {
    for h in [10u16, 12, 16, 24, 40] {
        let mut app = App::new();
        add_tool(&mut app);
        approval(&mut app, false);
        let frame = render(&app, 120, h);
        let last = last_row(&frame);
        assert!(
            last.contains("CHAT") || last.contains("/help"),
            "toolbar missing at height {h}; last row was {last:?}\n--- frame ---\n{frame}"
        );
    }
}

#[test]
fn many_tool_cards_never_hide_the_toolbar() {
    let mut app = App::new();
    for _ in 0..12 {
        add_tool(&mut app);
    }
    approval(&mut app, true);
    let frame = render(&app, 100, 24);
    let last = last_row(&frame);
    assert!(
        last.contains("CHAT") || last.contains("/help"),
        "toolbar missing after tool cards; last row was {last:?}\n--- frame ---\n{frame}"
    );
}

#[test]
fn toolbar_survives_after_approval_is_dismissed() {
    let mut app = App::new();
    add_tool(&mut app);
    approval(&mut app, false);
    app.active_approval = None;
    let frame = render(&app, 120, 20);
    let last = last_row(&frame);
    assert!(
        last.contains("CHAT") || last.contains("/help"),
        "toolbar missing after approval; last row was {last:?}\n--- frame ---\n{frame}"
    );
}
