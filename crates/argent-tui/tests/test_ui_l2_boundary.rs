use std::collections::HashMap;
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use argent_tui::app::{App, ApprovalData, CompletionItem, MessageItem};
use argent_tui::ui::draw_ui;

#[test]
fn test_ui_boundary_zero_and_tiny_dimensions() {
    let app = App::new();

    // 0x0 terminal
    let backend = TestBackend::new(0, 0);
    let mut terminal = Terminal::new(backend).unwrap();
    let res = terminal.draw(|f| draw_ui(f, &app));
    assert!(res.is_ok(), "draw_ui must not panic on 0x0 terminal");

    // 1x1 terminal
    let backend = TestBackend::new(1, 1);
    let mut terminal = Terminal::new(backend).unwrap();
    let res = terminal.draw(|f| draw_ui(f, &app));
    assert!(res.is_ok(), "draw_ui must not panic on 1x1 terminal");

    // 3x2 terminal
    let backend = TestBackend::new(3, 2);
    let mut terminal = Terminal::new(backend).unwrap();
    let res = terminal.draw(|f| draw_ui(f, &app));
    assert!(res.is_ok(), "draw_ui must not panic on 3x2 terminal");
}

#[test]
fn test_ui_render_all_message_types() {
    let mut app = App::new();

    // Fill with all types of messages
    app.messages.push(MessageItem::User("User question".to_string()));
    app.messages.push(MessageItem::Assistant {
        content: "Assistant response".to_string(),
        thinking: "Pondering the universe".to_string(),
        thinking_expanded: true,
        is_streaming_thinking: false,
        is_streaming_content: false,
        thinking_duration: Some(1.2),
        thinking_start: None,
        stream: Default::default(),
    });
    app.messages.push(MessageItem::Tool {
        name: "calculator".to_string(),
        args: HashMap::new(),
        result: Some("Result: 42".to_string()),
        duration_secs: Some(0.05),
        expanded: true,
        is_running: false,
    });
    app.messages.push(MessageItem::Tool {
        name: "failing_tool".to_string(),
        args: HashMap::new(),
        result: Some("Error: Connection refused".to_string()),
        duration_secs: Some(0.1),
        expanded: false,
        is_running: false,
    });
    app.messages.push(MessageItem::Diff {
        file: "src/main.rs".to_string(),
        diff: "+ fn new() {}\n- fn old() {}".to_string(),
    });
    app.messages.push(MessageItem::Notice("Info notification".to_string()));
    app.messages.push(MessageItem::Error { text: "Fatal error encountered".to_string(), hint: None });
    app.messages.push(MessageItem::Checkpoint {
        sha: "a1b2c3d".to_string(),
        label: "Saved checkpoint".to_string(),
    });

    let backend = TestBackend::new(100, 40);
    let mut terminal = Terminal::new(backend).unwrap();
    let res = terminal.draw(|f| draw_ui(f, &app));
    assert!(res.is_ok(), "draw_ui must successfully render all message types");
}

#[test]
fn test_ui_render_modals_and_completions() {
    let mut app = App::new();

    // 1. Approval modal
    app.active_approval = Some(ApprovalData {
        id: 1,
        action: "rm -rf /".to_string(),
        destructive: true,
        grant_key: None,
        prompt: None,
        selected: 0,
    });
    let backend = TestBackend::new(80, 24);
    let mut terminal = Terminal::new(backend).unwrap();
    assert!(terminal.draw(|f| draw_ui(f, &app)).is_ok());

    // 2. Command completions popup
    app.active_approval = None;
    app.show_completions = true;
    app.input = "/he".to_string();
    app.cursor_pos = 3;
    app.completions.push(CompletionItem {
        name: "/help".to_string(),
        help: "Show help commands".to_string(),
    });
    assert!(terminal.draw(|f| draw_ui(f, &app)).is_ok());

    // 3. File completions popup
    app.show_completions = false;
    app.show_file_completions = true;
    app.input = "@src".to_string();
    app.cursor_pos = 4;
    app.file_completions.push("src/main.rs".to_string());
    assert!(terminal.draw(|f| draw_ui(f, &app)).is_ok());
}
