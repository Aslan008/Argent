//! "Back to here": the button under the last message the person sent, and what the
//! `rewound` event does to the transcript and the prompt.

use argent_tui::app::{App, ClickAction, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

const LABEL: &str = "Вернуться сюда";

fn render(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| (0..buf.area.width).map(|x| buf.content[(y * buf.area.width + x) as usize].symbol()).collect::<String>())
        .collect()
}

fn chat() -> App {
    let mut app = App::new();
    app.status.can_rewind = true;
    app.messages.push(MessageItem::User("первое сообщение".into()));
    app.messages.push(MessageItem::assistant("первый ответ", ""));
    app.messages.push(MessageItem::User("второе сообщение".into()));
    app.messages.push(MessageItem::assistant("второй ответ", ""));
    app
}

fn button_rows(app: &App) -> Vec<usize> {
    render(app, 80, 24).iter().enumerate().filter(|(_, r)| r.contains(LABEL)).map(|(i, _)| i).collect()
}

#[test]
fn the_button_sits_under_the_last_message_sent_and_nowhere_else() {
    let app = chat();
    let screen = render(&app, 80, 24);
    let rows = button_rows(&app);
    assert_eq!(rows.len(), 1, "exactly one button:\n{}", screen.join("\n"));
    assert!(screen[rows[0] - 1].contains("второе сообщение"), "directly under the last user message");
    assert!(!screen.iter().any(|r| r.contains("первое сообщение") && r.contains(LABEL)));
}

#[test]
fn clicking_the_button_asks_for_the_rewind_action() {
    let app = chat();
    let row = button_rows(&app)[0] as u16;
    assert_eq!(app.get_clicked_action(3, row), Some(ClickAction::RewindLast));
    assert_ne!(app.get_clicked_action(3, row.saturating_sub(1)), Some(ClickAction::RewindLast));
}

#[test]
fn no_button_while_a_turn_runs_a_question_is_open_or_there_is_nothing_to_go_back_to() {
    let mut busy = chat();
    busy.is_busy = true;
    assert!(button_rows(&busy).is_empty());

    let mut command = chat();
    command.is_command_in_flight = true;
    assert!(button_rows(&command).is_empty());

    let mut nothing = chat();
    nothing.status.can_rewind = false;
    assert!(button_rows(&nothing).is_empty());
}

#[test]
fn a_slash_command_is_not_a_message_to_go_back_to() {
    let mut app = chat();
    app.messages.push(MessageItem::User("/model".into()));
    app.messages.push(MessageItem::User("!dir".into()));
    let screen = render(&app, 80, 24);
    let rows = button_rows(&app);
    assert_eq!(rows.len(), 1);
    assert!(screen[rows[0] - 1].contains("второе сообщение"), "the button stays under the last message for the agent");
}

#[test]
fn the_button_row_is_interface_not_text_a_selection_would_copy() {
    let mut app = chat();
    let _ = render(&app, 80, 24);
    // select from the start of the last user message to the end of the transcript
    assert!(app.select_all_transcript());
    let copied = app.selected_transcript_text().unwrap_or_default();
    assert!(copied.contains("второе сообщение"));
    assert!(!copied.contains(LABEL), "the button label must not end up on the clipboard: {copied:?}");
}

#[test]
fn the_rewound_event_drops_the_message_and_what_followed_and_refills_the_prompt() {
    let mut app = chat();
    app.messages.push(MessageItem::Notice("шум после ответа".into()));
    app.handle_event(ArgentEvent::Rewound { text: "второе сообщение".into(), detail: "Файлы вернулись.".into() });

    assert_eq!(app.input, "второе сообщение");
    assert_eq!(app.cursor_pos, "второе сообщение".chars().count());
    assert!(matches!(&app.messages[..], [
        MessageItem::User(a), MessageItem::Assistant { .. }, MessageItem::Notice(n)
    ] if a == "первое сообщение" && n.contains("Файлы вернулись.")));
    assert!(!app.status.can_rewind, "until the bridge says there is another step to go back to");
    assert!(button_rows(&app).is_empty());
}

#[test]
fn a_draft_in_the_prompt_is_kept_after_the_restored_text() {
    let mut app = chat();
    app.input = "черновик".into();
    app.cursor_pos = 8;
    app.handle_event(ArgentEvent::Rewound { text: "второе сообщение".into(), detail: String::new() });
    assert_eq!(app.input, "второе сообщение\nчерновик");
    assert_eq!(app.cursor_pos, "второе сообщение".chars().count());
}

#[test]
fn rewinding_twice_walks_back_one_message_each_time() {
    let mut app = chat();
    app.handle_event(ArgentEvent::Rewound { text: "второе сообщение".into(), detail: String::new() });
    app.input.clear();
    app.status.can_rewind = true; // the bridge's next state refresh
    assert_eq!(button_rows(&app).len(), 1);
    app.handle_event(ArgentEvent::Rewound { text: "первое сообщение".into(), detail: String::new() });
    assert!(!app.messages.iter().any(|m| matches!(m, MessageItem::User(_))));
    assert_eq!(app.input, "первое сообщение");
}

#[test]
fn a_rewind_event_for_text_that_is_not_in_the_transcript_still_cuts_at_the_last_message() {
    let mut app = chat();
    app.handle_event(ArgentEvent::Rewound { text: "другой текст".into(), detail: String::new() });
    assert!(matches!(&app.messages[..], [MessageItem::User(_), MessageItem::Assistant { .. }, MessageItem::Notice(_)]));
}

#[test]
fn it_survives_tiny_terminals() {
    let app = chat();
    for (w, h) in [(8, 4), (20, 6), (30, 10), (200, 60)] {
        let _ = render(&app, w, h);
    }
}
