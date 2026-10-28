//! Условия у сообщений в очереди, прокрутка панелей и списки подагентов.
//!
//! Очередь раньше знала только «ждать конца хода». Теперь сообщение может ждать
//! ошибки, успеха, таймера или зависшего шага — и не выстреливать, когда толку
//! от него всё равно не будет (пустой счёт у провайдера).

use std::time::{Duration, Instant};

use argent_tui::app::{
    is_balance_error, now_secs, parse_delay, parse_sub_todos, App, MessageItem, QueueCondition,
    QueueTrigger, QueuedMessage, SubTodos, Submit, TodoItem,
};
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

fn waiting(app: &mut App, text: &str, condition: QueueCondition) {
    app.message_queue.push(QueuedMessage {
        text: text.into(),
        condition: Some(condition),
    });
}

fn todo(content: &str, status: &str, since: Option<f64>) -> TodoItem {
    TodoItem {
        content: content.into(),
        status: status.into(),
        since,
    }
}

// ── условия ──────────────────────────────────────────────────────────────────

#[test]
fn a_message_waiting_for_an_error_does_not_go_out_with_the_next_turn() {
    let mut app = App::new();
    waiting(&mut app, "recover", QueueCondition::OnError { filter: None });
    app.message_queue.push(QueuedMessage::plain("ordinary"));

    // The ordinary one is next; the waiting one keeps its place behind it.
    assert_eq!(app.pop_next_queued_message().as_deref(), Some("ordinary"));
    assert_eq!(app.message_queue.len(), 1);
    assert!(!app.message_queue[0].is_ready());
}

#[test]
fn nothing_goes_out_when_only_waiting_messages_are_queued() {
    let mut app = App::new();
    waiting(&mut app, "recover", QueueCondition::OnError { filter: None });

    assert_eq!(app.pop_next_queued_message(), None);
    assert_eq!(app.message_queue.len(), 1, "the message stays for its trigger");
}

#[test]
fn an_error_releases_the_message_waiting_for_one() {
    let mut app = App::new();
    waiting(&mut app, "recover", QueueCondition::OnError { filter: None });

    assert_eq!(app.fire_error_conditions("Connection reset by peer"), 1);
    assert!(app.message_queue[0].is_ready());
}

#[test]
fn a_filtered_wait_ignores_the_wrong_failure() {
    let mut app = App::new();
    waiting(&mut app, "retry", QueueCondition::OnError { filter: Some("429".into()) });

    assert_eq!(app.fire_error_conditions("connection reset"), 0);
    assert!(!app.message_queue[0].is_ready());
    assert_eq!(app.fire_error_conditions("HTTP 429 Too Many Requests"), 1);
}

#[test]
fn an_empty_balance_never_fires_the_error_message() {
    let mut app = App::new();
    waiting(&mut app, "recover", QueueCondition::OnError { filter: None });
    let text = "Error code: 402 - {\"error\":{\"message\":\"Insufficient Balance\"}}";

    assert_eq!(app.fire_error_conditions(text), 0);
    assert!(!app.message_queue[0].is_ready(), "the message keeps waiting");

    // Said once, not once per failed turn.
    let notices = |app: &App| {
        app.messages
            .iter()
            .filter(|m| matches!(m, MessageItem::Notice(t) if t.contains("баланс")))
            .count()
    };
    assert_eq!(notices(&app), 1);
    app.fire_error_conditions(text);
    assert_eq!(notices(&app), 1);
}

#[test]
fn the_balance_check_knows_the_words_providers_use() {
    assert!(is_balance_error("Insufficient Balance"));
    assert!(is_balance_error("You exceeded your current quota"));
    assert!(is_balance_error("Error code: 402"));
    assert!(is_balance_error("Недостаточно средств на счёте"));
    assert!(!is_balance_error("Connection reset by peer"));
    assert!(!is_balance_error("tool 'write_file' raised ValueError"));
}

#[test]
fn success_releases_only_the_success_message() {
    let mut app = App::new();
    waiting(&mut app, "summarise", QueueCondition::OnSuccess);
    waiting(&mut app, "recover", QueueCondition::OnError { filter: None });

    assert_eq!(app.release_queued(QueueTrigger::Error, "boom"), 1);
    assert_eq!(app.release_queued(QueueTrigger::Success, ""), 1);
    assert!(app.message_queue.iter().all(|m| m.is_ready()));
}

#[test]
fn a_timer_releases_its_message_when_it_is_due() {
    let mut app = App::new();
    waiting(
        &mut app,
        "check the build",
        QueueCondition::At(Instant::now() + Duration::from_millis(30)),
    );

    assert_eq!(app.tick_queue(), 0, "not yet");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(app.tick_queue(), 1);
    assert!(app.message_queue[0].is_ready());
}

#[test]
fn a_hanging_step_releases_the_stuck_message() {
    let mut app = App::new();
    waiting(&mut app, "unstuck", QueueCondition::OnStuck);
    app.status.todos.push(todo("long step", "in_progress", Some(now_secs() - 900.0)));

    assert_eq!(app.tick_queue(), 1);
}

#[test]
fn a_quick_step_does_not_release_the_stuck_message() {
    let mut app = App::new();
    waiting(&mut app, "unstuck", QueueCondition::OnStuck);
    app.status.todos.push(todo("fresh step", "in_progress", Some(now_secs() - 5.0)));

    assert_eq!(app.tick_queue(), 0);
}

#[test]
fn delays_are_read_the_way_people_write_them() {
    assert_eq!(parse_delay("10m"), Some(Duration::from_secs(600)));
    assert_eq!(parse_delay("90s"), Some(Duration::from_secs(90)));
    assert_eq!(parse_delay("2h"), Some(Duration::from_secs(7200)));
    assert_eq!(parse_delay("1d"), Some(Duration::from_secs(86_400)));
    assert_eq!(parse_delay("10"), Some(Duration::from_secs(600)), "a bare number is minutes");
    assert_eq!(parse_delay("через"), None);
    assert_eq!(parse_delay("0"), None);
    assert_eq!(parse_delay(""), None);
}

#[test]
fn the_command_queues_a_message_for_the_next_error() {
    let mut app = App::new();

    assert!(matches!(app.handle_local_command("/queue on error:429 Продолжай"), Submit::Handled));
    assert_eq!(app.message_queue.len(), 1);
    assert_eq!(app.message_queue[0].text, "Продолжай");
    match &app.message_queue[0].condition {
        Some(QueueCondition::OnError { filter: Some(f) }) => assert_eq!(f, "429"),
        other => panic!("unexpected condition: {other:?}"),
    }
}

#[test]
fn the_command_queues_a_message_for_later() {
    let mut app = App::new();
    app.handle_local_command("/queue in 10m Проверь сборку");

    assert_eq!(app.message_queue.len(), 1);
    assert!(matches!(app.message_queue[0].condition, Some(QueueCondition::At(_))));
    assert_eq!(app.message_queue[0].text, "Проверь сборку");
}

#[test]
fn a_bad_delay_is_refused_with_the_format() {
    let mut app = App::new();
    app.handle_local_command("/queue in скоро Проверь");

    assert!(app.message_queue.is_empty());
    assert!(
        app.messages
            .iter()
            .any(|m| matches!(m, MessageItem::Error { text, .. } if text.contains("/queue in"))),
        "the person has to learn the format"
    );
}

// ── подсказка панели очереди ────────────────────────────────────────────────

#[test]
fn the_narrow_panel_still_says_how_to_edit() {
    let mut app = App::new();
    app.is_busy = true;
    app.message_queue.push(QueuedMessage::plain("hello"));

    // 40 columns is narrower than the header plus the hint: it used to be clipped
    // away, leaving a panel that never said a row could be clicked.
    let rows = frame(&app, 40, 24);
    assert!(
        rows.iter().any(|r| r.contains("клик")),
        "the hint must survive a narrow window:\n{}",
        rows.join("\n")
    );
}

#[test]
fn a_long_queue_says_where_the_rest_went() {
    let mut app = App::new();
    app.is_busy = true;
    for i in 1..=9 {
        app.message_queue.push(QueuedMessage::plain(format!("message {i}")));
    }

    let rows = frame(&app, 100, 30);
    assert!(
        rows.iter().any(|r| r.contains("ещё 3") && r.contains("/queue")),
        "the hidden tail must be named, with a way to reach it:\n{}",
        rows.join("\n")
    );
}

// ── прокрутка ───────────────────────────────────────────────────────────────

fn many_steps(app: &mut App) {
    for i in 1..=12 {
        app.status.todos.push(todo(&format!("step {i}"), "pending", None));
    }
    app.is_busy = true;
    // A turn is needed for the panel to exist at all.
    app.messages.push(MessageItem::User("go".into()));
}

#[test]
fn the_wheel_over_the_checklist_scrolls_the_checklist_not_the_chat() {
    let mut app = App::new();
    many_steps(&mut app);
    frame(&app, 100, 30);

    let (x, y, _, _) = app.todos_hover.get().expect("the panel must register itself");
    app.set_mouse_pos(x + 1, y + 1);
    let chat_before = app.scroll_offset;

    app.wheel(false, Instant::now());

    assert_eq!(app.scroll_offset, chat_before, "the chat must stay put");
    assert_ne!(app.todos_scroll, 0, "the checklist window moved");
}

#[test]
fn the_wheel_over_the_chat_still_scrolls_the_chat() {
    let mut app = App::new();
    many_steps(&mut app);
    for i in 0..40 {
        app.messages.push(MessageItem::User(format!("line {i}")));
    }
    frame(&app, 100, 30);

    // Somewhere in the transcript, well above the panels.
    app.set_mouse_pos(5, 2);
    app.wheel(true, Instant::now());

    assert_eq!(app.todos_scroll, 0, "the checklist must not move");
    assert!(app.scroll_offset > 0, "the chat scrolled");
}

#[test]
fn the_wheel_over_the_prompt_peeks_back_without_moving_the_caret() {
    let mut app = App::new();
    app.input = (1..=40).map(|i| format!("line {i}\n")).collect();
    app.cursor_pos = app.input.chars().count();
    frame(&app, 80, 20);

    let g = app.input_geom.get();
    assert!(g.base_skip > 0, "the draft is taller than the prompt");
    let caret = app.cursor_pos;

    app.set_mouse_pos(g.left + 1, g.top);
    app.wheel(true, Instant::now());

    assert_eq!(app.input_scroll, 1);
    assert_eq!(app.cursor_pos, caret, "the caret stays where it was");
}

// ── подагенты ───────────────────────────────────────────────────────────────

#[test]
fn subagent_lists_are_read_from_the_state() {
    let subs = parse_sub_todos(
        r#"{"sub-1":[{"content":"a","status":"completed"}],"sub-2":[]}"#,
    );
    assert_eq!(subs.len(), 1, "an empty list is not worth a row");
    assert_eq!(subs[0].name, "sub-1");
    assert_eq!(subs[0].items[0].content, "a");
}

#[test]
fn a_subagents_checklist_is_drawn_under_the_main_one() {
    let mut app = App::new();
    app.status.todos.push(todo("delegate", "in_progress", Some(now_secs())));
    app.status.todos_sub.push(SubTodos {
        name: "sub-1".into(),
        items: vec![todo("read", "completed", None), todo("write", "in_progress", None)],
    });
    app.todos_sub_open = vec![false];
    app.is_busy = true;

    let rows = frame(&app, 100, 30);
    let line = rows
        .iter()
        .find(|r| r.contains("sub-1"))
        .unwrap_or_else(|| panic!("no sub-agent row:\n{}", rows.join("\n")));
    assert!(line.contains("1/2"), "progress is the point: {line:?}");
    assert!(line.contains("write"), "the current step is named: {line:?}");
}

#[test]
fn an_opened_subagent_list_shows_its_steps() {
    let mut app = App::new();
    app.status.todos.push(todo("delegate", "in_progress", Some(now_secs())));
    app.status.todos_sub.push(SubTodos {
        name: "sub-1".into(),
        items: vec![todo("read the file", "completed", None)],
    });
    app.todos_sub_open = vec![true];
    app.is_busy = true;

    let rows = frame(&app, 100, 30);
    assert!(
        rows.iter().any(|r| r.contains("read the file")),
        "an open list must show its steps:\n{}",
        rows.join("\n")
    );
    assert!(
        !app.todos_sub_rows.borrow().is_empty(),
        "the header stays clickable"
    );
}
