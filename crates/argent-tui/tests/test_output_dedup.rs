//! A command's live output and its finished card are the same text.
//!
//! `run_command` prints while it runs; the bridge forwards every chunk as an `output`
//! event; `tool_end` then carries the whole result again. Keeping both drew the work
//! twice — the output as loose rows, and a card below it saying "13 строк" (seen in a
//! screenshot of a `Get-ChildItem` call).

use std::collections::HashMap;

use argent_tui::app::{App, MessageItem};
use argent_tui::events::ArgentEvent;

fn run_command_started(app: &mut App) {
    app.messages.push(MessageItem::User("покажи файлы".into()));
    app.handle_event(ArgentEvent::ToolStart {
        name: "run_command".into(),
        args: HashMap::new(),
    });
}

#[test]
fn the_live_copy_is_dropped_when_the_card_carries_the_same_text() {
    let mut app = App::new();
    run_command_started(&mut app);

    app.handle_event(ArgentEvent::Output("Name    Length\n".into()));
    app.handle_event(ArgentEvent::Output("a.pyd   2068480\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "Name    Length\na.pyd   2068480\n".into(),
        duration_secs: Some(0.2),
    });

    // One rendering of the work, not two: the card is the only thing left.
    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 0, "живой вывод остался дубликатом карточки: {:?}", app.messages.len());

    let cards = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Tool { .. }))
        .count();
    assert_eq!(cards, 1, "карточка на месте");
}

#[test]
fn a_truncated_live_tail_is_information_and_stays() {
    // The console shows a cap, `tool_end` carries everything. Removing the visible rows
    // would delete what the person can still see on screen.
    let mut app = App::new();
    run_command_started(&mut app);
    app.handle_event(ArgentEvent::Output("первые строки вывода\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "первые строки вывода\nи ещё двести\n".into(),
        duration_secs: Some(0.2),
    });

    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 1, "обрезанный живой хвост не трогаем");
}

#[test]
fn another_tools_output_is_not_touched() {
    let mut app = App::new();
    run_command_started(&mut app);
    app.handle_event(ArgentEvent::Output("вывод постороннего инструмента\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "grep_search".into(),
        result: "совсем другой результат\n".into(),
        duration_secs: None,
    });

    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 1, "чужой вывод не трогаем");
}
