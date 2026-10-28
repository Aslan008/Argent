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
fn the_live_copy_is_dropped_when_the_card_wraps_it_in_an_exit_header() {
    // `run_command` returns “Exit code: N / OUTPUT: <the child's bytes>”, and the live
    // stream carries only the child's text. An exact comparison never matched, so a
    // `python -m py_compile …` printed its whole output above the card and again inside
    // it (seen in a screenshot). Containment is the honest question.
    let mut app = App::new();
    run_command_started(&mut app);

    app.handle_event(ArgentEvent::Output("compile ok\n".into()));
    app.handle_event(ArgentEvent::Output("Loaded 6 global plugin(s)\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "Exit code: 0\nOUTPUT:\ncompile ok\nLoaded 6 global plugin(s)\n".into(),
        duration_secs: Some(0.9),
    });

    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 0, "вывод остался и вне карточки, и внутри неё");
}

#[test]
fn a_live_tail_the_card_also_carries_is_dropped() {
    // The console can cap its live rows; the card holds the whole result, and folded it
    // says “N строк · first line”. The card is the place where the output lives.
    let mut app = App::new();
    run_command_started(&mut app);
    app.handle_event(ArgentEvent::Output("первые строки вывода\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "Exit code: 0\nOUTPUT:\nпервые строки вывода\nи ещё двести\n".into(),
        duration_secs: Some(0.2),
    });

    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 0, "карточка содержит всё, что было в живом хвосте");
}

#[test]
fn colours_in_the_live_row_do_not_hide_it_from_the_card() {
    // The live rows carry the REPL's SGR codes, the result arrives cleaned: comparing
    // the raw bytes meant the duplicate survived whenever a line was coloured.
    let mut app = App::new();
    run_command_started(&mut app);
    app.handle_event(ArgentEvent::Output("\u{1b}[2m✓ ok\u{1b}[0m\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "Exit code: 0\nOUTPUT:\n✓ ok\n".into(),
        duration_secs: Some(0.2),
    });

    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 0, "цвет в живой строке спрятал дубликат от сравнения");
}

#[test]
fn the_approval_line_in_front_of_the_output_stays_while_the_output_goes() {
    // The exact case from the screenshot: the auto-approval is printed to the console
    // before the tool starts, so it is an `Output` item too, and it sits in front of
    // the live output. Comparing the whole block with the result could never match, and
    // the command's output stayed on screen twice. The approval line is a record, not
    // a duplicate: it stays.
    let mut app = App::new();
    app.messages.push(MessageItem::User("проверь синтаксис".into()));
    app.handle_event(ArgentEvent::Output(
        "✓ Авто-одобрено (автономный режим): выполнить команду: py_compile agent.py\n".into(),
    ));
    app.handle_event(ArgentEvent::ToolStart {
        name: "run_command".into(),
        args: HashMap::new(),
    });
    app.handle_event(ArgentEvent::Output("compile ok\n".into()));
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "Exit code: 0\nOUTPUT:\ncompile ok\n".into(),
        duration_secs: Some(0.9),
    });

    let outputs: Vec<&String> = app
        .messages
        .iter()
        .filter_map(|m| match m {
            MessageItem::Output(text) => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(outputs.len(), 1, "должна остаться только строка одобрения: {outputs:?}");
    assert!(outputs[0].contains("Авто-одобрено"), "{outputs:?}");
}

#[test]
fn a_live_tail_the_card_does_not_carry_stays() {
    // The result is truncated (`MAX_OUTPUT`), the console saw more: the live rows are
    // the only place that text exists, so they stay.
    let mut app = App::new();
    run_command_started(&mut app);
    app.handle_event(ArgentEvent::Output("строка, которой нет в результате\n".into()));

    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "Exit code: 0\nOUTPUT:\nсовсем другое\n".into(),
        duration_secs: Some(0.2),
    });

    let outputs = app
        .messages
        .iter()
        .filter(|m| matches!(m, MessageItem::Output(_)))
        .count();
    assert_eq!(outputs, 1, "уникальный живой вывод не трогаем");
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
