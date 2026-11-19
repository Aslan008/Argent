//! Мысли агента: пока он думает — видно, как закончил — свёрнуто.
//!
//! Развёрнутая цепочка рассуждений читается как новость ровно секунду, а потом
//! заслоняет то, ради чего она была: вызов инструмента и ответ. Поэтому блок
//! живёт раскрытым только пока пишется, а законченный сворачивается в строку
//! «Анализ · 12с», которую можно раскрыть кликом.

use argent_tui::app::{App, MessageItem};
use argent_tui::events::ArgentEvent;
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

/// Whether the newest reasoning block is drawn open.
fn thought_open(app: &App) -> Option<bool> {
    app.messages.iter().rev().find_map(|m| match m {
        MessageItem::Assistant { thinking, thinking_expanded, .. } if !thinking.is_empty() => {
            Some(*thinking_expanded)
        }
        _ => None,
    })
}

fn tool_start(app: &mut App) {
    app.handle_event(ArgentEvent::ToolStart {
        name: "read_file".into(),
        args: Default::default(),
    });
}

#[test]
fn the_live_thought_is_on_screen_while_it_streams() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("разбираю ui.rs".into()));

    let rows = frame(&app, 100, 30);
    assert!(
        rows.iter().any(|r| r.contains("разбираю ui.rs")),
        "мысль должна идти по мере поступления:\n{}",
        rows.join("\n")
    );
}

#[test]
fn the_thought_folds_when_the_tool_call_arrives() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("сначала прочитаю файл".into()));
    assert_eq!(thought_open(&app), Some(false), "пока стримится, флаг не в счёт");

    tool_start(&mut app);

    assert_eq!(thought_open(&app), Some(false), "мысль закончилась — блок свёрнут");
}

#[test]
fn the_thought_folds_when_the_answer_starts() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("теперь отвечу".into()));

    app.handle_event(ArgentEvent::Content("Готово.".into()));

    assert_eq!(thought_open(&app), Some(false));
}

#[test]
fn a_full_thinking_config_no_longer_keeps_finished_thoughts_open() {
    // The person's own setting said "full", which used to mean "не сворачивай":
    // the transcript stayed buried under the reasoning.
    let mut app = App::new();
    app.ui.thinking_mode = "full".to_string();

    app.handle_event(ArgentEvent::Thinking("долгая мысль".into()));
    tool_start(&mut app);

    assert_eq!(thought_open(&app), Some(false));
}

#[test]
fn the_folded_thought_opens_again_on_a_click() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("мысль".into()));
    app.handle_event(ArgentEvent::Content("ответ".into()));
    assert_eq!(thought_open(&app), Some(false));

    app.toggle_last_reasoning();

    assert_eq!(thought_open(&app), Some(true), "клик раскрывает законченную мысль");
}

#[test]
fn the_folded_block_keeps_the_whole_thought() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("первая часть, ".into()));
    app.handle_event(ArgentEvent::Thinking("вторая часть".into()));
    app.handle_event(ArgentEvent::Content("ответ".into()));

    let thinking = app
        .messages
        .iter()
        .find_map(|m| match m {
            MessageItem::Assistant { thinking, .. } => Some(thinking.clone()),
            _ => None,
        })
        .expect("an assistant block");
    assert_eq!(thinking, "первая часть, вторая часть");
}

#[test]
fn the_folded_block_remembers_how_long_it_took() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("мысль".into()));
    app.handle_event(ArgentEvent::Content("ответ".into()));

    let duration = app.messages.iter().find_map(|m| match m {
        MessageItem::Assistant { thinking_duration, .. } => Some(*thinking_duration),
        _ => None,
    });
    assert!(
        matches!(duration, Some(Some(s)) if s >= 0.0),
        "the collapsed line says how long the thought took: {duration:?}"
    );
}

#[test]
fn reasoning_that_resumes_after_the_answer_folds_on_its_own() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("первая мысль".into()));
    app.handle_event(ArgentEvent::Content("частичный ответ".into()));
    // Reasoning again, after the answer: a new block that folds its own way.
    app.handle_event(ArgentEvent::Thinking("вторая мысль".into()));
    tool_start(&mut app);

    // Both blocks are folded: the finished reasoning is history either way.
    let open: Vec<bool> = app
        .messages
        .iter()
        .filter_map(|m| match m {
            MessageItem::Assistant { thinking, thinking_expanded, .. } if !thinking.is_empty() => {
                Some(*thinking_expanded)
            }
            _ => None,
        })
        .collect();
    assert_eq!(open, vec![false, false]);
}

#[test]
fn the_second_pass_folds_as_well() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Rethink("перепроверяю решение".into()));
    // Opened by hand, as a person might: the end of the stream must still fold it.
    if let Some(MessageItem::Rethink { expanded, .. }) = app.messages.last_mut() {
        *expanded = true;
    }

    tool_start(&mut app);

    let expanded = app.messages.iter().find_map(|m| match m {
        MessageItem::Rethink { expanded, .. } => Some(*expanded),
        _ => None,
    });
    assert_eq!(expanded, Some(false), "вторая проходка тоже мысль");
}
