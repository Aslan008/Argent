//! A restored conversation reaches the screen as finished blocks and never looks like a turn.

use argent_tui::app::{App, MessageItem};
use argent_tui::events::{parse_py_event, ArgentEvent, HistoryItem};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

fn sample() -> Vec<HistoryItem> {
    let mut args = HashMap::new();
    args.insert("file_path".to_string(), serde_json::json!("menu.py"));
    vec![
        HistoryItem::Note("── восстановленный диалог ──".into()),
        HistoryItem::User("почини меню".into()),
        HistoryItem::Assistant { text: "Смотрю файл.".into(), thinking: "нужно прочитать".into() },
        HistoryItem::Tool { name: "read_file".into(), args, result: "def menu(): ...".into() },
        HistoryItem::Assistant { text: "Готово.".into(), thinking: String::new() },
    ]
}

#[test]
fn history_is_added_as_finished_blocks() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::History { items: sample(), replace: false });

    assert!(matches!(app.messages[0], MessageItem::System(_)));
    assert!(matches!(&app.messages[1], MessageItem::User(t) if t == "почини меню"));
    match &app.messages[3] {
        MessageItem::Tool { name, result, is_running, .. } => {
            assert_eq!(name, "read_file");
            assert_eq!(result.as_deref(), Some("def menu(): ..."));
            assert!(!is_running, "a restored call is not running");
        }
        other => panic!("expected a tool block, got {other:?}"),
    }
    assert_eq!(app.messages.len(), 5);
}

#[test]
fn history_does_not_make_the_ui_busy() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::History { items: sample(), replace: false });
    assert!(!app.is_busy, "a restored dialogue is not a turn in progress");
    assert!(app.messages.iter().all(|m| match m {
        MessageItem::Assistant { is_streaming_content, is_streaming_thinking, .. } =>
            !*is_streaming_content && !*is_streaming_thinking,
        _ => true,
    }));
}

#[test]
fn replace_clears_the_old_transcript_first() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Notice("old line".into()));
    app.handle_event(ArgentEvent::History { items: sample(), replace: true });
    assert_eq!(app.messages.len(), 5);
    assert!(!app.messages.iter().any(|m| matches!(m, MessageItem::Notice(t) if t == "old line")));
}

#[test]
fn without_replace_the_old_transcript_stays() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Banner("1.0".into()));
    app.handle_event(ArgentEvent::History { items: sample(), replace: false });
    assert!(matches!(app.messages[0], MessageItem::Banner(_)));
    assert_eq!(app.messages.len(), 6);
}

#[test]
fn the_python_event_is_parsed_block_by_block() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let user = PyDict::new_bound(py);
        user.set_item("role", "user").unwrap();
        user.set_item("text", "привет").unwrap();
        let tool = PyDict::new_bound(py);
        tool.set_item("role", "tool").unwrap();
        tool.set_item("name", "grep_search").unwrap();
        let args = PyDict::new_bound(py);
        args.set_item("pattern", "TODO").unwrap();
        tool.set_item("args", args).unwrap();
        tool.set_item("result", "3 matches").unwrap();
        let unknown = PyDict::new_bound(py);
        unknown.set_item("role", "mystery").unwrap();
        let items = PyList::new_bound(py, [user, tool, unknown]);

        let d = PyDict::new_bound(py);
        d.set_item("type", "history").unwrap();
        d.set_item("items", items).unwrap();
        d.set_item("replace", true).unwrap();

        match parse_py_event(&d).unwrap().unwrap() {
            ArgentEvent::History { items, replace } => {
                assert!(replace);
                assert_eq!(items.len(), 2, "an unknown role is skipped, not an error");
                assert_eq!(items[0], HistoryItem::User("привет".into()));
                match &items[1] {
                    HistoryItem::Tool { name, args, result } => {
                        assert_eq!(name, "grep_search");
                        assert_eq!(args.get("pattern").and_then(|v| v.as_str()), Some("TODO"));
                        assert_eq!(result, "3 matches");
                    }
                    other => panic!("expected a tool block, got {other:?}"),
                }
            }
            _ => panic!("expected a History event"),
        }
    });
}
