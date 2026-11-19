//! Кнопка «скопировать» на карточке ответа: копируется весь ответ.
//!
//! Ответ иногда приходит двумя соседними сегментами (продолжение после
//! переподключения). Заголовок — и кнопка на нём — стоит на первом, поэтому
//! копирование одного элемента отдавало половину ответа.

use argent_tui::app::{App, MessageItem};

#[test]
fn a_re_sent_answer_is_copied_whole() {
    let mut app = App::new();
    app.messages.push(MessageItem::assistant("первая часть", ""));
    app.messages.push(MessageItem::assistant("вторая часть", ""));
    let text = app.assistant_run_content(0).expect("есть что копировать");
    assert!(
        text.contains("первая часть") && text.contains("вторая часть"),
        "скопирована половина ответа: {text:?}"
    );
}

#[test]
fn the_click_on_either_segment_yields_the_same_text() {
    let mut app = App::new();
    app.messages.push(MessageItem::assistant("раз", ""));
    app.messages.push(MessageItem::assistant("два", ""));
    assert_eq!(
        app.assistant_run_content(0),
        app.assistant_run_content(1),
        "сегменты одного ответа должны копироваться одинаково"
    );
}

#[test]
fn a_user_message_is_not_an_answer() {
    let mut app = App::new();
    app.messages.push(MessageItem::User("привет".into()));
    assert_eq!(app.assistant_run_content(0), None);
}

#[test]
fn a_tool_between_answers_separates_the_runs() {
    let mut app = App::new();
    app.messages.push(MessageItem::assistant("до инструмента", ""));
    app.messages.push(MessageItem::Notice("пауза".into()));
    app.messages.push(MessageItem::assistant("после инструмента", ""));
    let first = app.assistant_run_content(0).expect("первый ответ");
    assert!(!first.contains("после инструмента"), "прогоны склеились: {first}");
}

#[test]
fn an_empty_answer_is_not_offered() {
    let mut app = App::new();
    app.messages.push(MessageItem::assistant("   ", ""));
    assert_eq!(app.assistant_run_content(0), None);
}
