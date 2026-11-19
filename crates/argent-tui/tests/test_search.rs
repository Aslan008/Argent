//! Поиск по транскрипту: найти место, где агент что-то сказал.
//!
//! Ctrl+R ищет то, что набирал пользователь (историю ввода). Вернуться к
//! абзацу ответа было нечем. Здесь проверяется сам поиск и то, что Enter
//! просит прокрутку к найденному сообщению.

use argent_tui::app::{App, MessageItem};

fn app_with_messages() -> App {
    let mut app = App::new();
    app.messages.push(MessageItem::User("как работает очередь?".into()));
    app.messages.push(MessageItem::Notice("заметка про кэш npx".into()));
    app.messages.push(MessageItem::User("и ещё про ОЧЕРЕДЬ".into()));
    app
}

#[test]
fn it_finds_messages_by_text_ignoring_case() {
    let mut app = app_with_messages();
    app.open_search();
    for c in "очередь".chars() {
        app.search_push(c);
    }
    let hits: Vec<usize> = app.search_matches().iter().map(|(i, _)| *i).collect();
    assert_eq!(hits, vec![0, 2], "регистр не должен мешать");
}

#[test]
fn an_empty_query_lists_nothing() {
    let mut app = app_with_messages();
    app.open_search();
    assert!(app.search_matches().is_empty(), "иначе список был бы всем чатом");
}

#[test]
fn enter_asks_to_scroll_to_the_hit() {
    let mut app = app_with_messages();
    app.open_search();
    for c in "кэш".chars() {
        app.search_push(c);
    }
    assert_eq!(app.search_sel, 0);
    assert!(app.search_accept());
    assert!(!app.search_is_open(), "после перехода поиск закрывается");
    assert_eq!(app.take_jump(), Some(1), "прыжок на второе сообщение");
    assert!(!app.auto_scroll, "автопрокрутка не должна вернуть вниз");
    assert_eq!(app.take_jump(), None, "прыжок одноразовый");
}

#[test]
fn a_jump_survives_the_next_frame() {
    // Прыжок задаёт прокрутку только на один кадр. Чтобы она осталась, смещение
    // должно быть ненулевым: `App::sync_scroll` записывает измеренное значение
    // обратно только тогда. С нулём прыжок откатывался на следующем кадре.
    let mut app = app_with_messages();
    app.open_search();
    for c in "кэш".chars() {
        app.search_push(c);
    }
    assert!(app.search_accept());
    assert!(
        app.scroll_offset > 0,
        "прыжок не переживёт следующий кадр: sync_scroll нечего записывать"
    );
}

#[test]
fn arrows_move_through_hits() {
    let mut app = app_with_messages();
    app.open_search();
    for c in "очередь".chars() {
        app.search_push(c);
    }
    app.search_move(1);
    assert_eq!(app.search_sel, 1);
    app.search_move(1);
    assert_eq!(app.search_sel, 0, "список закольцован");
}

#[test]
fn nothing_found_is_not_a_crash() {
    let mut app = app_with_messages();
    app.open_search();
    for c in "zzzz".chars() {
        app.search_push(c);
    }
    assert!(!app.search_accept());
    assert_eq!(app.take_jump(), None);
}

#[test]
fn backspacing_past_the_start_closes_it() {
    let mut app = app_with_messages();
    app.open_search();
    app.search_push('к');
    app.search_backspace();
    assert!(!app.search_is_open());
}
