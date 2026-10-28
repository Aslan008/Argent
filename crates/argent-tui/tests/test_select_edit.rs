//! Editing a free-text answer in a modal: the caret moves, not just the tail.
//!
//! The select modal used to append and delete at the end only, so picking
//! "Свой вариант" left no way to fix a typo in the middle of the answer.

use argent_tui::app::SelectModalData;

fn text_prompt() -> SelectModalData {
    SelectModalData::new("id".into(), "Введите свой вариант ❯ ".into(), Vec::new(), None)
}

fn typed(text: &str) -> SelectModalData {
    let mut s = text_prompt();
    for c in text.chars() {
        s.filter_insert(c);
    }
    s
}

#[test]
fn typing_starts_at_the_end() {
    let s = typed("hello");
    assert_eq!(s.filter, "hello");
    assert_eq!(s.filter_cursor, 5);
}

#[test]
fn a_char_is_inserted_where_the_caret_is() {
    let mut s = typed("helo");
    s.filter_left();
    s.filter_insert('l');
    assert_eq!(s.filter, "hello");
    assert_eq!(s.filter_cursor, 4, "каретка уехала вместе с вставленным символом");
}

#[test]
fn backspace_and_delete_work_on_either_side_of_the_caret() {
    let mut s = typed("abcd");
    s.filter_left();
    s.filter_left();
    s.filter_backspace();                 // удаляет 'b'
    assert_eq!((s.filter.as_str(), s.filter_cursor), ("acd", 1));
    s.filter_delete();                    // удаляет 'c'
    assert_eq!((s.filter.as_str(), s.filter_cursor), ("ad", 1));
}

#[test]
fn home_end_and_clear() {
    let mut s = typed("abcdef");
    s.filter_home();
    assert_eq!(s.filter_cursor, 0);
    s.filter_end();
    assert_eq!(s.filter_cursor, 6);
    s.filter_home();
    s.filter_insert('X');
    assert_eq!(s.filter, "Xabcdef");
    s.filter_clear();
    assert_eq!((s.filter.as_str(), s.filter_cursor), ("", 0));
}

#[test]
fn the_caret_never_walks_off_the_text() {
    let mut s = typed("ab");
    for _ in 0..10 {
        s.filter_right();
    }
    assert_eq!(s.filter_cursor, 2);
    for _ in 0..10 {
        s.filter_left();
    }
    assert_eq!(s.filter_cursor, 0);
    s.filter_backspace();                 // at 0: nothing to delete, no panic
    s.filter_delete();                    // deletes 'a'
    assert_eq!((s.filter.as_str(), s.filter_cursor), ("b", 0));
}

#[test]
fn a_cyrillic_answer_is_edited_by_chars_not_bytes() {
    let mut s = typed("привет");
    assert_eq!(s.filter_cursor, 6);
    s.filter_left();
    s.filter_left();
    s.filter_backspace();                 // удаляет 'в', а не половину буквы
    assert_eq!(s.filter, "приет");
    assert_eq!(s.filter_cursor, 3);
    s.filter_insert('в');
    assert_eq!(s.filter, "привет");
}

#[test]
fn the_filter_also_moves_the_selection_out_of_the_way() {
    // Editing the filter re-runs the match, so the highlighted option is never
    // one that the typed text no longer matches.
    let mut s = SelectModalData::new(
        "id".into(),
        "Выберите".into(),
        vec!["alpha".into(), "beta".into(), "gamma".into()],
        None,
    );
    s.filter_insert('g');
    let filtered = s.filtered_options();
    assert!(filtered.iter().any(|(i, _)| *i == s.selected_idx), "выбор внутри отфильтрованного списка");
    assert_eq!(s.filtered_options().len(), 1);
}
