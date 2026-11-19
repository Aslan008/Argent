//! The interface must never be able to freeze itself.
//!
//! A person reported a bug that only a restart cured: chat scrolling stopped, clicks and
//! fold toggles did nothing, and a hover hint stayed on screen forever. It arrived after
//! minimizing and restoring the window, and nothing about it pointed at a cause.
//!
//! Two states produced it together:
//!
//! * `App::focused == false` — set by `FocusLost`, cleared only by `FocusGained`. A
//!   minimized window can lose the regain, and then EVERY keystroke was dropped (the
//!   loop's first key arm did nothing but clear a selection). No typing, no Enter, no
//!   folds: exactly "the interface hung".
//! * `App::mouse_pos` left behind — the tooltip is drawn from it, so the last hint drawn
//!   before the window went away stayed on screen, attached to nothing.
//!
//! The fix does not depend on the terminal telling the truth about focus: an input event
//! is itself proof of focus, and losing focus forgets the pointer.

use argent_tui::app::{App, MessageItem};

fn app() -> App {
    App::new()
}

#[test]
fn a_key_or_click_is_proof_of_focus() {
    let mut app = app();
    app.focused = false; // as if FocusGained never arrived

    app.note_input();

    assert!(app.focused, "событие ввода возвращает фокус — иначе все клавиши мертвы");
}

#[test]
fn focus_is_not_changed_when_it_was_already_true() {
    let mut app = app();
    app.focused = true;

    app.note_input();

    assert!(app.focused);
}

#[test]
fn losing_focus_forgets_the_pointer() {
    let mut app = app();
    app.set_mouse_pos(10, 5);
    assert!(app.mouse_pos.is_some());

    app.forget_pointer();

    // No position, no hover hint: the tooltip cannot outlive the window it belonged to.
    assert!(app.mouse_pos.is_none(), "подсказка не должна переживать потерю фокуса");
}

#[test]
fn forgetting_a_pointer_that_was_already_forgotten_is_harmless() {
    let mut app = app();
    app.forget_pointer();
    app.forget_pointer();
    assert!(app.mouse_pos.is_none());
}

#[test]
fn the_answer_is_left_in_the_transcript() {
    let mut app = app();

    app.note_answer("вариант Б");

    let last = app.messages.last().expect("ответ должен оставить строку в чате");
    match last {
        MessageItem::Notice(text) => assert!(text.contains("вариант Б"), "{text}"),
        other => panic!("ожидалось Notice, получено {other:?}"),
    }
}

#[test]
fn a_dismissed_question_says_so_instead_of_vanishing() {
    let mut app = app();

    app.note_answer("");

    match app.messages.last().expect("строка есть") {
        MessageItem::Notice(text) => assert!(text.contains("без выбора"), "{text}"),
        other => panic!("ожидалось Notice, получено {other:?}"),
    }
}
