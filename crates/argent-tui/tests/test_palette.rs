//! Палитра команд: поиск по имени И по описанию, а не только по префиксу.
//!
//! Всплывашка `/` умеет искать только начало имени команды, поэтому «очередь»
//! не находила `/queue`. Здесь проверяется именно это: фильтр по описанию,
//! навигация по списку и подстановка команды в промпт.

use argent_tui::app::{App, CompletionItem};

fn app_with_commands() -> App {
    let mut app = App::new();
    app.completions = vec![
        CompletionItem { name: "/queue".into(), help: "очередь сообщений агента".into() },
        CompletionItem { name: "/model".into(), help: "выбрать модель".into() },
        CompletionItem { name: "/prices".into(), help: "стоимость запросов".into() },
    ];
    app
}

#[test]
fn it_opens_and_closes() {
    let mut app = app_with_commands();
    assert!(!app.palette_is_open());
    app.open_palette();
    assert!(app.palette_is_open());
    assert!(app.palette_matches().len() == 3, "пустой запрос показывает всё");
    app.palette_close();
    assert!(!app.palette_is_open());
}

#[test]
fn it_finds_a_command_by_what_it_does() {
    let mut app = app_with_commands();
    app.open_palette();
    for c in "очередь".chars() {
        app.palette_push(c);
    }
    let names: Vec<&str> = app.palette_matches().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["/queue"], "поиск по описанию не сработал");
}

#[test]
fn it_finds_a_command_by_its_name() {
    let mut app = app_with_commands();
    app.open_palette();
    for c in "pri".chars() {
        app.palette_push(c);
    }
    let names: Vec<&str> = app.palette_matches().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["/prices"]);
}

#[test]
fn enter_puts_the_command_into_the_prompt() {
    let mut app = app_with_commands();
    app.open_palette();
    for c in "model".chars() {
        app.palette_push(c);
    }
    assert!(app.palette_accept());
    assert!(!app.palette_is_open(), "после выбора палитра закрывается");
    assert_eq!(app.input, "/model ", "команда ждёт аргументов");
    assert_eq!(app.cursor_pos, app.input.chars().count());
}

#[test]
fn arrows_wrap_around_the_list() {
    let mut app = app_with_commands();
    app.open_palette();
    app.palette_move(-1);
    assert_eq!(app.palette_idx, 2, "вверх с первой строки уходит в конец");
    app.palette_move(1);
    assert_eq!(app.palette_idx, 0);
}

#[test]
fn backspacing_past_the_start_closes_it() {
    let mut app = app_with_commands();
    app.open_palette();
    app.palette_push('m');
    app.palette_backspace();
    assert!(!app.palette_is_open(), "пустой запрос + Backspace закрывает палитру");
}

#[test]
fn nothing_matching_is_not_a_crash() {
    let mut app = app_with_commands();
    app.open_palette();
    for c in "zzz".chars() {
        app.palette_push(c);
    }
    assert!(app.palette_matches().is_empty());
    assert!(!app.palette_accept(), "нечего выбирать");
}
