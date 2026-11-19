//! A finished turn folds its work into ONE line, and the answer stays where it was.
//!
//! The unit is the TURN and the whole of it: a call, a reasoning block, a progress note
//! between batches and the next call all belong to the same line, because the person reads
//! a question and an answer. Every finished turn folds, not only the newest one. What stays
//! out is the answer, the question, and work that came after the last text — an open
//! auto-run, which is not "before the answer". A turn with no text of its own (it ended on
//! a question, an error or a call) is left exactly as it is.

use std::collections::HashMap;

use argent_tui::app::{App, ClickAction, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, w: u16, h: u16) -> Vec<String> {
    let mut terminal = Terminal::new(TestBackend::new(w, h)).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| {
            (0..buf.area.width)
                .map(|x| buf.content[(y * buf.area.width + x) as usize].symbol())
                .collect::<String>()
        })
        .collect()
}

fn found(screen: &[String], needle: &str) -> Option<usize> {
    screen.iter().position(|r| r.contains(needle))
}

fn tool(_i: usize, result: &str) -> MessageItem {
    MessageItem::Tool {
        name: "run_command".into(),
        args: HashMap::new(),
        result: Some(result.into()),
        duration_secs: Some(2.0),
        expanded: true,
        is_running: false,
    }
}

fn app_with_turn(tools: usize) -> App {
    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай это".into()));
    for i in 0..tools {
        app.messages.push(tool(i, &format!("вывод {i}")));
    }
    app.messages.push(MessageItem::assistant("готово", ""));
    app
}

fn finish(app: &mut App) {
    app.handle_event(ArgentEvent::Done);
}

#[test]
fn the_work_of_a_finished_turn_folds_into_one_line() {
    let mut app = app_with_turn(4);
    assert!(found(&render(&app, 100, 30), "вывод 0").is_some(), "до ответа шаг виден");

    finish(&mut app);
    let screen = render(&app, 100, 30);

    assert!(found(&screen, "шаг").is_some(), "{screen:?}");
    assert!(found(&screen, "вывод 0").is_none(), "работа свёрнута");
    assert!(found(&screen, "готово").is_some(), "ответ на месте");
    assert!(found(&screen, "сделай это").is_some(), "вопрос на месте");
}

#[test]
fn the_line_says_how_much_work_it_hides() {
    let mut app = app_with_turn(4);
    finish(&mut app);

    let screen = render(&app, 100, 30);
    let row = found(&screen, "шаг").expect("строка хода");

    assert!(screen[row].contains('4'), "четыре шага: {:?}", screen[row]);
    assert!(screen[row].contains("шаг"), "{:?}", screen[row]);
    assert!(screen[row].contains("0:08"), "8 секунд по карточкам: {:?}", screen[row]);
}

#[test]
fn a_step_that_failed_is_named_on_the_line() {
    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай".into()));
    app.messages.push(tool(1, "вывод 1"));
    app.messages.push(tool(2, "Error: the file does not exist"));
    app.messages.push(tool(3, "вывод 3"));
    app.messages.push(MessageItem::assistant("готово", ""));
    finish(&mut app);

    let screen = render(&app, 100, 30);
    let row = found(&screen, "ошибк").expect("провал назван на самой строке");

    // Folded, but never hidden silently: the line says how many steps came back red.
    assert!(found(&screen, "the file does not exist").is_none(), "провал свёрнут: {screen:?}");
    assert!(screen[row].contains('1'), "одна ошибка: {:?}", screen[row]);
}

#[test]
fn a_short_turn_folds_too() {
    // A threshold left small turns sprawling: one call and an answer is still a turn.
    let mut app = app_with_turn(2);
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert!(found(&screen, "Ходы агента").is_some(), "{screen:?}");
    assert!(found(&screen, "вывод 0").is_none(), "работа короткого хода тоже свёрнута");
}

#[test]
fn a_click_opens_the_work_and_a_second_click_closes_it() {
    let mut app = app_with_turn(4);
    finish(&mut app);
    let screen = render(&app, 100, 30);
    let row = found(&screen, "шаг").expect("строка хода");

    let action = app.get_clicked_action(3, row as u16).expect("шапка — кнопка");
    match action {
        ClickAction::ToggleFold(idx) => app.toggle_fold(idx),
        other => panic!("ожидался ToggleFold, а не {other:?}"),
    }
    let opened = render(&app, 100, 30);
    assert!(found(&opened, "вывод 0").is_some(), "работа раскрыта: {opened:?}");
    assert!(found(&opened, "шаг").is_some(), "шапка остаётся, чтобы закрыть");

    let row = found(&opened, "шаг").expect("строка хода");
    match app.get_clicked_action(3, row as u16).expect("шапка всё ещё кнопка") {
        ClickAction::ToggleFold(idx) => app.toggle_fold(idx),
        other => panic!("ожидался ToggleFold, а не {other:?}"),
    }
    let closed = render(&app, 100, 30);
    assert!(found(&closed, "вывод 0").is_none(), "работа снова свёрнута");
}

#[test]
fn compact_view_off_turns_the_fold_off_and_on_brings_it_back() {
    use argent_tui::app::Submit;

    let mut app = app_with_turn(4);
    assert_eq!(app.handle_local_command("/compact-view off"), Submit::Handled);
    finish(&mut app);
    assert!(
        found(&render(&app, 100, 30), "вывод 0").is_some(),
        "без сворачивания шаг виден"
    );

    assert_eq!(app.handle_local_command("/compact-view on"), Submit::Handled);
    let screen = render(&app, 100, 30);
    assert!(found(&screen, "вывод 0").is_none(), "включили — и ход свернулся");
}

#[test]
fn the_switch_reports_its_own_state() {
    use argent_tui::app::Submit;

    let mut app = App::new();
    assert_eq!(app.handle_local_command("/compact-view"), Submit::Handled);

    let screen = render(&app, 100, 30);
    assert!(found(&screen, "включено").is_some(), "{screen:?}");
}

#[test]
fn a_fold_belongs_to_its_own_turn() {
    let mut app = app_with_turn(4);
    finish(&mut app);
    // A second turn: its own work folds on its own, and the first group is untouched.
    app.messages.push(MessageItem::User("и ещё".into()));
    for i in 0..4 {
        app.messages.push(tool(i, &format!("второй вывод {i}")));
    }
    app.messages.push(MessageItem::assistant("готово снова", ""));
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert!(found(&screen, "второй вывод 0").is_none());
    assert!(found(&screen, "готово снова").is_some());
    assert_eq!(
        screen.iter().filter(|r| r.contains("шаг")).count(),
        2,
        "по строке на ход: {screen:?}"
    );
}

#[test]
fn progress_text_between_batches_folds_with_the_work() {
    // "проверяю ещё раз" is a step, not a paragraph: the turn must read as one line.
    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай".into()));
    app.messages.push(tool(1, "вывод 1"));
    app.messages.push(tool(2, "вывод 2"));
    app.messages.push(MessageItem::assistant("проверяю ещё раз", ""));
    app.messages.push(tool(3, "вывод 3"));
    app.messages.push(MessageItem::assistant("готово", ""));
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert!(found(&screen, "готово").is_some(), "{screen:?}");
    assert!(found(&screen, "проверяю ещё раз").is_none(), "промежуточный текст тоже шаг");
    assert_eq!(
        screen.iter().filter(|r| r.contains("шаг")).count(),
        1,
        "одна строка на весь ход: {screen:?}"
    );
}

#[test]
fn work_after_the_answer_stays_visible() {
    // A vibe auto-continue: more work arrived after the last text. It is not "before the
    // answer", so it stays on screen while the steps that led to the answer fold.
    let mut app = app_with_turn(3);
    app.messages.push(tool(9, "продолжение"));
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert!(found(&screen, "продолжение").is_some(), "работа после ответа не прячется: {screen:?}");
    assert!(found(&screen, "вывод 0").is_none(), "шаги до ответа свёрнуты");
    assert!(found(&screen, "готово").is_some(), "ответ на месте");
}

#[test]
fn a_turn_that_ended_on_an_error_is_left_alone() {
    // The reason a turn died is the thing the person has to read: no fold at all.
    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай".into()));
    for i in 0..4 {
        app.messages.push(tool(i, &format!("вывод {i}")));
    }
    app.messages.push(MessageItem::Error { text: "провайдер упал".into(), hint: None });
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert!(found(&screen, "провайдер упал").is_some(), "{screen:?}");
    assert!(found(&screen, "вывод 0").is_some(), "без ответа сворачивать нечего");
}

#[test]
fn a_second_batch_of_the_same_turn_extends_the_same_line() {
    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай".into()));
    for i in 0..3 {
        app.messages.push(tool(i, &format!("вывод {i}")));
    }
    app.messages.push(MessageItem::assistant("первый ответ", ""));
    finish(&mut app);
    assert_eq!(
        render(&app, 100, 30).iter().filter(|r| r.contains("шаг")).count(),
        1
    );

    // The same turn goes on (no new User): the line covers the new batch too.
    for i in 3..6 {
        app.messages.push(tool(i, &format!("вывод {i}")));
    }
    app.messages.push(MessageItem::assistant("второй ответ", ""));
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert_eq!(
        screen.iter().filter(|r| r.contains("шаг")).count(),
        1,
        "один блок на ход: {screen:?}"
    );
    assert!(found(&screen, "второй ответ").is_some());
    assert!(found(&screen, "первый ответ").is_none(), "промежуточный ответ тоже шаг");
}

#[test]
fn every_finished_turn_folds_not_only_the_last_one() {
    // A history where only the newest turn is compact reads as a wall of blocks.
    let mut app = App::new();
    for turn in 0..3 {
        app.messages.push(MessageItem::User(format!("задача {turn}")));
        for i in 0..3 {
            app.messages.push(tool(i, &format!("вывод {turn}-{i}")));
        }
        app.messages.push(MessageItem::assistant(format!("ответ {turn}"), ""));
    }
    finish(&mut app);

    let screen = render(&app, 100, 30);

    assert_eq!(
        screen.iter().filter(|r| r.contains("Ходы агента")).count(),
        3,
        "по строке на каждый ход: {screen:?}"
    );
    for turn in 0..3 {
        assert!(found(&screen, &format!("ответ {turn}")).is_some(), "ответ {turn} виден");
        assert!(found(&screen, &format!("вывод {turn}-0")).is_none(), "шаги хода {turn} свёрнуты");
    }
}

#[test]
fn a_restored_history_folds_its_turns() {
    use argent_tui::events::HistoryItem;

    let mut app = App::new();
    app.apply_history(
        vec![
            HistoryItem::User("задача".into()),
            HistoryItem::Tool { name: "read_file".into(), args: HashMap::new(), result: "вывод".into() },
            HistoryItem::Assistant { text: "ответ".into(), thinking: String::new() },
        ],
        true,
    );

    let screen = render(&app, 100, 30);
    assert!(found(&screen, "Ходы агента").is_some(), "старый ход свёрнут при загрузке: {screen:?}");
    assert!(found(&screen, "вывод").is_none());
    assert!(found(&screen, "ответ").is_some());
}

#[test]
fn one_command_opens_and_closes_every_level() {
    use argent_tui::app::Submit;

    let mut app = App::new();
    app.messages.push(MessageItem::User("сделай".into()));
    app.messages.push(MessageItem::assistant("думаю", "размышление"));
    app.messages.push(tool(1, "вывод 1"));
    app.messages.push(tool(2, "вывод 2"));
    app.messages.push(MessageItem::assistant("готово", ""));
    finish(&mut app);
    assert!(found(&render(&app, 100, 40), "вывод 1").is_none(), "свёрнуто по умолчанию");

    assert_eq!(app.handle_local_command("/compact-view all"), Submit::Handled);
    let open = render(&app, 100, 40);
    assert!(found(&open, "вывод 1").is_some(), "карточки раскрыты: {open:?}");
    assert!(found(&open, "Ходы агента").is_some(), "строка хода остаётся, чтобы закрыть");

    assert_eq!(app.handle_local_command("/compact-view none"), Submit::Handled);
    let closed = render(&app, 100, 40);
    assert!(found(&closed, "вывод 1").is_none(), "карточки свёрнуты назад");
    assert!(found(&closed, "Ходы агента").is_some());
}

#[test]
fn a_running_turn_is_not_folded() {
    // The person is watching the output arrive: folding it would hide the work in flight.
    let mut app = App::new();
    app.messages.push(MessageItem::User("первая".into()));
    for i in 0..3 {
        app.messages.push(tool(i, &format!("старый вывод {i}")));
    }
    app.messages.push(MessageItem::assistant("первый ответ", ""));

    app.messages.push(MessageItem::User("вторая".into()));
    for i in 0..3 {
        app.messages.push(tool(i, &format!("живой вывод {i}")));
    }
    app.messages.push(MessageItem::assistant("ещё не финальный", ""));
    app.is_busy = true;

    app.fold_finished_turn();

    let screen = render(&app, 100, 30);
    assert!(found(&screen, "старый вывод 0").is_none(), "прошлый ход свёрнут");
    assert!(found(&screen, "живой вывод 0").is_some(), "идущий ход остаётся развёрнутым");
}
