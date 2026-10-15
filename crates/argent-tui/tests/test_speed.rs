//! The model's generation speed in the interface: measured from when the output
//! reached the sink, checked against the provider's own token count, drawn in the
//! live rows of the premium look and nowhere in the classic one (which is the REPL,
//! cell for cell, and the REPL shows none).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use argent_tui::app::App;
use argent_tui::events::ArgentEvent;
use argent_tui::theme::UiStyle;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn app_in(style: UiStyle) -> App {
    let mut app = App::new();
    app.ui.set_style(style);
    app
}

fn screen(app: &App) -> Vec<String> {
    let mut term = Terminal::new(TestBackend::new(110, 30)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    let buf = term.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect()
}

fn text(rows: &[String]) -> String {
    rows.join("\n")
}

/// `ago_ms` before now.
fn back(ago_ms: u64) -> Instant {
    Instant::now().checked_sub(Duration::from_millis(ago_ms)).expect("a clock that has run this long")
}

/// `n` pieces of "abcd" (1.08 estimated tokens each), `every_ms` apart, the first one
/// `ago_ms` ago, as the given event.
fn pour(app: &mut App, ago_ms: u64, n: u64, every_ms: u64, mut make: impl FnMut() -> ArgentEvent) {
    let t0 = back(ago_ms);
    for i in 0..n {
        app.handle_event_at(make(), t0 + Duration::from_millis(i * every_ms));
    }
}

fn content() -> ArgentEvent {
    ArgentEvent::Content("abcd".into())
}

fn row_with<'a>(rows: &'a [String], needle: &str) -> &'a str {
    rows.iter()
        .find(|r| r.contains(needle))
        .unwrap_or_else(|| panic!("`{needle}` is not on screen:\n{}", text(rows)))
}

// ── where it is drawn ────────────────────────────────────────────────────────

#[test]
fn the_premium_answer_row_shows_the_speed_while_the_model_writes() {
    let mut app = app_in(UiStyle::Premium);
    pour(&mut app, 1500, 28, 50, content);
    let rows = screen(&app);
    let live = row_with(&rows, "Генерация...");
    assert!(live.contains("tok/s"), "no speed on the live row:\n{}", text(&rows));
    assert!(live.contains("~"), "the first figure of a session is an estimate and says so");
}

#[test]
fn the_classic_look_stays_the_repls_own_and_shows_no_speed() {
    let mut app = app_in(UiStyle::Classic);
    pour(&mut app, 1500, 28, 50, content);
    let rows = screen(&app);
    assert!(app.speed.reading(Instant::now()).is_some(), "the speed is measured all the same");
    assert!(!text(&rows).contains("tok/s"), "classic must not differ from the REPL:\n{}", text(&rows));
    assert!(row_with(&rows, "Генерация...").contains("(0:0"), "the REPL's timer is still there");
}

#[test]
fn reasoning_shows_it_too() {
    let mut app = app_in(UiStyle::Premium);
    pour(&mut app, 1500, 28, 50, || ArgentEvent::Thinking("abcd".into()));
    let rows = screen(&app);
    assert!(row_with(&rows, "Анализ задачи").contains("tok/s"), "{}", text(&rows));
}

#[test]
fn a_tool_call_being_written_shows_it_in_its_panel_title() {
    let mut app = app_in(UiStyle::Premium);
    let mut bytes = 0;
    pour(&mut app, 1500, 28, 50, || {
        bytes += 4;
        ArgentEvent::ToolGenerating { name: "write_file".into(), bytes, delta: "abcd".into() }
    });
    let rows = screen(&app);
    let title = row_with(&rows, "Генерация: write_file");
    assert!(title.contains("tok/s"), "{title}");
    assert!(title.contains(" B | "), "the byte counter is still there: {title}");
}

#[test]
fn nothing_is_shown_before_there_is_enough_to_measure_or_while_a_tool_runs() {
    let mut app = app_in(UiStyle::Premium);
    app.handle_event_at(content(), back(50));
    assert!(!text(&screen(&app)).contains("tok/s"), "one piece is not a speed");

    let mut app = app_in(UiStyle::Premium);
    pour(&mut app, 1500, 28, 50, content);
    app.handle_event(ArgentEvent::ToolStart { name: "read_file".into(), args: HashMap::new() });
    let rows = screen(&app);
    assert!(rows.iter().any(|r| r.contains("read_file")), "the tool row is drawn:\n{}", text(&rows));
    assert!(!text(&rows).contains("tok/s"), "a running tool is not the model writing:\n{}", text(&rows));
}

// ── what it measures ─────────────────────────────────────────────────────────

#[test]
fn it_is_timed_by_when_the_output_arrived_not_by_when_it_was_handled() {
    let mut app = App::new();
    let t0 = Instant::now();
    // Handled in one go, but it arrived over five seconds: 20 pieces a second, 21.6 tok/s.
    for i in 0..100u64 {
        app.handle_event_at(content(), t0 + Duration::from_millis(i * 50));
    }
    let r = app.speed.reading(t0 + Duration::from_millis(4950)).expect("a reading");
    assert!((r.tokens_per_s - 21.6).abs() < 0.6, "{r:?}");
}

#[test]
fn the_providers_count_corrects_it_and_the_tilde_goes() {
    let mut app = app_in(UiStyle::Premium);
    // A request ten seconds ago, which the provider then counted as 1.5 times our estimate.
    pour(&mut app, 12_000, 100, 50, content);
    app.handle_event_at(
        ArgentEvent::Usage { tokens: 0, max: 0, percent: 0.0, completion: Some(162) },
        back(7_000),
    );
    assert!(app.speed.is_calibrated());
    pour(&mut app, 1500, 28, 50, content);
    let rows = screen(&app);
    let live = row_with(&rows, "Генерация...");
    assert!(live.contains("tok/s") && !live.contains('~'), "{live}");
}

#[test]
fn a_provider_report_does_not_blank_the_context_gauge() {
    let mut app = App::new();
    app.status.ctx_tokens = 4000;
    app.status.ctx_max = 32_000;
    app.status.ctx_percent = 12.5;
    app.handle_event(ArgentEvent::Usage { tokens: 0, max: 0, percent: 0.0, completion: Some(300) });
    assert_eq!(app.status.ctx_tokens, 4000);
    assert_eq!(app.status.ctx_max, 32_000);
    assert_eq!(app.status.ctx_percent, 12.5);
}

#[test]
fn what_was_learned_belongs_to_the_model_it_was_learned_from() {
    let state = |model: &str| HashMap::from([("model".to_string(), model.to_string()), ("provider".to_string(), "p".to_string())]);
    let mut app = App::new();
    app.update_status(state("model-a"));
    pour(&mut app, 12_000, 100, 50, content);
    app.handle_event_at(ArgentEvent::Usage { tokens: 0, max: 0, percent: 0.0, completion: Some(162) }, back(7_000));
    assert!(app.speed.is_calibrated());

    app.update_status(state("model-a"));
    assert!(app.speed.is_calibrated(), "the same model: nothing changes");
    app.update_status(state("model-b"));
    assert!(!app.speed.is_calibrated(), "another tokenizer: start over");
}
