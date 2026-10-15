//! A chat the user has scrolled back stays exactly where it is while output arrives
//! (tool calls and their results, streamed answers, a growing prompt, a question);
//! only a chat the user has scrolled to its very end follows new output.
//!
//! The question and the prompt sit right under the newest message while the user is
//! at the end of the chat, and at the bottom of the window while they are reading
//! further up.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use argent_tui::app::App;
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

const W: u16 = 60;
const H: u16 = 20;

fn long_chat() -> App {
    let mut app = App::new();
    for i in 0..400 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    settle(&app);
    app
}

fn rows(app: &App) -> Vec<String> {
    let mut term = Terminal::new(TestBackend::new(W, H)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    let buf = term.backend().buffer().clone();
    (0..buf.area.height)
        .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>())
        .collect()
}

/// The screen without its last column, where the scrollbar thumb moves as the chat grows.
fn body(app: &App) -> Vec<String> {
    rows(app).into_iter().map(|l| l.chars().take(W as usize - 1).collect()).collect()
}

fn settle(app: &App) {
    rows(app);
}

/// One frame, then the offset the frame measured is taken in, as the event loop does.
/// The part of a frame that is the chat, not the prompt and toolbar under it: a busy
/// agent hides an empty prompt, which gives the chat one more row at the bottom.
const TOP: usize = 14;

fn frame(app: &mut App) -> Vec<String> {
    let b = body(app);
    app.sync_scroll();
    b
}

fn scrolled_back() -> App {
    let mut app = long_chat();
    let t = Instant::now();
    for i in 0..4u64 {
        app.wheel(true, t + Duration::from_millis(500 * i));
        frame(&mut app);
    }
    assert!(app.scroll_offset > 0, "the chat is scrolled back");
    app
}

fn index_of(rows: &[String], needle: &str) -> usize {
    rows.iter().rposition(|r| r.contains(needle)).unwrap_or_else(|| panic!("`{needle}` is not on screen:\n{}", rows.join("\n")))
}

// ── a scrolled-back chat stays put ───────────────────────────────────────────

#[test]
fn output_arriving_below_does_not_move_what_is_being_read() {
    let mut app = scrolled_back();
    let before = frame(&mut app);
    for round in 0..3 {
        for i in 0..10 {
            app.handle_event(ArgentEvent::System(format!("late {round}-{i}")));
        }
        assert_eq!(frame(&mut app), before, "round {round}: the screen moved");
    }
    assert!(app.scroll_offset > 0 && !app.auto_scroll, "still scrolled back, not following");
}

#[test]
fn a_tool_call_and_its_result_do_not_move_it_either() {
    let mut app = scrolled_back();
    let before = frame(&mut app);
    app.handle_event(ArgentEvent::ToolStart { name: "read_file".into(), args: HashMap::new() });
    assert_eq!(frame(&mut app)[..TOP], before[..TOP], "a tool started");
    app.handle_event(ArgentEvent::ToolEnd { name: "read_file".into(), result: "a\nb\nc\nd\ne\nf".into(), duration_secs: Some(0.2) });
    assert_eq!(frame(&mut app)[..TOP], before[..TOP], "its result arrived");
    app.handle_event(ArgentEvent::Content("Here is the answer.\n\nSecond paragraph.\n\nThird.".into()));
    assert_eq!(frame(&mut app)[..TOP], before[..TOP], "an answer streamed in");
    app.handle_event(ArgentEvent::Done);
    assert_eq!(frame(&mut app)[..TOP], before[..TOP], "the turn ended");
}

#[test]
fn a_tool_that_streams_its_arguments_and_is_replaced_does_not_move_it() {
    let mut app = scrolled_back();
    let before = frame(&mut app);
    for i in 0..20 {
        app.handle_event(ArgentEvent::ToolGenerating { name: "write_file".into(), bytes: i * 40, delta: "x".repeat(40) });
        assert_eq!(frame(&mut app)[..TOP], before[..TOP], "argument chunk {i}");
    }
    app.handle_event(ArgentEvent::ToolStart { name: "write_file".into(), args: HashMap::new() });
    assert_eq!(frame(&mut app)[..TOP], before[..TOP], "the card turned into the running tool");
}

#[test]
fn the_prompt_growing_while_reading_does_not_move_the_chat_either() {
    let mut app = scrolled_back();
    let before = frame(&mut app);
    app.insert_str("one\ntwo\nthree\nfour\nfive");
    let after = frame(&mut app);
    // The bottom rows give way to the taller prompt; what is shown from the top stays.
    let kept = 6;
    assert_eq!(&after[..kept], &before[..kept], "the top of what is being read moved");
}

#[test]
fn reading_position_survives_a_message_being_inserted_above_the_running_tool() {
    let mut app = App::new();
    for i in 0..60 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    app.handle_event(ArgentEvent::ToolStart { name: "read_file".into(), args: HashMap::new() });
    settle(&app);
    let t = Instant::now();
    for i in 0..6u64 {
        app.wheel(true, t + Duration::from_millis(500 * i));
        frame(&mut app);
    }
    let before = frame(&mut app);
    // A notice printed while the tool runs lands above its placeholder.
    app.handle_event(ArgentEvent::Notice("something happened".into()));
    app.handle_event(ArgentEvent::Notice("and again".into()));
    assert_eq!(frame(&mut app), before);
}

// ── following the end ────────────────────────────────────────────────────────

#[test]
fn at_the_very_end_new_output_is_followed() {
    let mut app = long_chat();
    app.handle_event(ArgentEvent::System("brand new line".into()));
    let screen = frame(&mut app);
    assert!(screen.iter().any(|r| r.contains("brand new line")), "{}", screen.join("\n"));
    assert_eq!(app.scroll_offset, 0);
}

#[test]
fn scrolling_to_the_end_by_hand_starts_following_again() {
    let mut app = scrolled_back();
    app.handle_event(ArgentEvent::System("while away".into()));
    frame(&mut app);
    assert!(app.scroll_offset > 0, "still where it was");

    // The user takes the chat to its end themselves ...
    app.scroll_down(usize::MAX);
    assert!(frame(&mut app).iter().any(|r| r.contains("while away")), "and finds what came in meanwhile");
    assert_eq!(app.scroll_offset, 0);
    assert!(app.auto_scroll);
    // ... and from then on it follows.
    app.handle_event(ArgentEvent::System("after returning".into()));
    assert!(frame(&mut app).iter().any(|r| r.contains("after returning")));
}

#[test]
fn nothing_but_the_user_takes_a_scrolled_back_chat_to_the_end() {
    let mut app = scrolled_back();
    let offset = app.scroll_offset;
    for _ in 0..40 {
        app.handle_event(ArgentEvent::System("noise".into()));
        frame(&mut app);
    }
    app.handle_event(ArgentEvent::Done);
    frame(&mut app);
    assert!(app.scroll_offset >= offset, "it only ever drifts further from the end as output piles up: {} < {offset}", app.scroll_offset);
    assert!(!app.auto_scroll);
}

// ── the question and the prompt ──────────────────────────────────────────────

fn ask(app: &mut App) {
    app.handle_event(ArgentEvent::ApprovalRequest {
        id: 1,
        action: "rm build".into(),
        destructive: true,
        grant_key: None,
        prompt: None,
    });
}

#[test]
fn at_the_end_the_question_comes_right_after_the_last_message() {
    let mut app = App::new();
    for i in 0..5 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    ask(&mut app);
    let screen = frame(&mut app);
    let last = index_of(&screen, "line 004");
    let question = index_of(&screen, "Разрешить");
    assert!(question > last && question - last <= 4, "the question is not under the last message:\n{}", screen.join("\n"));
    // and in a chat longer than the window, the newest message is still the one above it
    let mut app = long_chat();
    ask(&mut app);
    let screen = frame(&mut app);
    let last = index_of(&screen, "line 399");
    let question = index_of(&screen, "Разрешить");
    assert!(question > last && question - last <= 4, "{}", screen.join("\n"));
}

#[test]
fn while_reading_further_up_the_question_is_pinned_at_the_bottom_and_covers_the_chat() {
    let mut app = scrolled_back();
    let before = frame(&mut app);
    ask(&mut app);
    let screen = frame(&mut app);
    assert_eq!(index_of(&screen, "Разрешить"), H as usize - 2, "just above the toolbar:\n{}", screen.join("\n"));
    let kept = 6;
    assert_eq!(&screen[..kept], &before[..kept], "the question must not move the text being read");
    assert!(!screen.iter().any(|r| r.contains("line 399")), "still not at the end of the chat");
}

#[test]
fn the_prompt_follows_the_same_rule() {
    // At the end of a short chat it sits right under the last message ...
    let mut app = App::new();
    for i in 0..5 {
        app.handle_event(ArgentEvent::System(format!("line {i:03}")));
    }
    let screen = frame(&mut app);
    assert_eq!(index_of(&screen, "❯"), index_of(&screen, "line 004") + 1, "{}", screen.join("\n"));
    // ... and while reading further up it is at the bottom of the window.
    let app = scrolled_back();
    let screen = body(&app);
    assert!(index_of(&screen, "❯") >= H as usize - 4, "{}", screen.join("\n"));
}
