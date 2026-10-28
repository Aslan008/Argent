//! The premium transcript: every kind of message has its own shape — your message a
//! bubble at the right edge, a thought quiet and dotted, a tool call a card with a
//! rail and a fill, one answer header per run — and copying and selecting take only
//! the text, never the dressing.

use std::collections::HashMap;

use argent_tui::app::{App, ApprovalData, MessageItem};
use argent_tui::theme::{Palette, UiStyle};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::Terminal;
use serde_json::json;

fn premium_app() -> App {
    let mut app = App::new();
    app.ui.set_style(UiStyle::Premium);
    app
}

fn draw(app: &App, w: u16, h: u16) -> Buffer {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    term.backend().buffer().clone()
}

fn row(buf: &Buffer, y: u16) -> String {
    (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>()
}

fn rows(buf: &Buffer) -> Vec<String> {
    (0..buf.area.height).map(|y| row(buf, y)).collect()
}

fn find(buf: &Buffer, needle: &str) -> u16 {
    rows(buf)
        .iter()
        .position(|r| r.contains(needle))
        .unwrap_or_else(|| panic!("`{needle}` is not on screen:\n{}", rows(buf).join("\n"))) as u16
}

fn col_of(r: &str, needle: &str) -> u16 {
    let byte = r.find(needle).unwrap_or_else(|| panic!("`{needle}` is not in {r:?}"));
    r[..byte].chars().count() as u16
}

fn tool(name: &str, args: &[(&str, &str)], result: &str) -> MessageItem {
    MessageItem::Tool {
        name: name.into(),
        args: args.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<HashMap<_, _>>(),
        result: Some(result.into()),
        duration_secs: Some(0.028),
        expanded: false,
        is_running: false,
    }
}

fn p() -> Palette {
    Palette::default()
}

// ── your message ─────────────────────────────────────────────────────────────

#[test]
fn your_message_is_a_bubble_at_the_right_edge() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("почини меню".into()));
    let buf = draw(&app, 60, 20);
    let y = find(&buf, "почини меню");
    let r = row(&buf, y);
    let x = col_of(&r, "почини меню");
    assert!(x > 40, "the text is at the right, not at the left: column {x}");
    assert_eq!(buf[(x, y)].bg, p().user_bg, "the bubble is tinted");
    assert_eq!(buf[(0, y)].bg, ratatui::style::Color::Reset, "the left of the row is the page");
    assert_eq!(buf[(58, y)].symbol(), "▐", "a slim bar closes it");
    assert_eq!(buf[(58, y)].fg, p().accent);
    assert_eq!(buf[(59, y)].symbol(), " ", "the last column stays free for the scrollbar");
}

#[test]
fn the_bubble_has_padding_rows_and_no_label() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("hello".into()));
    let buf = draw(&app, 60, 20);
    let y = find(&buf, "hello");
    assert!(!rows(&buf).iter().any(|r| r.contains("Вы")), "the bubble says who spoke by where it is");
    assert_eq!(buf[(58, y - 1)].symbol(), "▐", "padding above");
    assert_eq!(buf[(58, y + 1)].symbol(), "▐", "padding below");
    assert_eq!(buf[(50, y - 1)].bg, p().user_bg);
}

#[test]
fn a_long_message_wraps_inside_a_bubble_that_stays_off_the_left_margin() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("слово ".repeat(30)));
    let buf = draw(&app, 40, 24);
    let body: Vec<u16> = (0..24).filter(|&y| row(&buf, y).contains("слово")).collect();
    assert!(body.len() >= 4, "wrapped over several rows: {body:?}");
    for y in body {
        assert_eq!(buf[(38, y)].symbol(), "▐");
        assert_eq!(buf[(0, y)].bg, ratatui::style::Color::Reset);
        assert!(col_of(&row(&buf, y), "слово") >= 6, "the bubble is at most four fifths wide");
    }
}

#[test]
fn a_short_message_makes_a_short_bubble() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("да".into()));
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "да");
    assert!(col_of(&row(&buf, y), "да") > 70, "right-aligned however short it is");
}

#[test]
fn the_rewind_button_sits_under_the_bubble_at_the_right() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("go".into()));
    app.status.can_rewind = true;
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "Вернуться сюда");
    assert!(col_of(&row(&buf, y), "↩") > 40, "{:?}", row(&buf, y));
    assert_eq!(app.get_clicked_action(60, y), Some(argent_tui::app::ClickAction::RewindLast));
}

// ── a tool call ──────────────────────────────────────────────────────────────

#[test]
fn a_finished_tool_is_a_two_row_card_with_a_rail_and_a_fill() {
    let mut app = premium_app();
    app.messages.push(tool("read_file", &[("file_path", "agent.py")], "a\nb"));
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "read_file");
    let head = row(&buf, y);
    assert!(head.starts_with("   ▍ ✓ ▸ read_file"), "{head:?}");
    assert!(head.contains("28ms"), "the duration is on the right: {head:?}");
    assert_eq!(buf[(3, y)].symbol(), "▍");
    assert_eq!(buf[(3, y)].fg, p().tool);
    assert_eq!(buf[(40, y)].bg, p().surface, "the card is filled to the edge");
    assert_eq!(buf[(79, y)].bg, p().surface);
    let next = row(&buf, y + 1);
    assert!(next.contains("↳ 2 строк"), "{next:?}");
    assert_eq!(buf[(40, y + 1)].bg, p().surface);
}

#[test]
fn a_failed_tool_is_red_and_stays_collapsed_until_opened() {
    let mut app = premium_app();
    app.messages.push(tool("run_command", &[("command", "make")], "Error: exit 2\nboom on line 3"));
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "run_command");
    assert!(row(&buf, y).contains("✗"), "{:?}", row(&buf, y));
    assert!(row(&buf, y).contains("failed"));
    assert!(row(&buf, y).contains("▸"), "a failure starts folded");
    assert_eq!(buf[(3, y)].fg, p().danger, "the rail is red");
    assert_eq!(buf[(40, y)].bg, p().danger_soft, "the header is tinted red");
    assert!(!rows(&buf).iter().any(|r| r.contains("boom on line 3")), "output is hidden");

    let mut opened = premium_app();
    let mut t = tool("run_command", &[("command", "make")], "Error: exit 2\nboom on line 3");
    if let MessageItem::Tool { expanded, .. } = &mut t {
        *expanded = true;
    }
    opened.messages.push(t);
    let buf = draw(&opened, 80, 20);
    let out = find(&buf, "boom on line 3");
    assert_eq!(buf[(3, out)].fg, p().danger);
    assert_eq!(buf[(60, out)].bg, p().code, "the output is on the darker body");
}

#[test]
fn an_opened_command_shows_the_whole_command_not_only_the_clipped_header() {
    let mut app = premium_app();
    let tail = "FULL_COMMAND_TAIL";
    let command = format!("echo {} {tail}", "x".repeat(180));
    let mut t = tool("run_command", &[("command", &command)], "ok");
    if let MessageItem::Tool { expanded, .. } = &mut t {
        *expanded = true;
    }
    app.messages.push(t);
    let buf = draw(&app, 80, 30);
    let all = rows(&buf).join("\n");
    assert!(all.contains(tail), "the full command must be visible when open:\n{all}");
}

#[test]
fn an_opened_tool_shows_its_output_and_says_where_the_window_is() {
    let mut app = premium_app();
    let long: String = (0..40).map(|i| format!("line {i}\n")).collect();
    let mut t = tool("read_file", &[("file_path", "big.txt")], &long);
    if let MessageItem::Tool { expanded, .. } = &mut t {
        *expanded = true;
    }
    app.messages.push(t);
    let buf = draw(&app, 80, 40);
    assert!(rows(&buf).iter().any(|r| r.contains("line 0")));
    assert!(rows(&buf).iter().any(|r| r.contains("line 14")));
    assert!(!rows(&buf).iter().any(|r| r.contains("line 15")), "the REPL's own cap of fifteen rows");
    // The rest is not thrown away any more: the card is a window on the whole result,
    // and the line says where in it the person is.
    assert!(
        rows(&buf).iter().any(|r| r.contains("из 40")),
        "the window says how big the whole is"
    );
}

#[test]
fn a_running_tool_is_the_first_row_of_its_card() {
    let mut app = premium_app();
    app.messages.push(MessageItem::Tool {
        name: "grep_search".into(),
        args: [("query".to_string(), json!("needle"))].into_iter().collect(),
        result: None,
        duration_secs: None,
        expanded: false,
        is_running: true,
    });
    app.is_busy = true;
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "grep_search");
    assert!(row(&buf, y).starts_with("   ▍ "), "{:?}", row(&buf, y));
    assert_eq!(buf[(3, y)].fg, p().tool);
    assert_eq!(buf[(60, y)].bg, p().surface);
}

#[test]
fn consecutive_cards_sit_tight_one_under_the_other() {
    let mut app = premium_app();
    app.messages.push(tool("read_file", &[("file_path", "a.py")], "x"));
    app.messages.push(tool("read_file", &[("file_path", "b.py")], "y"));
    let buf = draw(&app, 80, 20);
    let a = find(&buf, "a.py");
    let b = find(&buf, "b.py");
    assert_eq!(b - a, 2, "header, summary, header: a run of calls is one block:\n{}", rows(&buf).join("\n"));
    assert_eq!(buf[(40, a + 1)].bg, p().surface, "no gap of the page's colour between the two");
}

#[test]
fn a_call_sits_tight_under_the_thought_that_led_to_it() {
    let mut app = premium_app();
    app.messages.push(MessageItem::assistant("", "I should look at the file."));
    app.messages.push(tool("read_file", &[("file_path", "a.py")], "x"));
    let buf = draw(&app, 80, 20);
    let t = find(&buf, "Анализ");
    let c = find(&buf, "read_file");
    assert_eq!(c, t + 1, "no gap between the thought and its call");
}

// ── thoughts ─────────────────────────────────────────────────────────────────

#[test]
fn a_thought_is_quiet_dotted_and_unfilled() {
    let mut app = premium_app();
    app.messages.push(MessageItem::assistant("", "First look.\nThe list is built too early."));
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "Анализ");
    let r = row(&buf, y);
    assert!(r.starts_with(" ┊ ▸ ◇ Анализ"), "{r:?}");
    assert!(r.contains("The list is built too early."), "a preview of the conclusion: {r:?}");
    assert_eq!(buf[(40, y)].bg, ratatui::style::Color::Reset, "no fill: a thought is not a card");
    assert_eq!(buf[(1, y)].fg, p().think);
}

#[test]
fn an_opened_thought_is_the_text_behind_the_dotted_rail() {
    let mut app = premium_app();
    app.messages.push(MessageItem::Assistant {
        content: String::new(),
        thinking: "line one\nline two".into(),
        thinking_expanded: true,
        is_streaming_thinking: false,
        is_streaming_content: false,
        thinking_duration: Some(2.0),
        thinking_start: None,
        stream: Default::default(),
    });
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "line one");
    assert!(row(&buf, y).starts_with(" ┊ "), "{:?}", row(&buf, y));
    assert!(row(&buf, find(&buf, "Анализ")).contains("2.0s"));
}

// ── answers ──────────────────────────────────────────────────────────────────

#[test]
fn an_answer_before_a_tool_call_is_not_labelled_as_incomplete() {
    let mut app = premium_app();
    app.messages.push(MessageItem::assistant("Сначала посмотрю файл.", ""));
    app.messages.push(tool("read_file", &[("file_path", "a.py")], "x"));
    app.messages.push(MessageItem::assistant("Готово.", ""));
    let buf = draw(&app, 80, 30);
    let all = rows(&buf);
    let headers: Vec<&String> = all.iter().filter(|r| r.contains("✦ Argent")).collect();
    assert_eq!(headers.len(), 2, "{all:#?}");
    assert!(
        headers.iter().all(|r| !r.contains("промежуточный")),
        "the old chip is gone: {headers:?}"
    );
}

#[test]
fn two_adjacent_assistant_items_share_one_header() {
    let mut app = premium_app();
    app.messages.push(MessageItem::assistant("Первая часть.", ""));
    app.messages.push(MessageItem::assistant("Продолжение.", ""));
    let buf = draw(&app, 80, 30);
    let all = rows(&buf);
    let headers = all.iter().filter(|r| r.contains("✦ Argent")).count();
    assert_eq!(headers, 1, "a re-sent segment is not a second speaker: {all:#?}");
    assert!(all.iter().any(|r| r.contains("Продолжение.")));
}

#[test]
fn the_spend_and_the_ignored_settings_sit_in_the_right_corner() {
    let mut app = premium_app();
    app.status.session_summary = "sum 12.3k tok · $0.0123".into();
    app.status.inert_label = "n/a: thinking, depth".into();
    let buf = draw(&app, 120, 12);
    let bar = rows(&buf).last().cloned().unwrap();
    let spend_at = bar.find("sum 12.3k tok").unwrap_or_else(|| panic!("no spend: {bar:?}"));
    let inert_at = bar.find("n/a: thinking").unwrap_or_else(|| panic!("no n/a note: {bar:?}"));
    assert!(spend_at < inert_at, "{bar:?}");
    assert!(inert_at > bar.len() / 2, "the note belongs in the right corner: {bar:?}");
    assert!(bar.trim_end().ends_with("/help"), "{bar:?}");
}

#[test]
fn the_spend_falls_back_to_the_raw_totals() {
    let mut app = premium_app();
    app.status.session_prompt = 800.0;
    app.status.session_completion = 200.0;
    app.status.session_cost = 0.0123;
    assert_eq!(argent_tui::ui::session_spend(&app.status), "sum 1.0k tok · $0.0123");
}

#[test]
fn nothing_is_invented_before_anything_was_spent() {
    let app = premium_app();
    assert_eq!(argent_tui::ui::session_spend(&app.status), "");
    let bar = rows(&draw(&app, 120, 12)).last().cloned().unwrap();
    assert!(!bar.contains("n/a:"), "no capability claim was made: {bar:?}");
}

#[test]
fn hovering_the_context_gauge_explains_the_budget() {
    let mut app = premium_app();
    app.status.ctx_tokens = 12_300;
    app.status.ctx_max = 32_768;
    app.status.ctx_percent = 37.5;
    app.status.ctx_breakdown =
        "Системный промпт   3.1k tok\nИнструменты (52)   2.4k tok\nИстория разговора   6.8k tok\n\nВсего   12.3k / 32.8k tok  ·  38%"
            .to_string();
    let first = draw(&app, 120, 20);
    let bar = rows(&first).last().cloned().unwrap();
    let col = bar.find("ctx ").expect("the ctx chip") as u16;
    app.mouse_pos = Some((col + 1, 19));

    let buf = draw(&app, 120, 20);
    let all = rows(&buf).join("\n");
    assert!(all.contains("Системный промпт"), "the breakdown is shown on hover:\n{all}");
    assert!(all.contains("Инструменты (52)"));
    assert!(all.contains("Всего"), "{all}");
}

#[test]
fn a_tool_card_offers_only_the_chevron_not_a_tooltip() {
    let mut app = premium_app();
    app.messages.push(tool("read_file", &[("file_path", "a.py")], "x"));
    let first = draw(&app, 100, 20);
    let y = find(&first, "read_file");
    app.mouse_pos = Some((20, y));
    let buf = draw(&app, 100, 20);
    let all = rows(&buf).join("\n");
    assert!(!all.contains("раскрыть/свернуть результат инструмента"), "no card tooltip:\n{all}");
    assert!(row(&buf, y).contains('▸'), "the chevron is the affordance: {all}");
}

#[test]
fn an_approval_is_a_framed_banner_that_cannot_be_missed() {
    let mut app = premium_app();
    app.active_approval = Some(ApprovalData {
        id: 1,
        action: "rm -rf build".into(),
        destructive: true,
        grant_key: None,
        prompt: None,
        selected: 0,
    });
    let buf = draw(&app, 100, 24);
    let all = rows(&buf).join("\n");
    assert!(all.contains("ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ"), "{all}");
    assert!(all.contains("ОПАСНОЕ ДЕЙСТВИЕ"), "a destructive call says so: {all}");
    assert!(all.contains('╭') && all.contains('╯'), "framed: {all}");
    assert!(all.contains("Esc — отклонить"), "{all}");
}

#[test]
fn a_stopped_turn_says_so_at_once() {
    let mut app = premium_app();
    app.messages.push(MessageItem::assistant("partial", ""));
    app.is_busy = true;
    app.is_cancelling = true;
    let buf = draw(&app, 80, 20);
    let all = rows(&buf).join("\n");
    assert!(all.contains("Останавливаю"), "the old spinner must not keep claiming generation:\n{all}");
}

#[test]
fn a_shorter_second_frame_does_not_leave_ghost_text() {
    let mut app = premium_app();
    app.messages.push(MessageItem::System("a long line that will disappear".into()));
    let mut term = Terminal::new(TestBackend::new(80, 20)).unwrap();
    term.draw(|f| draw_ui(f, &app)).unwrap();
    app.messages.clear();
    app.messages.push(MessageItem::System("x".into()));
    term.draw(|f| draw_ui(f, &app)).unwrap();
    let all = rows(term.backend().buffer()).join("\n");
    assert!(!all.contains("will disappear"), "stale characters survived:\n{all}");
}

#[test]
fn the_final_answer_header_carries_the_accent() {
    let mut app = premium_app();
    app.messages.push(MessageItem::assistant("Готово.", ""));
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "✦ Argent");
    let x = col_of(&row(&buf, y), "✦");
    assert_eq!(buf[(x, y)].fg, p().accent);
}

// ── changes and errors ───────────────────────────────────────────────────────

#[test]
fn a_diff_tints_added_and_removed_lines() {
    let mut app = premium_app();
    app.messages.push(MessageItem::Diff {
        file: "src/f.py".into(),
        diff: "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n ctx".into(),
    });
    let buf = draw(&app, 80, 20);
    let head = find(&buf, "diff f.py");
    assert!(row(&buf, head).contains("+1") && row(&buf, head).contains("−1"), "{:?}", row(&buf, head));
    let added = find(&buf, "+new");
    let removed = find(&buf, "-old");
    assert_eq!(buf[(40, added)].bg, p().ok_soft);
    assert_eq!(buf[(40, removed)].bg, p().danger_soft);
    assert_eq!(buf[(3, added)].fg, p().ok);
    assert_eq!(buf[(3, removed)].fg, p().danger);
}

#[test]
fn an_error_is_a_red_card_with_one_prefix_and_the_hint_under_it() {
    let mut app = premium_app();
    app.messages.push(MessageItem::Error { text: "boom".into(), hint: Some("Try again later".into()) });
    let buf = draw(&app, 80, 20);
    let y = find(&buf, "Ошибка: boom");
    assert!(row(&buf, y).starts_with("   ▍ ✗ "), "{:?}", row(&buf, y));
    assert_eq!(buf[(40, y)].bg, p().danger_soft);
    assert!(row(&buf, y + 1).contains("Try again later"));
    assert!(!rows(&buf).join("\n").contains("Ошибка: Ошибка"));
}

// ── copying and selecting ────────────────────────────────────────────────────

#[test]
fn copying_a_whole_turn_leaves_the_dressing_behind() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("почини меню".into()));
    app.messages.push(tool("read_file", &[("file_path", "a.py")], "x"));
    app.messages.push(MessageItem::assistant("Готово.", ""));
    draw(&app, 80, 30);
    assert!(app.select_all_transcript());
    let text = app.selected_transcript_text().expect("something selected");
    assert!(text.contains("почини меню"), "{text}");
    assert!(!text.contains('▐') && !text.contains('▍') && !text.contains("Вы"), "no dressing: {text}");
    assert!(text.contains("read_file: a.py"));
    assert!(text.contains("Готово."));
    for l in text.lines() {
        assert!(!l.starts_with(' ') || l.trim_start().starts_with('↳'), "left margin leaked: {l:?}");
    }
}

#[test]
fn the_highlight_covers_text_not_padding() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("select me".into()));
    draw(&app, 60, 20);
    assert!(app.select_all_transcript());
    let buf = draw(&app, 60, 20);
    let y = find(&buf, "select me");
    let x = col_of(&row(&buf, y), "select me");
    assert_eq!(buf[(x, y)].bg, p().selection, "the text is highlighted");
    assert_ne!(buf[(x - 1, y)].bg, p().selection, "not the padding at its side");
    assert_ne!(buf[(0, y)].bg, p().selection, "not the empty left of the row");
}

// ── the live region ──────────────────────────────────────────────────────────

#[test]
fn waiting_for_the_model_is_a_spinner_under_a_blank_row() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("go".into()));
    app.is_busy = true;
    let buf = draw(&app, 80, 24);
    let y = find(&buf, "Ожидание ответа");
    assert_eq!(row(&buf, y - 1).trim(), "", "set apart from your message");
}

#[test]
fn a_streaming_thought_sits_behind_the_dotted_rail() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("go".into()));
    app.messages.push(MessageItem::Assistant {
        content: String::new(),
        thinking: "weighing the options".into(),
        thinking_expanded: false,
        is_streaming_thinking: true,
        is_streaming_content: false,
        thinking_duration: None,
        thinking_start: None,
        stream: Default::default(),
    });
    app.is_busy = true;
    let buf = draw(&app, 80, 24);
    let y = find(&buf, "weighing the options");
    assert!(row(&buf, y).starts_with(" ┊ "), "{:?}", row(&buf, y));
    assert!(rows(&buf).iter().any(|r| r.contains("Анализ задачи")));
}

#[test]
fn the_streaming_tail_lines_up_with_the_committed_answer() {
    let mut app = premium_app();
    app.handle_event(argent_tui::events::ArgentEvent::Content("Первый абзац.\n\nвторой, ещё идёт".into()));
    app.is_busy = true;
    let buf = draw(&app, 80, 24);
    let committed = find(&buf, "Первый абзац.");
    let tail = find(&buf, "второй, ещё идёт");
    assert_eq!(col_of(&row(&buf, committed), "Первый"), col_of(&row(&buf, tail), "второй"));
}

// ── robustness ───────────────────────────────────────────────────────────────

#[test]
fn nothing_panics_or_overflows_at_any_size() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("слово ".repeat(40)));
    app.messages.push(MessageItem::assistant("текст", "мысль"));
    app.messages.push(tool("run_command", &[("command", &"x".repeat(200))], &"Error: ".repeat(40)));
    app.messages.push(MessageItem::Diff { file: "f".into(), diff: format!("+{}\n-{}", "y".repeat(300), "z".repeat(300)) });
    app.messages.push(MessageItem::Error { text: "e".repeat(300), hint: Some("h".repeat(300)) });
    for w in [8u16, 10, 16, 24, 40, 80, 160] {
        for h in [4u16, 6, 10, 24] {
            let buf = draw(&app, w, h);
            for y in 0..buf.area.height {
                assert_eq!(row(&buf, y).chars().count(), w as usize);
            }
        }
    }
}

#[test]
fn the_classic_look_is_untouched_by_all_of_this() {
    let mut app = App::new();
    app.messages.push(MessageItem::User("hello".into()));
    app.messages.push(tool("read_file", &[("file_path", "a.py")], "x"));
    let buf = draw(&app, 60, 12);
    let all = rows(&buf).join("\n");
    assert!(row(&buf, 0).starts_with("❯ hello"));
    assert!(all.contains("│ read_file: a.py"), "{all}");
    assert!(!all.contains('▐') && !all.contains('▍') && !all.contains('┊'));
}

#[test]
fn a_long_premium_session_costs_a_screenful_not_the_history() {
    let mut app = premium_app();
    for i in 0..1500 {
        app.messages.push(MessageItem::User(format!("question {i}")));
        app.messages.push(MessageItem::assistant(format!("answer {i}\n\nmore text"), format!("thought {i}")));
        app.messages.push(tool("read_file", &[("file_path", "a.py")], "x\ny"));
    }
    draw(&app, 100, 40); // cold: renders what is visible
    let t = std::time::Instant::now();
    for _ in 0..5 {
        draw(&app, 100, 40);
    }
    let per_frame = t.elapsed() / 5;
    assert!(per_frame < std::time::Duration::from_millis(250), "a frame took {per_frame:?}");
}

// ── a question of several lines ──────────────────────────────────────────────

#[test]
fn a_question_with_line_breaks_is_shown_whole() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("run it".into()));
    app.active_select = Some(argent_tui::app::SelectModalData::new(
        "q".into(),
        "Замечания критика:\n- первое замечание\n- второе замечание\n\nПродолжить?".into(),
        vec!["Да".into(), "Нет".into()],
        None,
    ));
    let buf = draw(&app, 70, 24);
    for needle in ["Замечания критика", "первое замечание", "второе замечание", "Продолжить?"] {
        assert!(rows(&buf).iter().any(|r| r.contains(needle)), "{needle} is on the screen:\n{}", rows(&buf).join("\n"));
    }
    assert!(rows(&buf).iter().any(|r| r.contains("Да")) && rows(&buf).iter().any(|r| r.contains("Нет")));
}
