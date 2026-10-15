//! The premium look of the native widgets: the prompt, the suggestion list, the
//! toolbar, what the user wrote, the "newest" chip — and that the classic look,
//! which the parity tests hold to the REPL, is left exactly as it was.

use argent_tui::app::{App, ClickAction, CompletionItem, MessageItem};
use argent_tui::events::ArgentEvent;
use argent_tui::theme::{Palette, UiStyle};
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::{Color, Modifier};
use ratatui::Terminal;

fn premium_app() -> App {
    let mut app = App::new();
    app.ui.set_style(UiStyle::Premium);
    app
}

fn draw(app: &App, w: u16, h: u16) -> (Buffer, Option<(u16, u16)>) {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| draw_ui(f, app)).unwrap();
    let cursor = term.get_cursor_position().ok().map(|p| (p.x, p.y));
    (term.backend().buffer().clone(), cursor)
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

/// The screen row of the suggestion list that shows `name`. The prompt row shows it as
/// well once Tab has previewed it, so `find` alone would land there.
fn list_row(buf: &Buffer, name: &str) -> u16 {
    rows(buf)
        .iter()
        .position(|r| r.starts_with(&format!(" ▸ {name}")) || r.starts_with(&format!("   {name}")))
        .unwrap_or_else(|| panic!("`{name}` is not in the list:\n{}", rows(buf).join("\n"))) as u16
}

fn command_app(names: &[&str]) -> App {
    let mut app = premium_app();
    for n in names {
        app.completions.push(CompletionItem { name: (*n).to_string(), help: format!("about {n}") });
    }
    app
}

fn surface() -> Color {
    Palette::default().surface
}

// ── the default is the REPL's look ───────────────────────────────────────────

#[test]
fn a_new_app_is_classic_and_draws_no_premium_chrome() {
    let app = App::new();
    assert_eq!(app.ui.style, UiStyle::Classic);
    let (buf, _) = draw(&app, 60, 12);
    let all = rows(&buf).join("\n");
    assert!(all.contains("❯ "), "the REPL prompt");
    assert!(!all.contains('▌') && !all.contains('▄'), "no premium panel marks:\n{all}");
}

// ── the prompt ───────────────────────────────────────────────────────────────

#[test]
fn the_premium_prompt_is_a_tinted_panel_with_an_accent_bar() {
    let app = premium_app();
    let (buf, _) = draw(&app, 60, 20);
    let y = find(&buf, "❯");
    assert!(row(&buf, y).starts_with("▌ ❯ "));
    assert_eq!(buf[(0, y)].fg, Palette::default().accent);
    assert_eq!(buf[(0, y)].bg, surface());
    assert_eq!(buf[(59, y)].bg, surface(), "the panel runs to the right edge");
    assert!(row(&buf, y).contains("Напишите задачу"), "a hint while the prompt is empty");
}

#[test]
fn the_panel_has_soft_edges_when_there_is_height_for_them() {
    let app = premium_app();
    let (tall, _) = draw(&app, 60, 20);
    let y = find(&tall, "❯");
    assert!(row(&tall, y - 1).starts_with("▄▄▄"));
    assert!(row(&tall, y + 1).starts_with("▀▀▀"));
    let (short, _) = draw(&app, 60, 10);
    assert!(!rows(&short).join("\n").contains('▄'), "a short terminal keeps its rows");
}

#[test]
fn the_cursor_sits_after_the_typed_text() {
    let mut app = premium_app();
    app.insert_str("привет");
    let (buf, cursor) = draw(&app, 60, 20);
    let y = find(&buf, "привет");
    assert_eq!(cursor, Some((4 + 6, y)), "bar, blank, glyph, blank, then six letters");
    assert!(!row(&buf, y).contains("Напишите задачу"), "the hint goes as soon as there is text");
}

#[test]
fn the_cursor_follows_a_wrapped_and_multiline_prompt() {
    let mut app = premium_app();
    app.insert_str("one\ntwo");
    let (buf, cursor) = draw(&app, 60, 20);
    let y = find(&buf, "two");
    assert_eq!(cursor, Some((4 + 3, y)));
    assert!(row(&buf, y).starts_with("▌   two"), "continuation rows keep the bar but not the glyph");
}

#[test]
fn selected_input_is_painted_with_the_selection_colour() {
    let mut app = premium_app();
    app.insert_str("select me");
    app.select_all_input();
    let (buf, _) = draw(&app, 60, 20);
    let y = find(&buf, "select me");
    assert_eq!(buf[(4, y)].bg, Palette::default().selection);
    assert_eq!(buf[(12, y)].bg, Palette::default().selection);
    assert_eq!(buf[(13, y)].bg, surface(), "the padding past the text is not selected");
}

#[test]
fn input_typed_while_the_agent_works_is_set_into_the_panel() {
    let mut app = premium_app();
    app.is_busy = true;
    app.insert_str("next task");
    let (buf, _) = draw(&app, 70, 20);
    let y = find(&buf, "Ввод на лету");
    assert!(row(&buf, y).starts_with("▌ "), "inside the panel");
    assert_eq!(buf[(0, y)].bg, surface());
}

// ── the suggestion list ──────────────────────────────────────────────────────

#[test]
fn the_list_lights_the_current_suggestion_and_names_the_keys() {
    let mut app = command_app(&["/help", "/history", "/model"]);
    app.insert_char('/');
    app.completion_next();
    let (buf, _) = draw(&app, 80, 24);
    let y = find(&buf, "/history");
    assert!(row(&buf, y).starts_with(" ▸ /history"), "the current item has the marker");
    assert_eq!(buf[(5, y)].bg, Palette::default().accent_soft);
    let other = find(&buf, "/model");
    assert_eq!(buf[(5, other)].bg, surface());
    let footer = row(&buf, find(&buf, "Shift+Tab"));
    assert!(footer.contains("Tab") && footer.contains("Enter") && footer.contains("Esc"));
    assert!(footer.trim_end().ends_with("2/3"), "which one of how many: {footer:?}");
}

#[test]
fn tab_keeps_the_list_open_and_moves_the_marker() {
    let mut app = command_app(&["/help", "/history", "/model"]);
    app.insert_str("/h");
    app.cycle_completion(true, false);
    let (buf, _) = draw(&app, 80, 24);
    assert!(row(&buf, list_row(&buf, "/help")).contains('▸'));
    app.cycle_completion(true, false);
    let (buf, _) = draw(&app, 80, 24);
    assert!(row(&buf, list_row(&buf, "/history")).contains('▸'));
    assert!(
        !row(&buf, list_row(&buf, "/help")).contains('▸'),
        "the first suggestion is still listed, but no longer marked"
    );
}

#[test]
fn a_long_list_scrolls_with_the_marker_and_counts() {
    let names: Vec<String> = (0..30).map(|i| format!("/cmd{i:02}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let mut app = command_app(&refs);
    app.insert_char('/');
    for _ in 0..20 {
        app.cycle_completion(true, false);
    }
    let (buf, _) = draw(&app, 80, 30);
    let y = list_row(&buf, "/cmd19");
    assert!(row(&buf, y).contains('▸'));
    assert!(row(&buf, find(&buf, "Shift+Tab")).trim_end().ends_with("20/30"));
    assert!(!rows(&buf).join("\n").contains("/cmd00"), "the window moved past the start");
}

#[test]
fn the_classic_list_is_untouched() {
    let mut app = App::new();
    app.completions.push(CompletionItem { name: "/help".into(), help: "h".into() });
    app.insert_char('/');
    let (buf, _) = draw(&app, 60, 12);
    let all = rows(&buf).join("\n");
    assert!(all.contains("/help") && !all.contains('▸') && !all.contains("Shift+Tab"));
}

// ── the toolbar ──────────────────────────────────────────────────────────────

#[test]
fn the_toolbar_is_chips_on_a_tinted_row() {
    let mut app = premium_app();
    app.status.mode = "CHAT".into();
    app.status.provider = "zai".into();
    app.status.model = "glm-5.1".into();
    let (buf, _) = draw(&app, 100, 20);
    let y = buf.area.height - 1;
    let bar = row(&buf, y);
    assert!(bar.contains(" CHAT ") && bar.contains("zai:glm-5.1") && bar.contains("/help"));
    let chip = bar.find("CHAT").unwrap() as u16;
    assert_eq!(buf[(chip, y)].bg, Palette::default().accent, "the mode is a filled chip");
    assert_eq!(buf[(99, y)].bg, surface(), "the row is tinted to the edge");
}

#[test]
fn the_toolbar_keeps_its_click_targets() {
    let mut app = premium_app();
    app.status.provider = "zai".into();
    app.status.model = "glm-5.1".into();
    app.status.thinking_level = "auto".into();
    let (buf, _) = draw(&app, 100, 20);
    let y = buf.area.height - 1;
    let x = row(&buf, y).find("zai:glm").unwrap() as u16;
    assert_eq!(app.get_clicked_action(x + 2, y), Some(ClickAction::OpenModelPicker));
    let t = row(&buf, y).find("think").unwrap() as u16;
    assert_eq!(app.get_clicked_action(t + 2, y), Some(ClickAction::CycleThinkingLevel));
}

#[test]
fn the_context_gauge_fills_and_changes_colour() {
    let mut app = premium_app();
    app.status.ctx_tokens = 9000;
    app.status.ctx_max = 10000;
    app.status.ctx_percent = 90.0;
    let (buf, _) = draw(&app, 120, 20);
    let y = buf.area.height - 1;
    let r = row(&buf, y);
    assert!(r.contains("▰▰▰▰▰▰") || r.contains("▰▰▰▰▰▱"), "{r}");
    let x = r.chars().position(|c| c == '▰').unwrap() as u16;
    assert_eq!(buf[(x, y)].fg, Palette::default().danger);
}

// ── what the user wrote ──────────────────────────────────────────────────────

#[test]
fn a_user_message_is_a_tinted_bubble_at_the_right_edge() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("почини меню".into()));
    let (buf, _) = draw(&app, 60, 20);
    let y = find(&buf, "почини меню");
    let x = row(&buf, y).find("почини").map(|b| row(&buf, y)[..b].chars().count() as u16).unwrap();
    assert!(x > 40, "right-aligned, not at the left: column {x}");
    assert_eq!(buf[(x, y)].bg, Palette::default().user_bg);
    assert_eq!(buf[(58, y)].symbol(), "▐", "closed by an accent bar");
    assert_eq!(buf[(58, y)].fg, Palette::default().accent);
    assert_ne!(buf[(0, y)].bg, Palette::default().user_bg, "the left of the row is the page");
}

#[test]
fn a_long_user_message_wraps_inside_the_bubble() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("слово ".repeat(30)));
    let (buf, _) = draw(&app, 40, 20);
    let bands: Vec<u16> = (0..20).filter(|&y| row(&buf, y).contains("слово")).collect();
    assert!(bands.len() >= 3, "wrapped over several rows: {bands:?}");
    for y in bands {
        assert_eq!(buf[(38, y)].symbol(), "▐");
    }
}

#[test]
fn copying_a_user_message_leaves_the_bar_behind() {
    let mut app = premium_app();
    app.messages.push(MessageItem::User("copy this exactly".into()));
    draw(&app, 60, 20);
    assert!(app.select_all_transcript());
    assert_eq!(app.selected_transcript_text().as_deref(), Some("copy this exactly"));
}

#[test]
fn classic_user_rows_are_the_repls() {
    let mut app = App::new();
    app.messages.push(MessageItem::User("hello".into()));
    let (buf, _) = draw(&app, 60, 12);
    assert!(row(&buf, 0).starts_with("❯ hello"));
}

// ── selection colour ─────────────────────────────────────────────────────────

#[test]
fn premium_selection_is_a_background_not_a_reversal() {
    let mut app = premium_app();
    app.handle_event(ArgentEvent::System("highlight me".into()));
    draw(&app, 60, 20);
    // The text starts two columns in (the answer column's indent), which is not text.
    assert!(app.select_cells((2, 0), (5, 0)));
    let (buf, _) = draw(&app, 60, 20);
    assert_eq!(buf[(2, 0)].bg, Palette::default().selection);
    assert_ne!(buf[(0, 0)].bg, Palette::default().selection, "the margin is never painted");
    assert!(!buf[(2, 0)].modifier.contains(Modifier::REVERSED));
    assert_ne!(buf[(10, 0)].bg, Palette::default().selection);
}

// ── the "newest" chip ────────────────────────────────────────────────────────

fn long_chat() -> App {
    let mut app = premium_app();
    for i in 0..60 {
        app.handle_event(ArgentEvent::System(format!("line {i:02}")));
    }
    app
}

#[test]
fn a_chip_offers_the_way_back_down_while_scrolled() {
    let mut app = long_chat();
    draw(&app, 80, 20);
    app.scroll_up(10);
    let (buf, _) = draw(&app, 80, 20);
    let y = find(&buf, "К последнему");
    let x = row(&buf, y).chars().position(|c| c == '↓').unwrap() as u16;
    assert_eq!(app.get_clicked_action(x, y), Some(ClickAction::ScrollToBottom));
}

#[test]
fn the_chip_is_gone_at_the_bottom() {
    let app = long_chat();
    let (buf, _) = draw(&app, 80, 20);
    assert!(!rows(&buf).join("\n").contains("К последнему"));
}

#[test]
fn the_classic_look_has_no_chip() {
    let mut app = App::new();
    for i in 0..60 {
        app.handle_event(ArgentEvent::System(format!("line {i:02}")));
    }
    draw(&app, 80, 20);
    app.scroll_up(10);
    let (buf, _) = draw(&app, 80, 20);
    assert!(!rows(&buf).join("\n").contains("К последнему"));
}

#[test]
fn the_bottom_of_the_chat_is_always_reachable() {
    let mut app = long_chat();
    draw(&app, 80, 20);
    app.scroll_up(usize::MAX);
    draw(&app, 80, 20);
    assert!(app.scroll_offset > 0);
    app.scroll_down(usize::MAX);
    let (buf, _) = draw(&app, 80, 20);
    assert_eq!(app.scroll_offset, 0);
    assert!(rows(&buf).join("\n").contains("line 59"), "the newest line is on screen");
}

// ── switching the look ───────────────────────────────────────────────────────

#[test]
fn the_ui_style_event_switches_the_look_and_drops_cached_blocks() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::System("a block".into()));
    draw(&app, 60, 12);
    assert!(!app.view_cache.borrow().is_empty());
    app.handle_event(ArgentEvent::UiStyle("premium".into()));
    assert_eq!(app.ui.style, UiStyle::Premium);
    assert!(app.view_cache.borrow().is_empty(), "blocks are drawn per style");
    app.handle_event(ArgentEvent::UiStyle("classic".into()));
    assert_eq!(app.ui.style, UiStyle::Classic);
}

#[test]
fn an_unknown_style_name_changes_nothing() {
    let mut app = premium_app();
    app.handle_event(ArgentEvent::UiStyle("neon".into()));
    assert_eq!(app.ui.style, UiStyle::Premium);
}

#[test]
fn the_markdown_request_carries_the_style() {
    use argent_tui::renderer::BlockRenderer;
    use ratatui::text::Line;
    use serde_json::Value;
    use std::sync::{Arc, Mutex};

    struct Spy(Arc<Mutex<Vec<String>>>);
    impl BlockRenderer for Spy {
        fn render(&self, kind: &str, data: &Value, _w: u16) -> Result<Vec<Line<'static>>, String> {
            if kind == "markdown" {
                self.0.lock().unwrap().push(data["style"].as_str().unwrap_or("none").to_string());
            }
            Ok(vec![Line::from("x")])
        }
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut app = premium_app();
    app.renderer = Box::new(Spy(seen.clone()));
    app.handle_event(ArgentEvent::Content("# Title\n\ntext\n\n".into()));
    app.handle_event(ArgentEvent::Done);
    draw(&app, 60, 20);
    assert!(seen.lock().unwrap().iter().all(|s| s == "premium"), "{:?}", seen.lock().unwrap());
    assert!(!seen.lock().unwrap().is_empty());

    seen.lock().unwrap().clear();
    app.handle_event(ArgentEvent::UiStyle("classic".into()));
    draw(&app, 60, 20);
    assert!(seen.lock().unwrap().iter().all(|s| s == "classic"), "a switch re-renders in the new look");
    assert!(!seen.lock().unwrap().is_empty());
}

// ── robustness ───────────────────────────────────────────────────────────────

#[test]
fn no_size_panics_with_everything_open() {
    let mut app = command_app(&["/help", "/history"]);
    app.messages.push(MessageItem::User("hello".into()));
    app.insert_str("/h");
    for w in [8u16, 9, 12, 20, 40, 120] {
        for h in [4u16, 5, 6, 8, 12, 20, 50] {
            draw(&app, w, h);
        }
    }
}

#[test]
fn the_panel_never_overflows_its_row() {
    let mut app = premium_app();
    app.insert_str(&"щ".repeat(200));
    let (buf, _) = draw(&app, 30, 20);
    for y in 0..buf.area.height {
        assert!(row(&buf, y).chars().count() <= 30);
    }
}
