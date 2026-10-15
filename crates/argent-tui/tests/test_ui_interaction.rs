//! LMTrust L1/L2/L9 for the interactive surfaces: collapsible reasoning
//! blocks, the `ask_user_questions` select modal and its free-text variant.
//!
//! Layer decision tree: these are stateful, event-driven surfaces, so the
//! high-ROI layers are L1 (contract of the new focus model), L2 (terminal-size
//! boundaries) and L9 (what must never be painted).

use argent_tui::app::{App, MessageItem, SelectModalData};
use argent_tui::events::ArgentEvent;
use argent_tui::ui::draw_ui;
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("terminal");
    terminal.draw(|f| draw_ui(f, app)).expect("draw");
    let buf = terminal.backend().buffer().clone();
    let area = buf.area;
    let mut out = String::new();
    for y in 0..area.height {
        for x in 0..area.width {
            out.push_str(buf.content[(y * area.width + x) as usize].symbol());
        }
        out.push('\n');
    }
    out
}

fn push_thinking(app: &mut App, text: &str) {
    app.messages.push(MessageItem::Assistant {
        content: format!("answer {text}"),
        thinking: text.to_string(),
        thinking_expanded: false,
        is_streaming_thinking: false,
        is_streaming_content: false,
        thinking_duration: Some(1.0),
        thinking_start: None,
        stream: Default::default(),
    });
}

fn expanded_states(app: &App) -> Vec<bool> {
    app.messages
        .iter()
        .filter_map(|m| match m {
            MessageItem::Assistant { thinking_expanded, thinking, .. } if !thinking.is_empty() => {
                Some(*thinking_expanded)
            }
            _ => None,
        })
        .collect()
}

// ── L1: collapsible-block focus model ──────────────────────────────────────

/// L1 x D7. Both reasoning blocks and tool cards are reachable, so several can
/// be open at once instead of only the newest.
#[test]
fn l1_every_collapsible_block_is_reachable() {
    let mut app = App::new();
    push_thinking(&mut app, "first thoughts");
    app.messages.push(MessageItem::Tool {
        name: "read_file".into(),
        args: Default::default(),
        result: Some("ok".into()),
        duration_secs: Some(0.1),
        expanded: false,
        is_running: false,
    });
    push_thinking(&mut app, "second thoughts");

    assert_eq!(app.collapsible_blocks().len(), 3, "expected 3 collapsible blocks");

    // Unpinned focus starts on the newest block.
    app.toggle_focused_block();
    assert_eq!(expanded_states(&app), vec![false, true], "newest reasoning should open");
    if let Some(MessageItem::Tool { expanded, .. }) = app.messages.get(1) {
        assert!(!*expanded, "an unrelated tool card must not be touched");
    }

    app.focus_prev_block();
    app.toggle_focused_block();
    if let Some(MessageItem::Tool { expanded, .. }) = app.messages.get(1) {
        assert!(expanded, "the middle tool card should be open");
    }
    assert_eq!(
        expanded_states(&app),
        vec![false, true],
        "only the focused block may change"
    );
}

/// L1 x D2. Ctrl+P from the newest block reaches the oldest, and focus wraps.
#[test]
fn l1_focus_navigation_wraps_in_both_directions() {
    let mut app = App::new();
    for i in 0..3 {
        push_thinking(&mut app, &format!("thoughts {i}"));
    }
    let blocks = app.collapsible_blocks();

    app.toggle_focused_block();
    assert!(app.is_focused_block(*blocks.last().unwrap()));

    app.focus_prev_block();
    assert!(app.is_focused_block(blocks[1]));
    app.focus_prev_block();
    assert!(app.is_focused_block(blocks[0]));
    app.focus_prev_block();
    assert!(app.is_focused_block(blocks[2]), "focus must wrap to the newest");
    app.focus_next_block();
    assert!(app.is_focused_block(blocks[0]), "focus must wrap to the oldest");
}

/// L1 x D2. Two blocks can be open independently — the previous model could
/// only ever expand one at a time.
#[test]
fn l1_several_blocks_can_be_open_independently() {
    let mut app = App::new();
    push_thinking(&mut app, "alpha");
    push_thinking(&mut app, "beta");
    push_thinking(&mut app, "gamma");

    app.toggle_focused_block();
    assert_eq!(expanded_states(&app), vec![false, false, true]);

    app.focus_prev_block();
    app.toggle_focused_block();
    assert_eq!(expanded_states(&app), vec![false, true, true]);

    app.focus_prev_block();
    app.toggle_focused_block();
    assert_eq!(expanded_states(&app), vec![true, true, true]);

    // And back down again, closing only what is focused.
    app.focus_next_block();
    app.toggle_focused_block();
    assert_eq!(expanded_states(&app), vec![true, false, true]);
}

/// L1 x D4. While the user has not chosen a block, a newly streamed reply
/// takes the selection so Tab always hits the latest.
#[test]
fn l1_unpinned_focus_follows_new_blocks() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("older".into()));
    let older = app.collapsible_blocks()[0];
    app.toggle_focused_block();
    assert!(app.is_focused_block(older));

    app.handle_event(ArgentEvent::ToolStart {
        name: "read_file".into(),
        args: Default::default(),
    });
    let blocks = app.collapsible_blocks();
    assert_eq!(blocks.len(), 2);
    app.toggle_focused_block();
    assert!(
        app.is_focused_block(blocks[1]),
        "an unpinned selection must follow the newest block"
    );
}

/// L1 x D4. Once the user navigates, the selection stays put: that is what
/// makes "open several blocks" usable.
#[test]
fn l1_pinned_focus_survives_new_blocks() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("older".into()));
    let older = app.collapsible_blocks()[0];

    app.focus_prev_block();
    assert!(app.focus_pinned);

    app.handle_event(ArgentEvent::ToolStart {
        name: "read_file".into(),
        args: Default::default(),
    });
    app.toggle_focused_block();

    assert!(
        app.is_focused_block(older),
        "a pinned selection must not be stolen by a new block"
    );
}

/// L1 x D2. A new turn unpins the selection again.
#[test]
fn l1_submitting_a_new_turn_unpins_the_focus() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Thinking("older".into()));
    app.focus_prev_block();
    assert!(app.focus_pinned);

    app.insert_str("next question");
    app.submit_input();

    assert!(!app.focus_pinned, "a new turn must hand focus back to the newest block");
}

/// L1 x D4. Toggling with nothing collapsible must be a safe no-op, not a panic.
#[test]
fn l1_toggle_with_no_collapsible_block_is_safe() {
    let mut app = App::new();
    assert!(app.collapsible_blocks().is_empty());
    assert!(!app.toggle_focused_block());
    assert!(!app.focus_next_block());
    assert!(!app.focus_prev_block());
    assert_eq!(app.focus_idx, None);

    app.insert_str("plain");
    app.submit_input();
    assert!(!app.toggle_focused_block(), "a user turn is not collapsible");

    // An assistant bubble with no reasoning is not collapsible either.
    app.handle_event(ArgentEvent::Content("no thinking here".into()));
    assert!(app.collapsible_blocks().is_empty());
    assert!(!app.toggle_focused_block());
}

/// L1 x D7. Bulk toggle covers every reasoning block, and only real reasoning.
#[test]
fn l1_toggle_all_reasoning_covers_every_block() {
    let mut app = App::new();
    push_thinking(&mut app, "a");
    push_thinking(&mut app, "b");
    app.messages.push(MessageItem::Assistant {
        content: "no reasoning".into(),
        thinking: String::new(),
        thinking_expanded: false,
        is_streaming_thinking: false,
        is_streaming_content: false,
        thinking_duration: None,
        thinking_start: None,
        stream: Default::default(),
    });

    assert_eq!(app.toggle_all_reasoning(true), 2);
    assert_eq!(expanded_states(&app), vec![true, true]);
    assert_eq!(app.toggle_all_reasoning(false), 2);
    assert_eq!(expanded_states(&app), vec![false, false]);
}

// ── L9: the focused block is visible, and Space is only a space ─────────────

/// L9 x D3. The user has to be able to see which block the keys will act on.
/// Collapsed reasoning is the REPL's one-line `◇ Анализ …` row; the focus marker
/// takes its gutter column.
#[test]
fn l9_the_focused_block_is_marked_in_the_frame() {
    let mut app = App::new();
    push_thinking(&mut app, "FIRST_BLOCK_MARKER");
    push_thinking(&mut app, "SECOND_BLOCK_MARKER");
    let blocks = app.collapsible_blocks();
    app.focus_prev_block();
    assert!(app.is_focused_block(blocks[0]));

    let frame = render(&app, 160, 50);
    let diamond_rows: Vec<&str> = frame.lines().filter(|l| l.contains('\u{25c7}')).collect();
    assert_eq!(diamond_rows.len(), 2, "one reasoning row per block:\n{frame}");

    let marker = '\u{25b8}';
    let marked: Vec<&str> = frame.lines().filter(|l| l.contains(marker)).collect();
    assert_eq!(marked.len(), 1, "exactly one block must be marked:\n{frame}");
    assert!(
        marked[0].contains("FIRST_BLOCK_MARKER"),
        "the marker is on the wrong block: {:?}\n{frame}",
        marked[0]
    );
}

/// L9 x D7. The toolbar is the REPL's, and it advertises nothing that is not
/// bound: no header strip, no Space shortcut.
#[test]
fn l9_toolbar_matches_the_repl_and_advertises_no_stale_keys() {
    let app = App::new();
    let frame = render(&app, 160, 50);
    let last = frame.lines().rfind(|l| !l.trim().is_empty()).unwrap_or("");
    assert!(last.contains("CHAT"), "toolbar missing the mode: {last:?}");
    assert!(last.contains("/help"), "toolbar missing /help: {last:?}");
    assert!(last.contains("tier:") && last.contains("cwd:"), "{last:?}");
    assert!(!frame.contains("Space"), "Space is not a shortcut:\n{frame}");
    assert!(!frame.contains("ARGENT CODER"), "the old header strip is gone:\n{frame}");
}

// ── L2: questionary-style prompts must fit the terminal ─────────────────────

fn select(options: Vec<&str>, default: Option<&str>) -> SelectModalData {
    SelectModalData::new(
        "sel".into(),
        "Выберите:".into(),
        options.into_iter().map(str::to_string).collect(),
        default.map(str::to_string),
    )
}

fn app_with_select(sel: SelectModalData) -> App {
    let mut app = App::new();
    app.active_select = Some(sel);
    app
}

/// L2 x D1. The question and its options stay inside the buffer at every size.
#[test]
fn l2_select_prompt_stays_inside_the_terminal() {
    let sel = select(vec!["a", "b", "c"], Some("b"));
    for (w, h) in [(10u16, 4u16), (12, 6), (20, 8), (40, 10), (60, 20), (80, 24), (120, 40), (200, 60)] {
        let app = app_with_select(sel.clone());
        let frame = render(&app, w, h);
        assert_eq!(frame.lines().count(), h as usize, "frame height at {w}x{h}");
        if h >= 6 && w >= 20 {
            assert!(frame.contains('?'), "no question mark at {w}x{h}\n{frame}");
            assert!(frame.contains('\u{bb}'), "no pointer at {w}x{h}\n{frame}");
        }
    }
}

/// L2 x D1. The free-text variant is a single row.
#[test]
fn l2_free_text_prompt_fits_every_terminal_size() {
    let mut sel = SelectModalData::new("sel".into(), "Свой вариант".into(), Vec::new(), None);
    sel.filter = "answer typed by the user".into();
    for (w, h) in [(10u16, 4u16), (12, 6), (20, 8), (40, 10), (80, 24), (120, 40)] {
        let app = app_with_select(sel.clone());
        let frame = render(&app, w, h);
        assert_eq!(frame.lines().count(), h as usize);
        if w >= 40 {
            assert!(frame.contains("answer typed"), "the answer is not shown at {w}x{h}\n{frame}");
        }
    }
}

/// L1 x D7. `draw_ui` survives every size with a prompt open.
#[test]
fn l2_prompt_rendering_never_panics() {
    let mut sel = select(vec!["alpha", "beta", "gamma"], Some("gamma"));
    for (w, h) in [(0u16, 0u16), (1, 1), (4, 4), (5, 5), (9, 7), (30, 9), (200, 60)] {
        let app = app_with_select(sel.clone());
        let _ = render(&app, w, h);
        sel.filter = "g".into();
        let app = app_with_select(sel.clone());
        let _ = render(&app, w, h);
    }
}

// ── L1/L9: the pointer follows the full-list index ──────────────────────────

fn pointer_rows(frame: &str) -> Vec<&str> {
    frame.lines().filter(|l| l.contains(" \u{bb} ")).collect()
}

/// L1 x D3. `selected_idx` addresses the FULL list. With a filter active the
/// pointer must still sit on the right row.
#[test]
fn l1_pointer_follows_the_selection_through_a_filter() {
    let mut sel = select(vec!["alpha", "beta", "gamma", "delta"], Some("gamma"));
    assert_eq!(sel.selected_idx, 2);
    sel.filter = "a".into();
    assert!(sel.filtered_options().len() > 1, "the filter should narrow the list");

    let frame = render(&app_with_select(sel), 100, 24);
    let rows = pointer_rows(&frame);
    assert_eq!(rows.len(), 1, "exactly one pointer expected:\n{frame}");
    assert!(rows[0].contains("gamma"), "the pointer left the selection:\n{frame}");
}

#[test]
fn l1_pointer_survives_a_filter_that_leaves_one_match() {
    let mut sel = select(vec!["alpha", "beta", "gamma", "delta"], Some("gamma"));
    sel.filter = "gam".into();
    let filtered = sel.filtered_options();
    assert_eq!((filtered.len(), filtered[0].0), (1, 2));

    let frame = render(&app_with_select(sel), 100, 24);
    let rows = pointer_rows(&frame);
    assert_eq!(rows.len(), 1, "{frame}");
    assert!(rows[0].contains("gamma"), "{frame}");
}

/// L1 x D3. Exactly one option carries the pointer.
#[test]
fn l9_exactly_one_row_carries_the_pointer() {
    let sel = select(vec!["one", "two", "three", "four"], Some("three"));
    let frame = render(&app_with_select(sel), 100, 24);
    let rows = pointer_rows(&frame);
    assert_eq!(rows.len(), 1, "{frame}");
    assert!(rows[0].contains("three"), "{frame}");
}

/// L9 x D3. A filtered-out selection must not leave a stray pointer on a row
/// that is not the first match.
#[test]
fn l9_no_pointer_on_a_filtered_out_row() {
    let mut sel = select(vec!["alpha", "beta"], Some("alpha"));
    sel.filter = "beta".into();
    let frame = render(&app_with_select(sel), 100, 24);
    assert!(
        !pointer_rows(&frame).iter().any(|l| l.contains("alpha")),
        "a hidden option still has the pointer:\n{frame}"
    );
}

// ── L9: escape sequences never reach the screen ────────────────────────────

/// L9 x D3. Captured command output carries SGR codes. Painting them produced
/// literal `[35;58;33M` fragments in the transcript.
#[test]
fn l9_ansi_codes_from_a_tool_result_never_reach_the_screen() {
    let mut app = App::new();
    let dirty = "\u{1b}[35;58;33m\u{1b}[35;57,33m\u{1b}[1;31mFATAL\u{1b}[0m: boom";
    app.handle_event(ArgentEvent::ToolStart { name: "run_command".into(), args: Default::default() });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: dirty.into(),
        duration_secs: Some(0.1),
    });

    let frame = render(&app, 160, 40);
    assert!(!frame.contains('\u{1b}'), "an escape byte survived into the frame");
    assert!(!frame.contains("[35;58;33"), "raw SGR parameters were painted as text:\n{frame}");
    assert!(frame.contains("FATAL"), "the actual message was lost:\n{frame}");
}

/// L9 x D3. The same for assistant text, which is where the digits showed up.
#[test]
fn l9_ansi_codes_from_an_assistant_reply_never_reach_the_screen() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::Content(
        "\u{1b}[1;35m\u{1b}[35;58;33m\x1b[5;58;33M[\u{1b}[34mhello\u{1b}[0m".into(),
    ));
    app.handle_event(ArgentEvent::Done);
    let frame = render(&app, 160, 40);
    assert!(!frame.contains('\u{1b}'));
    assert!(!frame.contains("58;33M"), "SGR digits painted:\n{frame}");
    assert!(frame.contains("hello"), "the reply was lost:\n{frame}");
}

/// L9 x D9. A carriage-return progress bar must not smear digits across lines.
#[test]
fn l9_carriage_return_progress_output_is_collapsed() {
    let mut app = App::new();
    app.handle_event(ArgentEvent::ToolStart { name: "run_command".into(), args: Default::default() });
    app.handle_event(ArgentEvent::ToolEnd {
        name: "run_command".into(),
        result: "10%\r55%\r100% done\r\n".into(),
        duration_secs: Some(0.1),
    });
    let frame = render(&app, 160, 40);
    assert!(frame.contains("100%"), "the final progress state was lost:\n{frame}");
    assert!(!frame.contains("10%"), "an intermediate redraw leaked:\n{frame}");
}

/// L9 x D3. Free-text answers are typed by the user but may arrive through a
/// paste carrying escape codes; the prompt must still render them cleanly.
#[test]
fn l9_free_text_prompt_shows_the_typed_answer() {
    let mut sel = SelectModalData::new("sel".into(), "Свой вариант".into(), Vec::new(), None);
    sel.filter = "\u{1b}[35;58;33mмой ответ".into();
    let frame = render(&app_with_select(sel), 100, 24);
    assert!(frame.contains("\u{043c}\u{043e}\u{0439}"), "the answer text is missing:\n{frame}");
    assert!(!frame.contains('\u{1b}'), "an escape byte reached the prompt");
}

/// L9 x D3. A very long free-text answer must stay readable instead of running
/// off the right edge: the tail (what is being typed) stays visible.
#[test]
fn l9_a_very_long_free_text_answer_shows_its_tail() {
    let mut sel = SelectModalData::new("sel".into(), "Свой вариант".into(), Vec::new(), None);
    sel.filter = format!("{}END", "x".repeat(500));
    let frame = render(&app_with_select(sel), 60, 24);
    assert!(frame.contains("END"), "the end of a long answer is not visible:\n{frame}");
    for row in frame.lines() {
        assert!(row.chars().count() <= 60, "a row runs past the terminal: {row:?}");
    }
}

// ── L9: the approval prompt uses the REPL's wording and honest choices ──────

fn app_with_approval(grant_key: Option<&str>, destructive: bool) -> App {
    let mut app = App::new();
    app.active_approval = Some(argent_tui::app::ApprovalData {
        id: 1,
        action: "powershell -c Get-CimInstance".into(),
        destructive,
        grant_key: grant_key.map(str::to_string),
        prompt: None,
        selected: 0,
    });
    app
}

fn offers_always(frame: &str) -> bool {
    frame.contains("всегда разрешать")
}

/// L9 x D4. `approval.py` only records a session grant when a `grant_key`
/// exists and the action is not destructive. Offering "always" otherwise made
/// the user choose it and get a denial back.
#[test]
fn l9_always_is_not_offered_without_a_session_grant_key() {
    let frame = render(&app_with_approval(None, false), 100, 24);
    assert!(!offers_always(&frame), "\"always\" must be hidden without a grant key:\n{frame}");
    assert!(frame.contains("(y/N)"), "the REPL asks a plain y/N question here:\n{frame}");
    assert!(frame.contains("Агент запрашивает разрешение"), "the request line is missing:\n{frame}");
    assert!(frame.contains("powershell -c Get-CimInstance"), "the action must be named:\n{frame}");
}

/// L9 x D4. Destructive actions are never covered by a session grant.
#[test]
fn l9_always_is_not_offered_for_a_destructive_action() {
    let frame = render(&app_with_approval(Some("rm"), true), 100, 24);
    assert!(!offers_always(&frame), "\"always\" must be hidden for a destructive action:\n{frame}");
    assert!(frame.contains('\u{25b2}'), "a destructive request is flagged with the warning glyph:\n{frame}");
}

/// L1 x D7. When a grant IS possible, the prompt says so and names the scope,
/// so "always" is an informed choice rather than a guess.
#[test]
fn l1_always_is_offered_and_scoped_when_a_grant_is_possible() {
    let frame = render(&app_with_approval(Some("powershell"), false), 100, 24);
    assert!(offers_always(&frame), "\"always\" must be offered when a grant can be stored:\n{frame}");
    assert!(frame.contains("'powershell'"), "the prompt must name what is being granted:\n{frame}");
    assert!(frame.contains("Use arrow keys"), "{frame}");
    assert_eq!(pointer_rows(&frame).len(), 1, "one choice carries the pointer:\n{frame}");
}

/// L2 x D3. The prompt renders at every size without painting outside the frame.
#[test]
fn l2_approval_prompt_never_panics_and_stays_inside_the_terminal() {
    for (w, h) in [(10u16, 4u16), (20, 8), (40, 12), (80, 24), (200, 60)] {
        for (gk, destructive) in [(Some("git"), false), (None, false), (Some("rm"), true)] {
            let frame = render(&app_with_approval(gk, destructive), w, h);
            assert_eq!(frame.lines().count(), h as usize);
            for row in frame.lines() {
                assert!(row.chars().count() <= w as usize, "row wider than {w}: {row:?}");
            }
        }
    }
}
