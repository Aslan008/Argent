//! Frame composition: the transcript, the prompt, and the bottom toolbar.
//!
//! The screen is laid out the way the REPL's terminal looks: the conversation
//! first, the `❯` prompt right under it, a completion menu below the prompt (as
//! prompt_toolkit draws it) and the one-line toolbar at the bottom. There is no
//! header, no bordered input box and no button strip — everything the REPL does
//! not show is reachable through keys and (invisible) click targets instead.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
    },
    Frame,
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::ansi::{clip_line, line_width};
use crate::app::{App, AppStatus, ApprovalData, ClickAction, ImagePreview, SelectModalData};
use crate::premium;
use crate::theme::toolbar_style;
use crate::view::{build_transcript, wrap_hard};

/// Rows prompt_toolkit reserves for its completion menu.
pub(crate) const MENU_ROWS: usize = 8;

// ── toolbar ─────────────────────────────────────────────────────────────────

/// `8600 -> 8.6k`, `900 -> 900` — the REPL toolbar's `_fmt_tokens`.
pub fn fmt_tokens(n: usize) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

/// `$0.012` below ten cents, `$1.23` above — the same split as `_fmt_cost`.
pub fn fmt_cost(cost: f64) -> String {
    if cost <= 0.0 {
        return String::new();
    }
    if cost < 0.1 {
        format!("${cost:.4}")
    } else {
        format!("${cost:.2}")
    }
}

/// The running session total for the toolbar's right corner.
///
/// `session_summary` is the line the REPL prints, so it is preferred and both
/// frontends say the same thing. The fallback covers a state that carried the
/// raw totals but no summary yet (an older bridge, a partially applied update).
pub fn session_spend(st: &AppStatus) -> String {
    if !st.session_summary.is_empty() {
        return st.session_summary.clone();
    }
    let total = (st.session_prompt + st.session_completion).round() as usize;
    if total == 0 {
        return String::new();
    }
    let cost = fmt_cost(st.session_cost);
    if cost.is_empty() {
        format!("sum {} tok", fmt_tokens(total))
    } else {
        format!("sum {} tok · {cost}", fmt_tokens(total))
    }
}

pub(crate) fn uptime_label(app: &App) -> String {    let mins = app.status.start_time.elapsed().as_secs() / 60;
    let hours = mins / 60;
    if hours > 0 {
        format!("{hours}h {}m", mins % 60)
    } else {
        format!("{mins}m")
    }
}

fn ctx_color(pct: i64) -> Color {
    if pct < 50 {
        Color::Rgb(0x4c, 0xaf, 0x50)
    } else if pct < 80 {
        Color::Rgb(0xff, 0xb3, 0x00)
    } else {
        Color::Rgb(0xe5, 0x39, 0x35)
    }
}

/// The REPL's bottom toolbar (`cli_prompt.build_bottom_toolbar`), as one line
/// plus the click targets hidden inside it.
///
///  CHAT | provider:model | tier:cloud | think:AUTO | cwd:Argent | ctx:12% 3.1k/32.8k | sum … | uptime:5m | /help
pub fn toolbar(app: &App, width: u16) -> (Line<'static>, Vec<(u16, u16, ClickAction)>) {
    let base = toolbar_style();
    let st = &app.status;
    let sep = || Span::styled(" | ", base);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut targets: Vec<(u16, u16, ClickAction)> = Vec::new();
    let mut x: u16 = 0;

    let push = |spans: &mut Vec<Span<'static>>, s: Span<'static>, x: &mut u16| {
        *x = x.saturating_add(UnicodeWidthStr::width(s.content.as_ref()) as u16);
        spans.push(s);
    };

    push(&mut spans, Span::styled(" ", base), &mut x);
    push(&mut spans, Span::styled(st.mode.clone(), base.add_modifier(Modifier::BOLD)), &mut x);
    push(&mut spans, sep(), &mut x);

    let pm = format!("{}:{}", st.provider, st.model);
    let x0 = x;
    push(&mut spans, Span::styled(pm, base), &mut x);
    targets.push((x0, x, ClickAction::OpenModelPicker));
    push(&mut spans, sep(), &mut x);

    push(&mut spans, Span::styled(format!("tier:{}", st.tier), base), &mut x);
    push(&mut spans, sep(), &mut x);

    let think = if st.thinking_label.is_empty() {
        st.thinking_level.to_uppercase()
    } else {
        st.thinking_label.clone()
    };
    if !think.is_empty() {
        let x0 = x;
        push(&mut spans, Span::styled(format!("think:{think}"), base), &mut x);
        targets.push((x0, x, ClickAction::CycleThinkingLevel));
        push(&mut spans, sep(), &mut x);
    }

    push(&mut spans, Span::styled(format!("cwd:{}", st.cwd), base), &mut x);

    if st.ctx_tokens > 0 || st.ctx_percent > 0.0 {
        push(&mut spans, sep(), &mut x);
        let pct = st.ctx_percent.round() as i64;
        push(&mut spans, Span::styled("ctx:", base), &mut x);
        push(&mut spans, Span::styled(format!("{pct}%"), base.fg(ctx_color(pct))), &mut x);
        if st.ctx_max > 0 {
            push(
                &mut spans,
                Span::styled(format!(" {}/{}", fmt_tokens(st.ctx_tokens), fmt_tokens(st.ctx_max)), base),
                &mut x,
            );
        }
    }
    if !st.session_summary.is_empty() {
        push(&mut spans, sep(), &mut x);
        push(&mut spans, Span::styled(st.session_summary.clone(), base), &mut x);
    }
    if !st.last_spend.is_empty() {
        push(&mut spans, sep(), &mut x);
        push(&mut spans, Span::styled(st.last_spend.clone(), base), &mut x);
    }
    push(&mut spans, sep(), &mut x);
    push(&mut spans, Span::styled(format!("uptime:{}", uptime_label(app)), base), &mut x);
    push(&mut spans, sep(), &mut x);
    push(&mut spans, Span::styled("/help", base.fg(Color::Rgb(0x88, 0x88, 0x88))), &mut x);
    push(&mut spans, Span::styled(" ", base), &mut x);

    let mut line = clip_line(Line::from(spans), width as usize);
    let used = line_width(&line);
    if used < width as usize {
        line.spans.push(Span::styled(" ".repeat(width as usize - used), base));
    }
    targets.retain(|(a, _, _)| *a < width);
    (line, targets)
}

// ── prompt ──────────────────────────────────────────────────────────────────

/// How the typed text is cut into screen rows: char ranges into the input.
pub(crate) struct InRow {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

/// Hard-wrap `input` into rows of at most `avail` columns, remembering which
/// characters each row holds (newlines belong to no row).
pub(crate) fn layout_input(input: &str, avail: usize) -> Vec<InRow> {
    let avail = avail.max(1);
    let mut rows = Vec::new();
    let mut base = 0usize;
    for logical in input.split('\n') {
        let chars: Vec<char> = logical.chars().collect();
        if chars.is_empty() {
            rows.push(InRow { start: base, end: base });
        } else {
            let mut start = 0usize;
            let mut w = 0usize;
            for (i, ch) in chars.iter().enumerate() {
                let cw = ch.width().unwrap_or(0);
                if w + cw > avail && i > start {
                    rows.push(InRow { start: base + start, end: base + i });
                    start = i;
                    w = 0;
                }
                w += cw;
            }
            rows.push(InRow { start: base + start, end: base + chars.len() });
        }
        base += chars.len() + 1;
    }
    rows
}

/// (row, column) of char index `cursor` in a layout; a cursor sitting at the end
/// of a wrapped row belongs to the start of the next one, like in a terminal.
pub(crate) fn cursor_in_layout(input: &str, rows: &[InRow], cursor: usize) -> (usize, usize) {
    let chars: Vec<char> = input.chars().collect();
    for (i, r) in rows.iter().enumerate() {
        let continues = rows.get(i + 1).is_some_and(|n| n.start == r.end);
        if cursor >= r.start && (cursor < r.end || (cursor == r.end && !continues)) {
            let col: usize = chars[r.start..cursor.min(chars.len())]
                .iter()
                .map(|c| c.width().unwrap_or(0))
                .sum();
            return (i, col);
        }
    }
    (rows.len().saturating_sub(1), 0)
}

pub(crate) struct PromptBlock {
    pub(crate) rows: Vec<Line<'static>>,
    /// Where the terminal cursor goes, relative to the first prompt row.
    pub(crate) cursor: Option<(u16, u16)>,
}

fn qmark() -> Span<'static> {
    Span::styled("?", Style::default().fg(Color::Rgb(0x5f, 0x81, 0x9d)))
}

fn bold() -> Style {
    Style::default().add_modifier(Modifier::BOLD)
}

/// A question's title as rows: wrapped to the width (it may hold line breaks and run long), cut
/// to `max_rows`. One row of a short title is exactly what the title was before.
fn title_rows(title: &str, width: usize, max_rows: usize) -> Vec<String> {
    // `? ` in front of the first row, two blanks under it on the others.
    let avail = width.saturating_sub(4).max(8);
    let mut rows: Vec<String> = crate::view::wrap_words(title.trim(), avail);
    while rows.last().is_some_and(|r| r.trim().is_empty()) && rows.len() > 1 {
        rows.pop();
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    let max_rows = max_rows.max(1);
    if rows.len() > max_rows {
        rows.truncate(max_rows);
        if let Some(last) = rows.last_mut() {
            last.push('…');
        }
    }
    rows
}

/// The title rows of a question as styled lines: the `?` mark on the first, the rest set in under
/// the text. `tail` is added after the last row (the typed text, or the hint of what keys do).
fn title_lines(rows: &[String], tail: Vec<Span<'static>>) -> Vec<Line<'static>> {
    let n = rows.len();
    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let mut spans = if i == 0 {
                vec![qmark(), Span::styled(format!(" {r} "), bold())]
            } else {
                vec![Span::styled(format!("  {r} "), bold())]
            };
            if i + 1 == n {
                spans.extend(tail.iter().cloned());
            }
            Line::from(spans)
        })
        .collect()
}

/// The `❯ ` prompt while idle, or the REPL's `❯ [Ввод на лету]:` echo while a
/// turn runs (nothing at all while busy and nothing typed — as in the REPL).
pub(crate) fn prompt_block_classic(app: &App, width: u16, max_rows: usize) -> PromptBlock {
    let theme = &app.ui.theme;
    let w = width as usize;

    if app.is_busy {
        // The echo of the queue prompt is not an editor: clicks do not belong in it.
        app.set_input_box(0, 1, 0, 0, 0);
        if app.input.is_empty() {
            return PromptBlock { rows: Vec::new(), cursor: None };
        }
        // cli_ui._render_queue_input_display: label, then the text with a fake
        // block cursor, continuation rows marked with a gutter.
        let label_style = Style::default().fg(Color::Rgb(0x58, 0xa6, 0xff)).add_modifier(Modifier::BOLD);
        let chars: Vec<char> = app.input.chars().collect();
        let cur = app.cursor_pos.min(chars.len());
        let mut spans: Vec<Span<'static>> = vec![Span::styled("❯ [Ввод на лету]:", label_style), Span::raw(" ")];
        let gutter = "  │ ";
        let mut col = UnicodeWidthStr::width("❯ [Ввод на лету]: ");
        let mut rows: Vec<Line<'static>> = Vec::new();
        let flush = |spans: &mut Vec<Span<'static>>, rows: &mut Vec<Line<'static>>| {
            rows.push(Line::from(std::mem::take(spans)));
        };
        let emit = |s: String, style: Style, spans: &mut Vec<Span<'static>>, rows: &mut Vec<Line<'static>>, col: &mut usize| {
            for ch in s.chars() {
                let cw = ch.width().unwrap_or(0);
                if *col + cw > w {
                    flush(spans, rows);
                    *col = 0;
                }
                spans.push(Span::styled(ch.to_string(), style));
                *col += cw;
            }
        };
        let plain = Style::default().fg(Color::White);
        let block = Style::default().fg(Color::Black).bg(Color::White);
        for (i, ch) in chars.iter().enumerate() {
            if i == cur {
                let shown = if *ch == '\n' { "↵".to_string() } else { ch.to_string() };
                emit(shown, block, &mut spans, &mut rows, &mut col);
                if *ch == '\n' {
                    flush(&mut spans, &mut rows);
                    col = 0;
                    emit(gutter.to_string(), plain, &mut spans, &mut rows, &mut col);
                }
                continue;
            }
            if *ch == '\n' {
                flush(&mut spans, &mut rows);
                col = 0;
                emit(gutter.to_string(), plain, &mut spans, &mut rows, &mut col);
            } else {
                emit(ch.to_string(), plain, &mut spans, &mut rows, &mut col);
            }
        }
        if cur >= chars.len() {
            emit(" ".to_string(), block, &mut spans, &mut rows, &mut col);
        }
        flush(&mut spans, &mut rows);
        rows.push(Line::from(Span::styled(
            "(Enter: в очередь · Shift+Enter: новая строка · ↑: вернуть из очереди · Esc: стереть)",
            Style::default().add_modifier(Modifier::DIM),
        )));
        let skip = rows.len().saturating_sub(max_rows);
        return PromptBlock { rows: rows.into_iter().skip(skip).collect(), cursor: None };
    }

    let avail = w.saturating_sub(2).max(1);
    let layout = layout_input(&app.input, avail);
    let (cur_row, cur_col) = cursor_in_layout(&app.input, &layout, app.cursor_pos);
    let (sel_lo, sel_hi) = match app.sel_anchor {
        Some(a) if a != app.cursor_pos => (a.min(app.cursor_pos), a.max(app.cursor_pos)),
        _ => (0, 0),
    };
    let chars: Vec<char> = app.input.chars().collect();
    let sym = theme.user_prompt_symbol.add_modifier(Modifier::BOLD);

    let mut rows: Vec<Line<'static>> = Vec::new();
    for (i, r) in layout.iter().enumerate() {
        let prefix = if i == 0 { "❯ " } else { "  " };
        let mut spans = vec![Span::styled(prefix, sym)];
        let mut seg = String::new();
        let mut seg_sel = false;
        for (k, ch) in chars[r.start..r.end].iter().enumerate() {
            let idx = r.start + k;
            let sel = sel_hi > sel_lo && idx >= sel_lo && idx < sel_hi;
            if sel != seg_sel && !seg.is_empty() {
                let st = if seg_sel { theme.user_prompt.add_modifier(Modifier::REVERSED) } else { theme.user_prompt };
                spans.push(Span::styled(std::mem::take(&mut seg), st));
            }
            seg_sel = sel;
            seg.push(*ch);
        }
        if !seg.is_empty() {
            let st = if seg_sel { theme.user_prompt.add_modifier(Modifier::REVERSED) } else { theme.user_prompt };
            spans.push(Span::styled(seg, st));
        }
        rows.push(Line::from(spans));
    }

    // Keep the cursor row visible when the text is taller than the allowance. The
    // wheel can then peek further back without moving the caret: the peek is
    // measured from there, so it never scrolls past the top of the draft.
    let base_skip = if rows.len() > max_rows {
        (cur_row + 1).saturating_sub(max_rows).min(rows.len() - max_rows)
    } else {
        0
    };
    app.set_input_base_skip(base_skip);
    let skip = base_skip.saturating_sub(app.input_scroll.min(base_skip));
    let rows: Vec<Line<'static>> = rows.into_iter().skip(skip).take(max_rows).collect();
    let cursor = Some(((2 + cur_col) as u16, cur_row.saturating_sub(skip) as u16));
    // The mouse turns a click into a caret position through this box.
    app.set_input_box(2, avail, skip, rows.len(), 0);
    PromptBlock { rows, cursor }
}

// ── completion menu (prompt_toolkit's `completion-menu`) ───────────────────

fn menu_style(current: bool) -> Style {
    if current {
        Style::default()
            .fg(Color::Rgb(0x88, 0x88, 0x88))
            .bg(Color::Rgb(0xff, 0xff, 0xff))
            .add_modifier(Modifier::REVERSED)
    } else {
        Style::default().fg(Color::Black).bg(Color::Rgb(0xbb, 0xbb, 0xbb))
    }
}

fn meta_style(current: bool) -> Style {
    let bg = if current { Color::Rgb(0xaa, 0xaa, 0xaa) } else { Color::Rgb(0x99, 0x99, 0x99) };
    Style::default().fg(Color::Black).bg(bg)
}

/// What the open completion menu lists — (name, description) — and which item
/// is current, or `None` when no menu is open.
pub(crate) fn menu_items(app: &App) -> Option<(Vec<(String, String)>, usize)> {
    let (items, cur): (Vec<(String, String)>, usize) = if app.show_completions && !app.completions.is_empty() {
        (
            app.get_matched_completions()
                .into_iter()
                .map(|c| (c.name.clone(), c.help.clone()))
                .collect(),
            app.completion_idx,
        )
    } else if app.show_file_completions && !app.file_completions.is_empty() {
        (
            app.file_completions.iter().map(|p| (format!("@{p}"), String::new())).collect(),
            app.file_completion_idx,
        )
    } else {
        return None;
    };
    if items.is_empty() {
        None
    } else {
        Some((items, cur))
    }
}

/// Rows of the open completion menu (commands or `@files`), if any.
fn menu_rows(app: &App, width: u16) -> Vec<Line<'static>> {
    let Some((items, cur)) = menu_items(app) else { return Vec::new() };
    let cur = cur.min(items.len() - 1);
    let name_w = items.iter().map(|(n, _)| UnicodeWidthStr::width(n.as_str())).max().unwrap_or(0) + 2;
    let meta_w = items.iter().map(|(_, m)| UnicodeWidthStr::width(m.as_str())).max().unwrap_or(0);
    let meta_w = if meta_w > 0 { meta_w + 2 } else { 0 };
    let max_total = (width as usize).saturating_sub(2);
    let top = if cur >= MENU_ROWS { cur + 1 - MENU_ROWS } else { 0 };

    let mut out = Vec::new();
    for (i, (name, meta)) in items.iter().enumerate().skip(top).take(MENU_ROWS) {
        let cur_row = i == cur;
        let mut spans = vec![Span::raw("  ")];
        let n = format!(" {name:<w$}", w = name_w - 1);
        spans.push(Span::styled(n, menu_style(cur_row)));
        if meta_w > 0 {
            spans.push(Span::styled(format!(" {meta:<w$}", w = meta_w - 1), meta_style(cur_row)));
        }
        out.push(clip_line(Line::from(spans), max_total + 2));
    }
    out
}

// ── the agent's checklist ───────────────────────────────────────────────────

/// Rows of the agent's checklist (`todo_write`), drawn between the chat and the prompt.
///
/// Finished work is history: it folds into one «✓ N готово» row, opened on a
/// click. Screen space then goes to what is left — the step being worked on and
/// the ones after it — carrying how long the current one has been running, since
/// that is the one number a person watching an unattended run needs.
pub(crate) fn todo_rows(app: &App, width: u16, max_rows: usize) -> Vec<Line<'static>> {
    let todos = &app.status.todos;
    let folded = app.todos_folded;
    // Filled again below; a stale row would make a click open a sub-agent that is
    // not on screen (a folded panel draws none of them).
    app.todos_sub_rows.borrow_mut().clear();
    if todos.is_empty() || max_rows < if folded { 1 } else { 2 } {
        return Vec::new();
    }
    let (muted, accent, ok, text) = if app.ui.is_premium() {
        let p = &app.ui.palette;
        (p.muted, p.accent, p.ok, p.text)
    } else {
        (Color::Rgb(0x88, 0x88, 0x88), Color::Rgb(0xff, 0xb3, 0x00), Color::Rgb(0x4c, 0xaf, 0x50), Color::Reset)
    };
    let done = todos.iter().filter(|t| t.is_done()).count();
    // The header is the fold button: `▾` open, `▸` folded. Folded, it also says what is going on now.
    let mut head = vec![
        Span::styled(if folded { "  ▸ Задачи " } else { "  ▾ Задачи " }, Style::default().fg(muted).add_modifier(Modifier::BOLD)),
        Span::styled(format!("{done}/{}", todos.len()), Style::default().fg(muted)),
    ];
    if folded {
        let now = todos.iter().find(|t| t.is_active()).or_else(|| todos.iter().find(|t| !t.is_done()));
        if let Some(t) = now {
            head.push(Span::styled("  ·  ", Style::default().fg(muted)));
            head.push(Span::styled(t.content.clone(), Style::default().fg(text)));
            if let Some((label, _)) = step_elapsed(t.since) {
                head.push(Span::styled(label, Style::default().fg(muted)));
            }
        }
        return vec![clip_line(Line::from(head), width as usize)];
    }
    let mut rows = vec![clip_line(Line::from(head), width as usize)];
    let mut room = max_rows - 1;

    if done > 0 && room > 0 {
        let open = app.todos_done_expanded;
        rows.push(clip_line(
            Line::from(vec![
                Span::styled(if open { "  ▾ " } else { "  ▸ " }, Style::default().fg(muted)),
                Span::styled(format!("✓ {done} готово"), Style::default().fg(ok)),
                Span::styled(
                    if open { "" } else { "  · клик — показать" },
                    Style::default().fg(muted),
                ),
            ]),
            width as usize,
        ));
        room -= 1;
        if open {
            for t in todos.iter().filter(|t| t.is_done()) {
                if room == 0 {
                    break;
                }
                rows.push(clip_line(
                    Line::from(vec![
                        Span::styled("  ✓ ", Style::default().fg(ok)),
                        Span::styled(
                            t.content.clone(),
                            Style::default().fg(muted).add_modifier(Modifier::CROSSED_OUT),
                        ),
                    ]),
                    width as usize,
                ));
                room -= 1;
            }
        }
    }

    let rest: Vec<&crate::app::TodoItem> = todos.iter().filter(|t| !t.is_done()).collect();
    if rest.is_empty() || room == 0 {
        return rows;
    }
    // The active step is the question this panel answers, so the window follows
    // it rather than starting from the top. Clipping is said out loud: a plan
    // with a silent hole is worse than a short one.
    let focus = rest.iter().position(|t| t.is_active()).unwrap_or(0);
    let fits = rest.len() <= room;
    let list_room = if fits { room } else { room.saturating_sub(1) }.max(1);
    let base_start = if rest.len() <= list_room {
        0
    } else {
        focus.saturating_sub(1).min(rest.len() - list_room)
    };
    let max_start = rest.len().saturating_sub(list_room);
    // How far the wheel may move this window: back to the first step, forward to
    // the last one. Past the list it does nothing, so a stray flick is harmless.
    app.todos_scroll_bounds
        .set((base_start as i32, (max_start - base_start) as i32));
    let shift = app
        .todos_scroll
        .clamp(-((max_start - base_start) as i32), base_start as i32);
    let start = ((base_start as i32) - shift).clamp(0, max_start as i32) as usize;
    let end = (start + list_room).min(rest.len());

    for t in rest.iter().take(end).skip(start) {
        let (mark, mark_style, text_style) = if t.is_active() {
            (
                "  ▶ ",
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
                Style::default().fg(text).add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ○ ", Style::default().fg(muted), Style::default().fg(text))
        };
        let mut spans = vec![
            Span::styled(mark, mark_style),
            Span::styled(t.content.clone(), text_style),
        ];
        if t.is_active() {
            if let Some((label, style)) = step_elapsed(t.since) {
                spans.push(Span::styled(label, style));
            }
        }
        rows.push(clip_line(Line::from(spans), width as usize));
    }

    let hidden_before = start;
    let hidden_after = rest.len() - end;
    if (hidden_before > 0 || hidden_after > 0) && rows.len() < max_rows {
        let mut parts = Vec::new();
        if hidden_before > 0 {
            parts.push(format!("{hidden_before} выше"));
        }
        if hidden_after > 0 {
            parts.push(format!("{hidden_after} ниже"));
        }
        if app.todos_scroll_bounds.get().0 > 0 || app.todos_scroll_bounds.get().1 > 0 {
            parts.push("колесо".to_string());
        }
        rows.push(clip_line(
            Line::from(Span::styled(
                format!("    … {}", parts.join(" · ")),
                Style::default().fg(muted),
            )),
            width as usize,
        ));
    }

    // Delegated work has a plan of its own. Showing only the main list left the
    // interesting part of an unattended run — what the sub-agent is doing — invisible.
    let mut sub_rows: Vec<(u16, usize)> = Vec::new();
    for (i, sub) in app.status.todos_sub.iter().enumerate() {
        if rows.len() >= max_rows {
            break;
        }
        let open = app.todos_sub_open.get(i).copied().unwrap_or(false);
        let mut spans = vec![
            Span::styled(if open { "  ▾ ↳ " } else { "  ▸ ↳ " }, Style::default().fg(muted)),
            Span::styled(
                sub.name.clone(),
                Style::default().fg(text).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {}/{}", sub.done(), sub.items.len()),
                Style::default().fg(muted),
            ),
        ];
        if let Some(t) = sub.current() {
            spans.push(Span::styled("  ·  ", Style::default().fg(muted)));
            spans.push(Span::styled(t.content.clone(), Style::default().fg(text)));
            if let Some((label, _)) = step_elapsed(t.since) {
                spans.push(Span::styled(label, Style::default().fg(muted)));
            }
        }
        sub_rows.push((rows.len() as u16, i));
        rows.push(clip_line(Line::from(spans), width as usize));
        if open {
            for t in &sub.items {
                if rows.len() >= max_rows {
                    break;
                }
                let (mark, style) = if t.is_done() {
                    ("    ✓ ", Style::default().fg(ok))
                } else if t.is_active() {
                    ("    ▶ ", Style::default().fg(accent).add_modifier(Modifier::BOLD))
                } else {
                    ("    ○ ", Style::default().fg(muted))
                };
                rows.push(clip_line(
                    Line::from(vec![
                        Span::styled(mark, style),
                        Span::styled(t.content.clone(), Style::default().fg(text)),
                    ]),
                    width as usize,
                ));
            }
        }
    }
    app.todos_sub_rows.borrow_mut().clear();
    app.todos_sub_rows.borrow_mut().extend(sub_rows);
    rows
}

/// After this long the current step stops looking like normal work.
const STEP_HANG_SECONDS: f64 = 600.0;

/// How long the step has been running; a warning once it looks stuck.
fn step_elapsed(since: Option<f64>) -> Option<(String, Style)> {
    let since = since?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    let secs = (now - since).max(0.0);
    let label = fmt_step_time(secs);
    if secs >= STEP_HANG_SECONDS {
        Some((
            format!("  ·  {label} · долго"),
            Style::default().fg(Color::Rgb(0xff, 0x9d, 0x00)).add_modifier(Modifier::BOLD),
        ))
    } else {
        Some((format!("  ·  {label}"), Style::default().add_modifier(Modifier::DIM)))
    }
}

pub(crate) fn fmt_step_time(secs: f64) -> String {
    let s = secs as u64;
    if s < 60 {
        format!("{s}с")
    } else if s < 3600 {
        format!("{}м {:02}с", s / 60, s % 60)
    } else {
        format!("{}ч {:02}м", s / 3600, (s % 3600) / 60)
    }
}

// ── background commands ────────────────────────────────────────────────────────

/// The background panel's rows plus the controls that live inside them.
pub(crate) struct BgPanel {
    pub rows: Vec<Line<'static>>,
    /// Row of the nested "Завершённые" fold header, when it is drawn.
    pub done_header: Option<usize>,
    /// Finished rows that open their output, each with the pid it opens.
    pub done_rows: Vec<(usize, String)>,
    /// Running rows, each with the pid whose window a click opens.
    pub task_rows: Vec<(usize, String)>,
    /// Running rows that carry the ⏹ stop button, each with the pid it stops.
    pub stop_rows: Vec<(usize, String)>,
}

impl BgPanel {
    fn empty() -> Self {
        Self {
            rows: Vec::new(),
            done_header: None,
            done_rows: Vec::new(),
            task_rows: Vec::new(),
            stop_rows: Vec::new(),
        }
    }
}

/// Rows of the background commands, drawn under the checklist.
///
/// Nothing at all when nothing runs: this is a glance at work in progress, not a
/// permanent panel. Folded it is one row that still says what is going on. Open,
/// it lists the running commands and a nested "Завершённые" fold — folded too,
/// because what has finished is history, not news.
pub(crate) fn background_rows(app: &App, width: u16, max_rows: usize) -> BgPanel {
    let tasks = &app.status.background;
    let folded = app.bg_folded;
    let done_total = app.status.background_done.max(app.status.bg_done.len());
    // A running command — or history worth opening — earns the row. Without this,
    // the panel would vanish the moment the last process exits, taking the way
    // back to its output with it.
    if (tasks.is_empty() && done_total == 0) || max_rows < if folded { 1 } else { 2 } {
        return BgPanel::empty();
    }
    let (muted, accent, text, ok, bad) = if app.ui.is_premium() {
        let p = &app.ui.palette;
        (p.muted, p.accent, p.text, p.ok, p.warn)
    } else {
        (
            Color::Rgb(0x88, 0x88, 0x88),
            Color::Rgb(0xff, 0xb3, 0x00),
            Color::Reset,
            Color::Rgb(0x4c, 0xaf, 0x50),
            Color::Rgb(0xe0, 0x52, 0x52),
        )
    };
    let mut head = vec![
        Span::styled(
            if folded { "  ▸ Фоновые задачи " } else { "  ▾ Фоновые задачи " },
            Style::default().fg(muted).add_modifier(Modifier::BOLD),
        ),
        Span::styled(tasks.len().to_string(), Style::default().fg(muted)),
    ];
    if folded {
        // Folded, the header still answers "what is going on?" without a click.
        let tail = match tasks.first() {
            Some(task) => Some(task.command.clone()),
            None => Some(format!("завершено {done_total}")),
        };
        if let Some(tail) = tail {
            head.push(Span::styled("  ·  ", Style::default().fg(muted)));
            head.push(Span::styled(tail, Style::default().fg(text)));
        }
        return BgPanel { rows: vec![clip_line(Line::from(head), width as usize)], ..BgPanel::empty() };
    }
    let mut rows = vec![clip_line(Line::from(head), width as usize)];
    let mut task_rows: Vec<(usize, String)> = Vec::new();
    let mut stop_rows: Vec<(usize, String)> = Vec::new();
    let room = max_rows.saturating_sub(1);
    let mut used = 0usize;
    // Reserve a row for the "… ещё" note whenever something has to be cut: a list
    // that silently stops at the edge looks complete.
    let mut visible = tasks.len().min(room);
    if tasks.len() > visible && room > 1 {
        visible = room - 1;
    }
    for task in tasks.iter().take(visible) {
        // A process with a terminal of its own can be answered; the glyph says so.
        let mark = if task.pty { "⌨" } else { "●" };
        task_rows.push((rows.len(), task.pid.clone()));
        // The ⏹ sits in the same column on every row, so it can be clicked without
        // guessing where the command text ends. Text is clipped to make room for it:
        // a stop button the command name can push off the edge is not a button.
        let text_room = (width as usize).saturating_sub(2);
        let head = clip_line(
            Line::from(vec![
                Span::styled(format!("  {mark} "), Style::default().fg(accent)),
                Span::styled(task.command.clone(), Style::default().fg(text)),
                Span::styled(
                    format!("  · pid {} · {}", task.pid, mmss(task.elapsed)),
                    Style::default().fg(muted),
                ),
            ]),
            text_room,
        );
        let pad = text_room.saturating_sub(head.width());
        let mut spans = head.spans.clone();
        spans.push(Span::raw(" ".repeat(pad)));
        // Armed twice in a row for the same row: bright red, it is about to happen.
        let armed = app.pending_stop.as_deref() == Some(task.pid.as_str());
        let colour = if armed { Color::Rgb(0xe0, 0x52, 0x52) } else { muted };
        spans.push(Span::styled(if armed { "⏹" } else { "⏹" }, Style::default().fg(colour)));
        stop_rows.push((rows.len(), task.pid.clone()));
        rows.push(Line::from(spans));
        used += 1;
    }
    if tasks.len() > visible && used < room {
        rows.push(clip_line(
            Line::from(Span::styled(
                format!("    … ещё {}", tasks.len() - visible),
                Style::default().fg(muted),
            )),
            width as usize,
        ));
        used += 1;
    }
    let mut done_header = None;
    let mut done_rows = Vec::new();
    // A header that could not show a single entry is worse than no header.
    if done_total > 0 && room.saturating_sub(used) >= 2 {
        done_header = Some(rows.len());
        rows.push(clip_line(
            Line::from(vec![
                Span::styled(
                    if app.bg_done_folded { "    ▸ Завершённые " } else { "    ▾ Завершённые " },
                    Style::default().fg(muted).add_modifier(Modifier::BOLD),
                ),
                Span::styled(done_total.to_string(), Style::default().fg(muted)),
            ]),
            width as usize,
        ));
        used += 1;
        if !app.bg_done_folded {
            let space = room.saturating_sub(used);
            // Newest first: the one that just finished is the interesting one.
            for item in app.status.bg_done.iter().rev().take(space) {
                let (mark, style) = if item.succeeded() {
                    ("✓", Style::default().fg(ok))
                } else {
                    ("✗", Style::default().fg(bad))
                };
                let code = item.code.map(|c| c.to_string()).unwrap_or_else(|| "?".to_string());
                done_rows.push((rows.len(), item.pid.clone()));
                rows.push(clip_line(
                    Line::from(vec![
                        Span::styled(format!("      {mark} "), style),
                        Span::styled(format!("{code}  "), style),
                        Span::styled(item.command.clone(), Style::default().fg(text)),
                        Span::styled(format!("  · {}", mmss(item.secs)), Style::default().fg(muted)),
                    ]),
                    width as usize,
                ));
            }
        }
    }
    BgPanel { rows, done_header, done_rows, task_rows, stop_rows }
}

/// `0:42` / `7s` — the shape the CLI already uses for a running job.
fn mmss(secs: u64) -> String {
    if secs >= 60 {
        format!("{}:{:02}", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

// ── questionary-style prompts ───────────────────────────────────────────────

fn answer_pointer(selected: bool) -> Span<'static> {
    Span::raw(if selected { " » " } else { "   " })
}

/// The approval question, drawn as a framed, unmissable block instead of a line
/// that blends into the transcript. A question the user has to answer must not
/// look like output they can scroll past.
fn approval_rows(app: &App, a: &ApprovalData, width: u16) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let avail = width as usize;
    if avail < 12 {
        return approval_body(app, a, width);
    }
    let inner = avail - 4;
    let border = Style::default().fg(p.warn);
    let label = if a.destructive {
        "! ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ — ОПАСНОЕ ДЕЙСТВИЕ"
    } else {
        "! ТРЕБУЕТСЯ ПОДТВЕРЖДЕНИЕ"
    };
    let label: String = label.chars().take(inner.saturating_sub(2)).collect();
    let pad = inner.saturating_sub(UnicodeWidthStr::width(label.as_str()) + 1);
    let mut rows = Vec::new();
    rows.push(Line::from(vec![
        Span::styled("╭─ ", border),
        Span::styled(
            label,
            Style::default().fg(p.warn).bg(p.surface_hi).add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" {}╮", "─".repeat(pad)), border),
    ]));
    for line in approval_body(app, a, inner as u16) {
        rows.push(box_line(line, inner, border));
    }
    rows.push(box_line(
        Line::from(Span::styled(
            "Enter — выбрать · ↑↓ — вариант · Esc — отклонить",
            Style::default().fg(p.muted).add_modifier(Modifier::DIM),
        )),
        inner,
        border,
    ));
    rows.push(Line::from(Span::styled(format!("╰{}╯", "─".repeat(avail - 2)), border)));
    rows
}

/// One body row with the frame's side bars added and the width filled exactly.
fn box_line(line: Line<'static>, inner: usize, border: Style) -> Line<'static> {
    let mut spans = vec![Span::styled("│ ", border)];
    let mut used = 2usize;
    for span in line.spans {
        used += UnicodeWidthStr::width(span.content.as_ref());
        spans.push(span);
    }
    if used < inner + 2 {
        spans.push(Span::raw(" ".repeat(inner + 2 - used)));
    }
    spans.push(Span::styled(" │", border));
    Line::from(spans)
}

/// The content of the question, in questionary's own wording.
fn approval_body(app: &App, a: &ApprovalData, width: u16) -> Vec<Line<'static>> {
    let p = a.resolved_prompt();
    let fp = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (a.id, a.destructive, a.action.len()).hash(&mut h);
        h.finish()
    };
    let header = app.cached_block(usize::MAX, 0, fp, width, || {
        (
            "approval_header",
            serde_json::json!({"action": a.action, "destructive": a.destructive}),
        )
    });
    let mut rows: Vec<Line<'static>> = header.iter().cloned().collect();
    let trows = title_rows(&p.question, width as usize, 8);
    if p.confirm {
        rows.extend(title_lines(&trows, vec![Span::raw("(y/N) ")]));
        return rows;
    }
    rows.extend(title_lines(&trows, vec![Span::raw("(Use arrow keys)")]));
    for (i, c) in p.choices.iter().enumerate() {
        rows.push(Line::from(vec![answer_pointer(i == a.selected), Span::raw(c.clone())]));
    }
    rows
}

/// A generic select / checkbox / text prompt from the bridge, in questionary's
/// look. Returns the rows and where the cursor goes (text prompts only).
fn select_rows(s: &SelectModalData, width: u16, max_rows: usize) -> (Vec<Line<'static>>, Option<(u16, u16)>) {
    let w = width as usize;
    if s.options.is_empty() {
        let typed = crate::ansi::strip_ansi(&s.filter);
        let trows = title_rows(&s.title, w, max_rows.saturating_sub(1).max(1));
        let last = trows.last().cloned().unwrap_or_default();
        // The typed text goes on the row of the last line of the title.
        let head = if trows.len() == 1 { format!("? {last} ") } else { format!("  {last} ") };
        let budget = w.saturating_sub(UnicodeWidthStr::width(head.as_str()) + 1).max(1);
        // The window follows the caret, not the end of the text: with the caret in
        // the middle the character under it has to stay on screen.
        let total = typed.chars().count();
        let caret = s.filter_cursor.min(total);
        let start = if total > budget {
            caret.saturating_sub(budget).min(total - budget)
        } else {
            0
        };
        let shown: String = typed.chars().skip(start).take(budget).collect();
        let tail = vec![Span::styled(
            shown.clone(),
            Style::default().fg(Color::Rgb(0xff, 0x9d, 0x00)).add_modifier(Modifier::BOLD),
        )];
        let rows = title_lines(&trows, tail);
        let before: String = shown.chars().take(caret - start).collect();
        let col = UnicodeWidthStr::width(head.as_str()) + UnicodeWidthStr::width(before.as_str());
        return (rows, Some((col as u16, (trows.len() - 1) as u16)));
    }

    let filtered = s.filtered_options();
    let long = s.options.len() > 12;
    let instruction = if s.multi {
        "(Use arrow keys to move, <space> to select)"
    } else if long {
        "(печатайте для поиска, ↑↓ выбор, Enter подтвердить)"
    } else {
        "(Use arrow keys)"
    };
    // A long title takes the rows it needs, but at least three are kept for the choices.
    let trows = title_rows(&s.title, w, max_rows.saturating_sub(3).max(1));
    let title_h = trows.len();
    let mut rows = title_lines(&trows, vec![Span::raw(instruction)]);

    let list_h = max_rows.saturating_sub(title_h + usize::from(!s.filter.is_empty())).max(1);
    let pos = filtered.iter().position(|(o, _)| *o == s.selected_idx).unwrap_or(0);
    let top = if pos >= list_h { pos + 1 - list_h } else { 0 };
    for (orig, opt) in filtered.iter().skip(top).take(list_h) {
        let pointed = *orig == s.selected_idx;
        let mut spans = vec![answer_pointer(pointed)];
        if s.multi {
            let on = s.checked.get(*orig).copied().unwrap_or(false);
            spans.push(Span::raw(if on { "● " } else { "○ " }));
        }
        spans.push(Span::raw((*opt).clone()));
        rows.push(clip_line(Line::from(spans), w));
    }
    if !s.filter.is_empty() {
        let style = if filtered.is_empty() {
            Style::default().fg(Color::Rgb(0xff, 0x00, 0x00)).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(0x00, 0xff, 0x00)).add_modifier(Modifier::BOLD)
        };
        let prefix = "  Search filter: ";
        rows.push(Line::from(Span::styled(format!("{prefix}{}", s.filter), style)));
        // The caret lives inside the typed text, so the arrows can walk it.
        let before: String = s.filter.chars().take(s.filter_cursor).collect();
        let col = UnicodeWidthStr::width(prefix) + UnicodeWidthStr::width(before.as_str());
        let filter_row = rows.len() - 1;
        return (rows, Some((col as u16, filter_row as u16)));
    }
    (rows, None)
}

// ── queue panel and attachment strip ────────────────────────────────────────

/// A clickable row of a panel drawn above the prompt: `(row, x, width, action)`.
pub(crate) type StripClick = (usize, u16, u16, ClickAction);

/// The queued messages, above the prompt where they can actually be reached.
///
/// The REPL's box used to sit inside the chat: it was a picture of the queue,
/// not a control — nothing could be clicked, and a person had to know that
/// `/queue edit 2 …` existed to change anything. Every row here is a button.
/// How many queued messages the panel shows at once. The rest are reachable with
/// the wheel over the panel, or with `/queue <№>`.
pub(crate) const QUEUE_ROWS: usize = 6;

pub(crate) fn queue_panel(app: &App, width: u16) -> (Vec<Line<'static>>, Vec<StripClick>) {
    let w = width as usize;
    if app.message_queue.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let dim = Style::default().add_modifier(Modifier::DIM);
    let n = app.message_queue.len();
    let head = format!("Очередь сообщений ({n})");
    let hint = if app.queue_edit.is_some() {
        "  · Enter — заменить · Esc — отменить"
    } else {
        "  · клик — править · ✕ — убрать"
    };
    // The hint used to share the header row, where a narrow window clipped it away
    // and left a panel that never said how to use it. It gets its own row now.
    let mut rows: Vec<Line<'static>> = Vec::new();
    if head.chars().count() + hint.chars().count() <= w {
        rows.push(Line::from(vec![
            Span::styled(head, Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(hint, dim),
        ]));
    } else {
        rows.push(Line::from(Span::styled(
            head,
            Style::default().add_modifier(Modifier::BOLD),
        )));
        if w >= 22 {
            rows.push(clip_line(
                Line::from(Span::styled(hint.trim_start().to_string(), dim)),
                w,
            ));
        }
    }

    let hidden = n.saturating_sub(QUEUE_ROWS);
    let top = app.queue_scroll.min(hidden);
    let mut clicks: Vec<StripClick> = Vec::new();
    for (i, msg) in app.message_queue.iter().enumerate().skip(top).take(QUEUE_ROWS) {
        let editing = app.queue_edit == Some(i);
        let num = format!("{}{}. ", if editing { "✎" } else { " " }, i + 1);
        // A waiting message says why it waits; everything after the text must fit.
        let badge = msg.badge().unwrap_or_default();
        let badge_w = if badge.is_empty() {
            0
        } else {
            badge.chars().count() + 2
        };
        let body_w = w.saturating_sub(num.chars().count() + badge_w + 1);
        let flat: String = msg.text.replace('\n', " ");
        let mut body: String = flat.chars().take(body_w).collect();
        while body.chars().count() < body_w {
            body.push(' ');
        }
        let style = if editing {
            Style::default().fg(Color::Rgb(0xff, 0x9d, 0x00))
        } else {
            Style::default().fg(Color::Indexed(7))
        };
        let mut spans = vec![
            Span::styled(num, dim),
            Span::styled(crate::ansi::strip_ansi(&body), style),
        ];
        if !badge.is_empty() {
            spans.push(Span::styled(
                format!(" {badge}"),
                Style::default()
                    .fg(Color::Rgb(0x58, 0xa6, 0xff))
                    .add_modifier(Modifier::DIM),
            ));
        }
        spans.push(Span::styled(
            "✕",
            Style::default().fg(Color::Rgb(0xff, 0x5f, 0x5f)).add_modifier(Modifier::BOLD),
        ));
        let row = rows.len();
        // The row edits; only the last cell deletes.
        clicks.push((row, 0, (w as u16).saturating_sub(1), ClickAction::EditQueue(i)));
        clicks.push((row, (w as u16).saturating_sub(1), 1, ClickAction::DropQueue(i)));
        rows.push(Line::from(spans));
    }
    if hidden > 0 {
        // Said out loud: a queue that silently hides its tail is a queue that loses
        // messages, and the wheel is not discoverable on its own.
        rows.push(clip_line(
            Line::from(Span::styled(
                format!("  …и ещё {hidden} · колесо или /queue"),
                dim,
            )),
            w,
        ));
    }

    (rows, clicks)
}

/// The pictures attached to the next message: a strip of previews.
///
/// Each one opens on a click and carries a ✕ in its corner, so an accidental
/// paste can be taken back without hunting for a command.
///
/// The preview is drawn from the wide frame Python sends, averaged down to the
/// room available. It used to be an 18-column thumb blown up to fill the panel,
/// which is what made a pasted screenshot look like a mosaic.
pub(crate) fn attachment_strip(app: &App, width: u16) -> (Vec<Line<'static>>, Vec<StripClick>) {
    let w = width as usize;
    if app.attachments.is_empty() || w == 0 {
        return (Vec::new(), Vec::new());
    }
    // A tall picture would otherwise eat the screen: a preview shows its top.
    const MAX_ROWS: usize = 8;
    // Wider than the old thumb, narrower than the whole screen: a preview is a
    // glance, but a coarse one is not a glance at anything.
    const MAX_COLS: usize = 56;

    let strip_cols = |a: &crate::app::Attachment| -> usize {
        let source = a.full.as_ref().unwrap_or(&a.thumb);
        source.cols.min(MAX_COLS).min(w.saturating_sub(1).max(1))
    };

    // Previews sit side by side until the width runs out.
    let mut groups: Vec<Vec<(usize, &crate::app::Attachment)>> = Vec::new();
    let mut cur: Vec<(usize, &crate::app::Attachment)> = Vec::new();
    let mut used = 0usize;
    for (i, a) in app.attachments.iter().enumerate() {
        let cw = strip_cols(a) + 1;
        if !cur.is_empty() && used + cw > w {
            groups.push(std::mem::take(&mut cur));
            used = 0;
        }
        used += cw;
        cur.push((i, a));
    }
    if !cur.is_empty() {
        groups.push(cur);
    }

    let mut rows: Vec<Line<'static>> = Vec::new();
    let mut shows: Vec<StripClick> = Vec::new();
    let mut drops: Vec<StripClick> = Vec::new();
    for group in groups {
        let rendered: Vec<(usize, &crate::app::Attachment, Vec<Line<'static>>)> = group
            .iter()
            .map(|(idx, a)| {
                let source = a.full.as_ref().unwrap_or(&a.thumb);
                (*idx, *a, crate::cards::image_lines_scaled(source, strip_cols(a)))
            })
            .collect();
        let h = rendered.iter().map(|(_, _, l)| l.len().min(MAX_ROWS)).max().unwrap_or(0);
        let mut cells: Vec<Vec<Span<'static>>> = vec![Vec::new(); h];
        let mut x = 0u16;
        for (idx, _a, lines) in &rendered {
            let cols = lines.first().map(|l| l.width()).unwrap_or(0);
            for (r, cell_row) in cells.iter_mut().enumerate() {
                match lines.get(r) {
                    Some(line) => cell_row.extend(line.spans.iter().cloned()),
                    None => cell_row.push(Span::raw(" ".repeat(cols))),
                }
                cell_row.push(Span::raw(" "));
            }
            // The ✕ takes the top-right pixel cell of the picture.
            if let Some(first) = cells.first_mut() {
                if let Some(corner) = first.iter_mut().rev().nth(1) {
                    *corner = Span::styled(
                        "✕",
                        Style::default().fg(Color::Rgb(0xff, 0x5f, 0x5f)).add_modifier(Modifier::BOLD),
                    );
                }
            }
            let base = rows.len();
            shows.push((base, x, cols as u16, ClickAction::ShowAttachment(*idx)));
            drops.push((base, x + cols as u16 - 1, 1, ClickAction::DropAttachment(*idx)));
            x += cols as u16 + 1;
        }
        for line in cells {
            rows.push(Line::from(line));
        }
    }
    // Registered last, so the corner cell belongs to the ✕ rather than to the
    // picture underneath it.
    let mut clicks = shows;
    clicks.extend(drops);
    (rows, clicks)
}

// ── frame ───────────────────────────────────────────────────────────────────

/// How selected transcript text is painted: reversed in the classic look, a soft
/// accent background (text keeps its colours) in the premium one.
fn selection_style(app: &App) -> Style {
    if app.ui.is_premium() {
        Style::default().bg(app.ui.palette.selection)
    } else {
        Style::default().add_modifier(Modifier::REVERSED)
    }
}

/// Write a row only if it lies inside the frame. ratatui indexes its buffer
/// directly, so a row computed one past the bottom edge (a short terminal with
/// a menu open, say) would otherwise panic instead of being clipped.
fn put_line(buf: &mut ratatui::buffer::Buffer, area: Rect, y: u16, line: &Line<'_>, width: u16) {
    if y >= area.y.saturating_add(area.height) {
        return;
    }
    buf.set_line(area.x, y, line, width);
}

/// Draw one frame.
pub fn draw_ui(f: &mut Frame, app: &App) {
    app.clear_clickables();
    app.clear_hoverables();
    let area = f.area();
    if area.width < 8 || area.height < 4 {
        return;
    }
    // A frame is a full redraw, not a patch: when a card shrinks (a restored
    // transcript replacing a longer live one, a tool result losing its stream),
    // `set_line` writes only the cells it owns. Without this reset the old
    // characters stay in the buffer and bleed into the next card ("targetSqoGT").
    f.buffer_mut().reset();
    // Nothing is editable again until a prompt says otherwise: a question or the
    // select modal replaces the prompt, and a click there must not move a caret
    // that is not on screen.
    app.set_input_box(0, 1, 0, 0, 0);
    let w = area.width;
    let h = area.height as usize;

    let premium = app.ui.is_premium();
    // Bottom stack: prompt (or a question), completion menu, toolbar.
    let max_prompt = (h * 4 / 10).clamp(3, 14);
    let (mut prompt_rows, mut cursor): (Vec<Line<'static>>, Option<(u16, u16)>) =
        if let Some(a) = app.active_approval.as_ref() {
            (approval_rows(app, a, w), None)
        } else if let Some(s) = app.active_select.as_ref() {
            select_rows(s, w, max_prompt.max(10))
        } else if premium {
            // Soft panel edges only where there is height to spare.
            let b = premium::prompt_block(app, w, max_prompt, h >= 14);
            (b.rows, b.cursor)
        } else {
            let b = prompt_block_classic(app, w, max_prompt);
            (b.rows, b.cursor)
        };
    let modal = app.active_approval.is_some() || app.active_select.is_some();
    let mut menu = if modal {
        Vec::new()
    } else if premium {
        premium::menu_rows(app, w)
    } else {
        menu_rows(app, w)
    };
    let toolbar_h = 1usize;
    // The menu yields to the prompt and the toolbar on a short terminal: with
    // both of those on screen there are only `h - 2` rows left for it.
    menu.truncate(h.saturating_sub(toolbar_h + 1));
    // The queue and the attached pictures sit between the checklist and the prompt.
    // They are measured HERE, before the bottom height is decided: drawn later without
    // reserving their rows, they pushed the prompt out of the area, and a draft vanished
    // exactly when a picture was attached.
    let (q_rows, q_clicks) = queue_panel(app, w);
    let (mut a_rows, mut a_clicks) = attachment_strip(app, w);
    // The agent's checklist sits between the chat and the prompt. It is the first thing to give way:
    // never in front of a question, and only with room to spare for the chat above it.
    // At most six rows (the header and five steps): the checklist is a glance, not the screen. A
    // click on its header folds it to that one row.
    let todo_room = h
        .saturating_sub(prompt_rows.len() + menu.len() + toolbar_h + q_rows.len() + a_rows.len() + 5)
        .min(6);
    let todo = if modal { Vec::new() } else { todo_rows(app, w, todo_room) };
    // Running background commands go right under the checklist and give way first
    // after it: five rows is already a lot for a list that is usually one item.
    //
    // Only two rows are held back for the chat. Holding four used to push the panel
    // off the screen entirely as soon as a draft wrapped to three lines — so it
    // vanished exactly while a person was writing, when the header earns its row.
    let bg_room = h
        .saturating_sub(prompt_rows.len() + menu.len() + toolbar_h + todo.len()
            + q_rows.len() + a_rows.len() + 2)
        .min(8);
    let bg = if modal {
        BgPanel::empty()
    } else {
        background_rows(app, w, bg_room)
    };
    // At the very least one transcript row survives, however tall the draft: a
    // panel with nothing above it is a wall, not a glance.
    let bg_rows = if h.saturating_sub(prompt_rows.len() + menu.len() + toolbar_h + 1) < bg.rows.len() {
        Vec::new()
    } else {
        bg.rows.clone()
    };
    let bg = BgPanel { rows: bg_rows, ..bg };
    // The stack may never squeeze the transcript out entirely.
    let reserved = prompt_rows.len() + menu.len() + toolbar_h;
    if reserved + 1 > h {
        let keep = h.saturating_sub(toolbar_h + menu.len() + 1).max(1);
        let skip = prompt_rows.len().saturating_sub(keep);
        prompt_rows = prompt_rows.into_iter().skip(skip).collect();
        cursor = cursor.map(|(x, y)| (x, y.saturating_sub(skip as u16)));
    }
    // A preview is worth less than the prompt: if the stack still does not fit, the strip
    // gives way row by row instead of pushing the draft off the screen.
    while !a_rows.is_empty()
        && todo.len() + bg.rows.len() + q_rows.len() + a_rows.len()
            + prompt_rows.len() + menu.len() + toolbar_h + 1 > h
    {
        a_rows.pop();
    }
    a_clicks.retain(|(row, _, _, _)| *row < a_rows.len());
    let bottom_h = todo.len() + bg.rows.len() + q_rows.len() + a_rows.len()
        + prompt_rows.len() + menu.len() + toolbar_h;
    let view_max = h.saturating_sub(bottom_h).max(1);
    // Empty space the user scrolled open under the newest output (`App::lower`);
    // the chat always keeps at least half of its rows.
    let over = app.overscroll.min(view_max / 2);
    let chat_max = view_max - over;

    let t = build_transcript(app, w, chat_max, app.scroll_offset);
    app.max_scroll
        .set(if t.exact { t.total.saturating_sub(chat_max) } else { t.total.max(t.scroll + chat_max) });
    app.scroll_used.set(Some(t.scroll));
    // Pin the text at the top of a scrolled-back view, so output that arrives below
    // it (or a prompt that grows) does not slide it. Rows of the live area have no
    // fixed place in the history and cannot be pinned.
    app.scroll_anchor.set(match t.rows.first() {
        Some(r) if t.scroll > 0 && r.anchor.0 < app.messages.len() => {
            Some(crate::app::ScrollAnchor { item: r.anchor.0, row: r.anchor.1, offset: t.scroll })
        }
        _ => None,
    });
    // Short conversations sit at the top with the prompt right under them (as in
    // the terminal); long ones fill the viewport.
    let view_h = if t.exact { t.total.min(chat_max) } else { chat_max };

    let buf = f.buffer_mut();
    let mut texts: Vec<String> = Vec::with_capacity(t.rows.len());
    for (i, row) in t.rows.iter().enumerate() {
        let y = area.y + i as u16;
        put_line(buf, area, y, &row.line, w);
        texts.push(crate::view::row_text(&row.line));
        if let Some(click) = &row.click {
            app.register_clickable(area.x, y, w, 1, click.clone());
        }
    }
    // A tool card is a window on its own output, and several can be on screen at once:
    // the wheel has to know which card the pointer is over.
    {
        let mut rects = app.card_rects.borrow_mut();
        rects.clear();
        let maxes = app.card_max.borrow();
        let mut spans: Vec<(usize, usize, usize)> = Vec::new(); // item, first row, last row
        for (i, row) in t.rows.iter().enumerate() {
            match spans.last_mut() {
                Some((item, _, last)) if *item == row.anchor.0 => *last = i,
                _ => spans.push((row.anchor.0, i, i)),
            }
        }
        for (item, first, last) in spans {
            if maxes.get(&item).copied().unwrap_or(0) > 0 {
                rects.push((area.x, area.y + first as u16, w, (last - first + 1) as u16, item));
            }
        }
    }
    {
        let mut vt = app.view_text.borrow_mut();
        vt.top = area.y;
        vt.anchors = t.rows.iter().map(|r| r.anchor).collect();
        vt.cols = t.rows.iter().map(|r| r.cols).collect();
        vt.rows = texts;
        vt.width = w;
        vt.view_h = view_h as u16;
    }

    // Mouse selection over the transcript: each visible row asks which of its
    // columns the selection covers, so it stays on the same text while scrolling.
    if let Some(sel) = app.mouse_sel {
        let style = selection_style(app);
        let vt = app.view_text.borrow();
        for (i, (item, row)) in vt.anchors.iter().enumerate() {
            // The rail, the margin and the padding of a card are not text: never painted.
            let cols = vt.cols.get(i).copied().unwrap_or(crate::view::Cols::ALL);
            let text_w = UnicodeWidthStr::width(vt.rows[i].trim_end()).min(cols.to);
            if let Some((from, to)) = crate::selection::selected_columns(&sel, *item, *row, text_w) {
                let from = from.max(cols.from);
                if from >= to {
                    continue;
                }
                let x = area.x + (from as u16).min(w.saturating_sub(1));
                let width = ((to - from) as u16).min(w.saturating_sub(x - area.x));
                if width > 0 {
                    buf.set_style(Rect { x, y: area.y + i as u16, width, height: 1 }, style);
                }
            }
        }
    }

    // Scroll position while looking at history.
    if t.scroll > 0 && view_h > 1 {
        let range = app.max_scroll.get().max(1);
        let mut state = ScrollbarState::new(range).position(range.saturating_sub(t.scroll.min(range)));
        let bar = Scrollbar::default()
            .orientation(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some(" "))
            .thumb_symbol("▐")
            .style(Style::default().fg(if premium {
                crate::theme::blend(app.ui.palette.accent, app.ui.palette.base, 0.6)
            } else {
                Color::Rgb(0x6e, 0x76, 0x81)
            }));
        f.render_stateful_widget(
            bar,
            Rect { x: area.x, y: area.y, width: area.width, height: view_h as u16 },
            &mut state,
        );
    }

    // While the chat is scrolled back (or scrolled past its end), a chip over the
    // last row of its area jumps to the newest.
    if premium && (t.scroll > 0 || over > 0) && view_h > 1 {
        let (chip, cx) = premium::jump_chip(&app.ui.palette, w);
        let chip_w = line_width(&chip) as u16;
        let cy = area.y + (view_h + over) as u16 - 1;
        f.buffer_mut().set_line(area.x + cx, cy, &chip, chip_w);
        app.register_clickable(area.x + cx, cy, chip_w, 1, ClickAction::ScrollToBottom);
    }

    // The checklist right under the transcript (and the empty space scrolled open), then the prompt.
    let todo_y = area.y + (view_h + over) as u16;
    for (i, row) in todo.iter().enumerate() {
        put_line(f.buffer_mut(), area, todo_y + i as u16, row, w);
    }
    if !todo.is_empty() && todo_y < area.y + area.height {
        // The header row is a button: it folds and unfolds the list.
        app.register_clickable(area.x, todo_y, w, 1, ClickAction::ToggleTodos);
        // The wheel over the panel moves its window, not the chat behind it.
        app.todos_hover.set(Some((area.x, todo_y, w, todo.len() as u16)));
        // Sub-agent rows open their own list.
        for (row, idx) in app.todos_sub_rows.borrow().iter() {
            let y = todo_y + row;
            if y < area.y + area.height {
                app.register_clickable(
                    area.x,
                    y,
                    w,
                    1,
                    ClickAction::ToggleSubTodos(*idx),
                );
            }
        }
        // Under it sits «✓ N готово», itself a button that opens the finished steps.
        let done_count = app.status.todos.iter().filter(|t| t.is_done()).count();
        if !app.todos_folded && done_count > 0 && todo.len() > 1 {
            app.register_clickable(area.x, todo_y + 1, w, 1, ClickAction::ToggleTodosDone);
        }
    } else {
        app.todos_hover.set(None);
        app.todos_scroll_bounds.set((0, 0));
    }
    // The background panel follows the checklist: its header, the nested
    // "Завершённые" fold and every finished row are click targets.
    let bg_y = todo_y + todo.len() as u16;
    for (i, row) in bg.rows.iter().enumerate() {
        put_line(f.buffer_mut(), area, bg_y + i as u16, row, w);
    }
    if !bg.rows.is_empty() && bg_y < area.y + area.height {
        app.register_clickable(area.x, bg_y, w, 1, ClickAction::ToggleBackground);
    }
    if let Some(idx) = bg.done_header {
        let y = bg_y + idx as u16;
        if y < area.y + area.height {
            app.register_clickable(area.x, y, w, 1, ClickAction::ToggleBackgroundDone);
        }
    }
    // Every row opens the window on that task — running or finished: a click is how
    // you look inside a job without printing a hundred lines into the chat.
    for (idx, pid) in bg.task_rows.iter().chain(bg.done_rows.iter()) {
        let y = bg_y + *idx as u16;
        if y < area.y + area.height {
            app.register_clickable(area.x, y, w, 1, ClickAction::OpenBackground(pid.clone()));
        }
    }
    // The ⏹ at the right edge of a running row. Registered after the row itself, so
    // it wins the click where they overlap.
    for (idx, pid) in bg.stop_rows.iter() {
        let y = bg_y + *idx as u16;
        if y < area.y + area.height && w >= 3 {
            app.register_clickable(area.x + w - 2, y, 2, 1, ClickAction::StopBackground(pid.clone()));
        }
    }
    // The queue panel: what goes out once this turn ends. Measured above with the
    // rest of the bottom stack; drawn here so its rows can be clicked.
    let qy = bg_y + bg.rows.len() as u16;
    if q_rows.is_empty() {
        app.queue_hover.set(None);
    } else {
        app.queue_hover.set(Some((area.x, qy, w, q_rows.len() as u16)));
    }
    for (i, row) in q_rows.iter().enumerate() {
        let y = qy + i as u16;
        if y < area.y + area.height {
            put_line(f.buffer_mut(), area, y, row, w);
        }
    }
    for (row, x, cw, action) in q_clicks {
        let y = qy + row as u16;
        if y < area.y + area.height {
            app.register_clickable(area.x + x, y, cw, 1, action);
        }
    }
    // The pictures attached to the next message, right above the prompt.
    let ay = qy + q_rows.len() as u16;
    for (i, row) in a_rows.iter().enumerate() {
        let y = ay + i as u16;
        if y < area.y + area.height {
            put_line(f.buffer_mut(), area, y, row, w);
        }
    }
    for (row, x, cw, action) in a_clicks {
        let y = ay + row as u16;
        if y < area.y + area.height {
            app.register_clickable(area.x + x, y, cw, 1, action);
        }
    }
    let py = ay + a_rows.len() as u16;
    app.set_input_origin(py + app.input_geom.get().pad, area.x);
    for (i, row) in prompt_rows.iter().enumerate() {
        put_line(f.buffer_mut(), area, py + i as u16, row, w);
    }
    // Completion menu under the prompt.
    let my = py + prompt_rows.len() as u16;
    for (i, row) in menu.iter().enumerate() {
        put_line(f.buffer_mut(), area, my + i as u16, row, w);
    }

    // Toolbar on the last row.
    let (bar, targets, hovers) = if premium {
        premium::toolbar(app, w)
    } else {
        let (line, targets) = toolbar(app, w);
        // The classic bar has no chip objects; attach the same spend hover to
        // the summary span by walking the laid-out row.
        let mut hovers = Vec::new();
        let summary = app.status.session_summary.as_str();
        let breakdown = app.status.spend_breakdown.as_str();
        if !summary.is_empty() && !breakdown.is_empty() {
            let mut x = 0u16;
            for span in &line.spans {
                let sw = UnicodeWidthStr::width(span.content.as_ref()) as u16;
                if span.content.as_ref() == summary {
                    hovers.push((x, sw, breakdown.to_string()));
                    break;
                }
                x = x.saturating_add(sw);
            }
        }
        (line, targets, hovers)
    };
    let ty = area.y + area.height - 1;
    put_line(f.buffer_mut(), area, ty, &bar, w);
    for (x0, x1, action) in targets {
        app.register_clickable(area.x + x0, ty, x1.saturating_sub(x0), 1, action);
    }
    for (x0, width, text) in hovers {
        app.register_hover(area.x + x0, ty, width, 1, text);
    }

    // The pointer is over a control that can explain itself: draw its hint.
    if let Some((mx, my)) = app.mouse_pos {
        if app.mouse_sel.is_none()
            && mx >= area.x
            && mx < area.x + area.width
            && my >= area.y
            && my < area.y + area.height
        {
            if let Some(text) = app.hover_text_at(mx, my) {
                draw_tooltip(f, app, mx, my, &text);
            }
        }
    }

    // The picture on the whole screen is drawn last: it is a look at one thing,
    // and the prompt underneath has no business showing through it.
    if let Some(prev) = app.image_view.as_ref() {
        draw_image_overlay(f, area, app, prev);
        return;
    }

    // The window on a background task sits above everything but a picture: the
    // person asked to look at one job, so it is not drawn behind the chat.
    if app.bg_view.is_some() {
        draw_bg_view(f, area, app);
    }

    if let Some((cx, cy)) = cursor {
        let y = py + cy;
        if cx < w && y < ty {
            f.set_cursor_position((area.x + cx, y));
        }
    }
}

/// The window on one background command: what it is doing, what it printed, and
/// the way out. Its output is a terminal's, not a card's — the newest line at the
/// bottom, the wheel to go back, `PgUp` for a page.
fn draw_bg_view(f: &mut Frame, area: Rect, app: &App) {
    let Some(view) = app.bg_view.as_ref() else { return };
    let pal = &app.ui.palette;
    if area.width < 24 || area.height < 7 {
        return;
    }
    let width = area.width.saturating_sub(4).min(120);
    let height = area.height.saturating_sub(2);
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    let rect = Rect { x, y, width, height };

    f.render_widget(Clear, rect);
    let title = format!(" {} ", view.headline());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(pal.accent))
        .title(Span::styled(
            clip_text(&title, width.saturating_sub(2) as usize),
            Style::default().fg(pal.accent).add_modifier(Modifier::BOLD),
        ));
    f.render_widget(block, rect);

    let inner_w = width.saturating_sub(2);
    // One line of state, the output, one line of keys.
    let body_h = height.saturating_sub(3);
    if inner_w == 0 || body_h == 0 {
        return;
    }
    app.bg_view_bounds.set((inner_w as usize, body_h as usize));

    let state = if !view.error.is_empty() {
        view.error.clone()
    } else {
        match view.status.as_str() {
            "running" => format!("▶ выполняется {} · вывод обновляется", mmss(view.elapsed)),
            "done" => format!(
                "✓ завершён · код {} · {}",
                view.code.map(|c| c.to_string()).unwrap_or_else(|| "?".to_string()),
                mmss(view.elapsed)
            ),
            "stopped" => "■ остановлен".to_string(),
            "opening" => "открываю окно…".to_string(),
            _ => "состояние неизвестно".to_string(),
        }
    };
    put_line(
        f.buffer_mut(),
        rect,
        y + 1,
        &Line::from(Span::styled(clip_text(&state, inner_w as usize), Style::default().fg(pal.muted))),
        inner_w,
    );

    let rows = view.window(inner_w as usize, body_h as usize);
    if rows.is_empty() || rows.iter().all(|r| r.trim().is_empty()) {
        let empty = if view.running() {
            "пока ничего не выведено — окно обновится само"
        } else {
            "вывод не сохранён (хвост держится только для последних задач)"
        };
        put_line(
            f.buffer_mut(),
            rect,
            y + 2,
            &Line::from(Span::styled(clip_text(empty, inner_w as usize), Style::default().fg(pal.muted))),
            inner_w,
        );
    } else {
        for (i, text) in rows.iter().enumerate() {
            let row = y + 2 + i as u16;
            if row >= y + height.saturating_sub(1) {
                break;
            }
            put_line(
                f.buffer_mut(),
                rect,
                row,
                &Line::from(Span::styled(text.clone(), Style::default().fg(pal.text))),
                inner_w,
            );
        }
    }

    let back = view.scroll;
    let hint = if back > 0 {
        format!("Esc — закрыть · колесо вниз / PgDn — к свежему (отступлено {back})")
    } else {
        "Esc — закрыть · колесо вверх / PgUp — прокрутить вывод".to_string()
    };
    put_line(
        f.buffer_mut(),
        rect,
        y + height.saturating_sub(1),
        &Line::from(Span::styled(clip_text(&hint, inner_w as usize), Style::default().fg(pal.muted))),
        inner_w,
    );
}

/// Truncate to `width` characters: the window clips rather than wraps its chrome.
fn clip_text(text: &str, width: usize) -> String {
    let clean = text.replace('\n', " ");
    clean.chars().take(width).collect()
}

/// A picture filling the screen, centred, with its caption and the way out.
fn draw_image_overlay(f: &mut Frame, area: Rect, app: &App, prev: &ImagePreview) {
    let pal = &app.ui.palette;
    f.render_widget(Clear, area);
    // Fit the picture to the window, do not crop it. Clipping a 320-column grid at the
    // window edge meant the right half of a screenshot simply was not on screen — and
    // what did fit was upscaled from whichever narrow grid was in the poll.
    let max_cols = area.width.max(1) as usize;
    let max_rows = area.height.saturating_sub(2).max(1) as usize;
    if prev.full.cols == 0 || prev.full.rows == 0 || area.height < 3 {
        return;
    }
    let mut cols = prev.full.cols.min(max_cols);
    let mut rows = ((prev.full.rows as f32) * (cols as f32) / (prev.full.cols as f32))
        .round()
        .max(1.0) as usize;
    if rows > max_rows {
        rows = max_rows;
        cols = ((prev.full.cols as f32) * (rows as f32) / (prev.full.rows as f32))
            .round()
            .max(1.0) as usize;
    }
    let x = area.x + (area.width.saturating_sub(cols as u16)) / 2;
    let y = area.y + (area.height.saturating_sub(rows as u16 + 2)) / 2;
    let lines = crate::cards::image_lines_scaled(&prev.full, cols);
    // No grid size in the caption: “120×34” read as “the picture is 120 pixels wide”,
    // which says how the terminal draws it, not how much of the frame is shown.
    let caption = if prev.caption.is_empty() {
        "Esc — закрыть".to_string()
    } else {
        format!("{}  ·  Esc — закрыть", prev.caption)
    };
    let caption: String = caption.chars().take(area.width as usize).collect();
    let caption_w = UnicodeWidthStr::width(caption.as_str()) as u16;
    let cx = area.x + (area.width.saturating_sub(caption_w)) / 2;
    let cy = (y + rows as u16 + 1).min(area.y + area.height - 1);
    let buf = f.buffer_mut();
    for (i, line) in lines.iter().enumerate().take(rows) {
        let row = y + i as u16;
        if row >= area.y + area.height {
            break;
        }
        buf.set_line(x, row, line, cols as u16);
    }
    buf.set_line(
        cx,
        cy,
        &Line::from(Span::styled(caption, Style::default().fg(pal.muted))),
        area.width,
    );
}

/// A small floating explanation under the pointer. Deliberately drawn on top of
/// everything (a tooltip, not a modal): it disappears when the pointer moves.
fn draw_tooltip(f: &mut Frame, app: &App, mx: u16, my: u16, text: &str) {
    let p = &app.ui.palette;
    let area = f.area();
    let all_lines: Vec<&str> = text.lines().collect();
    let visible = all_lines.iter().filter(|l| !l.trim().is_empty()).count();
    if visible == 0 {
        return;
    }
    let longest = all_lines.iter().map(|l| UnicodeWidthStr::width(*l)).max().unwrap_or(0);
    let width = (longest as u16 + 4).clamp(16, area.width.saturating_sub(2).max(16));
    let height = (all_lines.len() as u16 + 2).clamp(3, area.height);
    let x = mx.min(area.x + area.width.saturating_sub(width));
    let y = if my > height {
        my - height
    } else {
        (my + 1).min(area.y + area.height.saturating_sub(height))
    };
    // Never cover the toolbar: it is the one line that must stay readable while
    // a hint is open. If the hint would reach it, slide the hint up.
    let toolbar_row = area.y + area.height.saturating_sub(1);
    let y = y.min(toolbar_row.saturating_sub(height)).max(area.y);
    let rect = Rect { x, y, width, height };
    f.render_widget(Clear, rect);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(p.accent))
        .style(Style::default().bg(p.surface_hi).fg(p.text));
    let paragraph = Paragraph::new(text.to_string())
        .block(block)
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, rect);
}

// Kept for callers that draw a standalone note over the whole frame.
#[allow(dead_code)]
fn too_small_note(f: &mut Frame) {
    let area = f.area();
    let note: String = "Окно: терминал слишком мал.".chars().take(area.width.max(1) as usize).collect();
    f.render_widget(Clear, area);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1)])
        .split(area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(note, Style::default().fg(Color::Red)))).alignment(Alignment::Left),
        chunks[0],
    );
}

/// Wrap helper re-exported for tests that need the prompt's own row split.
pub fn wrap_prompt_text(text: &str, width: usize) -> Vec<String> {
    wrap_hard(text, width)
}

#[cfg(test)]
mod caret_tests {
    use super::*;

    #[test]
    fn the_free_text_prompt_puts_the_caret_where_the_cursor_is() {
        let mut s = SelectModalData::new("id".into(), "Ваш ответ ❯ ".into(), Vec::new(), None);
        for c in "abc".chars() {
            s.filter_insert(c);
        }
        let (x_end, y_end) = select_rows(&s, 80, 10).1.expect("текстовый промпт отдаёт каретку");

        s.filter_left();
        let (x_mid, y_mid) = select_rows(&s, 80, 10).1.expect("каретка есть и в середине");
        assert_eq!(y_mid, y_end, "каретка остаётся на той же строке");
        assert_eq!(x_mid, x_end - 1, "на один символ левее");

        s.filter_home();
        let (x_home, _) = select_rows(&s, 80, 10).1.expect("каретка есть в начале");
        assert!(x_home < x_mid);
    }

    #[test]
    fn the_filter_line_carries_the_caret_too() {
        let mut s = SelectModalData::new(
            "id".into(),
            "Выберите".into(),
            vec!["alpha".into(), "beta".into()],
            None,
        );
        s.filter_insert('a');
        let (rows1, caret1) = select_rows(&s, 80, 12);
        let (x1, row1) = caret1.expect("строка фильтра отдаёт каретку");
        // The filter line sits under the list, and the list shrinks as it filters,
        // so the caret follows the last row rather than a fixed one.
        assert_eq!(row1 as usize, rows1.len() - 1, "каретка на строке фильтра");

        s.filter_insert('l');
        let (rows2, caret2) = select_rows(&s, 80, 12);
        let (x2, row2) = caret2.expect("и после второго символа тоже");
        assert_eq!(row2 as usize, rows2.len() - 1);
        assert_eq!(x2, x1 + 1);
        s.filter_left();
        assert_eq!(select_rows(&s, 80, 12).1.unwrap().0, x1, "шаг влево возвращает каретку");
    }

    #[test]
    fn a_plain_choice_list_gets_no_caret() {
        let s = SelectModalData::new(
            "id".into(),
            "Выберите".into(),
            vec!["alpha".into(), "beta".into()],
            None,
        );
        assert!(select_rows(&s, 80, 12).1.is_none(), "пока ничего не набрано, каретке негде стоять");
    }
}
