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
    widgets::{Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
    Frame,
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::ansi::{clip_line, line_width};
use crate::app::{App, ApprovalData, ClickAction, SelectModalData};
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

pub(crate) fn uptime_label(app: &App) -> String {
    let mins = app.status.start_time.elapsed().as_secs() / 60;
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

    // Keep the cursor row visible when the text is taller than the allowance.
    let skip = if rows.len() > max_rows {
        (cur_row + 1).saturating_sub(max_rows).min(rows.len() - max_rows)
    } else {
        0
    };
    let rows: Vec<Line<'static>> = rows.into_iter().skip(skip).take(max_rows).collect();
    let cursor = Some(((2 + cur_col) as u16, cur_row.saturating_sub(skip) as u16));
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
/// A job of several steps is easier to trust when you can see which one the agent is on.
/// A long list shows the current step with its neighbours and a "… ещё N" row for the rest.
pub(crate) fn todo_rows(app: &App, width: u16, max_rows: usize) -> Vec<Line<'static>> {
    let todos = &app.status.todos;
    let folded = app.todos_folded;
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
        }
        return vec![clip_line(Line::from(head), width as usize)];
    }
    let mut rows = vec![clip_line(Line::from(head), width as usize)];

    let room = max_rows - 1;
    let (start, end, hidden) = if todos.len() <= room {
        (0, todos.len(), 0)
    } else {
        let visible = room.saturating_sub(1).max(1);
        let focus = todos.iter().position(|t| !t.is_done()).unwrap_or(todos.len() - 1);
        let start = focus.saturating_sub(1).min(todos.len() - visible);
        (start, start + visible, todos.len() - visible)
    };
    for item in &todos[start..end] {
        let (mark, mark_style, text_style) = if item.is_done() {
            (
                "  ✓ ",
                Style::default().fg(ok),
                Style::default().fg(muted).add_modifier(Modifier::CROSSED_OUT),
            )
        } else if item.is_active() {
            (
                "  ▶ ",
                Style::default().fg(accent).add_modifier(Modifier::BOLD),
                Style::default().fg(text).add_modifier(Modifier::BOLD),
            )
        } else {
            ("  ○ ", Style::default().fg(muted), Style::default().fg(text))
        };
        rows.push(clip_line(
            Line::from(vec![Span::styled(mark, mark_style), Span::styled(item.content.clone(), text_style)]),
            width as usize,
        ));
    }
    if hidden > 0 {
        rows.push(clip_line(
            Line::from(Span::styled(format!("    … ещё {hidden}"), Style::default().fg(muted))),
            width as usize,
        ));
    }
    rows
}

// ── questionary-style prompts ───────────────────────────────────────────────

fn answer_pointer(selected: bool) -> Span<'static> {
    Span::raw(if selected { " » " } else { "   " })
}

/// The approval question as questionary draws it, under the REPL's
/// `◇ Агент запрашивает разрешение: …` line.
fn approval_rows(app: &App, a: &ApprovalData, width: u16) -> Vec<Line<'static>> {
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
        let shown: String = {
            let n = typed.chars().count();
            if n > budget {
                typed.chars().skip(n - budget).collect()
            } else {
                typed.clone()
            }
        };
        let tail = vec![Span::styled(
            shown.clone(),
            Style::default().fg(Color::Rgb(0xff, 0x9d, 0x00)).add_modifier(Modifier::BOLD),
        )];
        let rows = title_lines(&trows, tail);
        let col = UnicodeWidthStr::width(head.as_str()) + UnicodeWidthStr::width(shown.as_str());
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
        rows.push(Line::from(Span::styled(format!("  Search filter: {}", s.filter), style)));
    }
    (rows, None)
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
    let area = f.area();
    if area.width < 8 || area.height < 4 {
        return;
    }
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
    // The agent's checklist sits between the chat and the prompt. It is the first thing to give way:
    // never in front of a question, and only with room to spare for the chat above it.
    // At most six rows (the header and five steps): the checklist is a glance, not the screen. A
    // click on its header folds it to that one row.
    let todo_room = h.saturating_sub(prompt_rows.len() + menu.len() + toolbar_h + 5).min(6);
    let todo = if modal { Vec::new() } else { todo_rows(app, w, todo_room) };
    // The stack may never squeeze the transcript out entirely.
    let reserved = prompt_rows.len() + menu.len() + toolbar_h;
    if reserved + 1 > h {
        let keep = h.saturating_sub(toolbar_h + menu.len() + 1).max(1);
        let skip = prompt_rows.len().saturating_sub(keep);
        prompt_rows = prompt_rows.into_iter().skip(skip).collect();
        cursor = cursor.map(|(x, y)| (x, y.saturating_sub(skip as u16)));
    }
    let bottom_h = todo.len() + prompt_rows.len() + menu.len() + toolbar_h;
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
    }
    let py = todo_y + todo.len() as u16;
    for (i, row) in prompt_rows.iter().enumerate() {
        put_line(f.buffer_mut(), area, py + i as u16, row, w);
    }
    // Completion menu under the prompt.
    let my = py + prompt_rows.len() as u16;
    for (i, row) in menu.iter().enumerate() {
        put_line(f.buffer_mut(), area, my + i as u16, row, w);
    }

    // Toolbar on the last row.
    let (bar, targets) = if premium { premium::toolbar(app, w) } else { toolbar(app, w) };
    let ty = area.y + area.height - 1;
    put_line(f.buffer_mut(), area, ty, &bar, w);
    for (x0, x1, action) in targets {
        app.register_clickable(area.x + x0, ty, x1.saturating_sub(x0), 1, action);
    }

    if let Some((cx, cy)) = cursor {
        let y = py + cy;
        if cx < w && y < ty {
            f.set_cursor_position((area.x + cx, y));
        }
    }
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
