//! The premium look of the native widgets (see `theme::UiStyle`).
//!
//! The classic look copies the terminal REPL cell for cell and the parity tests hold
//! it there; this module draws the same things — the prompt, the suggestion list,
//! the toolbar, what you wrote — as a designed surface. Same layout, same keys,
//! same click targets; only the dressing differs:
//!
//! * panels are tinted rectangles (`Palette::surface`), not bare text;
//! * a one-cell accent bar marks what is yours (the prompt; your messages are a
//!   bubble at the right edge, see `cards.rs`);
//! * the toolbar is a row of chips with a context gauge instead of a reversed strip;
//! * the suggestion list highlights the current item and says which keys move it.
//!
//! Everything is plain coloured cells: no layers, no per-frame allocation beyond
//! the rows themselves, so it is as cheap as the classic look.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::ansi::{clip_line, line_width};
use crate::app::{App, ClickAction};
use crate::theme::Palette;
use crate::ui::{cursor_in_layout, fmt_tokens, layout_input, menu_items, PromptBlock, MENU_ROWS};

/// `line` with `bg` behind every span that has no background of its own, padded
/// with blanks to exactly `width` columns.
pub fn panel_row(mut line: Line<'static>, width: usize, bg: Color) -> Line<'static> {
    line = clip_line(line, width);
    for span in line.spans.iter_mut() {
        if span.style.bg.is_none() {
            span.style = span.style.bg(bg);
        }
    }
    let used = line_width(&line);
    if used < width {
        line.spans.push(Span::styled(" ".repeat(width - used), Style::default().bg(bg)));
    }
    line
}

/// The soft top (`▄`) or bottom (`▀`) edge of a panel: half a cell of its colour,
/// which reads as a rounded, slightly translucent border without drawing a box.
pub fn edge_row(top: bool, width: usize, color: Color) -> Line<'static> {
    let glyph = if top { "▄" } else { "▀" };
    Line::from(Span::styled(glyph.repeat(width), Style::default().fg(color)))
}

/// A one-cell accent bar on the panel's background.
fn bar(p: &Palette) -> Span<'static> {
    Span::styled("▌", Style::default().fg(p.accent).bg(p.surface))
}

// ── prompt ──────────────────────────────────────────────────────────────────
// (Your messages in the transcript are drawn in `cards.rs`.)

/// Columns before the typed text: bar, blank, `❯`, blank.
pub const PROMPT_PREFIX: usize = 4;

const PLACEHOLDER: &str = "Напишите задачу… (/ — команды, @ — файлы, Shift+Enter — перенос строки)";

/// The prompt while the agent is idle: a tinted panel with soft edges, an accent
/// bar and `❯`, the text on `surface`, a dim hint while it is empty. While a turn
/// runs and something is typed, the REPL's `Ввод на лету` echo is drawn inside the
/// same panel.
pub(crate) fn prompt_block(app: &App, width: u16, max_rows: usize, edges: bool) -> PromptBlock {
    let p = &app.ui.palette;
    let w = width as usize;
    let edge = usize::from(edges);

    if app.is_busy {
        if app.input.is_empty() {
            return PromptBlock { rows: Vec::new(), cursor: None };
        }
        // The classic echo (label, text with a block cursor, the hint line), laid out
        // two columns narrower and then set into the panel.
        let inner = crate::ui::prompt_block_classic(app, width.saturating_sub(2), max_rows);
        let mut rows: Vec<Line<'static>> = Vec::new();
        if edges {
            rows.push(edge_row(true, w, p.surface));
        }
        for line in inner.rows {
            let mut spans = vec![bar(p), Span::styled(" ", Style::default().bg(p.surface))];
            spans.extend(line.spans);
            rows.push(panel_row(Line::from(spans), w, p.surface));
        }
        if edges {
            rows.push(edge_row(false, w, p.surface));
        }
        return PromptBlock { rows, cursor: None };
    }

    let avail = w.saturating_sub(PROMPT_PREFIX).max(1);
    let layout = layout_input(&app.input, avail);
    let (cur_row, cur_col) = cursor_in_layout(&app.input, &layout, app.cursor_pos);
    let (sel_lo, sel_hi) = match app.sel_anchor {
        Some(a) if a != app.cursor_pos => (a.min(app.cursor_pos), a.max(app.cursor_pos)),
        _ => (0, 0),
    };
    let chars: Vec<char> = app.input.chars().collect();
    let on = |fg: Color| Style::default().fg(fg).bg(p.surface);
    let glyph = Style::default().fg(p.accent).bg(p.surface).add_modifier(Modifier::BOLD);

    let mut rows: Vec<Line<'static>> = Vec::new();
    for (i, r) in layout.iter().enumerate() {
        let mut spans = vec![
            bar(p),
            Span::styled(" ", on(p.text)),
            Span::styled(if i == 0 { "❯" } else { " " }, glyph),
            Span::styled(" ", on(p.text)),
        ];
        let mut seg = String::new();
        let mut seg_sel = false;
        let flush = |seg: &mut String, sel: bool, spans: &mut Vec<Span<'static>>| {
            if seg.is_empty() {
                return;
            }
            let style = if sel { on(p.text).bg(p.selection) } else { on(p.text) };
            spans.push(Span::styled(std::mem::take(seg), style));
        };
        for (k, ch) in chars[r.start..r.end].iter().enumerate() {
            let idx = r.start + k;
            let sel = sel_hi > sel_lo && idx >= sel_lo && idx < sel_hi;
            if sel != seg_sel {
                flush(&mut seg, seg_sel, &mut spans);
            }
            seg_sel = sel;
            seg.push(*ch);
        }
        flush(&mut seg, seg_sel, &mut spans);
        if i == 0 && app.input.is_empty() {
            spans.push(Span::styled(
                PLACEHOLDER,
                Style::default().fg(p.muted).bg(p.surface).add_modifier(Modifier::ITALIC),
            ));
        }
        rows.push(panel_row(Line::from(spans), w, p.surface));
    }

    let skip = if rows.len() > max_rows {
        (cur_row + 1).saturating_sub(max_rows).min(rows.len() - max_rows)
    } else {
        0
    };
    let mut rows: Vec<Line<'static>> = rows.into_iter().skip(skip).take(max_rows).collect();
    if edges {
        rows.insert(0, edge_row(true, w, p.surface));
        rows.push(edge_row(false, w, p.surface));
    }
    let cursor = Some(((PROMPT_PREFIX + cur_col) as u16, (cur_row.saturating_sub(skip) + edge) as u16));
    PromptBlock { rows, cursor }
}

// ── suggestion list ─────────────────────────────────────────────────────────

/// The open suggestion list (commands or `@files`): a tinted panel with the current
/// item lit, then a footer that says which keys move through it.
pub fn menu_rows(app: &App, width: u16) -> Vec<Line<'static>> {
    let Some((items, cur)) = menu_items(app) else { return Vec::new() };
    let p = &app.ui.palette;
    let w = width as usize;
    let cur = cur.min(items.len() - 1);
    let name_w = items.iter().map(|(n, _)| UnicodeWidthStr::width(n.as_str())).max().unwrap_or(0);
    let top = if cur >= MENU_ROWS { cur + 1 - MENU_ROWS } else { 0 };

    let mut out: Vec<Line<'static>> = Vec::new();
    for (i, (name, meta)) in items.iter().enumerate().skip(top).take(MENU_ROWS) {
        let current = i == cur;
        let bg = if current { p.accent_soft } else { p.surface };
        let pad = " ".repeat(name_w.saturating_sub(UnicodeWidthStr::width(name.as_str())));
        let mut spans = vec![
            Span::styled(if current { " ▸ " } else { "   " }, Style::default().fg(p.accent).bg(bg).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("{name}{pad}"),
                Style::default()
                    .fg(if current { p.text } else { p.accent })
                    .bg(bg)
                    .add_modifier(if current { Modifier::BOLD } else { Modifier::empty() }),
            ),
        ];
        if !meta.is_empty() {
            spans.push(Span::styled("  ", Style::default().bg(bg)));
            spans.push(Span::styled(meta.clone(), Style::default().fg(if current { p.text } else { p.muted }).bg(bg)));
        }
        out.push(panel_row(Line::from(spans), w, bg));
    }

    let hint = Style::default().fg(p.muted);
    let key = Style::default().fg(p.text).add_modifier(Modifier::BOLD);
    let count = format!("{}/{} ", cur + 1, items.len());
    let mut footer = vec![
        Span::raw("  "),
        Span::styled("Tab", key),
        Span::styled(" ↓ далее  ", hint),
        Span::styled("Shift+Tab", key),
        Span::styled(" ↑ назад  ", hint),
        Span::styled("Enter", key),
        Span::styled(" принять  ", hint),
        Span::styled("Esc", key),
        Span::styled(" закрыть", hint),
    ];
    let used: usize = footer.iter().map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum();
    let room = w.saturating_sub(used + UnicodeWidthStr::width(count.as_str()));
    footer.push(Span::raw(" ".repeat(room)));
    footer.push(Span::styled(count, hint));
    out.push(clip_line(Line::from(footer), w));
    out
}

// ── toolbar ─────────────────────────────────────────────────────────────────

/// The context gauge: `cells` blocks, filled in proportion, coloured by how full
/// the window is.
pub fn ctx_gauge(p: &Palette, pct: i64, cells: usize) -> Vec<Span<'static>> {
    let used = ((pct.clamp(0, 100) as f64 / 100.0) * cells as f64).round() as usize;
    let color = if pct < 50 {
        p.ok
    } else if pct < 80 {
        p.warn
    } else {
        p.danger
    };
    vec![
        Span::styled("▰".repeat(used.min(cells)), Style::default().fg(color).bg(p.surface)),
        Span::styled("▱".repeat(cells - used.min(cells)), Style::default().fg(p.border).bg(p.surface)),
    ]
}

/// The bottom toolbar as chips on one tinted row, plus the click targets in it
/// (model, thinking level) — the same ones the classic bar has.
pub fn toolbar(app: &App, width: u16) -> (Line<'static>, Vec<(u16, u16, ClickAction)>) {
    let p = &app.ui.palette;
    let st = &app.status;
    let text = Style::default().fg(p.text).bg(p.surface);
    let muted = Style::default().fg(p.muted).bg(p.surface);
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut targets: Vec<(u16, u16, ClickAction)> = Vec::new();
    let mut x: u16 = 0;

    let push = |spans: &mut Vec<Span<'static>>, s: Span<'static>, x: &mut u16| {
        *x = x.saturating_add(UnicodeWidthStr::width(s.content.as_ref()) as u16);
        spans.push(s);
    };
    let gap = |spans: &mut Vec<Span<'static>>, x: &mut u16| {
        push(spans, Span::styled("  ", muted), x);
        push(spans, Span::styled("│", Style::default().fg(p.border).bg(p.surface)), x);
        push(spans, Span::styled("  ", muted), x);
    };

    push(&mut spans, Span::styled(" ", muted), &mut x);
    push(
        &mut spans,
        Span::styled(format!(" {} ", st.mode), Style::default().fg(p.base).bg(p.accent).add_modifier(Modifier::BOLD)),
        &mut x,
    );
    push(&mut spans, Span::styled("  ", muted), &mut x);

    let x0 = x;
    push(&mut spans, Span::styled(format!("{}:{}", st.provider, st.model), text), &mut x);
    targets.push((x0, x, ClickAction::OpenModelPicker));
    gap(&mut spans, &mut x);

    push(&mut spans, Span::styled(st.tier.clone(), muted), &mut x);

    let think = if st.thinking_label.is_empty() { st.thinking_level.to_uppercase() } else { st.thinking_label.clone() };
    if !think.is_empty() {
        gap(&mut spans, &mut x);
        let x0 = x;
        let lit = !think.eq_ignore_ascii_case("off");
        let style = if lit { Style::default().fg(p.warn).bg(p.surface) } else { muted };
        push(&mut spans, Span::styled(format!("think {think}"), style), &mut x);
        targets.push((x0, x, ClickAction::CycleThinkingLevel));
    }

    gap(&mut spans, &mut x);
    push(&mut spans, Span::styled(st.cwd.clone(), muted), &mut x);

    if st.ctx_tokens > 0 || st.ctx_percent > 0.0 {
        gap(&mut spans, &mut x);
        let pct = st.ctx_percent.round() as i64;
        push(&mut spans, Span::styled("ctx ", muted), &mut x);
        for s in ctx_gauge(p, pct, 6) {
            push(&mut spans, s, &mut x);
        }
        push(&mut spans, Span::styled(format!(" {pct}%"), text), &mut x);
        if st.ctx_max > 0 {
            push(
                &mut spans,
                Span::styled(format!(" {}/{}", fmt_tokens(st.ctx_tokens), fmt_tokens(st.ctx_max)), muted),
                &mut x,
            );
        }
    }
    if !st.session_summary.is_empty() {
        gap(&mut spans, &mut x);
        push(&mut spans, Span::styled(st.session_summary.clone(), muted), &mut x);
    }
    gap(&mut spans, &mut x);
    push(&mut spans, Span::styled(crate::ui::uptime_label(app), muted), &mut x);

    // `/help` hugs the right edge when there is room for it.
    let help = " /help ";
    let help_w = UnicodeWidthStr::width(help);
    let w = width as usize;
    let used = x as usize;
    if used + help_w < w {
        spans.push(Span::styled(" ".repeat(w - used - help_w), muted));
        spans.push(Span::styled(help, Style::default().fg(p.muted).bg(p.surface).add_modifier(Modifier::DIM)));
    }

    let line = panel_row(Line::from(spans), w, p.surface);
    targets.retain(|(a, _, _)| *a < width);
    (line, targets)
}

/// The chip that appears over the transcript while it is scrolled back.
pub fn jump_chip(p: &Palette, width: u16) -> (Line<'static>, u16) {
    let label = " ↓ К последнему · End ";
    let w = UnicodeWidthStr::width(label) as u16;
    let style = Style::default().fg(p.accent).bg(p.accent_soft).add_modifier(Modifier::BOLD);
    let x = width.saturating_sub(w + 2);
    (Line::from(Span::styled(label, style)), x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panel_row_is_exactly_as_wide_as_asked_and_tinted_throughout() {
        let bg = Color::Rgb(1, 2, 3);
        let row = panel_row(Line::from("hello"), 20, bg);
        assert_eq!(line_width(&row), 20);
        assert!(row.spans.iter().all(|s| s.style.bg == Some(bg)));
    }

    #[test]
    fn a_panel_row_keeps_a_span_that_has_its_own_background() {
        let row = panel_row(
            Line::from(vec![Span::styled("chip", Style::default().bg(Color::Red)), Span::raw(" rest")]),
            20,
            Color::Blue,
        );
        assert_eq!(row.spans[0].style.bg, Some(Color::Red));
        assert_eq!(row.spans[1].style.bg, Some(Color::Blue));
    }

    #[test]
    fn a_row_wider_than_the_panel_is_cut_not_overflowed() {
        let row = panel_row(Line::from("x".repeat(50)), 10, Color::Blue);
        assert_eq!(line_width(&row), 10);
    }

    #[test]
    fn the_gauge_fills_in_proportion_and_changes_colour_as_it_fills() {
        let p = Palette::default();
        let count = |pct: i64| {
            let g = ctx_gauge(&p, pct, 10);
            (g[0].content.chars().count(), g[1].content.chars().count(), g[0].style.fg)
        };
        assert_eq!(count(0).0, 0);
        assert_eq!(count(50).0, 5);
        assert_eq!(count(100).0, 10);
        assert_eq!(count(150).0, 10, "never past the end");
        assert_eq!(count(20).2, Some(p.ok));
        assert_eq!(count(60).2, Some(p.warn));
        assert_eq!(count(90).2, Some(p.danger));
        assert_eq!(count(37).0 + count(37).1, 10, "the gauge is always the same width");
    }

    #[test]
    fn edges_are_half_blocks_of_the_panel_colour() {
        let top = edge_row(true, 5, Color::Blue);
        assert_eq!(top.spans[0].content, "▄▄▄▄▄");
        assert_eq!(top.spans[0].style.fg, Some(Color::Blue));
        assert_eq!(edge_row(false, 3, Color::Blue).spans[0].content, "▀▀▀");
    }
}
