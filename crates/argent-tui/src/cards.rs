//! The premium transcript: every kind of message has a shape of its own.
//!
//! The classic look copies the terminal REPL cell for cell (the parity tests hold it
//! there). The premium look keeps the same messages, the same keys and the same click
//! targets and gives each kind of thing on the screen its own, unmistakable form:
//!
//! ```text
//!                                                  Вы
//!                         ┌─────────────────────────────────────────▐   your message:
//!                         │  почини меню инвентаря                  ▐   a tinted bubble
//!                         └─────────────────────────────────────────▐   at the right edge
//!
//!  ┊ ◇ Анализ · 3.2s  Нужно понять, почему список пуст.                  a thought: quiet,
//!                                                                         no fill, dotted rail
//!  ✦ Argent                                                              an answer: one header,
//!  Сначала посмотрю, как строится список.                                  however many tool calls
//!                                                                         follow it
//!
//!  ▍ ✓ read_file: Assets/InventoryMenu.cs                       28ms     a tool call: a card
//!  ▍   ↳ 2 строк · using UnityEngine;                                    with a rail and a fill
//!
//!  ✦ Argent                                                              the final answer
//!  Готово.
//! ```
//!
//! Nothing here is drawn per frame beyond the visible rows: composed blocks are cached
//! like the Python-rendered ones, keyed by message, slot, width and a fingerprint of
//! what they depend on.

use std::sync::Arc;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::{json, Value};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::ansi::{clip_line, glue_style, line_width, parse_ansi, strip_ansi};
use crate::app::{App, ClickAction, ImagePreview, MessageItem, RgbGrid};
use crate::premium::panel_row;
use crate::renderer::thought_preview;
use crate::theme::Palette;
use crate::toolrender;
use crate::view::{
    dur_bits, fmt_compact, hash_of, phase_elapsed, rewind_button_row, spinner_glyph, speed_suffix,
    text_fp, time_label, total_elapsed, wrap_words, Blocks, Cols, ItemView, Rows,
};

/// Columns to the left of an assistant's text.
const INDENT: usize = 2;
/// A card's left dressing: a margin (the "red line" that sets a card in from the
/// answer text), the rail, a blank.
const CARD_PRE: usize = 5;
/// Where a card's content (a tool's name, its result) starts: under the first glyph's text.
const CARD_TEXT: usize = 7;
/// Rows of a tool's output a card shows (the REPL's own cap).
const RESULT_ROWS: usize = 15;
/// Rows of a room's report a card shows (it is a list of findings, not a log).
const ROOM_RESULT_ROWS: usize = 40;
/// Rows of a diff a card shows.
const DIFF_ROWS: usize = 25;
/// Cap for an OPENED diff: every line up to this, then a note. It only exists so a
/// rewrite of one enormous file cannot make the renderer chew for seconds.
const MAX_DIFF_ROWS: usize = 2000;

const RAIL: &str = "▍";
const DOTS: &str = "┊";

// ── small helpers ───────────────────────────────────────────────────────────

fn rows_of(v: Vec<Line<'static>>) -> Rows {
    Arc::new(v)
}

fn fg(c: Color) -> Style {
    Style::default().fg(c)
}

fn width_of(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

fn blank_block() -> Rows {
    rows_of(vec![Line::default()])
}

/// Cut `text` into rows of at most `width` columns: at the last blank when the row
/// has one, mid-word when it has none. Spaces inside a row are kept (a pasted
/// stack trace or snippet must not be reflowed), unlike `wrap_words`.
pub fn wrap_soft(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for logical in text.split('\n') {
        let mut row = String::new();
        let mut row_w = 0usize;
        for ch in logical.chars() {
            let cw = ch.width().unwrap_or(0);
            if row_w + cw > width {
                // The row is full and the next character is a blank: that is the break.
                if ch == ' ' {
                    out.push(std::mem::take(&mut row));
                    row_w = 0;
                    continue;
                }
                match row.rfind(' ').filter(|&at| at > 0) {
                    Some(at) => {
                        let tail = row[at + 1..].to_string();
                        row.truncate(at);
                        out.push(std::mem::take(&mut row));
                        row = tail;
                        row_w = width_of(&row);
                    }
                    None => {
                        out.push(std::mem::take(&mut row));
                        row_w = 0;
                    }
                }
            }
            row.push(ch);
            row_w += cw;
        }
        out.push(row);
    }
    out
}

/// `rows` set `n` columns in from the left.
fn indented(rows: &[Line<'static>], n: usize) -> Vec<Line<'static>> {
    rows.iter()
        .map(|l| {
            let mut spans = Vec::with_capacity(l.spans.len() + 1);
            if !l.spans.is_empty() {
                spans.push(Span::raw(" ".repeat(n)));
            }
            spans.extend(l.spans.iter().cloned());
            Line::from(spans)
        })
        .collect()
}

/// `rows` with `prefix` in front of every row, and `dim` laid over their text.
fn railed(rows: &[Line<'static>], prefix: &[Span<'static>], dim: bool) -> Vec<Line<'static>> {
    rows.iter()
        .map(|l| {
            let mut spans: Vec<Span<'static>> = prefix.to_vec();
            for s in &l.spans {
                let style = if dim { s.style.add_modifier(Modifier::DIM) } else { s.style };
                spans.push(Span::styled(s.content.clone(), style));
            }
            Line::from(spans)
        })
        .collect()
}

// ── which rows get a blank row before them ──────────────────────────────────

/// True when the last row of the message is a blank one (an answer's markdown ends
/// with one), so nothing needs to be added after it.
fn ends_blank(m: &MessageItem) -> bool {
    matches!(m, MessageItem::Assistant { content, .. } if !content.is_empty())
}

/// A thought on its own: reasoning without an answer next to it, or a second-pass check.
fn is_thought(m: &MessageItem) -> bool {
    match m {
        MessageItem::Rethink { .. } => true,
        MessageItem::Assistant { content, thinking, .. } => content.is_empty() && !thinking.is_empty(),
        _ => false,
    }
}

/// A call or a change: a card.
fn is_card(m: &MessageItem) -> bool {
    matches!(
        m,
        MessageItem::Tool { .. } | MessageItem::Diff { .. } | MessageItem::ToolGenerating { .. }
    )
}

/// Does message `idx` start with a blank row to set it off from the one before?
///
/// Things are set apart by a blank row, with two exceptions that belong together: a
/// thought sits tight against the thought before it, and a call sits tight under the
/// thought that led to it.
pub(crate) fn lead_blank(app: &App, idx: usize) -> bool {
    if idx == 0 || idx > app.messages.len() {
        return false;
    }
    let prev = &app.messages[idx - 1];
    if ends_blank(prev) {
        return false;
    }
    match app.messages.get(idx) {
        // a thought sits against the thought before it; a call against the thought that led to it
        // and against the call before it (a run of calls is one block, not a column of islands)
        Some(cur) if is_thought(cur) => !is_thought(prev),
        Some(cur) if is_card(cur) => !(is_thought(prev) || is_card(prev)),
        // lines of the system (a room's progress, a notice) follow each other and the call they belong to
        Some(cur) if is_note(cur) => !(is_note(prev) || is_card(prev)),
        _ => true,
    }
}

/// A line of the system: a notice, a live output fragment or a system message.
fn is_note(m: &MessageItem) -> bool {
    matches!(
        m,
        MessageItem::Notice(_) | MessageItem::System(_) | MessageItem::Output(_)
    )
}

/// The first answer in a run carries the header; a second adjacent Assistant
/// item is its continuation (a re-sent segment), not another speaker.
fn answer_starts_here(app: &App, idx: usize) -> bool {
    !app.messages[..idx]
        .iter()
        .rev()
        .take_while(|m| matches!(m, MessageItem::Assistant { .. }))
        .any(|m| matches!(m, MessageItem::Assistant { content, .. } if !content.trim().is_empty()))
}

// ── the message list ────────────────────────────────────────────────────────

/// The rows of message `idx` at `width`, in the premium look.
pub(crate) fn item_view(app: &App, idx: usize, width: u16) -> ItemView {
    let mut blocks = Blocks::default();
    let mut chrome_tail = 0usize;
    let lead = usize::from(lead_blank(app, idx));
    if lead == 1 {
        blocks.add_cols(blank_block(), Cols::NONE);
    }

    match &app.messages[idx] {
        MessageItem::User(text) => {
            user_blocks(app, text, width, &mut blocks);
            if app.rewind_target() == Some(idx) {
                blocks.add_cols(rows_of(vec![rewind_button_row(app, width)]), Cols::NONE);
                blocks.click_last(ClickAction::RewindLast);
                chrome_tail = 1;
            }
        }

        MessageItem::Assistant {
            content,
            thinking,
            thinking_expanded,
            is_streaming_thinking,
            is_streaming_content,
            thinking_duration,
            stream,
            ..
        } => {
            let shown_thought = !thinking.is_empty() && !*is_streaming_thinking;
            if shown_thought {
                thought_blocks(
                    app,
                    idx,
                    0,
                    "reasoning",
                    thinking,
                    *thinking_expanded,
                    *thinking_duration,
                    width,
                    &mut blocks,
                );
                blocks.click_last(ClickAction::ToggleThinking(idx));
            }
            if !content.trim().is_empty() {
                if shown_thought {
                    blocks.add_cols(blank_block(), Cols::NONE);
                }
                if answer_starts_here(app, idx) {
                    blocks.add_cols(rows_of(vec![answer_header(&app.ui.palette)]), Cols::NONE);
                }
                let mut ranges = stream.chunks.clone();
                if !*is_streaming_content && stream.committed < content.len() {
                    ranges.push(stream.committed..content.len());
                }
                let style = app.ui.style.name();
                let inner = width.saturating_sub(INDENT as u16 + 1).max(8);
                for (i, r) in ranges.iter().enumerate() {
                    let Some(chunk) = content.get(r.clone()) else { continue };
                    let fp = hash_of((r.start, r.end, text_fp(chunk), style));
                    let rows = app.cached_rows(idx, 2000 + i as u16, fp, width, || {
                        let raw = app.cached_block(idx, 100 + i as u16, fp, inner, || {
                            ("markdown", json!({"text": chunk, "style": style}))
                        });
                        indented(&raw, INDENT)
                    });
                    blocks.add_cols(rows, Cols::from(INDENT));
                }
            }
        }

        MessageItem::Rethink { text, expanded, is_streaming, duration, .. } => {
            if !*is_streaming && !text.trim().is_empty() {
                thought_blocks(app, idx, 0, "rethink", text, *expanded, *duration, width, &mut blocks);
                blocks.click_last(ClickAction::ToggleThinking(idx));
            }
        }

        MessageItem::Tool { name, args, result, duration_secs, expanded, is_running } => {
            if *is_running {
                if !App::is_interactive_tool(name) {
                    blocks.add_cols(rows_of(tool_running_row(app, idx, name, args, width)), Cols::from(CARD_PRE));
                    blocks.click_last(ClickAction::ToggleTool(idx));
                }
            } else if name != "ask_user_questions" {
                // A room's report is the point of the call: it is shown, not folded into one row.
                let exp = *expanded || app.show_tool_results || app.ui.show_tool_results || name == "rooms";
                let res = result.as_deref().unwrap_or("");
                tool_blocks(app, idx, name, args, res, *duration_secs, exp, width, &mut blocks);
                blocks.click_last(ClickAction::ToggleTool(idx));
                // The picture rides as its own block: a click on it opens the image
                // on the whole screen, while a click on the card folds it. One block
                // carries one action, so the two cannot share rows.
                if let Some(prev) = app.previews.get(&idx) {
                    let fp = hash_of((prev.seq, exp));
                    let rows = app.cached_rows(idx, 21, fp, width, || {
                        image_card_rows(app, prev, exp, width as usize)
                    });
                    blocks.add_cols(rows, Cols::from(CARD_PRE));
                    blocks.click_last(ClickAction::ShowImage(idx));
                    // A terminal draws a picture as blocks of colour; the system
                    // viewer draws it as pixels. For a screenshot of text that is the
                    // difference between readable and not, so it gets its own row.
                    let p = &app.ui.palette;
                    let mut spans = card_prefix(p.tool, p.code);
                    spans.push(Span::styled(
                        "⧉ открыть в Windows  ·  настоящее разрешение".to_string(),
                        Style::default().fg(p.accent).bg(p.code).add_modifier(Modifier::BOLD),
                    ));
                    blocks.add_cols(
                        rows_of(vec![panel_row(Line::from(spans), width as usize, p.code)]),
                        Cols::from(CARD_PRE),
                    );
                    blocks.click_last(ClickAction::OpenImageExternally(idx));
                }
            }
        }

        MessageItem::Diff { file, diff, expanded } => {
            let rows = diff_card(&app.ui.palette, file, diff, *expanded, width);
            if !rows.is_empty() {
                blocks.add_cols(rows_of(rows), Cols::from(CARD_PRE));
                blocks.click_last(ClickAction::ToggleTool(idx));
            }
        }

        MessageItem::Notice(text) => {
            if text.contains('\u{1b}') {
                // Already styled by Rich (command output): drawn as it is.
                let rows: Vec<Line<'static>> =
                    parse_ansi(text).into_iter().map(|l| clip_line(l, width as usize)).collect();
                blocks.add(rows_of(rows));
            } else {
                gutter_block(app, idx, 0, "system", json!({"text": text}), text_fp(text), width, &mut blocks);
            }
        }

        // Live console output. The fragments were joined upstream (`push_output`),
        // so a row here ends where the REPL ended one, not where a read() stopped.
        MessageItem::Output(text) => {
            let rows: Vec<Line<'static>> = parse_ansi(&glue_style(text))
                .into_iter()
                .map(|l| clip_line(l, width as usize))
                .collect();
            blocks.add(rows_of(rows));
        }

        MessageItem::System(text) => {
            gutter_block(app, idx, 0, "system", json!({"text": text}), text_fp(text), width, &mut blocks);
        }

        MessageItem::Banner(version) => {
            let fp = text_fp(version);
            blocks.add(app.cached_block(idx, 0, fp, width, || ("banner", json!({"version": version}))));
        }

        MessageItem::Error { text, hint } => {
            blocks.add_cols(rows_of(error_card(&app.ui.palette, text, hint.as_deref(), width)), Cols::from(CARD_PRE));
        }

        MessageItem::Checkpoint { sha, .. } => {
            gutter_block(app, idx, 0, "checkpoint", json!({"sha": sha}), text_fp(sha), width, &mut blocks);
        }

        MessageItem::Snapshot { lines, context_line, summary, expanded } => {
            // One line and no context meter (modes `compact`/`off`): there is nothing
            // to open, so the REPL's text is drawn as it is — a button that opens onto
            // itself would be a lie about what a click does.
            let foldable = lines.len() > 1 || context_line.is_some();
            let fp = hash_of((
                summary.as_str(),
                *expanded,
                foldable,
                lines.iter().map(|l| text_fp(l)).collect::<Vec<_>>(),
                context_line.as_deref().map(text_fp),
            ));
            let rows = app.cached_rows(idx, 6, fp, width, || {
                if foldable {
                    snapshot_rows(app, idx, summary, lines, context_line.as_deref(), *expanded, width, INDENT)
                } else {
                    snapshot_detail_rows(app, idx, lines, context_line.as_deref(), width, INDENT)
                }
            });
            blocks.add(rows);
            if foldable {
                blocks.click_last(ClickAction::ToggleSnapshot(idx));
            }
        }

        MessageItem::ToolGenerating { name, bytes, text } => {
            blocks.add_cols(
                rows_of(tool_generating_card(app, name, *bytes, text, width)),
                Cols::from(CARD_PRE),
            );
        }
    }

    // A message that drew nothing leaves no gap behind it either.
    if blocks.0.iter().skip(lead).all(|(rows, _, _)| rows.is_empty()) {
        blocks.0.clear();
        chrome_tail = 0;
    }
    let len = blocks.0.iter().map(|(b, _, _)| b.len()).sum();
    ItemView {
        blocks: blocks.0,
        focused: app.is_focused_block(idx),
        len,
        chrome_tail,
        lead: if len > 0 { lead } else { 0 },
    }
}

/// A Python-rendered block set in from the left by the assistant's indent.
#[allow(clippy::too_many_arguments)]
fn gutter_block(
    app: &App,
    idx: usize,
    slot: u16,
    kind: &'static str,
    data: Value,
    fp: u64,
    width: u16,
    blocks: &mut Blocks,
) {
    let inner = width.saturating_sub(INDENT as u16).max(8);
    let rows = app.cached_rows(idx, 10 + slot, fp, width, || {
        let raw = app.cached_block(idx, slot, fp, inner, || (kind, data));
        indented(&raw, INDENT)
    });
    blocks.add_cols(rows, Cols::from(INDENT));
}

/// A finished turn's summary: one muted row that opens into the REPL's breakdown.
///
/// Folded, a long session is not a page of numbers between every answer. The `▸`
/// and the “клик — подробности” are the affordance: they say the row is a button
/// (`ClickAction::ToggleSnapshot` repeats it in the hover hint), because a line of
/// statistics does not look clickable on its own.
///
/// The numbers in `lines` come from the bridge, so both frontends still show the
/// same text; only the closed row is the terminal's own composition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn snapshot_rows(
    app: &App,
    idx: usize,
    summary: &str,
    lines: &[String],
    context_line: Option<&str>,
    expanded: bool,
    width: u16,
    indent: usize,
) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let w = width as usize;
    let hint = if expanded { "  ·  клик — свернуть" } else { "  ·  клик — подробности" };
    let mut spans = vec![
        Span::raw("  "),
        Span::styled(
            if expanded { "▾" } else { "▸" },
            Style::default().fg(p.accent).add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ];
    // The hint is the affordance and is never clipped; the numbers give way instead.
    let room = w.saturating_sub(3 + width_of(hint)).max(8);
    let shown = if width_of(summary) > room {
        let mut cut = String::new();
        let mut used = 0usize;
        for ch in summary.chars() {
            let c = ch.width().unwrap_or(0);
            if used + c > room.saturating_sub(1) {
                break;
            }
            cut.push(ch);
            used += c;
        }
        format!("{}…", cut.trim_end())
    } else {
        summary.to_string()
    };
    spans.push(Span::styled(shown, Style::default().fg(p.muted)));
    spans.push(Span::styled(
        hint,
        Style::default().fg(p.muted).add_modifier(Modifier::ITALIC),
    ));
    let mut out = vec![Line::from(spans)];
    if !expanded {
        return out;
    }
    out.extend(snapshot_detail_rows(app, idx, lines, context_line, width, indent));
    out
}

/// The REPL's own breakdown: the lines the bridge built, plus the context meter.
///
/// Shared by the opened summary and by the modes that have nothing to fold (one
/// line, no meter) — those draw the REPL's text directly.
pub(crate) fn snapshot_detail_rows(
    app: &App,
    idx: usize,
    lines: &[String],
    context_line: Option<&str>,
    width: u16,
    indent: usize,
) -> Vec<Line<'static>> {
    let inner = width.saturating_sub(indent as u16).max(8);
    let fp = hash_of((
        lines.iter().map(|l| text_fp(l)).collect::<Vec<_>>(),
        context_line.map(text_fp),
    ));
    let raw = app.cached_block(idx, 0, fp, inner, || {
        ("markup", json!({"lines": lines, "blank_after": true}))
    });
    let mut out = if indent == 0 { raw.to_vec() } else { indented(&raw, indent) };
    if let Some(ctx) = context_line {
        let raw_ctx = app.cached_block(idx, 1, text_fp(ctx), inner, || {
            ("markup", json!({"lines": [ctx], "blank_after": false}))
        });
        out.extend(if indent == 0 { raw_ctx.to_vec() } else { indented(&raw_ctx, indent) });
    }
    out
}

// ── what the user wrote ─────────────────────────────────────────────────────

/// Your message: a label and a tinted bubble at the right edge. The bubble is as
/// wide as its longest row (at most four fifths of the screen), the text inside is
/// left-aligned, and a slim accent bar closes it on the right.
fn user_blocks(app: &App, text: &str, width: u16, blocks: &mut Blocks) {
    let p = &app.ui.palette;
    let w = width as usize;
    let text = strip_ansi(text);
    let max_bubble = ((w * 4) / 5).min(w.saturating_sub(2)).max(5);
    let max_inner = max_bubble.saturating_sub(4).max(1);
    let rows = wrap_soft(text.trim_end_matches('\n'), max_inner);
    let inner = rows.iter().map(|r| width_of(r.trim_end())).max().unwrap_or(0).clamp(1, max_inner);
    let bubble_w = inner + 4;
    let pad = w.saturating_sub(bubble_w + 1);
    let bg = p.user_bg;

    let lead = || Span::raw(" ".repeat(pad));
    let bar = || Span::styled("▐", Style::default().fg(p.accent).bg(bg));
    let tint = |n: usize| Span::styled(" ".repeat(n), Style::default().bg(bg));
    let edge_row = || Line::from(vec![lead(), tint(bubble_w - 1), bar()]);

    blocks.add_cols(rows_of(vec![edge_row()]), Cols::NONE);
    let body: Vec<Line<'static>> = rows
        .iter()
        .map(|r| {
            let r = r.trim_end();
            let used = width_of(r);
            Line::from(vec![
                lead(),
                tint(2),
                Span::styled(r.to_string(), Style::default().fg(p.text).bg(bg)),
                tint(inner - used + 1),
                bar(),
            ])
        })
        .collect();
    blocks.add_cols(rows_of(body), Cols::span(pad + 2, pad + 2 + inner));
    blocks.add_cols(rows_of(vec![edge_row()]), Cols::NONE);
}

// ── answers, thoughts ───────────────────────────────────────────────────────

/// `✦ Argent`, once per answer run. The quiet "on the way" chip that used to sit
/// here labelled narration before a tool call as an incomplete answer; it said
/// nothing a reader needed and appeared over tool calls as if they were answers.
fn answer_header(p: &Palette) -> Line<'static> {
    Line::from(vec![
        Span::raw(" ".repeat(INDENT)),
        Span::styled("✦ Argent", Style::default().fg(p.accent).add_modifier(Modifier::BOLD)),
    ])
}

/// A thought, drawn quietly: dim, no fill, a dotted rail. Folded it is one row with
/// a preview; opened it is the reasoning itself. `kind` is `reasoning` or `rethink`.
#[allow(clippy::too_many_arguments)]
fn thought_blocks(
    app: &App,
    idx: usize,
    slot: u16,
    kind: &'static str,
    text: &str,
    expanded: bool,
    duration: Option<f64>,
    width: u16,
    blocks: &mut Blocks,
) {
    let p = &app.ui.palette;
    let (glyph, label, color) = if kind == "rethink" {
        ("◈", "Перепроверка", p.rethink)
    } else {
        ("◇", "Анализ", p.think)
    };
    let dur = match duration {
        Some(d) if d >= 0.1 => format!(" · {d:.1}s"),
        _ => String::new(),
    };
    let rail = vec![Span::raw(" "), Span::styled(DOTS, fg(color).add_modifier(Modifier::DIM)), Span::raw(" ")];
    let chevron = Span::styled(
        if expanded { "▾" } else { "▸" },
        fg(color).add_modifier(Modifier::DIM),
    );
    let head_spans = vec![
        chevron,
        Span::raw(" "),
        Span::styled(format!("{glyph} {label}{dur}"), fg(color).add_modifier(Modifier::BOLD)),
    ];
    let w = width as usize;

    if !expanded {
        let mut spans = rail;
        spans.extend(head_spans);
        let used: usize = spans.iter().map(|s| width_of(&s.content)).sum();
        // The gist goes in what is left of the row, with an ellipsis when it is cut.
        let room = w.saturating_sub(used + 3);
        let preview = thought_preview(&strip_ansi(text));
        if !preview.is_empty() && room >= 8 {
            let shown = if width_of(&preview) > room {
                let mut cut = String::new();
                let mut cw = 0;
                for ch in preview.chars() {
                    let c = ch.width().unwrap_or(0);
                    if cw + c > room - 1 {
                        break;
                    }
                    cut.push(ch);
                    cw += c;
                }
                format!("{}…", cut.trim_end())
            } else {
                preview
            };
            spans.push(Span::raw("  "));
            spans.push(Span::styled(
                shown,
                Style::default().fg(crate::theme::blend(color, p.text, 0.30)).add_modifier(Modifier::ITALIC),
            ));
        }
        blocks.add_cols(rows_of(vec![clip_line(Line::from(spans), w)]), Cols::from(CARD_PRE));
        return;
    }

    // Opened: the title, then the text through the same markdown the answer uses,
    // dimmed, behind the dotted rail.
    let inner = width.saturating_sub(CARD_PRE as u16 + 1).max(8);
    let fp = hash_of((text_fp(text), "open", dur_bits(duration), kind));
    let rows = app.cached_rows(idx, 30 + slot, fp, width, || {
        let style = app.ui.style.name();
        let raw = app.cached_block(idx, 40 + slot, fp, inner, || ("markdown", json!({"text": text, "style": style})));
        let mut v = vec![clip_line(Line::from([rail.clone(), head_spans].concat()), w)];
        let mut body: Vec<Line<'static>> = railed(&raw, &rail, true);
        while body.last().is_some_and(|l| line_width(l) <= CARD_PRE) {
            body.pop();
        }
        v.extend(body.into_iter().map(|l| clip_line(l, w)));
        v
    });
    blocks.add_cols(rows, Cols::from(CARD_PRE));
}

// ── tool calls ──────────────────────────────────────────────────────────────

fn tool_header_spans(
    app: &App,
    idx: usize,
    name: &str,
    args: &std::collections::HashMap<String, Value>,
    avail: usize,
) -> Vec<Span<'static>> {
    let fp = hash_of((name, args.len(), text_fp(&format!("{args:?}"))));
    let header = app.cached_block(idx, 1, fp, avail.clamp(20, 500) as u16, || {
        ("tool_header", json!({"name": name, "args": args}))
    });
    header.iter().flat_map(|l| l.spans.iter().cloned()).collect()
}

/// A card row's left dressing: a margin (two cells, so a card sits in from the
/// answer text like a paragraph's first line), the rail in `rail` colour, a blank.
fn card_prefix(rail: Color, bg: Color) -> Vec<Span<'static>> {
    vec![
        Span::styled("   ", Style::default().bg(bg)),
        Span::styled(RAIL, Style::default().fg(rail).bg(bg)),
        Span::styled(" ", Style::default().bg(bg)),
    ]
}

/// `⠋ read_file: agent.py (0:03)` while the tool runs: the card's first row, live.
fn tool_running_row(
    app: &App,
    idx: usize,
    name: &str,
    args: &std::collections::HashMap<String, Value>,
    width: u16,
) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let w = width as usize;
    let mut spans = card_prefix(p.tool, p.surface);
    spans.push(Span::styled(spinner_glyph(app), Style::default().fg(p.accent).add_modifier(Modifier::BOLD)));
    spans.push(Span::raw(" "));
    spans.extend(tool_header_spans(app, idx, name, args, w.saturating_sub(CARD_PRE + 12)));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(format!("({})", fmt_compact(phase_elapsed(app))), fg(p.muted)));
    vec![panel_row(Line::from(spans), w, p.surface)]
}

/// A finished tool call. Folded: two rows on a tinted card (what was called, what
/// came back, how long it took). Failed, or opened: the card gets a body with the
/// output, in red for a failure.
#[allow(clippy::too_many_arguments)]
fn tool_blocks(
    app: &App,
    idx: usize,
    name: &str,
    args: &std::collections::HashMap<String, Value>,
    result: &str,
    duration: Option<f64>,
    expanded: bool,
    width: u16,
    blocks: &mut Blocks,
) {
    let failed = toolrender::result_is_error(result);
    // The picture under the card is part of what is drawn, so its arrival has to
    // invalidate the cached rows.
    let preview_seq = app.previews.get(&idx).map(|p| p.seq).unwrap_or(0);
    // The window's position is part of the cache key: a scrolled card is a different
    // picture of the same result.
    let scroll_off = if expanded {
        app.card_scroll.get(&idx).copied().unwrap_or(0)
    } else {
        0
    };
    let fp = hash_of((
        text_fp(result),
        expanded,
        dur_bits(duration),
        name,
        text_fp(&format!("{args:?}")),
        preview_seq,
        scroll_off as u64,
    ));
    // Measured out here, on every draw: the wheel needs to know how far the card can
    // scroll even when the rows themselves come from the cache.
    if expanded {
        let cap = if name == "rooms" { ROOM_RESULT_ROWS } else { RESULT_ROWS };
        app.card_max
            .borrow_mut()
            .insert(idx, result.lines().count().saturating_sub(cap));
    } else {
        app.card_max.borrow_mut().insert(idx, 0);
    }
    let rows = app.cached_rows(idx, 20, fp, width, || {
        tool_rows(app, idx, name, args, result, duration, failed, expanded, width)
    });
    blocks.add_cols(rows, Cols::from(CARD_PRE));
}

#[allow(clippy::too_many_arguments)]
fn tool_rows(
    app: &App,
    idx: usize,
    name: &str,
    args: &std::collections::HashMap<String, Value>,
    result: &str,
    duration: Option<f64>,
    failed: bool,
    expanded: bool,
    width: u16,
) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let w = width as usize;
    let dur = duration.map(toolrender::format_duration);
    let (state_color, head_bg) = if failed { (p.danger, p.danger_soft) } else { (p.ok, p.surface) };
    let rail = if failed { p.danger } else { p.tool };
    let glyph = if failed { "✗" } else { "✓" };

    let right = match (&dur, failed || expanded) {
        (Some(d), true) => format!("{} · {d}", if failed { "failed" } else { "finished" }),
        (Some(d), false) => d.clone(),
        (None, true) => (if failed { "failed" } else { "finished" }).to_string(),
        (None, false) => String::new(),
    };
    let right_w = width_of(&right);

    let mut head = card_prefix(rail, head_bg);
    head.push(Span::styled(glyph, Style::default().fg(state_color).add_modifier(Modifier::BOLD)));
    head.push(Span::raw(" "));
    // The affordance: this card opens. ▸ folded, ▾ open. A failure no longer
    // opens itself: a failed command can print thousands of lines, and the user
    // asked to see output only after explicitly expanding the card.
    let open = expanded;
    head.push(Span::styled(
        if open { "▾" } else { "▸" },
        Style::default().fg(p.muted).bg(head_bg),
    ));
    head.push(Span::raw(" "));
    let room = w.saturating_sub(CARD_PRE + 4 + right_w + 3);
    let mut name_spans = tool_header_spans(app, idx, name, args, room);
    let name_w: usize = name_spans.iter().map(|s| width_of(&s.content)).sum();
    if name_w > room {
        // A long target is cut, not wrapped: the card keeps its shape.
        let l = clip_line(Line::from(name_spans), room.saturating_sub(1));
        name_spans = l.spans;
        let style = name_spans.last().map(|s| s.style).unwrap_or_default();
        name_spans.push(Span::styled("…", style));
    }
    let used = CARD_PRE + 4 + name_spans.iter().map(|s| width_of(&s.content)).sum::<usize>();
    head.extend(name_spans);
    if right_w > 0 {
        head.push(Span::raw(" ".repeat(w.saturating_sub(used + right_w + 1))));
        head.push(Span::styled(right, fg(if failed { p.danger } else { p.muted })));
    }
    let mut out = vec![panel_row(Line::from(head), w, head_bg)];

    if !expanded {
        // The summary row: what came back. Failures stay collapsed too; the
        // header already says "failed" and the red rail makes it visible.
        let fp = hash_of((text_fp(result), name));
        let summary = app.cached_block(idx, 2, fp, width.saturating_sub(CARD_TEXT as u16 + 1).max(20), || {
            ("tool_summary", json!({"name": name, "result": result, "args": args}))
        });
        let mut spans = card_prefix(rail, p.surface);
        spans.push(Span::styled("  ↳ ", Style::default().fg(p.muted).bg(p.surface)));
        let text: String = summary.iter().flat_map(|l| l.spans.iter().map(|s| s.content.as_ref())).collect();
        let summary_fg = if failed {
            p.danger
        } else {
            Color::Rgb(0xc9, 0xd1, 0xd9)
        };
        spans.push(Span::styled(text, Style::default().fg(summary_fg).bg(p.surface)));
        out.push(panel_row(Line::from(spans), w, p.surface));
        return out;
    }

    // Header targets are clipped so the card keeps its shape; once the card is
    // open, show the full command here, wrapped, instead of leaving it cut.
    if let Some(command) = args.get("command").and_then(|v| v.as_str()) {
        let command = strip_ansi(command);
        if !command.trim().is_empty() {
            let mut first = true;
            for part in wrap_soft(&command.replace('\t', "    "), w.saturating_sub(CARD_TEXT + 1)) {
                let mut spans = card_prefix(rail, p.code);
                spans.push(Span::styled(
                    if first { "  $ " } else { "    " },
                    Style::default().fg(p.muted).bg(p.code),
                ));
                spans.push(Span::styled(
                    part,
                    Style::default().fg(Color::Rgb(0xe6, 0xed, 0xf3)).bg(p.code),
                ));
                out.push(panel_row(Line::from(spans), w, p.code));
                first = false;
            }
        }
    }

    // The output, on the darker body of the card. The card is a WINDOW on the whole
    // result: no line is dropped, only moved out of sight, so a thousand-line run can be
    // read to its last line without leaving the chat (the wheel over the card moves it —
    // see `App::wheel`; PgUp/PgDn when the card has the focus).
    let all: Vec<&str> = result.lines().collect();
    let cap = if name == "rooms" { ROOM_RESULT_ROWS } else { RESULT_ROWS };
    let total = all.len();
    let max_off = total.saturating_sub(cap);
    let off = app.card_scroll.get(&idx).copied().unwrap_or(0).min(max_off);
    let mut body_rows: Vec<String> = Vec::new();
    for l in &all[off..(off + cap).min(total)] {
        let clean = strip_ansi(l);
        body_rows.extend(wrap_soft(&clean.replace('\t', "    "), w.saturating_sub(CARD_TEXT + 1)));
    }
    let text_style = Style::default().fg(Color::Rgb(0xc9, 0xd1, 0xd9)).bg(p.code);
    for r in body_rows {
        let mut spans = card_prefix(rail, p.code);
        spans.push(Span::styled("  ", Style::default().bg(p.code)));
        spans.push(Span::styled(r, text_style));
        out.push(panel_row(Line::from(spans), w, p.code));
    }
    if max_off > 0 {
        let last = (off + cap).min(total);
        let mut spans = card_prefix(rail, p.code);
        spans.push(Span::styled("  ", Style::default().bg(p.code)));
        spans.push(Span::styled(
            format!("{}-{} из {total} · колесо — листать", off + 1, last),
            Style::default().fg(p.muted).bg(p.code).add_modifier(Modifier::ITALIC),
        ));
        out.push(panel_row(Line::from(spans), w, p.code));
    }
    out
}

/// How wide the folded preview is drawn, at most. The wide frame is scaled down
/// to this, which is cheap and stays sharp; stretching a small grid was the bug.
const FOLDED_PREVIEW_COLS: usize = 120;

/// The picture under a card: a compact strip while folded, the whole card width
/// when opened.
///
/// Both are drawn from the wide frame (`full`), never from the small thumbnail.
/// The thumbnail is 26 columns; spread over a card a hundred columns wide it was
/// the mosaic people pointed at, and shrinking a wide grid loses far less than
/// stretching a narrow one. Once the card is opened the bridge swaps `full` for a
/// screen-sized grid, so the same code keeps the picture sharp at any width.
pub(crate) fn image_card_rows(app: &App, prev: &ImagePreview, expanded: bool, width: usize) -> Vec<Line<'static>> {
    let palette = &app.ui.palette;
    let room = width.saturating_sub(CARD_TEXT + 1).max(4);
    // Folded it is a reminder that a picture is there, not a place to read it.
    let cols = if expanded { room } else { room.min(FOLDED_PREVIEW_COLS) };
    let mut label = String::from("  ");
    if !prev.caption.is_empty() {
        label.push_str(&prev.caption);
        label.push_str("  ·  ");
    }
    if !expanded {
        // No “26×7”: that is how many characters the frame takes, not the picture's size.
        label.push_str("клик — на весь экран");
    }
    let mut spans = card_prefix(palette.tool, palette.code);
    spans.push(Span::styled(label, Style::default().fg(palette.muted).bg(palette.code)));
    let mut out = vec![panel_row(Line::from(spans), width, palette.code)];
    // The frame is wider than the card: average it down instead of cutting the
    // right-hand side off.
    for line in image_lines_scaled(&prev.full, cols) {
        let mut spans = card_prefix(palette.tool, palette.code);
        spans.extend(line.spans);
        out.push(panel_row(Line::from(spans), width, palette.code));
    }
    // The button: the only way to look at a screenshot as a picture rather than as
    // blocks of colour. Kept as its own row so the click can mean this and not the
    // full-screen overlay.
    out
}

/// A grid as half-block rows, clipped to `cols`.
///
/// The upper pixel is the glyph's colour, the lower one the background: a cell
/// shows two pixels, which is what keeps a picture from looking stretched on a
/// screen whose characters are about twice as tall as they are wide.
pub fn image_lines(grid: &RgbGrid, cols: usize) -> Vec<Line<'static>> {
    let cols = cols.min(grid.cols);
    let mut rows = Vec::with_capacity(grid.rows);
    for r in 0..grid.rows {
        let mut spans = Vec::with_capacity(cols);
        for c in 0..cols {
            let top = grid.pixel(c, r * 2);
            let bottom = grid.pixel(c, r * 2 + 1);
            spans.push(Span::styled("▀", Style::default().fg(top).bg(bottom)));
        }
        rows.push(Line::from(spans));
    }
    rows
}

/// The colour of a box of source pixels, averaged.
///
/// Averaging rather than sampling is the whole point: picking one pixel out of a
/// 6×6 block is what made a downscaled screenshot look like a mosaic.
fn box_color(grid: &RgbGrid, x0: usize, x1: usize, y0: usize, y1: usize) -> Color {
    let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
    for y in y0..y1.min(grid.rows * 2) {
        for x in x0..x1.min(grid.cols) {
            let i = (y * grid.cols + x) * 3;
            if let Some(px) = grid.data.get(i..i + 3) {
                r += px[0] as u64;
                g += px[1] as u64;
                b += px[2] as u64;
                n += 1;
            }
        }
    }
    if n == 0 {
        return Color::Reset;
    }
    Color::Rgb((r / n) as u8, (g / n) as u8, (b / n) as u8)
}

/// The same picture at another width, averaged rather than cropped or stretched.
///
/// Python sends fixed widths (a small thumb, a wide frame). A thumb blown up to
/// fill the panel is blocky and a wide frame squeezed without resampling loses
/// detail, so the strip draws the wide frame through this: every output cell is
/// the mean of the source pixels it covers.
pub fn image_lines_scaled(grid: &RgbGrid, target_cols: usize) -> Vec<Line<'static>> {
    let cols = target_cols.max(1).min(grid.cols.max(1));
    if cols == grid.cols {
        return image_lines(grid, cols);
    }
    let rows = ((grid.rows as f32) * (cols as f32) / (grid.cols as f32)).round().max(1.0) as usize;
    let src_x = grid.cols as f32 / cols as f32;
    let src_y = (grid.rows * 2) as f32 / (rows * 2) as f32;
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut spans = Vec::with_capacity(cols);
        for c in 0..cols {
            let x0 = (c as f32 * src_x) as usize;
            let x1 = (((c + 1) as f32 * src_x).ceil() as usize).max(x0 + 1);
            let top = r * 2;
            let bottom = r * 2 + 1;
            let top_y0 = (top as f32 * src_y) as usize;
            let top_y1 = (((top + 1) as f32 * src_y).ceil() as usize).max(top_y0 + 1);
            let bot_y0 = (bottom as f32 * src_y) as usize;
            let bot_y1 = (((bottom + 1) as f32 * src_y).ceil() as usize).max(bot_y0 + 1);
            spans.push(Span::styled(
                "\u{2580}",
                Style::default()
                    .fg(box_color(grid, x0, x1, top_y0, top_y1))
                    .bg(box_color(grid, x0, x1, bot_y0, bot_y1)),
            ));
        }
        out.push(Line::from(spans));
    }
    out
}

/// `1 строка`, `2 строки`, `5 строк` (the noun only).
fn plural_rows(n: usize) -> &'static str {
    match (n % 10, n % 100) {
        (_, 11..=14) => "строк",
        (1, _) => "строка",
        (2..=4, _) => "строки",
        _ => "строк",
    }
}

/// The card shown while the model writes a tool call's arguments.
fn tool_generating_card(app: &App, name: &str, bytes: usize, text: &str, width: u16) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let w = width as usize;
    let label = time_label(phase_elapsed(app), total_elapsed(app));
    let speed = speed_suffix(app, " | ");
    let mut head = card_prefix(p.tool, p.surface);
    head.push(Span::styled("✎", Style::default().fg(p.accent).bg(p.surface)));
    head.push(Span::styled(" ", Style::default().bg(p.surface)));
    head.push(Span::styled(
        format!("Генерация: {name}"),
        Style::default().fg(p.text).bg(p.surface).add_modifier(Modifier::BOLD),
    ));
    head.push(Span::styled(
        format!(" ({bytes} B | {label}{speed})"),
        Style::default().fg(p.muted).bg(p.surface),
    ));
    let mut out = vec![panel_row(Line::from(head), w, p.surface)];

    let chars: Vec<char> = text.chars().collect();
    let shown: String = chars[chars.len().saturating_sub(1000)..].iter().collect();
    let shown = if chars.len() > 1000 { format!("...\n{shown}") } else { shown };
    let shown = if shown.is_empty() { "...".to_string() } else { shown };
    for r in wrap_words(&shown, w.saturating_sub(CARD_TEXT + 1)) {
        let mut spans = card_prefix(p.tool, p.code);
        spans.push(Span::styled("  ", Style::default().bg(p.code)));
        spans.push(Span::styled(r, Style::default().fg(p.ok).bg(p.code).add_modifier(Modifier::DIM)));
        out.push(panel_row(Line::from(spans), w, p.code));
    }
    out
}

// ── a change, an error ──────────────────────────────────────────────────────

/// A file change: a header with the counts, then the diff with added lines on green
/// and removed lines on red.
fn diff_card(p: &Palette, file: &str, diff: &str, expanded: bool, width: u16) -> Vec<Line<'static>> {
    let diff = strip_ansi(diff);
    if diff.trim().is_empty() {
        return Vec::new();
    }
    let w = width as usize;
    let base = std::path::Path::new(file)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "file".to_string());
    let raw: Vec<&str> = diff.trim().lines().collect();
    let added = raw.iter().filter(|l| l.starts_with('+') && !l.starts_with("+++")).count();
    let removed = raw.iter().filter(|l| l.starts_with('-') && !l.starts_with("---")).count();

    let mut head = card_prefix(p.tool, p.surface);
    let on = |c: Color| Style::default().fg(c).bg(p.surface);
    // The arrow is the affordance: a folded card says it can open, an open one says
    // it can close. Without it the row read as a plain caption.
    head.push(Span::styled(if expanded { "▾ " } else { "▸ " }, on(p.muted)));
    head.push(Span::styled("±", on(p.accent).add_modifier(Modifier::BOLD)));
    head.push(Span::styled(" ", on(p.text)));
    head.push(Span::styled(format!("diff {base}"), on(p.text).add_modifier(Modifier::BOLD)));
    head.push(Span::styled("  ", on(p.text)));
    head.push(Span::styled(format!("+{added}"), on(p.ok).add_modifier(Modifier::BOLD)));
    head.push(Span::styled(" ", on(p.text)));
    head.push(Span::styled(format!("−{removed}"), on(p.danger).add_modifier(Modifier::BOLD)));
    if !expanded {
        // Collapsed the card is its summary row and nothing else: a long change must
        // not flood the transcript, and "… ещё N строк" was worse than useless —
        // reviewing needs the whole change at once or none of it.
        head.push(Span::styled("  клик — показать", on(p.muted)));
        return vec![panel_row(Line::from(head), w, p.surface)];
    }
    let mut out = vec![panel_row(Line::from(head), w, p.surface)];

    // Opened, the card shows the change to the end. The cap only guards the renderer
    // against a rewrite of one enormous file.
    let shown = raw.len().min(MAX_DIFF_ROWS);
    for l in raw.iter().take(shown) {
        let (rail, bg, color) = if l.starts_with('+') && !l.starts_with("+++") {
            (p.ok, p.ok_soft, p.text)
        } else if l.starts_with('-') && !l.starts_with("---") {
            (p.danger, p.danger_soft, p.text)
        } else if l.starts_with("@@") {
            (p.border, p.code, p.accent)
        } else {
            (p.border, p.code, p.muted)
        };
        let mut spans = card_prefix(rail, bg);
        spans.push(Span::styled(" ", Style::default().bg(bg)));
        spans.push(Span::styled(l.replace('\t', "    "), Style::default().fg(color).bg(bg)));
        out.push(panel_row(Line::from(spans), w, bg));
    }
    let hidden = raw.len().saturating_sub(shown);
    // No "ещё N строк" in the normal case: opened, the card shows the whole change.
    // The note survives only for a file too large to draw.
    let note = if hidden > 0 {
        Some(format!("… ещё {} {} (файл слишком большой)", hidden, plural_rows(hidden)))
    } else {
        Some("▾ свернуть — клик".to_string())
    };
    if let Some(note) = note {
        let mut spans = card_prefix(p.border, p.code);
        spans.push(Span::styled(" ", Style::default().bg(p.code)));
        spans.push(Span::styled(
            note,
            Style::default().fg(p.muted).bg(p.code).add_modifier(Modifier::ITALIC),
        ));
        out.push(panel_row(Line::from(spans), w, p.code));
    }
    out
}

/// An error: a red-tinted card with one prefix, and the hint, if there is one, dim
/// under it.
fn error_card(p: &Palette, text: &str, hint: Option<&str>, width: u16) -> Vec<Line<'static>> {
    let w = width as usize;
    let clean = strip_ansi(text);
    let lower = clean.trim_start().to_lowercase();
    let body = if lower.starts_with("error") || lower.starts_with("ошибка") {
        clean
    } else {
        format!("Ошибка: {clean}")
    };
    let bg = p.danger_soft;
    let room = w.saturating_sub(CARD_TEXT + 1).max(4);
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut first = true;
    for r in wrap_soft(body.trim_end(), room) {
        let mut spans = card_prefix(p.danger, bg);
        spans.push(Span::styled(if first { "✗ " } else { "  " }, Style::default().fg(p.danger).bg(bg).add_modifier(Modifier::BOLD)));
        spans.push(Span::styled(r, Style::default().fg(p.text).bg(bg).add_modifier(Modifier::BOLD)));
        first = false;
        out.push(panel_row(Line::from(spans), w, bg));
    }
    if let Some(h) = hint.filter(|h| !h.trim().is_empty()) {
        for r in wrap_soft(strip_ansi(h).trim_end(), room) {
            let mut spans = card_prefix(p.danger, bg);
            spans.push(Span::styled("  ", Style::default().bg(bg)));
            spans.push(Span::styled(r, Style::default().fg(p.muted).bg(bg)));
            out.push(panel_row(Line::from(spans), w, bg));
        }
    }
    out
}

// ── the live region ─────────────────────────────────────────────────────────

fn premium_spinner(app: &App, text: String) -> Line<'static> {
    let p = &app.ui.palette;
    Line::from(vec![
        Span::raw(" "),
        Span::styled(spinner_glyph(app), Style::default().fg(p.accent)),
        Span::styled(format!(" {text}"), Style::default().fg(p.muted)),
    ])
}

/// Rows for whatever the agent is doing right now, below the last message (the
/// premium look; `view::live_rows` has already ruled out "nothing is happening").
pub(crate) fn live_rows(app: &App, width: u16, height: usize) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let w = width as usize;
    let mut rows: Vec<Line<'static>> = Vec::new();
    let last = app.messages.len().saturating_sub(1);
    // A stop was acknowledged: say so at once instead of leaving the old
    // spinner ("Генерация...") looking like the turn is still producing.
    if app.is_cancelling {
        if app.messages.last().is_some_and(|m| !ends_blank(m)) {
            rows.push(Line::default());
        }
        rows.push(premium_spinner(app, "Останавливаю… (Ctrl+C ещё раз — выйти)".to_string()));
        return rows;
    }
    // The message being written is not on screen yet, so what is above it decides
    // whether a blank row is needed; a spinner waits under a message that is.
    let gap_thought = lead_blank(app, last);
    let gap_wait = app.messages.last().is_some_and(|m| !ends_blank(m));

    match app.messages.last() {
        Some(MessageItem::Assistant { is_streaming_thinking: true, thinking, .. }) => {
            if gap_thought {
                rows.push(Line::default());
            }
            rows.extend(thought_live(app, thinking, "◇ Анализ…", p.think, "Анализ задачи...", w, height));
        }
        Some(MessageItem::Rethink { is_streaming: true, text, .. }) => {
            if gap_thought {
                rows.push(Line::default());
            }
            rows.extend(thought_live(app, text, "◈ Перепроверка…", p.rethink, "Стресс-тест решения...", w, height));
        }
        Some(MessageItem::Assistant { is_streaming_content: true, content, stream, .. }) => {
            let max_lines = height.saturating_sub(8).max(3);
            let pending = stream.pending(content);
            let tail: Vec<&str> = pending.lines().collect();
            let tail = tail[tail.len().saturating_sub(max_lines)..].join("\n");
            let tail = tail.trim_matches('\n');
            if !tail.is_empty() {
                for r in wrap_words(tail, w.saturating_sub(INDENT + 1)) {
                    rows.push(Line::from(vec![
                        Span::raw(" ".repeat(INDENT)),
                        Span::styled(r, app.ui.theme.assistant_text),
                    ]));
                }
            }
            let label = time_label(phase_elapsed(app), total_elapsed(app));
            let speed = speed_suffix(app, " · ");
            rows.push(premium_spinner(app, format!("Генерация... ({label}{speed})")));
        }
        // Their own rows come from the message list.
        Some(MessageItem::ToolGenerating { .. }) => {}
        Some(MessageItem::Tool { is_running: true, .. }) => {}
        _ => {
            if gap_wait {
                rows.push(Line::default());
            }
            rows.push(premium_spinner(app, format!("Ожидание ответа... ({})", fmt_compact(total_elapsed(app)))));
        }
    }
    rows
}

/// The reasoning as it arrives: the last rows of it behind the dotted rail, then
/// the spinner.
fn thought_live(
    app: &App,
    text: &str,
    title: &str,
    color: Color,
    spinner_label: &str,
    width: usize,
    height: usize,
) -> Vec<Line<'static>> {
    let p = &app.ui.palette;
    let rail = vec![Span::raw(" "), Span::styled(DOTS, fg(color).add_modifier(Modifier::DIM)), Span::raw(" ")];
    let mut rows = vec![Line::from([rail.clone(), vec![Span::styled(title.to_string(), fg(color).add_modifier(Modifier::BOLD))]].concat())];
    let max_lines = height.saturating_sub(6).max(3);
    let inner = width.saturating_sub(CARD_PRE + 1).max(4);
    let chars: Vec<char> = text.chars().collect();
    let keep = max_lines * inner;
    let tail: String = chars[chars.len().saturating_sub(keep)..].iter().collect();
    let lines: Vec<&str> = tail.lines().collect();
    let shown = lines[lines.len().saturating_sub(max_lines)..].join("\n");
    let shown = if shown.is_empty() { "...".to_string() } else { shown };
    let wrapped = wrap_words(&shown, inner);
    let from = wrapped.len().saturating_sub(max_lines);
    for r in &wrapped[from..] {
        let mut spans = rail.clone();
        spans.push(Span::styled(r.clone(), Style::default().fg(p.muted).add_modifier(Modifier::ITALIC)));
        rows.push(Line::from(spans));
    }
    let label = time_label(phase_elapsed(app), total_elapsed(app));
    let speed = speed_suffix(app, " · ");
    rows.push(premium_spinner(app, format!("{spinner_label} ({label}{speed})")));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_wrap_breaks_at_blanks_and_keeps_every_character() {
        assert_eq!(wrap_soft("one two three", 7), vec!["one two", "three"]);
        let long = wrap_soft(&"x".repeat(20), 8);
        assert_eq!(long.concat(), "x".repeat(20));
        assert!(long.iter().all(|r| width_of(r) <= 8));
    }

    #[test]
    fn soft_wrap_keeps_inner_spacing_and_blank_lines() {
        assert_eq!(wrap_soft("a   b\n\nc", 20), vec!["a   b", "", "c"]);
    }

    #[test]
    fn soft_wrap_counts_wide_characters_as_two_columns() {
        let rows = wrap_soft("\u{4e2d}\u{4e2d}\u{4e2d}", 4);
        assert_eq!(rows, vec!["\u{4e2d}\u{4e2d}", "\u{4e2d}"]);
    }

    #[test]
    fn plural_forms_follow_russian_rules() {
        assert_eq!(plural_rows(1), "строка");
        assert_eq!(plural_rows(3), "строки");
        assert_eq!(plural_rows(5), "строк");
        assert_eq!(plural_rows(11), "строк");
        assert_eq!(plural_rows(21), "строка");
    }
}
