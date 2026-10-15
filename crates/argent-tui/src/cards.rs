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
//!  ✦ Argent  промежуточный ответ                                          an answer on the way
//!  Сначала посмотрю, как строится список.
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

use crate::ansi::{clip_line, line_width, parse_ansi, strip_ansi};
use crate::app::{App, ClickAction, MessageItem};
use crate::premium::panel_row;
use crate::renderer::thought_preview;
use crate::theme::Palette;
use crate::toolrender;
use crate::view::{
    dur_bits, fmt_compact, hash_of, phase_elapsed, queue_rows, rewind_button_row, spinner_glyph, speed_suffix,
    text_fp, time_label, total_elapsed, wrap_words, Blocks, Cols, ItemView, Rows,
};

/// Columns to the left of an assistant's text.
const INDENT: usize = 2;
/// A card's left dressing: a margin, the rail, a blank.
const CARD_PRE: usize = 3;
/// Where a card's content (a tool's name, its result) starts: under the first glyph's text.
const CARD_TEXT: usize = 5;
/// Rows of a tool's output a card shows (the REPL's own cap).
const RESULT_ROWS: usize = 15;
/// Rows of a room's report a card shows (it is a list of findings, not a log).
const ROOM_RESULT_ROWS: usize = 40;
/// Rows of a diff a card shows.
const DIFF_ROWS: usize = 25;

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

/// A line of the system: a notice or a system message.
fn is_note(m: &MessageItem) -> bool {
    matches!(m, MessageItem::Notice(_) | MessageItem::System(_))
}

/// An answer that is followed, in the same turn, by a tool call is one on the way
/// to the final answer, not the answer.
fn is_intermediate(app: &App, idx: usize) -> bool {
    for m in app.messages.iter().skip(idx + 1) {
        match m {
            MessageItem::User(_) => return false,
            MessageItem::Tool { .. } | MessageItem::Diff { .. } | MessageItem::ToolGenerating { .. } => return true,
            _ => {}
        }
    }
    false
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
            if !content.is_empty() {
                if shown_thought {
                    blocks.add_cols(blank_block(), Cols::NONE);
                }
                let intermediate = is_intermediate(app, idx);
                blocks.add_cols(rows_of(vec![answer_header(&app.ui.palette, intermediate)]), Cols::NONE);
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
            }
        }

        MessageItem::Diff { file, diff } => {
            let rows = diff_card(&app.ui.palette, file, diff, width);
            if !rows.is_empty() {
                blocks.add_cols(rows_of(rows), Cols::from(CARD_PRE));
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

        MessageItem::Snapshot { lines, context_line } => {
            let fp = hash_of((lines.iter().map(|l| text_fp(l)).collect::<Vec<_>>(), context_line.as_deref().map(text_fp)));
            gutter_block(
                app,
                idx,
                0,
                "markup",
                json!({"lines": lines, "blank_after": true}),
                fp,
                width,
                &mut blocks,
            );
            if let Some(ctx) = context_line {
                gutter_block(
                    app,
                    idx,
                    1,
                    "markup",
                    json!({"lines": [ctx], "blank_after": false}),
                    text_fp(ctx),
                    width,
                    &mut blocks,
                );
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

/// `✦ Argent`, and a chip when this is not yet the answer.
fn answer_header(p: &Palette, intermediate: bool) -> Line<'static> {
    let mut spans = vec![Span::raw(" ".repeat(INDENT))];
    if intermediate {
        let quiet = crate::theme::blend(p.accent, p.muted, 0.45);
        spans.push(Span::styled("✦ Argent", Style::default().fg(quiet).add_modifier(Modifier::BOLD)));
        spans.push(Span::raw("  "));
        spans.push(Span::styled(" промежуточный ответ ", Style::default().fg(quiet).bg(p.surface_hi)));
    } else {
        spans.push(Span::styled("✦ Argent", Style::default().fg(p.accent).add_modifier(Modifier::BOLD)));
    }
    Line::from(spans)
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
    let head = Span::styled(format!("{glyph} {label}{dur}"), fg(color).add_modifier(Modifier::BOLD));
    let w = width as usize;

    if !expanded {
        let mut spans = rail;
        spans.push(head);
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
        let mut v = vec![clip_line(Line::from([rail.clone(), vec![head.clone()]].concat()), w)];
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

/// A card row's left dressing: a margin, the rail in `rail` colour, a blank, all on
/// the row's fill.
fn card_prefix(rail: Color, bg: Color) -> Vec<Span<'static>> {
    vec![
        Span::styled(" ", Style::default().bg(bg)),
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
    let fp = hash_of((text_fp(result), expanded, dur_bits(duration), name, text_fp(&format!("{args:?}"))));
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
    let room = w.saturating_sub(CARD_PRE + 2 + right_w + 3);
    let mut name_spans = tool_header_spans(app, idx, name, args, room);
    let name_w: usize = name_spans.iter().map(|s| width_of(&s.content)).sum();
    if name_w > room {
        // A long target is cut, not wrapped: the card keeps its shape.
        let l = clip_line(Line::from(name_spans), room.saturating_sub(1));
        name_spans = l.spans;
        let style = name_spans.last().map(|s| s.style).unwrap_or_default();
        name_spans.push(Span::styled("…", style));
    }
    let used = CARD_PRE + 2 + name_spans.iter().map(|s| width_of(&s.content)).sum::<usize>();
    head.extend(name_spans);
    if right_w > 0 {
        head.push(Span::raw(" ".repeat(w.saturating_sub(used + right_w + 1))));
        head.push(Span::styled(right, fg(if failed { p.danger } else { p.muted })));
    }
    let mut out = vec![panel_row(Line::from(head), w, head_bg)];

    if !failed && !expanded {
        // The summary row: what came back.
        let fp = hash_of((text_fp(result), name));
        let summary = app.cached_block(idx, 2, fp, width.saturating_sub(CARD_TEXT as u16 + 1).max(20), || {
            ("tool_summary", json!({"name": name, "result": result, "args": args}))
        });
        let mut spans = card_prefix(rail, p.surface);
        spans.push(Span::styled("  ↳ ", Style::default().fg(p.muted).bg(p.surface)));
        let text: String = summary.iter().flat_map(|l| l.spans.iter().map(|s| s.content.as_ref())).collect();
        spans.push(Span::styled(text, Style::default().fg(Color::Rgb(0xc9, 0xd1, 0xd9)).bg(p.surface)));
        out.push(panel_row(Line::from(spans), w, p.surface));
        return out;
    }

    // The output, on the darker body of the card.
    let all: Vec<&str> = result.lines().collect();
    let cap = if name == "rooms" { ROOM_RESULT_ROWS } else { RESULT_ROWS };
    let hidden = all.len().saturating_sub(cap);
    let mut body_rows: Vec<String> = Vec::new();
    for l in all.iter().take(cap) {
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
    if hidden > 0 {
        let mut spans = card_prefix(rail, p.code);
        spans.push(Span::styled("  ", Style::default().bg(p.code)));
        spans.push(Span::styled(
            format!("… ещё {hidden} {}", plural_rows(hidden)),
            Style::default().fg(p.muted).bg(p.code).add_modifier(Modifier::ITALIC),
        ));
        out.push(panel_row(Line::from(spans), w, p.code));
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
fn diff_card(p: &Palette, file: &str, diff: &str, width: u16) -> Vec<Line<'static>> {
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
    head.push(Span::styled("±", on(p.accent).add_modifier(Modifier::BOLD)));
    head.push(Span::styled(" ", on(p.text)));
    head.push(Span::styled(format!("diff {base}"), on(p.text).add_modifier(Modifier::BOLD)));
    head.push(Span::styled("  ", on(p.text)));
    head.push(Span::styled(format!("+{added}"), on(p.ok).add_modifier(Modifier::BOLD)));
    head.push(Span::styled(" ", on(p.text)));
    head.push(Span::styled(format!("−{removed}"), on(p.danger).add_modifier(Modifier::BOLD)));
    let mut out = vec![panel_row(Line::from(head), w, p.surface)];

    for l in raw.iter().take(DIFF_ROWS) {
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
    if raw.len() > DIFF_ROWS {
        let mut spans = card_prefix(p.border, p.code);
        spans.push(Span::styled(" ", Style::default().bg(p.code)));
        spans.push(Span::styled(
            format!("… ещё {} {} — /diff {base}", raw.len() - DIFF_ROWS, plural_rows(raw.len() - DIFF_ROWS)),
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
    rows.extend(queue_rows(app, w));
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
