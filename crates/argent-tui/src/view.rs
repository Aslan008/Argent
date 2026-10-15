//! The transcript: turning the message list into terminal rows.
//!
//! Static blocks (an answer chunk, a finished tool, a diff, ...) are produced by
//! the block renderer — in production the REPL's own Python renderer — and
//! cached per (message, slot) with a cheap content fingerprint, so a block is
//! rendered once per terminal width instead of on every frame.
//!
//! Live parts (the spinner lines, the streaming tail, the queue box) are drawn
//! natively from the same strings and styles the REPL's `Live` displays use
//! (`src/cli/cli_ui.py`), and change every frame, so they are never cached.
//!
//! Only the rows that can be seen are built: items are walked from the newest
//! backwards until the viewport (plus the scroll offset) is covered, so a long
//! session costs the same per frame as a short one.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::{json, Value};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::ansi::{clip_line, line_width, parse_ansi, strip_ansi, style_from_rich};
use crate::app::{App, ClickAction, MessageItem};
use crate::renderer::{BlockRenderer, FallbackRenderer};
use crate::theme::UiStyle;

/// A rendered block: shared, so handing it out each frame copies nothing.
pub type Rows = Arc<Vec<Line<'static>>>;

/// The columns of a row that are the message's own text: what a selection paints
/// and a copy takes. Everything outside is dressing: a rail, a margin, the padding
/// of a bubble. It is left out, wherever the layout puts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cols {
    pub from: usize,
    pub to: usize,
}

impl Cols {
    /// The whole row is text (the classic look, which is the REPL's own rows).
    pub const ALL: Cols = Cols { from: 0, to: usize::MAX };
    /// A row that is decoration only: a label, a blank, the padding of a card.
    pub const NONE: Cols = Cols { from: 0, to: 0 };

    /// Text from column `from` to the end of the row.
    pub fn from(from: usize) -> Cols {
        Cols { from, to: usize::MAX }
    }

    /// Text in `from..to`.
    pub fn span(from: usize, to: usize) -> Cols {
        Cols { from, to }
    }

    pub fn is_none(&self) -> bool {
        self.to <= self.from
    }
}

struct Entry {
    fp: u64,
    width: u16,
    rows: Rows,
}

/// Rendered blocks keyed by (message index, slot).
#[derive(Default)]
pub struct ViewCache {
    map: HashMap<(usize, u16), Entry>,
}

impl ViewCache {
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

pub(crate) fn hash_of<T: Hash>(v: T) -> u64 {
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

/// A cheap fingerprint of a string that grows by appending: its length plus its
/// first and last bytes. Enough to notice any streamed change without hashing
/// megabytes per frame.
pub(crate) fn text_fp(s: &str) -> u64 {
    let b = s.as_bytes();
    let mut h = DefaultHasher::new();
    b.len().hash(&mut h);
    b[..b.len().min(24)].hash(&mut h);
    b[b.len().saturating_sub(24)..].hash(&mut h);
    h.finish()
}

pub(crate) fn dur_bits(d: Option<f64>) -> u64 {
    d.map(f64::to_bits).unwrap_or(0)
}

// ── small text helpers ──────────────────────────────────────────────────────

/// `0:08`, `1:24`, `1:12:05` — the REPL's `_format_time_compact`.
pub fn fmt_compact(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    let (m, s) = (s / 60, s % 60);
    let (h, m) = (m / 60, m % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// `этап 0:03 · всего 0:08`, or just the total when the phase IS the turn.
pub fn time_label(phase: f64, total: f64) -> String {
    if phase < total - 1.0 {
        format!("этап {} · всего {}", fmt_compact(phase), fmt_compact(total))
    } else {
        fmt_compact(total)
    }
}

/// Greedy word wrap by display width; blank lines and leading indentation of a
/// line are kept, and an over-long word is hard-split. Mirrors how Rich wraps a
/// plain `Text`.
pub fn wrap_words(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        if raw.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let indent: String = raw.chars().take_while(|c| *c == ' ').collect();
        let mut line = indent.clone();
        let mut line_w = UnicodeWidthStr::width(line.as_str());
        let mut first = true;
        for word in raw.split_whitespace() {
            let ww = UnicodeWidthStr::width(word);
            let sep = usize::from(!first && line_w > UnicodeWidthStr::width(indent.as_str()));
            if line_w + sep + ww <= width {
                if sep == 1 {
                    line.push(' ');
                }
                line.push_str(word);
                line_w += sep + ww;
            } else if ww <= width {
                out.push(std::mem::take(&mut line));
                line = word.to_string();
                line_w = ww;
            } else {
                // A word wider than the screen: fill the current row, then cut.
                for ch in word.chars() {
                    let cw = ch.width().unwrap_or(0);
                    if line_w + cw > width {
                        out.push(std::mem::take(&mut line));
                        line_w = 0;
                    }
                    line.push(ch);
                    line_w += cw;
                }
            }
            first = false;
        }
        out.push(line);
    }
    out
}

/// Cut `text` into rows of at most `width` columns, breaking mid-word like a
/// terminal does (the typed prompt is wrapped that way, not by words).
pub fn wrap_hard(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for raw in text.split('\n') {
        let mut row = String::new();
        let mut w = 0usize;
        for ch in raw.chars() {
            let cw = ch.width().unwrap_or(0);
            if w + cw > width {
                out.push(std::mem::take(&mut row));
                w = 0;
            }
            row.push(ch);
            w += cw;
        }
        out.push(row);
    }
    out
}

// ── native building blocks ──────────────────────────────────────────────────

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Rich's `dots` spinner: ten frames, 80 ms apart.
pub fn spinner_glyph(app: &App) -> &'static str {
    if let Some(i) = app.spinner_frame.get() {
        return SPINNER[i % SPINNER.len()];
    }
    let ms = app.status.start_time.elapsed().as_millis() as usize;
    SPINNER[(ms / 80) % SPINNER.len()]
}

pub(crate) fn dim_cyan() -> Style {
    Style::default().fg(Color::Indexed(6)).add_modifier(Modifier::DIM)
}

/// `⠋  <text>` — a Rich `Spinner("dots", text=" …")`: the frame, a space, then
/// the text, which itself starts with a space.
pub(crate) fn spinner_row(app: &App, text: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = vec![Span::raw(spinner_glyph(app)), Span::raw(" ")];
    spans.extend(text);
    Line::from(spans)
}

pub(crate) fn total_elapsed(app: &App) -> f64 {
    app.turn_started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0)
}

pub(crate) fn phase_elapsed(app: &App) -> f64 {
    app.phase_started.elapsed().as_secs_f64()
}

/// The model's generation speed (`48 tok/s`, `~48 tok/s` while it is still an
/// estimate) for a live row, joined to the row's timers with `sep`; empty when
/// there is none to show.
///
/// Only the premium look has it: the classic rows are the REPL's own, cell for cell,
/// and the REPL shows no speed. And only while the model is writing — a tool
/// running or a reply being awaited has no speed, so the figure is simply absent.
pub(crate) fn speed_suffix(app: &App, sep: &str) -> String {
    if app.ui.style != UiStyle::Premium {
        return String::new();
    }
    match app.speed.reading(std::time::Instant::now()) {
        Some(r) => format!("{sep}{}", r.label()),
        None => String::new(),
    }
}

/// A rounded box spanning the whole width, like Rich's default `Panel`.
pub fn full_panel(
    title: Vec<Span<'static>>,
    body: Vec<Line<'static>>,
    border: Style,
    width: usize,
) -> Vec<Line<'static>> {
    let width = width.max(8);
    let inner = width - 4;
    let title_w: usize = title.iter().map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum();
    let fill = inner.saturating_sub(title_w + 2);
    let (left, right) = (fill / 2, fill - fill / 2);

    let mut top = vec![
        Span::styled("╭─", border),
        Span::styled("─".repeat(left), border),
        Span::styled(" ", border),
    ];
    top.extend(title);
    top.push(Span::styled(" ", border));
    top.push(Span::styled("─".repeat(right), border));
    top.push(Span::styled("─╮", border));

    let mut out = vec![Line::from(top)];
    for l in body {
        let mut l = clip_line(l, inner);
        let w = line_width(&l);
        let mut spans = vec![Span::styled("│", border), Span::raw(" ")];
        spans.append(&mut l.spans);
        spans.push(Span::raw(" ".repeat(inner - w)));
        spans.push(Span::raw(" "));
        spans.push(Span::styled("│", border));
        out.push(Line::from(spans));
    }
    out.push(Line::from(Span::styled(format!("╰{}╯", "─".repeat(width - 2)), border)));
    out
}

/// The REPL's `Очередь сообщений` box (cli_ui `_render_queue_input_display`).
pub(crate) fn queue_rows(app: &App, width: usize) -> Vec<Line<'static>> {
    if app.message_queue.is_empty() {
        return Vec::new();
    }
    let border = Style::default().fg(Color::Indexed(6)).add_modifier(Modifier::DIM);
    // Rich lays the title's `bold cyan` over the panel's `cyan dim`, so the title
    // is dim as well — the REPL's box shows exactly that.
    let title = vec![Span::styled(
        format!("Очередь сообщений ({})", app.message_queue.len()),
        Style::default().fg(Color::Indexed(6)).add_modifier(Modifier::BOLD | Modifier::DIM),
    )];
    let body: Vec<Line<'static>> = app
        .message_queue
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let mut short: String = item.replace('\n', " ");
            if short.chars().count() > 65 {
                short = format!("{}…", short.chars().take(62).collect::<String>());
            }
            Line::from(vec![
                Span::raw(" "),
                Span::styled(format!("{}.", i + 1), Style::default().add_modifier(Modifier::DIM)),
                Span::raw(" "),
                Span::styled(strip_ansi(&short), Style::default().fg(Color::Indexed(7))),
            ])
        })
        .collect();
    full_panel(title, body, border, width)
}

/// What the user typed, as the REPL echoes it: `❯ ` then the text, continuation
/// rows indented to line up under it. (The premium look draws it in `cards.rs`.)
fn user_rows(app: &App, text: &str, width: u16) -> Vec<Line<'static>> {
    let text = strip_ansi(text);
    let theme = &app.ui.theme;
    let budget = (width as usize).saturating_sub(2).max(1);
    let mut out = Vec::new();
    let mut first = true;
    for logical in text.split('\n') {
        for row in wrap_hard(logical, budget) {
            let prefix = if first { "❯ " } else { "  " };
            first = false;
            out.push(Line::from(vec![
                Span::styled(prefix, theme.user_prompt_symbol.add_modifier(Modifier::BOLD)),
                Span::styled(row, theme.user_prompt),
            ]));
        }
    }
    out
}

// ── per-item views ──────────────────────────────────────────────────────────

/// Collects the blocks of one message, remembering which of them are buttons and
/// which columns of each are text.
#[derive(Default)]
pub(crate) struct Blocks(pub(crate) Vec<(Rows, Option<ClickAction>, Cols)>);

impl Blocks {
    pub(crate) fn add(&mut self, rows: Rows) {
        self.0.push((rows, None, Cols::ALL));
    }

    /// A block whose text is only part of each row.
    pub(crate) fn add_cols(&mut self, rows: Rows, cols: Cols) {
        self.0.push((rows, None, cols));
    }

    /// Make the block added last a click target.
    pub(crate) fn click_last(&mut self, action: ClickAction) {
        if let Some(last) = self.0.last_mut() {
            last.1 = Some(action);
        }
    }
}

/// One message turned into rows: a few cached blocks, each with what a click on
/// it does (only the reasoning line and the tool card are buttons — the answer
/// text is not).
pub struct ItemView {
    pub blocks: Vec<(Rows, Option<ClickAction>, Cols)>,
    pub focused: bool,
    pub len: usize,
    /// How many rows at the end are interface (a button), not the message's text.
    pub chrome_tail: usize,
    /// How many rows at the start are spacing (a blank before the message): the focus
    /// marker belongs to the first row after them.
    pub lead: usize,
}

impl ItemView {
    /// All rows of the message, in order.
    pub fn lines(&self) -> Vec<Line<'static>> {
        self.blocks.iter().flat_map(|(rows, _, _)| rows.iter().cloned()).collect()
    }

    /// For each row of `lines()`, the columns that are the message's text.
    pub fn cols(&self) -> Vec<Cols> {
        self.blocks.iter().flat_map(|(rows, _, cols)| std::iter::repeat_n(*cols, rows.len())).collect()
    }

    /// The rows that are the message's own text: what selecting and copying should see.
    pub fn content_lines(&self) -> Vec<Line<'static>> {
        let mut all = self.lines();
        all.truncate(all.len().saturating_sub(self.chrome_tail));
        all
    }
}

/// `  ↩ Вернуться сюда (диалог и файлы)`: the button under the last message sent.
/// The premium look sets it under the bubble, at the right edge, where the message is.
pub(crate) fn rewind_button_row(app: &App, width: u16) -> Line<'static> {
    let label_style = if app.ui.is_premium() {
        Style::default().fg(app.ui.palette.accent).add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default().fg(Color::Indexed(6)).add_modifier(Modifier::UNDERLINED)
    };
    let hint_style = Style::default().add_modifier(Modifier::DIM);
    let mut spans = vec![
        Span::styled("↩ Вернуться сюда", label_style),
        Span::styled(" (диалог и файлы)", hint_style),
    ];
    let lead = if app.ui.is_premium() {
        let w: usize = spans.iter().map(|s| UnicodeWidthStr::width(s.content.as_ref())).sum();
        (width as usize).saturating_sub(w + 2).max(2)
    } else {
        2
    };
    spans.insert(0, Span::raw(" ".repeat(lead)));
    clip_line(Line::from(spans), width as usize)
}

impl App {
    /// A block from the cache, rendering it first when needed.
    ///
    /// `fp` fingerprints everything the block depends on; `make` builds the
    /// request lazily so a hit costs no allocation. A failed render falls back to
    /// the Rust approximation and is remembered once for the UI to report.
    pub fn cached_block(
        &self,
        idx: usize,
        slot: u16,
        fp: u64,
        width: u16,
        make: impl FnOnce() -> (&'static str, Value),
    ) -> Rows {
        if let Some(e) = self.view_cache.borrow().map.get(&(idx, slot)) {
            if e.fp == fp && e.width == width {
                return e.rows.clone();
            }
        }
        let (kind, data) = make();
        let rows = match self.renderer.render(kind, &data, width) {
            Ok(r) => r,
            Err(err) => {
                let mut slot_err = self.render_error.borrow_mut();
                if slot_err.is_none() {
                    *slot_err = Some(format!("{kind}: {err}"));
                }
                FallbackRenderer.render(kind, &data, width).unwrap_or_default()
            }
        };
        let rows: Rows = Arc::new(rows);
        self.view_cache
            .borrow_mut()
            .map
            .insert((idx, slot), Entry { fp, width, rows: rows.clone() });
        rows
    }

    /// A composed block (rows built in Rust out of other blocks) from the same cache.
    ///
    /// Same contract as `cached_block`: `fp` fingerprints everything the rows depend
    /// on, and a hit costs nothing.
    pub fn cached_rows(
        &self,
        idx: usize,
        slot: u16,
        fp: u64,
        width: u16,
        make: impl FnOnce() -> Vec<Line<'static>>,
    ) -> Rows {
        if let Some(e) = self.view_cache.borrow().map.get(&(idx, slot)) {
            if e.fp == fp && e.width == width {
                return e.rows.clone();
            }
        }
        let rows: Rows = Arc::new(make());
        self.view_cache
            .borrow_mut()
            .map
            .insert((idx, slot), Entry { fp, width, rows: rows.clone() });
        rows
    }

    /// Drop cache entries that belong to messages that no longer exist.
    fn prune_view_cache(&self) {
        let n = self.messages.len();
        let mut cache = self.view_cache.borrow_mut();
        if cache.map.len() > n.saturating_mul(6) + 64 {
            cache.map.retain(|(idx, _), _| *idx < n);
        }
    }
}

pub(crate) fn mark_focus(rows: &mut [Line<'static>]) {
    // The first row of a collapsible block carries a gutter marker in column 0.
    // Rows that start with an indent give up one of its spaces; a row that
    // starts flush (a full reasoning tree) is reversed instead, so nothing shifts.
    if let Some(first) = rows.first_mut() {
        if let Some(sp) = first.spans.first_mut() {
            let content = sp.content.to_string();
            if content.starts_with(' ') {
                let rest: String = content.chars().skip(1).collect();
                let marker = Span::styled("▸", Style::default().fg(Color::Indexed(6)).add_modifier(Modifier::BOLD));
                let mut spans = vec![marker];
                if !rest.is_empty() {
                    spans.push(Span::styled(rest, sp.style));
                }
                spans.extend(first.spans.drain(1..));
                first.spans = spans;
                return;
            }
        }
        for sp in first.spans.iter_mut() {
            sp.style = sp.style.add_modifier(Modifier::REVERSED);
        }
    }
}

/// The rows of message `idx` at `width`.
pub fn item_view(app: &App, idx: usize, width: u16) -> ItemView {
    if app.ui.is_premium() {
        return crate::cards::item_view(app, idx, width);
    }
    let mut blocks = Blocks::default();
    let mut chrome_tail = 0usize;

    match &app.messages[idx] {
        MessageItem::User(text) => {
            blocks.add(Arc::new(user_rows(app, text, width)));
            if app.rewind_target() == Some(idx) {
                blocks.add(Arc::new(vec![rewind_button_row(app, width)]));
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
            if !thinking.is_empty() && !*is_streaming_thinking {
                let mode = if *thinking_expanded { "full" } else { "summary" };
                let fp = hash_of((text_fp(thinking), mode, dur_bits(*thinking_duration)));
                blocks.add(app.cached_block(idx, 0, fp, width, || {
                    (
                        "reasoning",
                        json!({"text": thinking, "duration": thinking_duration, "mode": mode}),
                    )
                }));
                blocks.click_last(ClickAction::ToggleThinking(idx));
            }
            if !content.is_empty() {
                blocks.add(app.cached_block(idx, 1, 1, width, || ("assistant_header", json!({}))));
                let mut ranges = stream.chunks.clone();
                if !*is_streaming_content && stream.committed < content.len() {
                    ranges.push(stream.committed..content.len());
                }
                // The look is part of the request (and of the cache key): the same
                // markdown is drawn as plain REPL text or as designed blocks.
                let style = app.ui.style.name();
                for (i, r) in ranges.iter().enumerate() {
                    let Some(chunk) = content.get(r.clone()) else { continue };
                    let fp = hash_of((r.start, r.end, text_fp(chunk), style));
                    blocks.add(app.cached_block(idx, 100 + i as u16, fp, width, || {
                        ("markdown", json!({"text": chunk, "style": style}))
                    }));
                }
            }
        }

        MessageItem::Rethink { text, expanded, is_streaming, duration, .. } => {
            if !*is_streaming && !text.trim().is_empty() {
                let mode = if *expanded { "full" } else { "summary" };
                let fp = hash_of((text_fp(text), mode, dur_bits(*duration)));
                blocks.add(app.cached_block(idx, 0, fp, width, || {
                    ("rethink", json!({"text": text, "duration": duration, "mode": mode}))
                }));
                blocks.click_last(ClickAction::ToggleThinking(idx));
            }
        }

        MessageItem::Tool { name, args, result, duration_secs, expanded, is_running } => {
            if *is_running {
                blocks.add(Arc::new(tool_running_rows(app, idx, name, args, width)));
            } else {
                let exp = *expanded || app.show_tool_results || app.ui.show_tool_results;
                let res = result.as_deref().unwrap_or("");
                let fp = hash_of((text_fp(res), exp, dur_bits(*duration_secs), name.as_str()));
                blocks.add(app.cached_block(idx, 0, fp, width, || {
                    (
                        "tool_end",
                        json!({"name": name, "result": res, "duration": duration_secs,
                               "args": args, "expanded": exp}),
                    )
                }));
            }
            blocks.click_last(ClickAction::ToggleTool(idx));
        }

        MessageItem::Diff { file, diff } => {
            let fp = hash_of((text_fp(file), text_fp(diff)));
            blocks.add(app.cached_block(idx, 0, fp, width, || {
                ("diff", json!({"file": file, "diff": diff}))
            }));
        }

        MessageItem::Notice(text) => {
            if text.contains('\u{1b}') {
                // Already styled by Rich (command output): draw it as it is.
                let rows: Vec<Line<'static>> = parse_ansi(text)
                    .into_iter()
                    .map(|l| clip_line(l, width as usize))
                    .collect();
                blocks.add(Arc::new(rows));
            } else {
                let fp = text_fp(text);
                blocks.add(app.cached_block(idx, 0, fp, width, || ("system", json!({"text": text}))));
            }
        }

        MessageItem::System(text) => {
            let fp = text_fp(text);
            blocks.add(app.cached_block(idx, 0, fp, width, || ("system", json!({"text": text}))));
        }

        MessageItem::Banner(version) => {
            let fp = text_fp(version);
            blocks.add(app.cached_block(idx, 0, fp, width, || ("banner", json!({"version": version}))));
        }

        MessageItem::Error { text, hint } => {
            let fp = hash_of((text_fp(text), hint.as_deref().map(text_fp)));
            blocks.add(app.cached_block(idx, 0, fp, width, || {
                ("error", json!({"text": text, "hint": hint}))
            }));
        }

        MessageItem::Checkpoint { sha, .. } => {
            let fp = text_fp(sha);
            blocks.add(app.cached_block(idx, 0, fp, width, || ("checkpoint", json!({"sha": sha}))));
        }

        MessageItem::Snapshot { lines, context_line } => {
            let fp = hash_of((lines.iter().map(|l| text_fp(l)).collect::<Vec<_>>(), context_line.as_deref().map(text_fp)));
            blocks.add(app.cached_block(idx, 0, fp, width, || {
                ("markup", json!({"lines": lines, "blank_after": true}))
            }));
            if let Some(ctx) = context_line {
                blocks.add(app.cached_block(idx, 1, text_fp(ctx), width, || {
                    ("markup", json!({"lines": [ctx], "blank_after": false}))
                }));
            }
        }

        MessageItem::ToolGenerating { name, bytes, text } => {
            blocks.add(Arc::new(tool_generating_rows(app, name, *bytes, text, width)));
        }
    }

    let len = blocks.0.iter().map(|(b, _, _)| b.len()).sum();
    ItemView { blocks: blocks.0, focused: app.is_focused_block(idx), len, chrome_tail, lead: 0 }
}

/// `⠋   ⠋ name: target (0:03)` — the REPL's `_ToolWaitDisplay`, including its
/// second, static spinner glyph.
fn tool_running_rows(
    app: &App,
    idx: usize,
    name: &str,
    args: &std::collections::HashMap<String, Value>,
    width: u16,
) -> Vec<Line<'static>> {
    if App::is_interactive_tool(name) {
        // The tool draws its own prompt; a spinner would fight it (as in the REPL).
        return Vec::new();
    }
    let fp = hash_of((name, args.len(), text_fp(&format!("{args:?}"))));
    let header = app.cached_block(idx, 1, fp, width, || {
        ("tool_header", json!({"name": name, "args": args}))
    });
    let mut text: Vec<Span<'static>> = vec![
        Span::raw("  "),
        Span::styled("⠋", Style::default().fg(Color::Indexed(6)).add_modifier(Modifier::BOLD)),
        Span::raw(" "),
    ];
    for l in header.iter() {
        text.extend(l.spans.iter().cloned());
    }
    text.push(Span::raw(" "));
    text.push(Span::styled(
        format!("({})", fmt_compact(phase_elapsed(app))),
        Style::default().add_modifier(Modifier::DIM),
    ));
    vec![clip_line(spinner_row(app, text), width as usize)]
}

/// The REPL's `_ToolGenDisplay`: a cyan box while the model writes a tool call.
fn tool_generating_rows(app: &App, name: &str, bytes: usize, text: &str, width: u16) -> Vec<Line<'static>> {
    let label = time_label(phase_elapsed(app), total_elapsed(app));
    let speed = speed_suffix(app, " | ");
    let title = vec![Span::styled(
        format!("Генерация: {name} ({bytes} B | {label}{speed})"),
        dim_cyan(),
    )];
    // The last 1000 characters of the arguments being written, as the REPL shows
    // them (an ellipsis row marks the cut), or `...` before any text arrived.
    let chars: Vec<char> = text.chars().collect();
    let shown: String = chars[chars.len().saturating_sub(1000)..].iter().collect();
    let shown = if chars.len() > 1000 { format!("...\n{shown}") } else { shown };
    let shown = if shown.is_empty() { "...".to_string() } else { shown };
    let green = Style::default().fg(Color::Indexed(2)).add_modifier(Modifier::DIM);
    let body: Vec<Line<'static>> = wrap_words(&shown, (width as usize).saturating_sub(4))
        .into_iter()
        .map(|r| Line::from(Span::styled(r, green)))
        .collect();
    full_panel(title, body, Style::default().fg(Color::Indexed(6)), width as usize)
}

// ── the live region ─────────────────────────────────────────────────────────

/// Rows for whatever the agent is doing right now, below the last message.
pub fn live_rows(app: &App, width: u16, height: usize) -> Vec<Line<'static>> {
    let w = width as usize;
    let mut rows: Vec<Line<'static>> = Vec::new();
    // A slash command prints as it goes; the REPL shows no spinner for it.
    if app.is_command_in_flight {
        return rows;
    }
    // An answer that is still streaming keeps its tail on screen even if the
    // busy flag is not up (a replacement arrives before the first delta, say):
    // hiding the tail would hide text that has already been received.
    let tail_open = matches!(
        app.messages.last(),
        Some(MessageItem::Assistant { is_streaming_content: true, .. })
    );
    if !app.is_busy && !tail_open {
        return rows;
    }
    if app.ui.is_premium() {
        return crate::cards::live_rows(app, width, height);
    }
    let theme = &app.ui.theme;

    match app.messages.last() {
        Some(MessageItem::Assistant { is_streaming_thinking: true, thinking, .. }) => {
            rows.extend(reasoning_live(
                app,
                thinking,
                theme.reasoning_label,
                theme.reasoning_text,
                "Анализ:",
                "Анализ задачи...",
                w,
                height,
            ));
        }
        Some(MessageItem::Rethink { is_streaming: true, text, .. }) => {
            rows.extend(reasoning_live(
                app,
                text,
                style_from_rich("bold #d18f52"),
                style_from_rich("italic #ffe0b2"),
                "Перепроверка (Rethink):",
                "Стресс-тест решения...",
                w,
                height,
            ));
        }
        Some(MessageItem::Assistant { is_streaming_content: true, content, stream, .. }) => {
            let max_lines = height.saturating_sub(8).max(3);
            let pending = stream.pending(content);
            let tail: Vec<&str> = pending.lines().collect();
            let tail = tail[tail.len().saturating_sub(max_lines)..].join("\n");
            let tail = tail.trim_matches('\n');
            if !tail.is_empty() {
                for r in wrap_words(tail, w) {
                    rows.push(Line::from(Span::styled(r, theme.assistant_text)));
                }
            }
            let label = time_label(phase_elapsed(app), total_elapsed(app));
            let speed = speed_suffix(app, " · ");
            rows.push(spinner_row(
                app,
                vec![Span::styled(format!(" Генерация... ({label}{speed})"), dim_cyan())],
            ));
        }
        // Their own rows come from the message list.
        Some(MessageItem::ToolGenerating { .. }) => {}
        Some(MessageItem::Tool { is_running: true, .. }) => {}
        _ => {
            rows.push(spinner_row(
                app,
                vec![Span::styled(
                    format!(" Ожидание ответа... ({})", fmt_compact(total_elapsed(app))),
                    dim_cyan(),
                )],
            ));
        }
    }
    rows.extend(queue_rows(app, w));
    rows
}

#[allow(clippy::too_many_arguments)]
fn reasoning_live(
    app: &App,
    text: &str,
    label_style: Style,
    text_style: Style,
    title: &str,
    spinner_label: &str,
    width: usize,
    height: usize,
) -> Vec<Line<'static>> {
    let mut rows = vec![
        Line::default(),
        Line::from(Span::styled(title.to_string(), label_style)),
        Line::default(),
    ];
    let max_lines = height.saturating_sub(6).max(3);
    let chars: Vec<char> = text.chars().collect();
    let keep = max_lines * width.max(1);
    let tail: String = chars[chars.len().saturating_sub(keep)..].iter().collect();
    let lines: Vec<&str> = tail.lines().collect();
    let shown = lines[lines.len().saturating_sub(max_lines)..].join("\n");
    let shown = if shown.is_empty() { "...".to_string() } else { shown };
    let wrapped = wrap_words(&shown, width);
    let from = wrapped.len().saturating_sub(max_lines);
    for r in &wrapped[from..] {
        rows.push(Line::from(Span::styled(r.clone(), text_style)));
    }
    rows.push(Line::default());
    let label = time_label(phase_elapsed(app), total_elapsed(app));
    let speed = speed_suffix(app, " · ");
    rows.push(spinner_row(
        app,
        vec![Span::styled(format!(" {spinner_label} ({label}{speed})"), dim_cyan())],
    ));
    rows
}

// ── assembling the visible window ───────────────────────────────────────────

/// One screen row of the transcript.
pub struct Row {
    pub line: Line<'static>,
    pub click: Option<ClickAction>,
    /// Where the row comes from: the message index (`messages.len()` for the live
    /// rows below the last message) and the row's index inside it. A selection is
    /// kept in these terms, so it does not move when the chat scrolls.
    pub anchor: (usize, usize),
    /// Which columns are the row's text (a selection paints and copies only these).
    pub cols: Cols,
}

/// The visible slice of the transcript plus what the scroll logic needs.
pub struct Transcript {
    pub rows: Vec<Row>,
    /// Rows known so far (everything, when `exact`).
    pub total: usize,
    /// True once the oldest message has been reached, so `total` is the real
    /// height and the scroll limit is exact.
    pub exact: bool,
    /// The scroll offset actually used (clamped once the limit is known).
    pub scroll: usize,
}

/// Build the rows that fit a viewport of `height` rows, scrolled `scroll` rows
/// up from the newest output, at `width` columns.
pub fn build_transcript(app: &App, width: u16, height: usize, scroll: usize) -> Transcript {
    app.prune_view_cache();
    let live = live_rows(app, width, height);
    // A chat the user has scrolled back stays put while output arrives below it.
    let scroll = match app.scroll_anchor.get() {
        Some(a) if scroll > 0 && a.offset == scroll => anchored_scroll(app, &a, width, height, live.len()).unwrap_or(scroll),
        _ => scroll,
    };
    let need = height.saturating_add(scroll);

    let mut collected: Vec<(ItemView, usize)> = Vec::new();
    let mut running = live.len();
    let mut idx = app.messages.len();
    while idx > 0 && running < need {
        idx -= 1;
        let v = item_view(app, idx, width);
        running += v.len;
        collected.push((v, idx));
    }
    let exact = idx == 0;
    collected.reverse();

    // Positions below are relative to the oldest row we hold; anything older is
    // outside the window by construction (we stop collecting once it is covered).
    let total = running;
    // Once the oldest message is on hand the scroll limit is exact: never show a
    // window that starts above the first row.
    let scroll = if exact { scroll.min(total.saturating_sub(height)) } else { scroll };
    let first = total.saturating_sub(height.saturating_add(scroll));
    let last = total.saturating_sub(scroll).max(first);

    let mut out: Vec<Row> = Vec::with_capacity(height);
    let mut pos = 0usize;
    for (view, item) in &collected {
        if pos + view.len <= first || pos >= last {
            pos += view.len;
            continue;
        }
        // Materialise this item's rows only when some of them are visible.
        let mut item_rows: Vec<Line<'static>> = view.lines();
        if view.focused {
            let lead = view.lead.min(item_rows.len());
            mark_focus(&mut item_rows[lead..]);
        }
        let clicks = view
            .blocks
            .iter()
            .flat_map(|(rows, click, cols)| std::iter::repeat_n((click.clone(), *cols), rows.len()));
        for (i, (line, (click, cols))) in item_rows.into_iter().zip(clicks).enumerate() {
            let at = pos + i;
            if at >= first && at < last {
                out.push(Row { line, click, anchor: (*item, i), cols });
            }
        }
        pos += view.len;
    }
    for (i, line) in live.into_iter().enumerate() {
        let at = pos + i;
        if at >= first && at < last {
            out.push(Row { line, click: None, anchor: (app.messages.len(), i), cols: Cols::ALL });
        }
    }

    Transcript { rows: out, total, exact, scroll }
}

/// The scroll offset that keeps the anchored row at the top of a viewport of
/// `height` rows: the rows below it, less the rows the viewport shows from it down.
/// `None` when the anchor no longer points at anything.
///
/// Only the messages BELOW the anchor are measured, and they are the ones a walk up
/// from the newest output has to build anyway to reach the viewport, so this costs
/// about what drawing the scrolled-back chat did before.
fn anchored_scroll(app: &App, a: &crate::app::ScrollAnchor, width: u16, height: usize, live_len: usize) -> Option<usize> {
    if a.item >= app.messages.len() {
        return None;
    }
    let own = item_view(app, a.item, width).len;
    if own == 0 {
        return None;
    }
    let mut below = live_len + (own - 1 - a.row.min(own - 1));
    for idx in (a.item + 1..app.messages.len()).rev() {
        below += item_view(app, idx, width).len;
    }
    Some(below.saturating_sub(height.saturating_sub(1)))
}

/// Plain text of a row, for copying a selection.
pub fn row_text(line: &Line<'_>) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_time_format() {
        assert_eq!(fmt_compact(8.4), "0:08");
        assert_eq!(fmt_compact(84.0), "1:24");
        assert_eq!(fmt_compact(4325.0), "1:12:05");
        assert_eq!(fmt_compact(-3.0), "0:00");
    }

    #[test]
    fn phase_label_only_when_it_differs_from_the_turn() {
        assert_eq!(time_label(8.0, 8.5), "0:08");
        assert_eq!(time_label(3.0, 8.0), "этап 0:03 · всего 0:08");
    }

    #[test]
    fn word_wrap_keeps_blank_lines_and_never_loses_text() {
        let rows = wrap_words("one two three\n\nfour", 7);
        assert_eq!(rows, vec!["one two", "three", "", "four"]);
        let long = wrap_words(&"x".repeat(20), 8);
        assert_eq!(long.concat(), "x".repeat(20));
        assert!(long.iter().all(|r| r.len() <= 8));
    }

    #[test]
    fn hard_wrap_breaks_mid_word_and_counts_columns() {
        assert_eq!(wrap_hard("abcdef", 4), vec!["abcd", "ef"]);
        assert_eq!(wrap_hard("a\nb", 4), vec!["a", "b"]);
        assert_eq!(wrap_hard("\u{4e2d}\u{4e2d}\u{4e2d}", 4), vec!["\u{4e2d}\u{4e2d}", "\u{4e2d}"]);
    }

    #[test]
    fn full_panel_spans_exactly_the_width() {
        let rows = full_panel(
            vec![Span::raw("Title")],
            vec![Line::from("body")],
            Style::default(),
            40,
        );
        assert_eq!(rows.len(), 3);
        for r in &rows {
            assert_eq!(line_width(r), 40, "{}", row_text(r));
        }
        assert!(row_text(&rows[0]).starts_with("╭─"));
        assert!(row_text(&rows[0]).contains(" Title "));
        assert!(row_text(&rows[2]).starts_with("╰"));
    }

    #[test]
    fn title_is_centred_with_the_odd_cell_on_the_right() {
        // Matches Rich: 9 fill on the left, 10 on the right for a 55-wide title.
        let title = "x".repeat(55);
        let rows = full_panel(vec![Span::raw(title)], vec![], Style::default(), 80);
        let top = row_text(&rows[0]);
        let after_corner: String = top.chars().skip(2).collect();
        let left = after_corner.chars().take_while(|c| *c == '─').count();
        assert_eq!(left, 9, "{top}");
    }
}
