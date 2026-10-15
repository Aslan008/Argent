//! Markdown rendering for the TUI.
//!
//! The legacy CLI leaned on `rich.markdown.Markdown` (markdown-it-py) plus
//! Pygments. A terminal has no rich-text engine, so the supported subset is
//! rendered here directly into ratatui lines: headings, fenced code, block
//! quotes, lists, rules, tables and inline emphasis/code.
//!
//! Inline markers that are not recognised are emitted verbatim — the same
//! "data, not markup" rule the CLI applies to tool results.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

/// Matches the CLI's `#181a1f` code / diff background.
pub const CODE_BG: Color = Color::Rgb(24, 26, 31);
const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::Rgb(125, 133, 144);
const CODE_FG: Color = Color::Rgb(163, 190, 140);
const LINK: Color = Color::Rgb(121, 192, 255);

/// Greedy word wrap honouring unicode display width.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    let mut line = String::new();
    let mut line_w = 0usize;

    for word in text.split_whitespace() {
        let w = UnicodeWidthStr::width(word);
        if line.is_empty() {
            line.push_str(word);
            line_w = w;
        } else if line_w + 1 + w <= width {
            line.push(' ');
            line.push_str(word);
            line_w += 1 + w;
        } else {
            out.push(std::mem::take(&mut line));
            line.push_str(word);
            line_w = w;
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// A word too long for one line has to be hard-split or it would be clipped.
fn hard_split(word: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for ch in word.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1);
        if cur_w + cw > width {
            out.push(std::mem::take(&mut cur));
            cur_w = 0;
        }
        cur.push(ch);
        cur_w += cw;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Split text into wrappable lines, hard-splitting over-long words.
fn wrap_hard(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    for para in wrap(text, width) {
        if UnicodeWidthStr::width(para.as_str()) <= width.max(1) {
            out.push(para);
        } else {
            out.extend(hard_split(&para, width));
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Render inline markdown (`**bold**`, `*italic*`, `` `code` ``, `~~strike~~`)
/// into styled spans.
pub fn inline(text: &str, base: Style) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut i = 0usize;

    let flush = |buf: &mut String, spans: &mut Vec<Span<'static>>| {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), base));
        }
    };

    while i < chars.len() {
        let c = chars[i];

        // `code`
        if c == '`' {
            if let Some(end) = find_marker(&chars, i + 1, '`') {
                flush(&mut buf, &mut spans);
                let inner: String = chars[i + 1..end].iter().collect();
                spans.push(Span::styled(inner, base.fg(CODE_FG)));
                i = end + 1;
                continue;
            }
        }

        // ~~strike~~
        if c == '~' && chars.get(i + 1) == Some(&'~') {
            if let Some(end) = find_seq(&chars, i + 2, "~~") {
                flush(&mut buf, &mut spans);
                let inner: String = chars[i + 2..end].iter().collect();
                spans.push(Span::styled(inner, base.add_modifier(Modifier::CROSSED_OUT)));
                i = end + 2;
                continue;
            }
        }

        // **bold**
        if c == '*' && chars.get(i + 1) == Some(&'*') {
            if let Some(end) = find_seq(&chars, i + 2, "**") {
                flush(&mut buf, &mut spans);
                let inner: String = chars[i + 2..end].iter().collect();
                for (n, piece) in inline(&inner, base.add_modifier(Modifier::BOLD)).into_iter().enumerate() {
                    if n == 0 {
                        spans.push(Span::styled(piece.content, base.fg(Color::White).add_modifier(Modifier::BOLD)));
                    } else {
                        spans.push(piece);
                    }
                }
                i = end + 2;
                continue;
            }
        }

        // *italic*
        if c == '*' {
            if let Some(end) = find_marker(&chars, i + 1, '*') {
                flush(&mut buf, &mut spans);
                let inner: String = chars[i + 1..end].iter().collect();
                spans.extend(inline(&inner, base.add_modifier(Modifier::ITALIC)));
                i = end + 1;
                continue;
            }
        }

        // [text](url)
        if c == '[' {
            if let Some((label, url, after)) = parse_link(&chars, i) {
                flush(&mut buf, &mut spans);
                spans.extend(inline(&label, base.fg(LINK).add_modifier(Modifier::UNDERLINED)));
                spans.push(Span::styled(format!(" ({url})"), base.fg(MUTED)));
                i = after;
                continue;
            }
        }

        buf.push(c);
        i += 1;
    }
    flush(&mut buf, &mut spans);
    spans
}

fn find_marker(chars: &[char], from: usize, marker: char) -> Option<usize> {
    (from..chars.len()).find(|&j| chars[j] == marker)
}

fn find_seq(chars: &[char], from: usize, marker: &str) -> Option<usize> {
    let m: Vec<char> = marker.chars().collect();
    if m.is_empty() || chars.len() < m.len() {
        return None;
    }
    (from..=chars.len() - m.len()).find(|&j| chars[j..j + m.len()] == m[..])
}

/// Returns `(label, url, index_after)`.
fn parse_link(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    let close = (start + 1..chars.len()).find(|&j| chars[j] == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let paren = (close + 2..chars.len()).find(|&j| chars[j] == ')')?;
    let label: String = chars[start + 1..close].iter().collect();
    let url: String = chars[close + 2..paren].iter().collect();
    Some((label, url, paren + 1))
}

// ── lightweight syntax highlighting ────────────────────────────────────────

const KW_RUST: &[&str] = &["fn", "let", "mut", "struct", "enum", "impl", "trait", "use", "mod",
    "pub", "const", "static", "match", "if", "else", "for", "while", "loop", "return", "self",
    "Some", "None", "Ok", "Err", "true", "false", "async", "await", "move", "ref", "where",
    "break", "continue", "unsafe", "dyn", "type", "as", "in"];
const KW_PY: &[&str] = &["def", "class", "return", "if", "elif", "else", "for", "while", "in",
    "import", "from", "as", "with", "try", "except", "finally", "raise", "lambda", "yield",
    "async", "await", "pass", "break", "continue", "and", "or", "not", "is", "None", "True",
    "False", "self", "global", "nonlocal", "assert", "del"];
const KW_JS: &[&str] = &["const", "let", "var", "function", "return", "if", "else", "for",
    "while", "do", "switch", "case", "break", "continue", "new", "class", "extends", "import",
    "from", "export", "default", "async", "await", "try", "catch", "finally", "throw", "typeof",
    "instanceof", "this", "null", "undefined", "true", "false", "interface", "type", "enum"];

fn keywords_for(lang: &str) -> &'static [&'static str] {
    let l = lang.to_ascii_lowercase();
    if l.contains("rs") || l.contains("rust") {
        KW_RUST
    } else if l.starts_with("py") || l.contains("python") {
        KW_PY
    } else if l.contains("js") || l.contains("ts") || l.contains("node") || l.contains("json") {
        KW_JS
    } else {
        &[]
    }
}

fn line_comment(lang: &str) -> &'static str {
    let l = lang.to_ascii_lowercase();
    if l.contains("py") || l.contains("sh") || l.contains("bash") || l.contains("yaml") {
        "#"
    } else if l.contains("sql") {
        "--"
    } else {
        "//"
    }
}

/// Tokenise one line into coloured spans.
///
/// Deliberately small: it recognises comments, strings, numbers and keywords for
/// the common families. Pulling in a full syntax engine for a terminal TUI
/// costs tens of megabytes of tables and buys little at these font sizes.
pub fn highlight_line(line: &str, lang: &str) -> Vec<Span<'static>> {
    let kws = keywords_for(lang);
    let comment = line_comment(lang);
    let bytes: Vec<char> = line.chars().collect();
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut i = 0usize;

    let style = |b: &mut String, out: &mut Vec<Span<'static>>, s: Style| {
        if !b.is_empty() {
            out.push(Span::styled(std::mem::take(b), s));
        }
    };

    while i < bytes.len() {
        let c = bytes[i];

        // Comment to end of line.
        if !comment.is_empty() && bytes[i..].iter().collect::<String>().starts_with(comment) {
            style(&mut buf, &mut out, Style::default().fg(CODE_FG));
            let rest: String = bytes[i..].iter().collect();
            out.push(Span::styled(rest, Style::default().fg(Color::Rgb(106, 115, 129)).add_modifier(Modifier::ITALIC)));
            return out;
        }

        // String literal, with backslash escapes.
        if c == '"' || c == '\'' || c == '`' {
            style(&mut buf, &mut out, Style::default().fg(CODE_FG));
            let quote = c;
            let mut s = String::new();
            s.push(c);
            i += 1;
            while i < bytes.len() {
                let ch = bytes[i];
                if ch == '\\' && i + 1 < bytes.len() {
                    s.push(ch);
                    s.push(bytes[i + 1]);
                    i += 2;
                    continue;
                }
                s.push(ch);
                i += 1;
                if ch == quote {
                    break;
                }
            }
            out.push(Span::styled(s, Style::default().fg(Color::Rgb(206, 145, 120))));
            continue;
        }

        // Number.
        if c.is_ascii_digit() {
            style(&mut buf, &mut out, Style::default().fg(CODE_FG));
            let mut n = String::new();
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == '.' || bytes[i] == '_') {
                n.push(bytes[i]);
                i += 1;
            }
            out.push(Span::styled(n, Style::default().fg(Color::Rgb(181, 206, 168))));
            continue;
        }

        // Identifier / keyword.
        if c.is_alphabetic() || c == '_' {
            style(&mut buf, &mut out, Style::default().fg(Color::Rgb(220, 223, 228)));
            let mut w = String::new();
            while i < bytes.len() && (bytes[i].is_alphanumeric() || bytes[i] == '_') {
                w.push(bytes[i]);
                i += 1;
            }
            let is_kw = kws.contains(&w.as_str());
            out.push(Span::styled(
                w,
                if is_kw {
                    Style::default().fg(Color::Rgb(197, 134, 192)).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Rgb(220, 223, 228))
                },
            ));
            continue;
        }

        // Punctuation / operator.
        if "(){}[];,:.<>=+-*/%&|!?^~#@".contains(c) {
            style(&mut buf, &mut out, Style::default().fg(CODE_FG));
            out.push(Span::styled(c.to_string(), Style::default().fg(Color::Rgb(140, 148, 158))));
            i += 1;
            continue;
        }

        buf.push(c);
        i += 1;
    }
    style(&mut buf, &mut out, Style::default().fg(Color::Rgb(220, 223, 228)));
    out
}

fn heading_prefix(level: usize) -> &'static str {
    match level {
        1 => "█ ",
        2 => "▓▓ ",
        _ => "▒ ",
    }
}

/// A fenced code block open/close test. Accepts ``` and ~~~, 3+ chars, with an
/// optional info string — matching the CLI's stream splitter rather than its
/// narrower render-time regex.
fn fence_of(line: &str) -> Option<(char, usize, String)> {
    let t = line.trim_start();
    let ch = t.chars().next()?;
    if ch != '`' && ch != '~' {
        return None;
    }
    let n = t.chars().take_while(|&c| c == ch).count();
    if n < 3 {
        return None;
    }
    let info: String = t.chars().skip(n).collect();
    let info = info.trim().to_string();
    // An info string on a backtick fence may not itself contain a backtick.
    if ch == '`' && info.contains('`') {
        return None;
    }
    Some((ch, n, info))
}

fn is_rule(line: &str) -> bool {
    let t: String = line.trim().chars().filter(|c| !c.is_whitespace()).collect();
    t.len() >= 3
        && (t.chars().all(|c| c == '-')
            || t.chars().all(|c| c == '*')
            || t.chars().all(|c| c == '_'))
}

/// For a list item, returns `(indent, content_col, content, bullet)`.
///
/// `bullet` is `"•"` for unordered items and the original `"1."` marker for
/// ordered ones. Slicing uses `content_col` so multi-byte markers are handled —
/// indexing by the rendered bullet length instead would eat characters.
fn list_item(line: &str) -> Option<(usize, usize, String, String)> {
    let indent = line.len() - line.trim_start().len();
    let t = line.trim_start();
    for marker in ["- ", "* ", "+ "] {
        if let Some(rest) = t.strip_prefix(marker) {
            return Some((indent, indent + marker.len(), rest.to_string(), "•".to_string()));
        }
    }
    let digits: String = t.chars().take_while(char::is_ascii_digit).collect();
    if !digits.is_empty() {
        let mlen = digits.len() + 2; // "N."
        if let Some(rest) = t[digits.len()..].strip_prefix(". ") {
            return Some((
                indent,
                indent + mlen,
                rest.to_string(),
                format!("{digits}."),
            ));
        }
    }
    None
}

fn quote_prefix(line: &str) -> Option<usize> {
    let indent = line.len() - line.trim_start().len();
    let t = line.trim_start();
    if let Some(rest) = t.strip_prefix("> ") {
        return Some(indent + 2.min(rest.len()));
    }
    if t == ">" {
        return Some(indent + 1);
    }
    None
}

/// True when `line` opens or closes a pipe table (second line is the delimiter).
fn table_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.contains('|')
}

fn table_delimiter(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|')
        && t.contains('-')
        && t
            .trim_matches('|')
            .split('|')
            .all(|c| c.trim().chars().all(|ch| ch == '-' || ch == ':' || ch.is_whitespace()))
}

/// Split a `| a | b |` row into trimmed cells.
fn table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|c| c.trim().to_string())
        .collect()
}

/// Render markdown into ratatui lines fitting `width` columns.
pub fn render(text: &str, width: usize) -> Vec<Line<'static>> {
    let width = width.max(20);
    let base = Style::default();
    let mut out: Vec<Line<'static>> = Vec::new();
    // Model replies and tool output carry SGR codes; ratatui has no escape
    // engine, so they would be painted as literal `[35;58;33M` text.
    let cleaned;
    let text = if text.contains('\u{1b}') || text.contains('\r') {
        cleaned = crate::ansi::strip_ansi(text);
        cleaned.as_str()
    } else {
        text
    };
    let src: Vec<&str> = text.lines().collect();
    let mut i = 0usize;

    while i < src.len() {
        let line = src[i];

        // ── fenced code block ───────────────────────────────────────────────
        // Drawn like the REPL's `_render_code_blocks`: a `── lang ──` title, then
        // the code on the #181a1f background with a cell of padding either side.
        // No side bars and no "copy" strip — the REPL has neither.
        if let Some((ch, n, info)) = fence_of(line) {
            let label = if info.is_empty() { "text".to_string() } else { info.clone() };
            let dim_rule = Style::default().add_modifier(Modifier::DIM);
            out.push(Line::from(vec![
                Span::styled("──", dim_rule),
                Span::raw(" "),
                Span::styled(label.clone(), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled("──", dim_rule),
            ]));

            let inner = width.saturating_sub(2).max(8);
            let bg = Style::default().bg(CODE_BG);
            let code_style = Style::default().fg(CODE_FG).bg(CODE_BG);
            let mut j = i + 1;
            while j < src.len() {
                if let Some((c2, n2, _)) = fence_of(src[j]) {
                    if c2 == ch && n2 >= n {
                        break;
                    }
                }
                let code_line = src[j].replace('\t', "    ");
                let cut = if UnicodeWidthStr::width(code_line.as_str()) > inner {
                    format!("{}…", take_cols(&code_line, inner.saturating_sub(1)))
                } else {
                    code_line
                };
                let mut spans = vec![Span::styled(" ", bg)];
                if label == "text" {
                    spans.push(Span::styled(cut.clone(), code_style));
                } else {
                    for sp in highlight_line(&cut, &label) {
                        spans.push(Span::styled(sp.content.to_string(), sp.style.bg(CODE_BG)));
                    }
                }
                let used = 1 + UnicodeWidthStr::width(cut.as_str());
                spans.push(Span::styled(" ".repeat(width.saturating_sub(used)), bg));
                out.push(Line::from(spans));
                j += 1;
            }
            i = j + 1;
            continue;
        }

        // ── blank ───────────────────────────────────────────────────────────
        if line.trim().is_empty() {
            if !out.is_empty() {
                out.push(Line::default());
            }
            i += 1;
            continue;
        }

        // ── table ───────────────────────────────────────────────────────────
        if table_row(line) && i + 1 < src.len() && table_delimiter(src[i + 1]) {
            let header = table_cells(line);
            let mut rows: Vec<Vec<String>> = vec![header.clone()];
            let mut j = i + 2;
            while j < src.len() && table_row(src[j]) {
                rows.push(table_cells(src[j]));
                j += 1;
            }
            let aligns: Vec<Alignment> = table_cells(src[i + 1])
                .iter()
                .map(|c| {
                    let l = c.starts_with(':');
                    let r = c.ends_with(':');
                    match (l, r) {
                        (true, true) => Alignment::Center,
                        (true, false) => Alignment::Left,
                        (false, true) => Alignment::Right,
                        (false, false) => Alignment::Left,
                    }
                })
                .collect();

            // Column widths come from the widest cell across header AND body.
            //
            // A rendered row is `│ ` + (cell + ` │ `) per column, i.e.
            // 2 + sum(widths) + 3 * cols columns wide. A row wider than the
            // target gets wrapped by the viewport, which shreds the table:
            // borders land mid-line and the left edge is cut off. So both the
            // column count and the cell widths are clamped to fit.
            let natural_cols = rows.iter().map(Vec::len).max().unwrap_or(1).max(1);
            // Every column costs 3 columns of separator plus at least 1 of cell.
            let max_cols = (width.saturating_sub(2) / 4).max(1);
            let cols = natural_cols.min(max_cols);
            let dropped = natural_cols - cols;

            let mut widths = vec![0usize; cols];
            for r in &rows {
                for (w, cell) in widths.iter_mut().zip(r.iter()) {
                    *w = (*w).max(UnicodeWidthStr::width(cell.as_str()));
                }
            }
            // Shave the widest columns until the row fits the target width.
            let budget = width.saturating_sub(2 + 3 * cols);
            loop {
                let total: usize = widths.iter().sum();
                if total <= budget {
                    break;
                }
                let Some((idx, widest)) = widths
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, w)| **w)
                    .map(|(i, w)| (i, *w))
                else {
                    break;
                };
                if widest == 0 {
                    break;
                }
                widths[idx] = widest - 1;
            }

            let sep = |v: &[String], header_row: bool| -> Line<'static> {
                let mut s = vec![Span::styled("│ ", Style::default().fg(MUTED))];
                for (c, cw) in widths.iter().enumerate() {
                    let mut raw = v.get(c).map(String::as_str).unwrap_or("");
                    // Say so rather than silently dropping the tail columns.
                    if dropped > 0 && c + 1 == cols {
                        raw = "\u{2026}";
                    }
                    let cell = align_cell(raw, *cw, aligns.get(c).copied().unwrap_or(Alignment::Left));
                    let style = if header_row {
                        Style::default().fg(Color::Rgb(121, 192, 255)).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::White)
                    };
                    s.push(Span::styled(cell, style));
                    s.push(Span::styled(" │ ", Style::default().fg(MUTED)));
                }
                Line::from(s)
            };
            out.push(sep(&rows[0], true));
            // The rule must span exactly the same columns as a data row.
            let row_w = 2 + widths.iter().sum::<usize>() + 3 * cols;
            out.push(Line::from(Span::styled(
                "\u{2500}".repeat(row_w.min(width).max(1)),
                Style::default().fg(MUTED),
            )));
            for r in &rows[1..] {
                out.push(sep(r, false));
            }
            i = j;
            continue;
        }

        // ── heading ─────────────────────────────────────────────────────────
        let t = line.trim_start();
        if t.starts_with('#') {
            let level = t.chars().take_while(|&c| c == '#').count();
            if level <= 6 && t[level..].starts_with(' ') {
                let text = t[level..].trim();
                let style = match level {
                    1 => Style::default().fg(Color::Rgb(88, 166, 255)).add_modifier(Modifier::BOLD),
                    2 => Style::default().fg(Color::Rgb(121, 192, 255)).add_modifier(Modifier::BOLD),
                    _ => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                };
                out.push(Line::from(vec![
                    Span::styled(heading_prefix(level), style),
                    Span::styled(text.to_string(), style),
                ]));
                i += 1;
                continue;
            }
        }

        // ── rule ────────────────────────────────────────────────────────────
        if is_rule(line) {
            out.push(Line::from(Span::styled(
                "─".repeat(width.min(78)),
                Style::default().fg(MUTED),
            )));
            i += 1;
            continue;
        }

        // ── block quote (consume the whole run) ─────────────────────────────
        if quote_prefix(line).is_some() {
            let mut buf: Vec<String> = Vec::new();
            while i < src.len() {
                match quote_prefix(src[i]) {
                    Some(p) => {
                        buf.push(src[i][p..].to_string());
                        i += 1;
                    }
                    None => {
                        // Lazy continuation of the previous quoted line.
                        if src[i].trim().is_empty() || list_item(src[i]).is_some() {
                            break;
                        }
                        buf.push(src[i].to_string());
                        i += 1;
                    }
                }
            }
            for (n, piece) in wrap_hard(&buf.join(" "), width.saturating_sub(4).max(8)).into_iter().enumerate() {
                let bar = if n == 0 { "▎" } else { " " };
                out.push(Line::from(vec![
                    Span::styled(format!("{bar} "), Style::default().fg(ACCENT)),
                    Span::styled(piece, Style::default().fg(Color::Rgb(200, 205, 212)).add_modifier(Modifier::ITALIC)),
                ]));
            }
            continue;
        }

        // ── list (consume the whole run, preserving nesting indent) ─────────
        if list_item(line).is_some() {
            while i < src.len() {
                let Some((indent, content_col, content, bullet)) = list_item(src[i]) else { break };
                let avail = width.saturating_sub(content_col).max(8);
                for (n, piece) in wrap_hard(&content, avail).into_iter().enumerate() {
                    let head = if n == 0 {
                        format!("{}{} ", " ".repeat(indent), bullet)
                    } else {
                        " ".repeat(content_col)
                    };
                    out.push(Line::from(vec![
                        Span::styled(head, Style::default().fg(ACCENT)),
                        Span::styled(piece, Style::default()),
                    ]));
                }
                i += 1;
            }
            continue;
        }

        // ── paragraph ───────────────────────────────────────────────────────
        let mut buf = String::from(line.trim());
        i += 1;
        while i < src.len() {
            let nxt = src[i];
            if nxt.trim().is_empty()
                || fence_of(nxt).is_some()
                || nxt.trim_start().starts_with('#')
                || list_item(nxt).is_some()
                || quote_prefix(nxt).is_some()
                || is_rule(nxt)
                || table_row(nxt)
            {
                break;
            }
            buf.push(' ');
            buf.push_str(nxt.trim());
            i += 1;
        }
        for piece in wrap_hard(&buf, width) {
            out.push(Line::from(inline(&piece, base)));
        }
    }

    out
}

#[derive(Clone, Copy)]
enum Alignment {
    Left,
    Center,
    Right,
}

fn take_cols(s: &str, cols: usize) -> String {
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(1);
        if w + cw > cols {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out
}

fn pad_to(s: &str, width: usize) -> String {
    let w = UnicodeWidthStr::width(s);
    if w >= width {
        take_cols(s, width)
    } else {
        format!("{}{}", s, " ".repeat(width - w))
    }
}

fn align_cell(s: &str, width: usize, align: Alignment) -> String {
    let w = UnicodeWidthStr::width(s);
    if w >= width {
        return take_cols(s, width);
    }
    match align {
        Alignment::Left => pad_to(s, width),
        Alignment::Right => format!("{}{}", " ".repeat(width - w), s),
        Alignment::Center => {
            let total = width - w;
            format!("{}{}{}", " ".repeat(total / 2), s, " ".repeat(total - total / 2))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(lines: &[Line<'static>]) -> String {
        lines.iter().map(|l| {
            l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()
        }).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn heading_and_paragraph() {
        let out = render("# Title\n\nHello world", 40);
        let text = plain(&out);
        assert!(text.contains("Title"), "got {text:?}");
        assert!(text.contains("Hello world"), "got {text:?}");
    }

    #[test]
    fn fenced_code_block_with_language() {
        let out = render("intro\n```python\nx = 1\n```\nafter", 40);
        let text = plain(&out);
        assert!(text.contains("── python ──"), "got {text:?}");
        assert!(text.contains("x = 1"), "got {text:?}");
        assert!(text.contains("after"), "got {text:?}");
    }

    #[test]
    fn tilde_fence_is_recognised() {
        let out = render("~~~sh\nls\n~~~", 40);
        assert!(plain(&out).contains("ls"));
        assert!(plain(&out).contains("── sh ──"));
    }

    #[test]
    fn inline_code_and_bold() {
        let spans = inline("a **b** `c`", Style::default());
        assert!(spans.iter().any(|s| s.content == "b"));
        assert!(spans.iter().any(|s| s.content == "c"));
    }

    #[test]
    fn unterminated_marker_stays_literal() {
        let spans = inline("a `unclosed", Style::default());
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "a `unclosed");
    }

    #[test]
    fn long_line_is_hard_split_not_dropped() {
        let word = "x".repeat(200);
        let out = render(&word, 30);
        let joined: String = out.iter().flat_map(|l| l.spans.iter()).map(|s| s.content.as_ref()).collect();
        assert_eq!(joined.chars().filter(|c| *c == 'x').count(), 200, "characters were lost");
    }

    #[test]
    fn list_and_quote_and_rule() {
        let out = render("- one\n- two\n\n> quoted\n\n---", 40);
        let text = plain(&out);
        assert!(text.contains("one"), "{text:?}");
        assert!(text.contains("•"), "{text:?}");
        assert!(text.contains("quoted"), "{text:?}");
    }

    #[test]
    fn table_renders_cells() {
        let out = render("| a | b |\n|---|---|\n| 1 | 2 |", 40);
        let text = plain(&out);
        assert!(text.contains('a') && text.contains('2'), "{text:?}");
    }

    #[test]
    fn highlighting_marks_keywords_strings_and_comments() {
        let spans = highlight_line("let x = \"hi\"; // note", "rust");
        let kw = spans.iter().any(|s| s.content == "let" && s.style.add_modifier.contains(Modifier::BOLD));
        assert!(kw, "keyword not emphasised");
        let string = spans
            .iter()
            .find(|s| s.content.contains('"'))
            .map(|s| s.content.clone())
            .unwrap_or_default();
        assert_eq!(string, "\"hi\"", "string should keep its quotes: {string:?}");
        let comment = spans
            .iter()
            .any(|s| s.content.starts_with("//") && s.content.contains("note"));
        assert!(comment, "comment not recognised");
    }

    #[test]
    fn highlighting_is_inert_for_plain_text() {
        let spans = highlight_line("hello world", "text");
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "hello world");
    }

    #[test]
    fn highlighting_never_drops_characters() {
        for (line, lang) in [
            ("fn f(x: i32) -> i32 { x + 1 }", "rust"),
            ("def f(x):\n    return 'a'  # c", "python"),
            ("const a = `tpl`; // x", "js"),
            ("a\"b\\\"c", "rust"),
        ] {
            let joined: String = highlight_line(line, lang)
                .iter()
                .map(|s| s.content.as_ref())
                .collect();
            assert_eq!(joined, line, "highlighting altered the text for {lang}");
        }
    }

    #[test]
    fn code_block_content_is_preserved_with_a_language() {
        let out = render("```rust\nlet v = vec![1];\n```", 60);
        let text = plain(&out);
        assert!(text.contains("let v = vec![1];"), "got {text:?}");
    }

    #[test]
    fn never_panics_on_pathological_input() {
        for w in [1usize, 2, 5, 20, 200] {
            let _ = render("# \n```\n\n|\n>\n- \n1. \n~~~", w);
            let _ = render(&"a".repeat(1000), w);
            let _ = render("```\n", w);
        }
    }
}
#[cfg(test)]
mod width_tests {
    use super::*;

    fn widths(out: &[Line<'static>]) -> Vec<usize> {
        out.iter().map(|l| l.width()).collect()
    }

    #[test]
    fn a_table_never_exceeds_the_requested_width() {
        // The shape from a real transcript: a cell far wider than the terminal.
        let src = "| Параметр | Значение |\n|---|---|\n| Процесс (PID 3420) | `nvcontainer.exe` |\n| Состояние | Running |";
        for w in [20usize, 24, 30, 40, 60, 80, 120] {
            let out = render(src, w);
            let over: Vec<usize> = widths(&out).into_iter().filter(|x| *x > w).collect();
            assert!(
                over.is_empty(),
                "requested width {w} but produced rows of {over:?}"
            );
        }
    }

    #[test]
    fn a_wide_table_is_clamped_rather_than_wrapped() {
        let cell = "x".repeat(200);
        let src = format!("| a | b |\n|---|---|\n| {cell} | y |");
        let out = render(&src, 30);
        for row in widths(&out) {
            assert!(row <= 30, "row of {row} columns at width 30");
        }
    }

    #[test]
    fn a_very_wide_table_still_fits() {
        // 12 columns is more than a narrow terminal can hold.
        let mut src = String::from("|");
        for i in 0..12 {
            src.push_str(&format!(" c{i} |"));
        }
        src.push_str("\n|");
        for _ in 0..12 {
            src.push_str(" --- |");
        }
        src.push_str("\n|");
        for i in 0..12 {
            src.push_str(&format!(" value{i} |"));
        }
        for w in [20usize, 24, 40, 200] {
            for row in widths(&render(&src, w)) {
                assert!(row <= w, "row of {row} columns at width {w}");
            }
        }
    }
}
