//! Terminal escape-sequence removal.
//!
//! Tool output, diffs and model replies routinely carry SGR colour codes. The
//! legacy CLI leaned on `rich`, which interprets them; this frontend paints into
//! ratatui cells and has no escape-sequence engine, so the bytes would be
//! rendered as literal text — a captured `ESC[35;58;33m` shows up on screen as
//! `[35;58;33M`, which is where the "random digits" in the transcript come from.
//!
//! Everything the renderer draws goes through [`strip_ansi`] first.

/// CSI: `ESC [` params `0x30..=0x3F` intermediates `0x20..=0x2F` final `0x40..=0x7E`.
const CSI: char = '[';
/// String-terminated introducers: OSC, DCS, SOS, PM, APC. All end at BEL or ST.
const ST_STRING_INTRODUCERS: [char; 5] = [']', 'P', 'X', '^', '_'];
const BEL: char = '\u{7}';
const ESC: char = '\u{1b}';
const ST_ESCAPE: char = '\\';

/// Remove terminal escape sequences and collapse carriage-return redraws.
///
/// A bare `\r` means the producer rewrote the line it was on (progress bars,
/// spinners), so only what follows it survives — which is what a real terminal
/// would end up displaying.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == ESC {
            strip_one_sequence(&mut chars, &mut out);
            continue;
        }
        if c == '\r' {
            // CRLF is an ordinary line ending, not a redraw — treating it as
            // one would wipe the line and report an empty result.
            if chars.peek() == Some(&'\n') {
                out.push('\n');
                chars.next();
                continue;
            }
            // Bare CR: the producer rewrote the line it was on (progress bars,
            // spinners), so only what follows survives.
            match out.rfind('\n') {
                Some(nl) => out.truncate(nl + 1),
                None => out.clear(),
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Consume one escape sequence starting after the ESC, appending its
/// non-escape payload (there is none) to `out`.
fn strip_one_sequence(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, _out: &mut String) {
    match chars.peek().copied() {
        // CSI — parameterised, terminated by a byte in 0x40..=0x7E.
        Some(CSI) => {
            chars.next();
            for c in chars.by_ref() {
                if ('\u{40}'..='\u{7e}').contains(&c) {
                    break;
                }
            }
        }
        // String sequences — run until BEL or ESC \ (ST).
        Some(c) if ST_STRING_INTRODUCERS.contains(&c) => {
            chars.next();
            while let Some(c) = chars.next() {
                if c == BEL {
                    break;
                }
                if c == ESC && chars.peek() == Some(&ST_ESCAPE) {
                    chars.next();
                    break;
                }
            }
        }
        // Charset designators and other two-byte escapes may carry
        // intermediates, e.g. `ESC ( B`.
        Some(c) if ('\u{20}'..='\u{2f}').contains(&c) => {
            chars.next();
            for c in chars.by_ref() {
                if ('\u{30}'..='\u{7e}').contains(&c) {
                    break;
                }
            }
        }
        // Anything else after ESC is a complete two-byte sequence.
        Some(_) => {
            chars.next();
        }
        None => {}
    }
}

/// Convenience wrapper for callers that already own a `String`.
pub fn strip_ansi_owned(s: String) -> String {
    strip_ansi(&s)
}

// ── ANSI -> styled lines ─────────────────────────────────────────────────────
//
// The REPL's blocks are rendered by Rich (see `src/bridge/render.py`) and arrive
// here as SGR-coloured text. Only SGR is interpreted; every other escape is
// dropped, because this frontend owns the cursor and the screen.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Apply one SGR parameter list (the `a;b;c` of `ESC[a;b;cm`) to `st`.
fn apply_sgr(mut st: Style, p: &[i64]) -> Style {
    if p.is_empty() {
        return Style::default();
    }
    let mut i = 0usize;
    while i < p.len() {
        match p[i] {
            0 => st = Style::default(),
            1 => st = st.add_modifier(Modifier::BOLD),
            2 => st = st.add_modifier(Modifier::DIM),
            3 => st = st.add_modifier(Modifier::ITALIC),
            4 | 21 => st = st.add_modifier(Modifier::UNDERLINED),
            7 => st = st.add_modifier(Modifier::REVERSED),
            9 => st = st.add_modifier(Modifier::CROSSED_OUT),
            22 => st = st.remove_modifier(Modifier::BOLD | Modifier::DIM),
            23 => st = st.remove_modifier(Modifier::ITALIC),
            24 => st = st.remove_modifier(Modifier::UNDERLINED),
            27 => st = st.remove_modifier(Modifier::REVERSED),
            29 => st = st.remove_modifier(Modifier::CROSSED_OUT),
            n @ 30..=37 => st = st.fg(Color::Indexed((n - 30) as u8)),
            39 => st.fg = None,
            n @ 40..=47 => st = st.bg(Color::Indexed((n - 40) as u8)),
            49 => st.bg = None,
            n @ 90..=97 => st = st.fg(Color::Indexed((n - 90 + 8) as u8)),
            n @ 100..=107 => st = st.bg(Color::Indexed((n - 100 + 8) as u8)),
            n @ (38 | 48) => {
                let color = match p.get(i + 1) {
                    Some(5) => {
                        let c = p.get(i + 2).map(|v| Color::Indexed((*v).clamp(0, 255) as u8));
                        i += 2;
                        c
                    }
                    Some(2) => {
                        let rgb = (p.get(i + 2), p.get(i + 3), p.get(i + 4));
                        i += 4;
                        match rgb {
                            (Some(r), Some(g), Some(b)) => Some(Color::Rgb(
                                (*r).clamp(0, 255) as u8,
                                (*g).clamp(0, 255) as u8,
                                (*b).clamp(0, 255) as u8,
                            )),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(c) = color {
                    st = if n == 38 { st.fg(c) } else { st.bg(c) };
                }
            }
            _ => {}
        }
        i += 1;
    }
    st
}

/// Parse SGR-coloured text into styled lines, one per `\n`.
///
/// A trailing newline does not produce an extra empty line, so the output of a
/// Rich `print` maps one-to-one onto rows. Tabs become four spaces and any other
/// control character is dropped: a stray byte must never move the cursor.
pub fn parse_ansi(text: &str) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut style = Style::default();
    let mut chars = text.chars().peekable();

    fn flush(buf: &mut String, spans: &mut Vec<Span<'static>>, style: Style) {
        if !buf.is_empty() {
            spans.push(Span::styled(std::mem::take(buf), style));
        }
    }

    while let Some(c) = chars.next() {
        match c {
            ESC => {
                if chars.peek() == Some(&CSI) {
                    chars.next();
                    let mut params: Vec<i64> = Vec::new();
                    let mut cur = String::new();
                    let mut fin = None;
                    for c in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&c) {
                            fin = Some(c);
                            break;
                        }
                        if c == ';' || c == ':' {
                            params.push(cur.parse().unwrap_or(0));
                            cur.clear();
                        } else {
                            cur.push(c);
                        }
                    }
                    if fin == Some('m') {
                        params.push(cur.parse().unwrap_or(0));
                        flush(&mut buf, &mut spans, style);
                        style = apply_sgr(style, &params);
                    }
                } else {
                    let mut sink = String::new();
                    strip_one_sequence(&mut chars, &mut sink);
                }
            }
            '\n' => {
                flush(&mut buf, &mut spans, style);
                lines.push(Line::from(std::mem::take(&mut spans)));
            }
            '\t' => buf.push_str("    "),
            c if c.is_control() => {}
            c => buf.push(c),
        }
    }
    flush(&mut buf, &mut spans, style);
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    lines
}

/// Rich colour name -> ANSI index (the 16 named colours).
fn rich_named_color(name: &str) -> Option<Color> {
    let (bright, base) = match name.strip_prefix("bright_") {
        Some(rest) => (true, rest),
        None => (false, name),
    };
    let idx = match base {
        "black" => 0u8,
        "red" => 1,
        "green" => 2,
        "yellow" => 3,
        "blue" => 4,
        "magenta" => 5,
        "cyan" => 6,
        "white" => 7,
        _ => return None,
    };
    Some(Color::Indexed(if bright { idx + 8 } else { idx }))
}

fn rich_color(tok: &str) -> Option<Color> {
    if let Some(hex) = tok.strip_prefix('#') {
        if hex.len() == 6 {
            let v = u32::from_str_radix(hex, 16).ok()?;
            return Some(Color::Rgb((v >> 16) as u8, (v >> 8) as u8, v as u8));
        }
        return None;
    }
    rich_named_color(tok)
}

/// Parse a Rich style string such as `"bold #58a6ff"` or `"dim #8b949e"`
/// (the form `theme.yaml` uses) into a ratatui style. Unknown tokens are
/// ignored; `"default"` and the empty string give the terminal default.
pub fn style_from_rich(spec: &str) -> Style {
    let mut st = Style::default();
    let mut on_next = false;
    for tok in spec.split_whitespace() {
        if on_next {
            if let Some(c) = rich_color(tok) {
                st = st.bg(c);
            }
            on_next = false;
            continue;
        }
        match tok {
            "on" => on_next = true,
            "bold" | "b" => st = st.add_modifier(Modifier::BOLD),
            "dim" | "d" => st = st.add_modifier(Modifier::DIM),
            "italic" | "i" => st = st.add_modifier(Modifier::ITALIC),
            "underline" | "u" => st = st.add_modifier(Modifier::UNDERLINED),
            "reverse" | "r" => st = st.add_modifier(Modifier::REVERSED),
            "strike" | "s" => st = st.add_modifier(Modifier::CROSSED_OUT),
            "default" => {}
            other => {
                if let Some(c) = rich_color(other) {
                    st = st.fg(c);
                }
            }
        }
    }
    st
}

/// Cut `line` to at most `max` columns, keeping each span's style. A wide glyph
/// that would straddle the limit is dropped rather than split.
pub fn clip_line(line: Line<'static>, max: usize) -> Line<'static> {
    if line_width(&line) <= max {
        return line;
    }
    let mut used = 0usize;
    let mut out: Vec<Span<'static>> = Vec::new();
    for span in line.spans {
        let mut kept = String::new();
        for ch in span.content.chars() {
            let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if used + w > max {
                if !kept.is_empty() {
                    out.push(Span::styled(kept, span.style));
                }
                return Line::from(out);
            }
            kept.push(ch);
            used += w;
        }
        out.push(Span::styled(kept, span.style));
    }
    Line::from(out)
}

/// Display width of a styled line in terminal columns.
pub fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgr_colour_codes_are_removed() {
        assert_eq!(strip_ansi("\u{1b}[35;58;33mhello\u{1b}[0m"), "hello");
        assert_eq!(strip_ansi("\u{1b}[1;31mred\u{1b}[0m plain"), "red plain");
    }

    #[test]
    fn screen_and_cursor_commands_are_removed() {
        assert_eq!(strip_ansi("a\u{1b}[2Jb\u{1b}[Hc"), "abc");
        assert_eq!(strip_ansi("\u{1b}[?25la\u{1b}[?25h"), "a");
    }

    #[test]
    fn osc_hyperlinks_keep_only_the_label() {
        // ESC ] 8 ; ; https://example.com ESC \ label ESC ] 8 ; ; ESC \
        let linked = "\u{1b}]8;;https://example.com\u{1b}\\label\u{1b}]8;;\u{1b}\\";
        assert_eq!(strip_ansi(linked), "label");
    }

    #[test]
    fn osc_terminated_by_bel_is_removed() {
        assert_eq!(strip_ansi("\u{1b}]0;window title\u{7}body"), "body");
    }

    #[test]
    fn charset_designators_are_removed() {
        assert_eq!(strip_ansi("\u{1b}(Btext"), "text");
        // ESC = (DECKPAM) is a complete two-byte escape and takes no
        // parameters, so what follows it is ordinary text.
        assert_eq!(strip_ansi("\u{1b}=1m x"), "1m x");
    }

    #[test]
    fn carriage_return_redraw_keeps_only_the_final_pass() {
        assert_eq!(strip_ansi("10%\r50%\r100%"), "100%");
        assert_eq!(strip_ansi("first\n10%\r20%\nlast"), "first\n20%\nlast");
    }

    #[test]
    fn crlf_is_a_line_ending_not_a_redraw() {
        // Windows-style output must survive intact.
        assert_eq!(strip_ansi("one\r\ntwo\r\n"), "one\ntwo\n");
        assert_eq!(strip_ansi("done\r\n"), "done\n");
        // A redraw immediately before a newline still keeps the final pass.
        assert_eq!(strip_ansi("10%\r100% done\r\nnext"), "100% done\nnext");
    }

    #[test]
    fn plain_text_is_untouched() {
        for s in ["", "hello", "a\nb", "Таблица: 5;58;33", "0.0% (0/0)"] {
            assert_eq!(strip_ansi(s), s);
        }
    }

    #[test]
    fn a_lone_escape_at_the_end_does_not_panic() {
        assert_eq!(strip_ansi("abc\u{1b}"), "abc");
        assert_eq!(strip_ansi("\u{1b}"), "");
    }

    #[test]
    fn no_escape_byte_ever_survives() {
        let dirty = "\u{1b}[1mbold\u{1b}[0m \u{1b}]8;;u\u{1b}\\x\u{1b}]8;;\u{1b}\\ \u{1b}(B\u{1b}[35;58;33m";
        assert!(!strip_ansi(dirty).contains(ESC), "{:?}", strip_ansi(dirty));
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect()
    }

    #[test]
    fn plain_text_maps_one_line_per_newline() {
        let l = parse_ansi("a\nb\n");
        assert_eq!(text(&l), vec!["a", "b"]);
    }

    #[test]
    fn a_blank_row_is_preserved() {
        assert_eq!(text(&parse_ansi("a\n\nb\n")), vec!["a", "", "b"]);
    }

    #[test]
    fn missing_final_newline_still_yields_the_last_line() {
        assert_eq!(text(&parse_ansi("a\nb")), vec!["a", "b"]);
        assert!(parse_ansi("").is_empty());
    }

    #[test]
    fn truecolor_foreground_and_background() {
        let l = parse_ansi("\u{1b}[38;2;1;2;3;48;2;4;5;6mX\u{1b}[0mY\n");
        let s = &l[0].spans;
        assert_eq!(s[0].content, "X");
        assert_eq!(s[0].style.fg, Some(Color::Rgb(1, 2, 3)));
        assert_eq!(s[0].style.bg, Some(Color::Rgb(4, 5, 6)));
        assert_eq!(s[1].content, "Y");
        assert_eq!(s[1].style, Style::default());
    }

    #[test]
    fn modifiers_and_their_resets() {
        let l = parse_ansi("\u{1b}[1;2;3;4;7;9mA\u{1b}[22;23;24;27;29mB\n");
        let a = l[0].spans[0].style.add_modifier;
        for m in [
            Modifier::BOLD,
            Modifier::DIM,
            Modifier::ITALIC,
            Modifier::UNDERLINED,
            Modifier::REVERSED,
            Modifier::CROSSED_OUT,
        ] {
            assert!(a.contains(m), "missing {m:?}");
        }
        assert!(l[0].spans[1].style.add_modifier.is_empty());
    }

    #[test]
    fn basic_and_bright_colours_use_the_ansi_palette() {
        let l = parse_ansi("\u{1b}[36mc\u{1b}[91mr\u{1b}[44mb\u{1b}[39;49mx\n");
        assert_eq!(l[0].spans[0].style.fg, Some(Color::Indexed(6)));
        assert_eq!(l[0].spans[1].style.fg, Some(Color::Indexed(9)));
        assert_eq!(l[0].spans[2].style.bg, Some(Color::Indexed(4)));
        assert_eq!(l[0].spans[3].style.fg, None);
        assert_eq!(l[0].spans[3].style.bg, None);
    }

    #[test]
    fn indexed_256_colour() {
        let l = parse_ansi("\u{1b}[38;5;75mX\n");
        assert_eq!(l[0].spans[0].style.fg, Some(Color::Indexed(75)));
    }

    #[test]
    fn empty_sgr_is_a_reset() {
        let l = parse_ansi("\u{1b}[1mA\u{1b}[mB\n");
        assert!(l[0].spans[1].style.add_modifier.is_empty());
    }

    #[test]
    fn style_survives_across_text_until_reset() {
        let l = parse_ansi("\u{1b}[1;31mone two\u{1b}[0m\n");
        assert_eq!(l[0].spans.len(), 1);
        assert_eq!(l[0].spans[0].content, "one two");
    }

    #[test]
    fn non_sgr_escapes_are_dropped_not_interpreted() {
        let l = parse_ansi("a\u{1b}[2Jb\u{1b}[?25lc\u{1b}[1;1Hd\n");
        assert_eq!(text(&l), vec!["abcd"]);
    }

    #[test]
    fn hyperlinks_keep_their_label_and_style() {
        let st = "\u{1b}\\";
        let linked = format!(
            "\u{1b}]8;id=1;https://e.com{st}\u{1b}[4mlink\u{1b}[0m\u{1b}]8;;{st}!\n"
        );
        let l = parse_ansi(&linked);
        assert_eq!(text(&l), vec!["link!"]);
        assert!(l[0].spans[0].style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn control_characters_never_reach_the_screen() {
        let l = parse_ansi("a\u{7}b\u{8}c\r\td\n");
        assert_eq!(text(&l), vec!["abc    d"]);
    }

    #[test]
    fn truncated_sequences_do_not_panic() {
        for s in ["\u{1b}", "\u{1b}[", "\u{1b}[38;2;1", "\u{1b}]8;;", "x\u{1b}[3"] {
            let _ = parse_ansi(s);
        }
    }

    #[test]
    fn rich_style_strings() {
        let s = style_from_rich("bold #58a6ff");
        assert_eq!(s.fg, Some(Color::Rgb(0x58, 0xa6, 0xff)));
        assert!(s.add_modifier.contains(Modifier::BOLD));
        let d = style_from_rich("dim #8b949e");
        assert!(d.add_modifier.contains(Modifier::DIM));
        assert_eq!(style_from_rich("default"), Style::default());
        assert_eq!(style_from_rich("bright_white").fg, Some(Color::Indexed(15)));
        assert_eq!(style_from_rich("cyan on #181a1f").bg, Some(Color::Rgb(24, 26, 31)));
        assert_eq!(style_from_rich("nonsense").fg, None);
    }

    #[test]
    fn line_width_counts_columns_not_chars() {
        let l = parse_ansi("ab\u{4e2d}\n"); // a wide CJK glyph is two columns
        assert_eq!(line_width(&l[0]), 4);
    }
}
