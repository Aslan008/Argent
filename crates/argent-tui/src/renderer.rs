//! How a visual block becomes terminal rows.
//!
//! In production every block is rendered by the Python side with the REPL's own
//! code (Rich + Pygments), so the TUI looks exactly like the REPL by
//! construction — see `src/bridge/render.py`. This module defines the seam:
//!
//! * [`BlockRenderer`] — `render(kind, data, width) -> rows`.
//! * [`FallbackRenderer`] — a small pure-Rust approximation. It is used when no
//!   bridge is attached (unit tests, a headless run) and when a Python render
//!   fails, so a render error degrades the look of one block instead of losing it.
//!
//! The block kinds and their `data` keys are the ones `render.py::BLOCKS` knows.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;
use std::collections::HashMap;

use crate::ansi::strip_ansi;
use crate::markdown;
use crate::toolrender;

/// Renders one block to styled rows, already wrapped to `width` columns.
pub trait BlockRenderer: Send {
    fn render(&self, kind: &str, data: &Value, width: u16) -> Result<Vec<Line<'static>>, String>;
}

/// Remove Rich markup tags (`[dim]`, `[/bold]`, `[#fff]`) from `s`, keeping
/// escaped brackets (`\[`) as literal text.
pub fn strip_markup(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && chars.get(i + 1) == Some(&'[') {
            out.push('[');
            i += 2;
            continue;
        }
        if c == '[' {
            if let Some(close) = (i + 1..chars.len()).find(|&j| chars[j] == ']') {
                let inner: String = chars[i + 1..close].iter().collect();
                let tagish = !inner.is_empty()
                    && inner
                        .chars()
                        .all(|x| x.is_ascii_alphanumeric() || "_#/ .-".contains(x))
                    && (inner.starts_with('/')
                        || inner.starts_with('#')
                        || inner.chars().next().is_some_and(|x| x.is_ascii_lowercase()));
                if tagish {
                    i = close + 1;
                    continue;
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn s_str<'a>(d: &'a Value, key: &str) -> &'a str {
    d.get(key).and_then(Value::as_str).unwrap_or("")
}

fn s_f64(d: &Value, key: &str) -> Option<f64> {
    d.get(key).and_then(Value::as_f64)
}

fn dim() -> Style {
    Style::default().fg(Color::Rgb(0x8b, 0x94, 0x9e)).add_modifier(Modifier::DIM)
}

fn args_map(d: &Value) -> HashMap<String, Value> {
    d.get("args")
        .and_then(Value::as_object)
        .map(|o| o.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

fn blank() -> Line<'static> {
    Line::default()
}

/// A rounded box around `body`, like Rich's default `Panel` (`expand=False`).
fn panel(title: Vec<Span<'static>>, body: Vec<Line<'static>>, border: Style, max_w: usize) -> Vec<Line<'static>> {
    let title_w: usize = title.iter().map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref())).sum();
    let body_w = body.iter().map(crate::ansi::line_width).max().unwrap_or(0);
    let inner = body_w.max(title_w + 2).min(max_w.saturating_sub(4)).max(1);
    let mut out = Vec::with_capacity(body.len() + 2);

    let mut top = vec![Span::styled("╭─ ", border)];
    top.extend(title);
    let used = 3 + title_w + 1;
    top.push(Span::styled(
        format!(" {}╮", "─".repeat((inner + 4).saturating_sub(used + 1))),
        border,
    ));
    out.push(Line::from(top));

    for l in body {
        let mut l = crate::ansi::clip_line(l, inner);
        let w = crate::ansi::line_width(&l);
        let mut spans = vec![Span::styled("│ ", border)];
        spans.append(&mut l.spans);
        if w < inner {
            spans.push(Span::raw(" ".repeat(inner - w)));
        }
        spans.push(Span::styled(" │", border));
        out.push(Line::from(spans));
    }
    out.push(Line::from(Span::styled(format!("╰{}╯", "─".repeat(inner + 2)), border)));
    out
}

/// The pure-Rust approximation of the Python renderer.
#[derive(Debug, Default, Clone, Copy)]
pub struct FallbackRenderer;

impl BlockRenderer for FallbackRenderer {
    fn render(&self, kind: &str, data: &Value, width: u16) -> Result<Vec<Line<'static>>, String> {
        let w = width.max(20) as usize;
        let accent = Style::default().fg(Color::Rgb(0x58, 0xa6, 0xff));
        let rows = match kind {
            "user_markup" => vec![Line::from(vec![
                Span::styled("❯ ", accent),
                Span::styled(strip_markup(s_str(data, "text")), Style::default().fg(Color::White)),
            ])],
            "assistant_header" => vec![Line::from(Span::styled(
                "✦ Argent",
                accent.add_modifier(Modifier::BOLD),
            ))],
            "markdown" => {
                let text = s_str(data, "text").trim_matches('\n');
                if text.trim().is_empty() {
                    return Ok(Vec::new());
                }
                let mut v = markdown::render(text, w);
                v.push(blank());
                v
            }
            "reasoning" | "rethink" => reasoning_rows(kind, data, w),
            "tool_end" => tool_end_rows(data, w),
            "tool_header" => {
                let header = toolrender::tool_header(s_str(data, "name"), &args_map(data));
                let name = s_str(data, "name");
                let mut spans = Vec::new();
                match header.strip_prefix(&format!("{name}: ")) {
                    Some(target) => {
                        spans.push(Span::styled(
                            name.to_string(),
                            Style::default().fg(Color::Rgb(0x79, 0xc0, 0xff)).add_modifier(Modifier::BOLD),
                        ));
                        spans.push(Span::styled(":", dim()));
                        spans.push(Span::styled(
                            format!(" {target}"),
                            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                        ));
                    }
                    None => spans.push(Span::styled(
                        header,
                        Style::default().fg(Color::Rgb(0x79, 0xc0, 0xff)).add_modifier(Modifier::BOLD),
                    )),
                }
                vec![Line::from(spans)]
            }
            "tool_summary" => vec![Line::from(Span::raw(toolrender::tool_result_summary(
                s_str(data, "name"),
                &strip_ansi(s_str(data, "result")),
            )))],
            "diff" => diff_rows(data, w),
            "error" => {
                let text = strip_ansi(s_str(data, "text"));
                let lower = text.trim_start().to_lowercase();
                let body = if lower.starts_with("error") || lower.starts_with("ошибка") {
                    text
                } else {
                    format!("Ошибка: {text}")
                };
                let mut v: Vec<Line<'static>> = body
                    .lines()
                    .map(|l| {
                        Line::from(Span::styled(
                            l.to_string(),
                            Style::default().fg(Color::Rgb(0xf8, 0x51, 0x49)).add_modifier(Modifier::BOLD),
                        ))
                    })
                    .collect();
                let hint = s_str(data, "hint");
                if !hint.is_empty() {
                    v.push(Line::from(Span::styled(hint.to_string(), Style::default().add_modifier(Modifier::DIM))));
                }
                v
            }
            "system" => strip_markup(&strip_ansi(s_str(data, "text")))
                .split('\n')
                .map(|l| Line::from(Span::styled(l.to_string(), dim())))
                .collect(),
            "checkpoint" => vec![Line::from(Span::styled(
                format!("◇ Чекпоинт {} создан — откат: /rewind", s_str(data, "sha")),
                Style::default().add_modifier(Modifier::DIM),
            ))],
            "banner" => {
                let title = format!(" Argent Coder v{} ", s_str(data, "version"));
                let tw = unicode_width::UnicodeWidthStr::width(title.as_str());
                let left = w.saturating_sub(tw) / 2;
                let right = w.saturating_sub(tw + left);
                vec![Line::from(vec![
                    Span::styled("─".repeat(left), Style::default()),
                    Span::styled(title, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::styled("─".repeat(right), Style::default()),
                ])]
            }
            "approval_header" => {
                let destructive = data.get("destructive").and_then(Value::as_bool).unwrap_or(false);
                let (icon, color) = if destructive {
                    ("▲ ", Color::Indexed(1))
                } else {
                    ("◇ ", Color::Indexed(3))
                };
                vec![
                    blank(),
                    Line::from(Span::styled(
                        format!("{icon}Агент запрашивает разрешение: {}", strip_ansi(s_str(data, "action"))),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    )),
                ]
            }
            "context_usage" => {
                let t = s_f64(data, "tokens").unwrap_or(0.0) as i64;
                let m = s_f64(data, "max").unwrap_or(0.0) as i64;
                let p = s_f64(data, "percent").unwrap_or(0.0);
                vec![Line::from(Span::styled(
                    format!("Context: {p:.1}% ({t}/{m} tokens)"),
                    Style::default().add_modifier(Modifier::DIM),
                ))]
            }
            "markup" => {
                let mut v: Vec<Line<'static>> = data
                    .get("lines")
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .map(|l| {
                                Line::from(Span::styled(
                                    strip_markup(l),
                                    Style::default().add_modifier(Modifier::DIM),
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if data.get("blank_after").and_then(Value::as_bool).unwrap_or(false) {
                    v.push(blank());
                }
                v
            }
            other => return Err(format!("unknown block kind: {other}")),
        };
        Ok(rows)
    }
}

/// The one-line gist of a reasoning text (its last sentence), as the REPL shows it
/// next to the folded `◇ Анализ`.
pub fn thought_preview(text: &str) -> String {
    preview(text)
}

fn preview(text: &str) -> String {
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let Some(tail) = lines.last() else { return String::new() };
    let mut t = if tail.chars().count() < 25 && lines.len() > 1 {
        format!("{} {}", lines[lines.len() - 2], tail)
    } else {
        (*tail).to_string()
    };
    t = t
        .trim_start_matches(|c: char| "#*>-. ".contains(c) || c.is_ascii_digit())
        .trim()
        .to_string();
    if t.chars().count() > 120 {
        t = format!("{}…", t.chars().take(119).collect::<String>());
    }
    t
}

fn reasoning_rows(kind: &str, data: &Value, w: usize) -> Vec<Line<'static>> {
    let text = strip_ansi(s_str(data, "text"));
    if text.trim().is_empty() {
        return Vec::new();
    }
    let (glyph, label, color) = if kind == "rethink" {
        ("◈", "Перепроверка", Color::Rgb(0xd1, 0x8f, 0x52))
    } else {
        ("◇", "Анализ", Color::Rgb(0x7d, 0x85, 0x90))
    };
    let dur = match s_f64(data, "duration") {
        Some(d) if d >= 0.1 => format!(" · {d:.1}s"),
        _ => String::new(),
    };
    if s_str(data, "mode") != "full" {
        let mut spans = vec![Span::styled(
            format!("  {glyph} {label}{dur}"),
            Style::default().fg(color).add_modifier(Modifier::DIM),
        )];
        let p = preview(&text);
        if !p.is_empty() {
            spans.push(Span::styled(
                format!(" ({p})"),
                Style::default().add_modifier(Modifier::DIM | Modifier::ITALIC),
            ));
        }
        return vec![Line::from(spans)];
    }
    let mut out = vec![Line::from(Span::styled(
        format!("{glyph} {label}"),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ))];
    for (n, l) in markdown::render(&text, w.saturating_sub(4)).into_iter().enumerate() {
        let mut spans = vec![Span::raw(if n == 0 { "└── " } else { "    " })];
        spans.extend(l.spans);
        out.push(Line::from(spans));
    }
    out.push(blank());
    out
}

fn tool_end_rows(data: &Value, w: usize) -> Vec<Line<'static>> {
    let name = s_str(data, "name");
    let result = strip_ansi(s_str(data, "result"));
    let args = args_map(data);
    let failed = toolrender::result_is_error(&result);
    let expanded = data.get("expanded").and_then(Value::as_bool).unwrap_or(false);
    let dur = s_f64(data, "duration").map(toolrender::format_duration);
    let bar = Style::default().fg(Color::Rgb(0x38, 0x8b, 0xfd)).add_modifier(Modifier::DIM);

    if !failed && !expanded {
        let header = toolrender::tool_header(name, &args);
        let mut head = vec![Span::styled("  │ ", bar)];
        match header.strip_prefix(&format!("{name}: ")) {
            Some(target) => {
                head.push(Span::styled(
                    name.to_string(),
                    Style::default().fg(Color::Rgb(0x79, 0xc0, 0xff)).add_modifier(Modifier::BOLD),
                ));
                head.push(Span::styled(":", dim()));
                head.push(Span::styled(
                    format!(" {target}"),
                    Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
                ));
            }
            None => head.push(Span::styled(
                header,
                Style::default().fg(Color::Rgb(0x79, 0xc0, 0xff)).add_modifier(Modifier::BOLD),
            )),
        }
        let mut tail = vec![
            Span::styled("  │ ", bar),
            Span::styled("↳ ", dim()),
            Span::styled(
                toolrender::tool_result_summary(name, &result),
                Style::default().fg(Color::Rgb(0xc9, 0xd1, 0xd9)),
            ),
        ];
        if let Some(d) = dur {
            tail.push(Span::raw(format!(" · {d}")));
        }
        return vec![Line::from(head), Line::from(tail)];
    }

    let all: Vec<&str> = result.lines().collect();
    let cap = 15usize;
    let mut body: Vec<Line<'static>> = all
        .iter()
        .take(cap)
        .map(|l| Line::from(Span::styled((*l).to_string(), Style::default().fg(Color::Rgb(0xc9, 0xd1, 0xd9)))))
        .collect();
    if all.len() > cap {
        body.push(Line::from(format!("... [{} lines hidden in UI]", all.len() - cap)));
    }
    let (icon, label, border) = if failed {
        ("✗", "failed", Style::default().fg(Color::Rgb(0xf8, 0x51, 0x49)).add_modifier(Modifier::BOLD))
    } else {
        ("✓", "finished", Style::default().fg(Color::Rgb(0x30, 0x36, 0x3d)).add_modifier(Modifier::DIM))
    };
    let time = dur.map(|d| format!(" ({d})")).unwrap_or_default();
    let title = vec![
        Span::styled(format!("{icon} "), Style::default().fg(if failed { Color::Rgb(0xf8, 0x51, 0x49) } else { Color::Rgb(0x3f, 0xb9, 0x50) })),
        Span::styled(name.to_string(), Style::default().fg(Color::Rgb(0x79, 0xc0, 0xff)).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {label}{time}"), Style::default().add_modifier(Modifier::DIM)),
    ];
    panel(title, body, border, w.min(100))
}

fn diff_rows(data: &Value, w: usize) -> Vec<Line<'static>> {
    let diff = strip_ansi(s_str(data, "diff"));
    if diff.trim().is_empty() {
        return Vec::new();
    }
    let base = std::path::Path::new(s_str(data, "file"))
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "file".to_string());
    let raw: Vec<&str> = diff.trim().lines().collect();
    let added = raw.iter().filter(|l| l.starts_with('+') && !l.starts_with("+++")).count();
    let removed = raw.iter().filter(|l| l.starts_with('-') && !l.starts_with("---")).count();
    let max = 25usize;
    let mut body: Vec<Line<'static>> = raw
        .iter()
        .take(max)
        .map(|l| {
            let col = if l.starts_with('+') {
                Color::Rgb(0x98, 0xc3, 0x79)
            } else if l.starts_with('-') {
                Color::Rgb(0xe0, 0x6c, 0x75)
            } else {
                Color::Rgb(0xab, 0xb2, 0xbf)
            };
            Line::from(Span::styled(
                format!(" {l}"),
                Style::default().fg(col).bg(markdown::CODE_BG),
            ))
        })
        .collect();
    if raw.len() > max {
        body.push(Line::from(format!(
            "... [{} more lines hidden — use /diff {base} for full view]",
            raw.len() - max
        )));
    }
    let border = Style::default().fg(Color::Rgb(0x38, 0x8b, 0xfd)).add_modifier(Modifier::DIM);
    let title = vec![
        Span::styled(format!("diff {base}"), Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD | Modifier::DIM)),
        Span::styled(" (", border),
        Span::styled(format!("+{added}"), Style::default().fg(Color::Green).add_modifier(Modifier::DIM)),
        Span::styled(", ", border),
        Span::styled(format!("-{removed}"), Style::default().fg(Color::Red).add_modifier(Modifier::DIM)),
        Span::styled(")", border),
    ];
    let mut out = vec![blank()];
    out.extend(panel(title, body, border, w.min(104)));
    out.push(blank());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text(lines: &[Line<'_>]) -> String {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn r(kind: &str, data: Value) -> String {
        text(&FallbackRenderer.render(kind, &data, 80).expect("render"))
    }

    #[test]
    fn markup_is_stripped_but_escaped_brackets_survive() {
        assert_eq!(strip_markup("[dim]a[/dim] [bold #fff]b[/]"), "a b");
        assert_eq!(strip_markup("x \\[not a tag] y"), "x [not a tag] y");
        assert_eq!(strip_markup("[System: note]"), "[System: note]");
    }

    #[test]
    fn header_and_banner() {
        assert_eq!(r("assistant_header", json!({})), "✦ Argent");
        assert!(r("banner", json!({"version": "1.2"})).contains("Argent Coder v1.2"));
    }

    #[test]
    fn compact_tool_block_is_two_rows() {
        let t = r("tool_end", json!({"name":"read_file","result":"a\nb","duration":0.028,
            "args":{"file_path":"agent.py"}}));
        let rows: Vec<&str> = t.lines().collect();
        assert_eq!(rows.len(), 2, "{t}");
        assert!(rows[0].contains("│ read_file: agent.py"));
        assert!(rows[1].contains("↳ 2 строк · a") && rows[1].contains("28ms"));
    }

    #[test]
    fn failed_tool_is_boxed() {
        let t = r("tool_end", json!({"name":"read_file","result":"Error: nope","duration":0.004,
            "args":{}}));
        assert!(t.contains("╭") && t.contains("✗") && t.contains("failed") && t.contains("Error: nope"));
    }

    #[test]
    fn reasoning_summary_and_full() {
        let s = r("reasoning", json!({"text":"first\nthe conclusion here","duration":3.2,"mode":"summary"}));
        assert!(s.contains("◇ Анализ · 3.2s") && s.contains("the conclusion here"));
        let f = r("reasoning", json!({"text":"first\nsecond","mode":"full"}));
        assert!(f.contains("└──") && f.contains("first") && f.contains("second"));
    }

    #[test]
    fn markdown_chunk_ends_with_a_blank_row() {
        let rows = FallbackRenderer.render("markdown", &json!({"text":"hello"}), 40).unwrap();
        assert_eq!(text(&rows[rows.len() - 1..]), "");
        assert!(text(&rows).contains("hello"));
    }

    #[test]
    fn empty_markdown_renders_nothing() {
        assert!(FallbackRenderer.render("markdown", &json!({"text":"\n\n"}), 40).unwrap().is_empty());
    }

    #[test]
    fn error_has_one_prefix() {
        assert!(r("error", json!({"text":"boom"})).starts_with("Ошибка: boom"));
        assert!(!r("error", json!({"text":"Error: boom"})).contains("Ошибка:"));
    }

    #[test]
    fn unknown_kind_is_an_error() {
        assert!(FallbackRenderer.render("nope", &json!({}), 40).is_err());
    }

    #[test]
    fn diff_counts_changes() {
        let t = r("diff", json!({"file":"src/f.py","diff":"--- a\n+++ b\n-old\n+new"}));
        assert!(t.contains("diff f.py (+1, -1)") && t.contains("-old") && t.contains("+new"));
    }

    #[test]
    fn no_row_exceeds_the_width() {
        let wide = "x".repeat(300);
        for (kind, data) in [
            ("tool_end", json!({"name":"run_command","result": wide, "expanded": true, "args":{}})),
            ("diff", json!({"file":"f","diff": format!("+{wide}")})),
        ] {
            let rows = FallbackRenderer.render(kind, &data, 60).unwrap();
            for l in &rows {
                assert!(crate::ansi::line_width(l) <= 60 + 4, "{kind}: {}", crate::ansi::line_width(l));
            }
        }
    }
}
