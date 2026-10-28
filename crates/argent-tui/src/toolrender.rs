//! Human-readable tool call rendering, ported from the legacy CLI
//! (`ui.py::_format_tool_header` and `ui.py::_format_tool_summary`).
//!
//! Raw argument dicts are unreadable in a terminal; the CLI turns them into one
//! salient target (a path, a query, a command, a URL) and collapses results to
//! a single informative line. The same is done here so the two frontends read
//! identically.

use crate::ansi::strip_ansi;
use serde_json::Value;

fn truncate_chars(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        return s.to_string();
    }
    let head: String = s.chars().take(limit.saturating_sub(1)).collect();
    format!("{head}…")
}

fn arg_str(args: &std::collections::HashMap<String, Value>, key: &str) -> Option<String> {
    args.get(key).and_then(|v| match v {
        Value::String(s) => Some(strip_ansi(s)),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    })
}

/// An argument as an integer, whether it arrived as a JSON number or as text.
fn int_arg(args: &std::collections::HashMap<String, Value>, key: &str) -> Option<i64> {
    match args.get(key)? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// An argument rendered the way Python's `f"{value}"` would, `None` when it is
/// missing or falsy (empty, zero, null).
fn truthy_text(args: &std::collections::HashMap<String, Value>, key: &str) -> Option<String> {
    match args.get(key)? {
        Value::Null => None,
        Value::String(s) if s.is_empty() => None,
        Value::Number(n) if n.as_f64() == Some(0.0) => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// Trailing `:10-50` / `:5+`, exactly as the REPL's `_format_tool_header`
/// builds it: `offset`/`limit` read as "start, count" and print `start-(start+count)`.
fn line_suffix(args: &std::collections::HashMap<String, Value>) -> String {
    if args.contains_key("offset") || args.contains_key("limit") {
        let off = int_arg(args, "offset").filter(|v| *v != 0).unwrap_or(1);
        return match int_arg(args, "limit") {
            Some(lim) => format!(":{off}-{}", off + lim),
            None => format!(":{off}+"),
        };
    }
    if args.contains_key("start_line") || args.contains_key("end_line") {
        let s = args
            .get("start_line")
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_else(|| "1".to_string());
        return match truthy_text(args, "end_line") {
            Some(e) => format!(":{s}-{e}"),
            None => format!(":{s}+"),
        };
    }
    String::new()
}

/// One-line description of what a tool is about to do.
pub fn tool_header(
    name: &str,
    args: &std::collections::HashMap<String, Value>,
) -> String {
    // 1. Search / query tools.
    for key in ["query", "pattern", "Query", "search_term"] {
        if let Some(q) = arg_str(args, key) {
            let mut s = format!("{name}: {}", truncate_chars(&q, 40));
            if let Some(p) = arg_str(args, "path") {
                if let Some(base) = std::path::Path::new(&p).file_name() {
                    s.push_str(&format!(" in {}", base.to_string_lossy()));
                }
            }
            return s;
        }
    }

    // 2. URL tools.
    for key in ["url", "link", "target_url", "uri", "Url"] {
        if let Some(u) = arg_str(args, key) {
            return format!("{name}: {}", truncate_chars(&u, 250));
        }
    }

    // 3. File tools.
    for key in ["file_path", "path", "target_file", "file"] {
        if let Some(p) = arg_str(args, key) {
            return format!("{name}: {p}{}", line_suffix(args));
        }
    }

    // 4. Shell commands.
    if let Some(cmd) = arg_str(args, "command") {
        let flat = cmd.replace('\n', " ");
        return format!("{name}: {}", truncate_chars(&flat, 55));
    }

    // 5. Browser tools.
    if name.starts_with("browser_") {
        for key in ["url", "selector", "text", "expression", "script"] {
            if let Some(v) = arg_str(args, key) {
                let limit = if key == "url" { 250 } else { 45 };
                return format!("{name}: {}", truncate_chars(&v, limit));
            }
        }
    }

    // 6. MCP passthrough.
    if let Some(t) = arg_str(args, "tool_name") {
        return format!("{name}: {t}");
    }

    // 7. Fallback: first two scalar-ish args.
    let scalars: Vec<String> = args
        .iter()
        .filter(|(_, v)| !matches!(v, Value::Object(_) | Value::Array(_)))
        .map(|(k, v)| format!("{k}={}", strip_ansi(&v.to_string())))
        .take(2)
        .collect();
    if !scalars.is_empty() {
        return format!("{name}: {}", truncate_chars(&scalars.join(", "), 45));
    }

    name.to_string()
}

/// Collapse a tool result to a single line, mirroring the CLI's summariser.
pub fn tool_result_summary(name: &str, result: &str) -> String {
    // Captured command output routinely carries SGR codes. Painting them would
    // show `[35;58;33M` as literal text, so they are removed up front.
    let owned;
    let trimmed = if result.contains('\u{1b}') || result.contains('\r') {
        owned = strip_ansi(result);
        owned.trim()
    } else {
        result.trim()
    };
    if trimmed.is_empty() {
        return "пусто".to_string();
    }
    // `Exit code: 1` / `OUTPUT:` is the envelope `run_command` wraps the result in:
    // it says nothing about what happened, so it is never the line the summary quotes.
    let lines: Vec<&str> = trimmed.lines().filter(|l| !is_meta_line(l)).collect();
    if lines.is_empty() {
        return match exit_code_of(trimmed) {
            Some(code) if code != 0 => format!("exit {code}"),
            _ => "пусто".to_string(),
        };
    }

    // Commands: prefer a test summary line, then a change-counter line.
    if name == "run_command" || name == "start_background_command" {
        for l in &lines {
            let low = l.to_lowercase();
            if (low.contains("passed") || low.contains("failed"))
                && (l.contains('=') || l.contains(','))
            {
                return truncate_chars(l.trim_start_matches(['=', ' ', '\t']), 80);
            }
        }
        for l in &lines {
            let low = l.to_lowercase();
            if low.contains("file changed")
                || low.contains("files changed")
                || low.contains("insertions(+)")
            {
                return truncate_chars(l, 80);
            }
        }
    }

    // File writers.
    if matches!(
        name,
        "write_file" | "replace_in_file" | "multi_replace_in_file"
    ) {
        for l in &lines {
            if l.contains("Successfully wrote") {
                return truncate_chars(l.trim(), 80);
            }
        }
        if lines.len() == 1 && lines[0].trim().len() <= 80 {
            return lines[0].trim().to_string();
        }
        return "успешно применено".to_string();
    }

    // Browser actions report their value through a `JS result:` marker.
    if name.starts_with("browser_") {
        if let Some(idx) = trimmed.find("JS result:") {
            let v = trimmed[idx + "JS result:".len()..].trim();
            if v.is_empty() || v == "null" || v == "undefined" || v == "__ARGENT_UNDEFINED__" {
                return "Действие выполнено".to_string();
            }
            return truncate_chars(v, 60);
        }
    }

    // A command that failed says so with its exit code, and that is what the
    // compact row should report — the first line of output is often just noise.
    if let Some(code) = exit_code_of(trimmed) {
        if code != 0 {
            let first = lines.first().map(|l| l.trim()).unwrap_or("");
            return if first.is_empty() {
                format!("exit {code}")
            } else {
                format!("exit {code} · {}", truncate_chars(first, 40))
            };
        }
    }

    if lines.len() == 1 {
        return truncate_chars(lines[0].trim(), 60);
    }

    let first = lines[0].trim();
    format!(
        "{} строк · {}",
        lines.len(),
        truncate_chars(first, 50)
    )
}

/// Duration in the CLI's format: `450ms` under a second, otherwise `1.2s`.
pub fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{}ms", (secs * 1000.0).round() as i64)
    } else {
        format!("{secs:.1}s")
    }
}

/// The `Exit code: N` line `run_command` reports first in a result, when present.
pub fn exit_code_of(text: &str) -> Option<i32> {
    text.lines().take(3).find_map(|line| {
        let line = line.trim();
        let rest = line
            .strip_prefix("Exit code:")
            .or_else(|| line.strip_prefix("exit code:"))?;
        rest.trim().parse::<i32>().ok()
    })
}

/// `Exit code: 1` / `OUTPUT:` — the envelope `run_command` adds, not content.
fn is_meta_line(line: &str) -> bool {
    let t = line.trim();
    t == "OUTPUT:" || exit_code_of(t).is_some()
}

/// True when a tool result signals failure (the CLI's prefix check, plus the exit
/// code a failed shell command reports).
pub fn result_is_error(result: &str) -> bool {
    let cleaned;
    let text = if result.contains('\u{1b}') || result.contains('\r') {
        cleaned = strip_ansi(result);
        cleaned.as_str()
    } else {
        result
    };
    if text.trim_start().to_lowercase().starts_with("error") {
        return true;
    }
    // The prefix alone missed every failed shell command: `run_command` opens the
    // result with `Exit code: 1`, so a broken build was drawn as finished.
    matches!(exit_code_of(text), Some(code) if code != 0)
}

/// `12 строк` / `3 файла` for the compact result gutter.
pub fn count_files_hint(result: &str) -> Option<String> {
    let n = result.lines().filter(|l| l.trim_start().starts_with("--- ")).count();
    if n > 0 {
        Some(format!("{n} файлов"))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn m(pairs: Vec<(&str, Value)>) -> std::collections::HashMap<String, Value> {
        pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn header_prefers_query() {
        let a = m(vec![("query", json!("needle")), ("path", json!("src/app.rs"))]);
        assert_eq!(tool_header("grep_search", &a), "grep_search: needle in app.rs");
    }

    #[test]
    fn header_truncates_long_query() {
        // The CLI caps the *query* at 40 columns; the "grep_search: " prefix
        // is not part of that budget.
        let a = m(vec![("query", json!("x".repeat(80)))]);
        let h = tool_header("grep_search", &a);
        let q = h.strip_prefix("grep_search: ").expect("prefix");
        assert!(q.chars().count() <= 40, "query not truncated: {q}");
        assert!(q.ends_with('…'), "{q}");
    }

    #[test]
    fn header_includes_line_range() {
        // The REPL prints `offset-(offset+limit)`; pinned so the two never differ.
        let a = m(vec![("file_path", json!("a.py")), ("offset", json!(10)), ("limit", json!(5))]);
        assert_eq!(tool_header("read_file", &a), "read_file: a.py:10-15");
    }

    #[test]
    fn header_open_ended_range() {
        let a = m(vec![("file_path", json!("a.py")), ("offset", json!(3))]);
        assert!(tool_header("read_file", &a).ends_with(":3+"), "{:?}", tool_header("read_file", &a));
    }

    #[test]
    fn header_range_survives_text_numbers() {
        // Bridges that stringify arguments must still get the line range.
        let a = m(vec![("file_path", json!("a.py")), ("offset", json!("10")), ("limit", json!("40"))]);
        assert_eq!(tool_header("read_file", &a), "read_file: a.py:10-50");
    }

    #[test]
    fn header_start_end_line_range() {
        let a = m(vec![("path", json!("a.py")), ("start_line", json!(3)), ("end_line", json!(9))]);
        assert_eq!(tool_header("read_file", &a), "read_file: a.py:3-9");
        let b = m(vec![("path", json!("a.py")), ("start_line", json!(3))]);
        assert_eq!(tool_header("read_file", &b), "read_file: a.py:3+");
    }

    #[test]
    fn limit_alone_counts_from_line_one() {
        let a = m(vec![("file_path", json!("a.py")), ("limit", json!(20))]);
        assert_eq!(tool_header("read_file", &a), "read_file: a.py:1-21");
    }

    #[test]
    fn header_flattens_multiline_command() {
        let a = m(vec![("command", json!("line1\nline2"))]);
        assert_eq!(tool_header("run_command", &a), "run_command: line1 line2");
    }

    #[test]
    fn header_url_is_kept_long_for_clickability() {
        let a = m(vec![("url", json!("https://example.com/".to_string() + &"p".repeat(300)))]);
        assert!(tool_header("browser_open", &a).chars().count() > 200);
    }

    #[test]
    fn header_falls_back_to_name() {
        assert_eq!(tool_header("noop", &m(vec![])), "noop");
    }

    #[test]
    fn summary_prefers_test_line() {
        let s = tool_result_summary("run_command", "noise\nTests: 3 passed, 0 failed = 5s\nmore");
        assert_eq!(s, "Tests: 3 passed, 0 failed = 5s");
    }

    #[test]
    fn summary_counts_lines_when_multiline() {
        let s = tool_result_summary("grep_search", "first line\nsecond\nthird");
        assert!(s.starts_with("3 строк · "), "{s}");
    }

    #[test]
    fn summary_single_line_truncated() {
        let s = tool_result_summary("read_file", &"y".repeat(100));
        assert!(s.chars().count() <= 60, "{s}");
    }

    #[test]
    fn summary_empty() {
        assert_eq!(tool_result_summary("read_file", "   \n "), "пусто");
    }

    #[test]
    fn browser_null_result_is_friendly() {
        assert_eq!(
            tool_result_summary("browser_click", "JS result: null"),
            "Действие выполнено"
        );
    }

    #[test]
    fn duration_format() {
        assert_eq!(format_duration(0.45), "450ms");
        assert_eq!(format_duration(1.24), "1.2s");
    }

    #[test]
    fn error_detection_is_prefix_based() {
        assert!(result_is_error("  ERROR: boom"));
        assert!(!result_is_error("no error here"));
    }

    #[test]
    fn a_non_zero_exit_code_means_failure() {
        // `run_command` opens a failed result with the code; the prefix check alone
        // drew a green tick over every broken build.
        let failed = "Exit code: 1\nOUTPUT:\nbox/mod.rs:3: error: boom\n";
        assert!(result_is_error(failed));
        assert!(!result_is_error("Exit code: 0\nOUTPUT:\nall good\n"));
        assert_eq!(exit_code_of(failed), Some(1));
        assert_eq!(exit_code_of("no code here"), None);
    }

    #[test]
    fn the_exit_code_line_is_never_the_summary() {
        let failed = "Exit code: 1\nOUTPUT:\nbox/mod.rs:3: error: boom\nsecond\n";
        let s = tool_result_summary("run_command", failed);
        assert!(s.starts_with("exit 1 · "), "{s}");
        assert!(!s.contains("Exit code"), "{s}");
    }

    #[test]
    fn a_result_of_nothing_but_an_exit_code_still_summarises() {
        assert_eq!(tool_result_summary("run_command", "Exit code: 2"), "exit 2");
        assert_eq!(tool_result_summary("run_command", "Exit code: 0\nOUTPUT:\n"), "пусто");
    }
}