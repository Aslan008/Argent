//! Append-only segmentation of a streaming markdown answer.
//!
//! A direct port of `src/cli/stream_markdown.py`. The REPL commits COMPLETE
//! markdown blocks once (they are never redrawn) and keeps only the incomplete
//! trailing block as a live tail. The TUI does the same so a block is rendered
//! exactly once per width instead of re-parsing the whole growing answer on
//! every frame, and so a chunk looks identical in both frontends.
//!
//! `tests/test_stream_golden.rs` replays vectors produced by the real Python
//! splitter, so any drift between the two shows up as a failing test.

use std::ops::Range;

/// Split `text` into lines the way Python's `str.splitlines(keepends=True)` does
/// (the separators that matter for model output: `\n`, `\r\n`, `\r`, VT, FF,
/// FS/GS/RS, NEL, LS, PS).
fn split_lines_keepends(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut iter = text.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        let is_break = matches!(
            c,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}'
                | '\u{2028}' | '\u{2029}'
        );
        if !is_break {
            continue;
        }
        let mut end = i + c.len_utf8();
        if c == '\r' {
            if let Some(&(j, '\n')) = iter.peek() {
                end = j + 1;
                iter.next();
            }
        }
        out.push(&text[start..end]);
        start = end;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// `^[ \t]*(`{3,}|~{3,})` — returns the fence marker's first char when the line
/// opens or closes a fenced code block.
fn fence_char(line: &str) -> Option<char> {
    let t = line.trim_start_matches([' ', '\t']);
    let c = t.chars().next()?;
    if c != '`' && c != '~' {
        return None;
    }
    let run = t.chars().take_while(|&x| x == c).count();
    if run >= 3 {
        Some(c)
    } else {
        None
    }
}

/// Offset (bytes) up to which `text` is a run of COMPLETE markdown blocks.
///
/// A safe boundary is a blank line outside a code fence, or the end of a closed
/// fence. Everything after it is an incomplete trailing block that must stay
/// pending — committing it could render a broken fragment (e.g. the inside of an
/// unterminated code fence).
pub fn last_safe_boundary(text: &str) -> usize {
    let mut in_fence = false;
    let mut open_char = '`';
    let mut offset = 0usize;
    let mut last_safe = 0usize;
    for line in split_lines_keepends(text) {
        if let Some(ch) = fence_char(line) {
            if !in_fence {
                in_fence = true;
                open_char = ch;
                offset += line.len();
                continue;
            }
            if ch == open_char {
                in_fence = false;
                offset += line.len();
                last_safe = offset;
                continue;
            }
        }
        offset += line.len();
        if !in_fence && line.trim().is_empty() {
            last_safe = offset;
        }
    }
    last_safe
}

/// Which prefix of a growing buffer has been committed, as a list of chunks.
///
/// The buffer itself lives with the caller (the assistant message's `content`);
/// this only remembers the boundaries, so nothing is stored twice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamState {
    /// Bytes of the buffer already committed.
    pub committed: usize,
    /// Committed chunks, in order, as byte ranges of the buffer.
    pub chunks: Vec<Range<usize>>,
}

impl StreamState {
    /// Commit whatever has become safe since the last call. Returns whether a
    /// new chunk was added.
    pub fn advance(&mut self, buffer: &str) -> bool {
        if self.committed > buffer.len() || !buffer.is_char_boundary(self.committed) {
            // The buffer shrank or was rewritten under us: start over rather
            // than slice into the middle of a character.
            *self = Self::default();
        }
        let region = &buffer[self.committed..];
        let boundary = last_safe_boundary(region);
        if boundary == 0 {
            return false;
        }
        self.chunks.push(self.committed..self.committed + boundary);
        self.committed += boundary;
        true
    }

    /// The stream declared the whole content to be `buffer` (e.g. a raw
    /// tool-call JSON was cut out of it). The REPL cannot un-print what it
    /// already committed and keeps it when it is still a prefix; here nothing is
    /// printed for good, so the segmentation is simply recomputed.
    pub fn replace(&mut self, buffer: &str) {
        *self = Self::default();
        self.advance(buffer);
    }

    /// The uncommitted tail (the incomplete trailing block).
    pub fn pending<'a>(&self, buffer: &'a str) -> &'a str {
        buffer.get(self.committed..).unwrap_or("")
    }

    /// Commit everything that is left. Returns whether a chunk was added.
    pub fn finalize(&mut self, buffer: &str) -> bool {
        if self.committed > buffer.len() || !buffer.is_char_boundary(self.committed) {
            *self = Self::default();
        }
        if self.committed >= buffer.len() {
            return false;
        }
        self.chunks.push(self.committed..buffer.len());
        self.committed = buffer.len();
        true
    }

    /// The text of chunk `i`.
    pub fn chunk<'a>(&self, buffer: &'a str, i: usize) -> &'a str {
        self.chunks
            .get(i)
            .and_then(|r| buffer.get(r.clone()))
            .unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(deltas: &[&str]) -> (String, StreamState, Vec<String>) {
        let mut buf = String::new();
        let mut st = StreamState::default();
        let mut committed = Vec::new();
        for d in deltas {
            buf.push_str(d);
            if st.advance(&buf) {
                committed.push(st.chunk(&buf, st.chunks.len() - 1).to_string());
            }
        }
        (buf, st, committed)
    }

    #[test]
    fn nothing_commits_until_a_blank_line() {
        let (_, st, c) = feed_all(&["one line", " still one"]);
        assert!(c.is_empty());
        assert_eq!(st.committed, 0);
    }

    #[test]
    fn a_blank_line_commits_the_block_before_it() {
        let (buf, st, c) = feed_all(&["Hello\n\nWor"]);
        assert_eq!(c, vec!["Hello\n\n"]);
        assert_eq!(st.pending(&buf), "Wor");
    }

    #[test]
    fn a_blank_line_inside_a_fence_is_not_a_boundary() {
        let (buf, st, c) = feed_all(&["```\na\n\nb\n"]);
        assert!(c.is_empty());
        assert_eq!(st.pending(&buf), "```\na\n\nb\n");
    }

    #[test]
    fn closing_the_fence_commits_it() {
        let (_, _, c) = feed_all(&["```\na\n", "```\n"]);
        assert_eq!(c, vec!["```\na\n```\n"]);
    }

    #[test]
    fn finalize_flushes_the_tail_once() {
        let (buf, mut st, _) = feed_all(&["tail without blank"]);
        assert!(st.finalize(&buf));
        assert_eq!(st.chunk(&buf, 0), "tail without blank");
        assert!(!st.finalize(&buf), "a second finalize adds nothing");
    }

    #[test]
    fn replace_recomputes_the_segmentation() {
        let mut st = StreamState::default();
        let mut buf = String::from("a\n\nb");
        st.advance(&buf);
        buf = "x\n\ny\n\nz".to_string();
        st.replace(&buf);
        assert_eq!(st.chunks.len(), 1);
        assert_eq!(st.pending(&buf), "z");
        assert_eq!(st.chunk(&buf, 0), "x\n\ny\n\n");
    }

    #[test]
    fn a_shrunken_buffer_resets_instead_of_panicking() {
        let mut st = StreamState::default();
        st.advance("aaaa\n\nbbbb");
        assert!(st.committed > 0);
        // The buffer is now shorter than what was committed.
        st.advance("a");
        assert_eq!(st.committed, 0);
    }

    #[test]
    fn multibyte_text_never_splits_a_character() {
        let (buf, st, c) = feed_all(&["Привет\n\nмир"]);
        assert_eq!(c, vec!["Привет\n\n"]);
        assert_eq!(st.pending(&buf), "мир");
    }

    #[test]
    fn lines_split_like_python_splitlines() {
        assert_eq!(split_lines_keepends("a\r\nb\rc\nd"), vec!["a\r\n", "b\r", "c\n", "d"]);
        assert_eq!(split_lines_keepends(""), Vec::<&str>::new());
        assert_eq!(split_lines_keepends("\n\n"), vec!["\n", "\n"]);
    }
}
