//! Selecting text in the transcript with the mouse.
//!
//! A selection used to remember the *screen cells* it began and ended on. The
//! text under those cells changes whenever the chat scrolls, so a drag that
//! carried on while the wheel turned, or a selection that was scrolled away from,
//! copied other text than the one highlighted. Both ends are now positions in
//! the conversation itself — the message, the row inside it, the column — which
//! scrolling cannot move.
//!
//! The same positions drive the highlight (each visible row asks which of its
//! columns are selected), the copy (rows are rebuilt from the messages, so text
//! that has scrolled out of view is still copied), double click (a word), triple
//! click (a row), Ctrl+A (everything) and dragging past the edge (the chat scrolls
//! on its own and the selection keeps growing).

use std::time::{Duration, Instant};

use unicode_width::UnicodeWidthChar;

use crate::app::{slice_columns, App};
use crate::view::{item_view, live_rows, row_text, Cols};

/// A place in the conversation that does not move when the view scrolls.
///
/// `item` is the message index; the rows of whatever is still being drawn below
/// the last message (spinner, streaming tail) count as one more item with index
/// `messages.len()`. `row` counts rows inside the item and `col` display columns
/// inside the row. Positions compare in reading order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TextPos {
    pub item: usize,
    pub row: usize,
    pub col: usize,
}

/// A selection over the transcript: where the press began and where the pointer is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MouseSelection {
    pub anchor: TextPos,
    pub head: TextPos,
    /// A word, a row or everything was selected (double / triple click, Ctrl+A):
    /// it is kept even if both ends are the same cell, which a plain click is not.
    pub multi: bool,
}

impl MouseSelection {
    /// The two ends in reading order.
    pub fn ordered(&self) -> (TextPos, TextPos) {
        if self.anchor <= self.head {
            (self.anchor, self.head)
        } else {
            (self.head, self.anchor)
        }
    }
}

/// Two presses closer together than this, on the same row, are a double click.
const MULTI_CLICK: Duration = Duration::from_millis(450);

/// How often a drag held past the edge of the chat scrolls it by one row.
const EDGE_SCROLL: Duration = Duration::from_millis(45);

/// The columns of a transcript row that `sel` covers, as `start..end` (end
/// exclusive), or `None` when the row is outside it. `text_width` is the width of
/// the row's text without trailing blanks, so a highlight never paints padding.
pub fn selected_columns(
    sel: &MouseSelection,
    item: usize,
    row: usize,
    text_width: usize,
) -> Option<(usize, usize)> {
    let (a, b) = sel.ordered();
    let here = (item, row);
    if here < (a.item, a.row) || here > (b.item, b.row) {
        return None;
    }
    let start = if here == (a.item, a.row) { a.col } else { 0 };
    let end = if here == (b.item, b.row) { b.col.saturating_add(1) } else { usize::MAX };
    let end = end.min(text_width);
    (start < end).then_some((start, end))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Word,
    Space,
    Other,
}

fn class_of(c: char) -> Class {
    if c.is_alphanumeric() || c == '_' {
        Class::Word
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// Display columns (first, last — both included) of the run of like characters
/// under `col`: a word, a run of blanks or a run of punctuation. `None` past the
/// end of the text.
pub fn word_bounds(text: &str, col: usize) -> Option<(usize, usize)> {
    let mut cells: Vec<(usize, usize, Class)> = Vec::new();
    let mut at = 0usize;
    for ch in text.chars() {
        let w = ch.width().unwrap_or(0);
        if w == 0 {
            continue; // a combining mark belongs to the character before it
        }
        cells.push((at, at + w - 1, class_of(ch)));
        at += w;
    }
    let i = cells.iter().position(|&(s, e, _)| col >= s && col <= e)?;
    let kind = cells[i].2;
    let mut lo = i;
    while lo > 0 && cells[lo - 1].2 == kind {
        lo -= 1;
    }
    let mut hi = i;
    while hi + 1 < cells.len() && cells[hi + 1].2 == kind {
        hi += 1;
    }
    Some((cells[lo].0, cells[hi].1))
}

impl App {
    /// True when screen row `y` shows transcript text.
    pub fn in_transcript(&self, y: u16) -> bool {
        let vt = self.view_text.borrow();
        y >= vt.top && ((y - vt.top) as usize) < vt.rows.len()
    }

    /// The transcript position under screen cell (`x`, `y`) as of the last frame.
    ///
    /// A pointer above the first visible row maps to the start of that row and one
    /// below the last to its end, so a drag that leaves the chat keeps extending
    /// the selection to the edge instead of stopping.
    pub fn pos_at(&self, x: u16, y: u16) -> Option<TextPos> {
        let vt = self.view_text.borrow();
        if vt.anchors.is_empty() {
            return None;
        }
        let last = vt.anchors.len() - 1;
        let (rel, col) = if y < vt.top {
            (0, 0)
        } else if (y - vt.top) as usize > last {
            (last, usize::MAX)
        } else {
            ((y - vt.top) as usize, x as usize)
        };
        let (item, row) = vt.anchors[rel];
        Some(TextPos { item, row, col })
    }

    /// How many presses in a row this one is (1, 2 or 3, then it starts over).
    pub fn click_count(&mut self, x: u16, y: u16, now: Instant) -> u8 {
        let count = match self.last_click {
            Some((at, lx, ly, n)) if now.duration_since(at) <= MULTI_CLICK && ly == y && lx.abs_diff(x) <= 1 => {
                if n >= 3 {
                    1
                } else {
                    n + 1
                }
            }
            _ => 1,
        };
        self.last_click = Some((now, x, y, count));
        count
    }

    /// A press on the transcript: start a selection there, or select the word
    /// (`clicks == 2`) or the row (`clicks >= 3`) under it.
    pub fn begin_selection(&mut self, x: u16, y: u16, clicks: u8) -> bool {
        let Some(pos) = self.pos_at(x, y) else {
            self.mouse_sel = None;
            return false;
        };
        let (anchor, head, multi) = match clicks {
            0 | 1 => (pos, pos, false),
            2 => {
                let text = self.view_row_text(y);
                match word_bounds(&text, pos.col) {
                    Some((from, to)) => {
                        (TextPos { col: from, ..pos }, TextPos { col: to, ..pos }, true)
                    }
                    None => (pos, pos, false),
                }
            }
            _ => (TextPos { col: 0, ..pos }, TextPos { col: usize::MAX, ..pos }, true),
        };
        self.mouse_sel = Some(MouseSelection { anchor, head, multi });
        self.selecting = true;
        self.last_drag = Some((x, y));
        self.press_row = Some(y);
        self.drag_edge = None;
        true
    }

    /// The pointer moved with the button down.
    pub fn drag_selection(&mut self, x: u16, y: u16) {
        if !self.selecting {
            return;
        }
        self.last_drag = Some((x, y));
        self.drag_edge = {
            let vt = self.view_text.borrow();
            // The chat starts at the very top of the window, so the pointer cannot
            // go above it: sitting on the first row counts as "past the top", but
            // only for a drag that began lower, or a press there would scroll.
            let began_lower = self.press_row.is_some_and(|press_y| press_y > vt.top);
            if y < vt.top || (y == vt.top && began_lower) {
                Some(-1)
            } else if y >= vt.top.saturating_add(vt.view_h) {
                Some(1)
            } else {
                None
            }
        };
        if let Some(pos) = self.pos_at(x, y) {
            if let Some(sel) = self.mouse_sel.as_mut() {
                sel.head = pos;
            }
        }
    }

    /// The button went up. Returns the selected text (to be copied), or `None`
    /// after a plain click, which selects nothing.
    pub fn finish_selection(&mut self) -> Option<String> {
        self.selecting = false;
        self.drag_edge = None;
        self.last_drag = None;
        let sel = self.mouse_sel?;
        if !sel.multi && sel.anchor == sel.head {
            self.mouse_sel = None;
            return None;
        }
        let text = self.selected_transcript_text();
        if text.is_none() {
            self.mouse_sel = None;
        }
        text
    }

    /// One step of a drag that is still going: keep the selection's end under the
    /// pointer after the view moved, and scroll on when the pointer is held past
    /// the top or bottom of the chat. True when something changed (redraw).
    pub fn selection_tick(&mut self, now: Instant) -> bool {
        if !self.selecting {
            return false;
        }
        let mut changed = false;
        if let Some((x, y)) = self.last_drag {
            if let Some(pos) = self.pos_at(x, y) {
                if let Some(sel) = self.mouse_sel.as_mut() {
                    if sel.head != pos {
                        sel.head = pos;
                        changed = true;
                    }
                }
            }
        }
        let before = self.scroll_offset;
        let due = self.last_edge_scroll.is_none_or(|t| now.duration_since(t) >= EDGE_SCROLL);
        if due {
            match self.drag_edge {
                Some(-1) => self.scroll_up(1),
                Some(1) => self.scroll_down(1),
                _ => {}
            }
            if self.drag_edge.is_some() {
                self.last_edge_scroll = Some(now);
            }
        }
        changed || self.scroll_offset != before
    }

    /// Select the screen cells from `from` to `to` (as they were last drawn).
    pub fn select_cells(&mut self, from: (u16, u16), to: (u16, u16)) -> bool {
        match (self.pos_at(from.0, from.1), self.pos_at(to.0, to.1)) {
            (Some(anchor), Some(head)) => {
                self.mouse_sel = Some(MouseSelection { anchor, head, multi: false });
                true
            }
            _ => false,
        }
    }

    /// Ctrl+A with nothing typed: select the whole conversation.
    pub fn select_all_transcript(&mut self) -> bool {
        if self.messages.is_empty() {
            return false;
        }
        self.mouse_sel = Some(MouseSelection {
            anchor: TextPos { item: 0, row: 0, col: 0 },
            head: TextPos { item: self.messages.len(), row: usize::MAX, col: usize::MAX },
            multi: true,
        });
        self.selecting = false;
        true
    }

    /// Forget a selection whose messages are gone (a cleared conversation).
    pub fn drop_stale_selection(&mut self) {
        if let Some(sel) = self.mouse_sel {
            let (a, b) = sel.ordered();
            if a.item > self.messages.len() || b.item > self.messages.len() {
                self.mouse_sel = None;
                self.selecting = false;
            }
        }
    }

    fn view_row_text(&self, y: u16) -> String {
        let vt = self.view_text.borrow();
        y.checked_sub(vt.top)
            .and_then(|rel| vt.rows.get(rel as usize))
            .cloned()
            .unwrap_or_default()
    }

    /// The rows of one item as plain text, laid out at the width on screen, each with
    /// the columns that are text.
    fn item_texts(&self, item: usize, width: u16, height: usize) -> Vec<(String, Cols)> {
        if item < self.messages.len() {
            let view = item_view(self, item, width);
            let keep = view.len.saturating_sub(view.chrome_tail);
            let cols = view.cols();
            view.content_lines()
                .iter()
                .zip(cols)
                .take(keep)
                .map(|(l, c)| (row_text(l), c))
                .collect()
        } else {
            live_rows(self, width, height).iter().map(|l| (row_text(l), Cols::ALL)).collect()
        }
    }

    /// The text under the selection, one line per row with trailing blanks trimmed
    /// (the REPL pads rows to the full width).
    ///
    /// Rows are rebuilt from the messages rather than read off the screen, so a
    /// selection that reaches past the visible part still copies all of it.
    pub fn selected_transcript_text(&self) -> Option<String> {
        let sel = self.mouse_sel?;
        let (a, b) = sel.ordered();
        let (width, height) = {
            let vt = self.view_text.borrow();
            (vt.width, vt.view_h as usize)
        };
        if width == 0 {
            return None;
        }
        let mut lines: Vec<String> = Vec::new();
        for item in a.item..=b.item.min(self.messages.len()) {
            let texts = self.item_texts(item, width, height.max(1));
            let Some(last) = texts.len().checked_sub(1) else { continue };
            let first_row = if item == a.item { a.row } else { 0 };
            let last_row = if item == b.item { b.row.min(last) } else { last };
            for (r, (text, cols)) in texts.iter().enumerate().take(last_row + 1).skip(first_row) {
                // A rail, a margin, the padding of a bubble: decoration, not what was written.
                if cols.is_none() {
                    continue;
                }
                let from = if item == a.item && r == a.row { a.col.max(cols.from) } else { cols.from };
                let to = if item == b.item && r == b.row { b.col.saturating_add(1) } else { usize::MAX };
                lines.push(slice_columns(text, from, to.min(cols.to)).trim_end().to_string());
            }
        }
        let text = lines.join("\n");
        let text = text.trim_matches('\n');
        if text.trim().is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(a: (usize, usize, usize), b: (usize, usize, usize)) -> MouseSelection {
        MouseSelection {
            anchor: TextPos { item: a.0, row: a.1, col: a.2 },
            head: TextPos { item: b.0, row: b.1, col: b.2 },
            multi: false,
        }
    }

    #[test]
    fn positions_compare_in_reading_order() {
        let a = TextPos { item: 1, row: 5, col: 0 };
        let b = TextPos { item: 2, row: 0, col: 0 };
        assert!(a < b);
        assert!(TextPos { item: 1, row: 2, col: 9 } < TextPos { item: 1, row: 3, col: 0 });
    }

    #[test]
    fn a_selection_is_the_same_whichever_way_it_was_dragged() {
        let forward = sel((0, 0, 3), (2, 1, 4));
        let backward = sel((2, 1, 4), (0, 0, 3));
        assert_eq!(forward.ordered(), backward.ordered());
    }

    #[test]
    fn columns_cover_the_start_row_from_the_press_and_the_end_row_to_the_pointer() {
        let s = sel((1, 0, 4), (1, 2, 6));
        assert_eq!(selected_columns(&s, 1, 0, 20), Some((4, 20)), "first row: from the press to the end");
        assert_eq!(selected_columns(&s, 1, 1, 20), Some((0, 20)), "a row in the middle: all of it");
        assert_eq!(selected_columns(&s, 1, 2, 20), Some((0, 7)), "last row: up to the pointer, included");
        assert_eq!(selected_columns(&s, 0, 9, 20), None, "before the selection");
        assert_eq!(selected_columns(&s, 1, 3, 20), None, "after the selection");
    }

    #[test]
    fn a_highlight_never_paints_padding_past_the_text() {
        let s = sel((0, 0, 0), (0, 0, 80));
        assert_eq!(selected_columns(&s, 0, 0, 12), Some((0, 12)));
        assert_eq!(selected_columns(&s, 0, 0, 0), None, "an empty row has nothing to paint");
    }

    #[test]
    fn a_pointer_far_to_the_right_of_a_row_selects_to_its_end() {
        let s = sel((0, 0, 2), (0, 1, usize::MAX));
        assert_eq!(selected_columns(&s, 0, 1, 9), Some((0, 9)));
    }

    #[test]
    fn a_word_is_a_run_of_like_characters() {
        let t = "let foo_bar = baz(1);";
        assert_eq!(word_bounds(t, 5), Some((4, 10)), "foo_bar");
        assert_eq!(word_bounds(t, 3), Some((3, 3)), "the blank between words");
        assert_eq!(word_bounds(t, 12), Some((12, 12)), "punctuation on its own");
        assert_eq!(word_bounds(t, 100), None, "past the end");
    }

    #[test]
    fn words_are_found_by_display_column_not_by_char_index() {
        // Each of these takes two columns.
        let t = "\u{4e2d}\u{6587} ab";
        assert_eq!(word_bounds(t, 1), Some((0, 3)), "either column of a wide char hits the word");
        assert_eq!(word_bounds(t, 5), Some((5, 6)));
    }

    #[test]
    fn cyrillic_words_select_whole() {
        assert_eq!(word_bounds("привет мир", 2), Some((0, 5)));
    }
}
