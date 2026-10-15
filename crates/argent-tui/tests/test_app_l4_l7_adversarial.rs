//! LMTrust L4 (Adversarial) x L7 (Compositional) for the `App` edit engine.
//!
//! Layer decision tree: `App` is a stateful object whose edits are driven by
//! keystrokes, so the high-ROI layers are L4 (what if the buffer invariants are
//! lies?) and L7 (what if two things go wrong at once — here, multibyte text
//! combined with the kill ring).
//!
//! Every "does not panic" assertion below is a real assertion: a panic in this
//! crate happens inside the event loop, so it takes the whole TUI down rather
//! than failing a keystroke.

use argent_tui::app::App;
use proptest::prelude::*;

/// Every editing entry point that is reachable from the key handler in
/// `lib.rs`, collapsed into one enum so a property test can hammer all of them.
#[derive(Debug, Clone)]
enum Op {
    Insert(char),
    InsertStr(String),
    Backspace,
    Delete,
    DeleteWordLeft,
    DeleteWordRight,
    DeleteToStart,
    DeleteToEnd,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    Home,
    End,
    TransposeChars,
    TransposeWords,
    Yank,
    YankPop,
    Undo,
    ExtendSelection(i32),
    SetCursor(usize),
}

fn seed(s: &str, cursor: usize) -> App {
    let mut app = App::new();
    app.input = s.to_string();
    app.cursor_pos = cursor.min(s.chars().count());
    app
}

fn apply(app: &mut App, op: &Op) {
    match op {
        Op::Insert(c) => app.insert_char(*c),
        Op::InsertStr(s) => app.insert_str(s),
        Op::Backspace => app.backspace(),
        Op::Delete => app.delete_char(),
        Op::DeleteWordLeft => app.delete_word_left(),
        Op::DeleteWordRight => app.delete_word_right(),
        Op::DeleteToStart => app.delete_to_start(),
        Op::DeleteToEnd => app.delete_to_end(),
        Op::MoveLeft => app.move_cursor_left(),
        Op::MoveRight => app.move_cursor_right(),
        Op::MoveUp => {
            app.move_cursor_up();
        }
        Op::MoveDown => {
            app.move_cursor_down();
        }
        Op::Home => app.move_cursor_home(),
        Op::End => app.move_cursor_end(),
        Op::TransposeChars => app.transpose_chars(),
        Op::TransposeWords => app.transpose_words(),
        Op::Yank => {
            app.yank();
        }
        Op::YankPop => {
            app.yank_pop();
        }
        Op::Undo => {
            app.undo();
        }
        Op::ExtendSelection(n) => app.extend_selection(*n as isize),
        Op::SetCursor(p) => app.set_cursor(*p, false),
    }
}

// ── L4: the buffer invariants, violated one at a time ───────────────────────

/// L4 x D2. The core invariant every editor method relies on:
/// `0 <= cursor_pos <= input.chars().count()`.
///
/// `transpose_chars` clamps its working index with `min(n)` and then indexes
/// `chars[c]`, which is one past the end when the cursor sits at the end of the
/// buffer. `n < 2` is the only guard, and a two-character buffer passes it.
#[test]
fn l4_transpose_chars_at_end_of_buffer_does_not_panic() {
    let mut app = seed("ab", 2);
    assert_eq!(app.cursor_pos, 2, "cursor sits at the end of the buffer");
    app.transpose_chars();
}

/// L4 x D2. Same defect, one character longer, to show it is not a 2-char
/// special case.
#[test]
fn l4_transpose_chars_at_end_of_longer_buffer_does_not_panic() {
    let mut app = seed("hello", 5);
    app.transpose_chars();
}

/// L4 x D2. Multi-byte input at the end of the buffer.
#[test]
fn l4_transpose_chars_at_end_of_multibyte_buffer_does_not_panic() {
    let mut app = seed("\u{65e5}\u{672c}\u{8a9e}", 3);
    app.transpose_chars();
}

/// L4 x D2. `delete_word_left` indexes `chars[idx - 1]` with an unclamped
/// `cursor_pos`. An out-of-range cursor is a panic, not a no-op.
#[test]
fn l4_delete_word_left_tolerates_an_out_of_range_cursor() {
    let mut app = seed("abc", 99);
    app.delete_word_left();
}

/// L4 x D2. Same for `delete_word_right`, which indexes `chars[idx]`.
#[test]
fn l4_delete_word_right_tolerates_an_out_of_range_cursor() {
    let mut app = seed("abc", 99);
    app.delete_word_right();
}

/// L4 x D2. `move_word_left` walks `chars[idx - 1]` with no clamp either.
#[test]
fn l4_move_word_left_tolerates_an_out_of_range_cursor() {
    let mut app = seed("abc def", 99);
    app.move_word_left();
}

/// L4 x D2. `yank_pop` assigns a byte offset to `cursor_pos`. `rfind` returns a
/// byte index; `cursor_pos` is a char index. On multibyte input the value is
/// out of range, and the next word-wise delete panics.
#[test]
fn l4_yank_pop_keeps_the_cursor_inside_the_buffer() {
    let mut app = seed("\u{65e5}\u{672c}\u{8a9e}abc", 6);
    app.kill_ring = vec!["zz".into(), "abc".into()];
    app.yanked = Some("abc".into());

    app.yank_pop();

    assert!(
        app.cursor_pos <= app.input.chars().count(),
        "cursor_pos {} escaped a {}-char buffer",
        app.cursor_pos,
        app.input.chars().count()
    );
}

/// L4 x D3. `yank_pop` slices the buffer with a byte offset read as a char
/// offset, so text outside the yanked region is destroyed or duplicated.
#[test]
fn l4_yank_pop_does_not_corrupt_multibyte_text() {
    let mut app = seed("\u{e9}abc", 2);
    app.kill_ring = vec!["zz".into(), "abc".into()];
    app.yanked = Some("abc".into());

    app.yank_pop();

    assert_eq!(
        app.input, "\u{e9}zz",
        "yank_pop mangled the surrounding text (byte offset used as char offset)"
    );
}

/// L4 x D3. The kill ring must round-trip: kill two words, yank them back.
#[test]
fn l4_kill_ring_restores_the_exact_removed_text() {
    let mut app = seed("alpha beta", 10);
    app.delete_to_start();
    assert_eq!(app.input, "");
    assert!(app.yank());
    assert_eq!(app.input, "alpha beta");
}

/// L4 x D10. `record_undo` caps the stack at 200 by dropping the oldest frame.
/// The bound must hold and the newest frame must survive.
#[test]
fn l4_undo_stack_is_bounded_and_keeps_the_newest_frame() {
    let mut app = App::new();
    for _ in 0..260 {
        app.insert_char('x');
    }
    assert!(app.undo_stack.len() <= 200, "stack grew to {}", app.undo_stack.len());
    assert!(app.undo(), "the newest frame must still be undoable");
    assert_eq!(app.input.chars().count(), 259);
}

/// L4 x D3. `undo` restores a cursor that was valid for the older, possibly
/// shorter, buffer. It clamps — this pins that it keeps clamping.
#[test]
fn l4_undo_clamps_the_cursor_to_the_restored_buffer() {
    let mut app = App::new();
    app.insert_str("hello");
    app.move_cursor_home();
    app.insert_str("XY");
    assert!(app.undo());
    assert!(app.cursor_pos <= app.input.chars().count());
    assert_eq!(app.input, "hello");
}

// ── L7: two failures at once ───────────────────────────────────────────────

/// L7 x D6. The realistic chain: `yank_pop` leaves the cursor out of range,
/// then the user presses Ctrl+Backspace. The second defect is only reachable
/// because the first one did not hold an invariant.
#[test]
fn l7_yank_pop_then_word_delete_does_not_panic() {
    let mut app = seed("\u{65e5}\u{672c}\u{8a9e}abc", 6);
    app.kill_ring = vec!["zz".into(), "abc".into()];
    app.yanked = Some("abc".into());
    app.yank_pop();

    app.delete_word_left();
    app.delete_word_right();
    app.move_word_left();
}

/// L7 x D3. Selection maths over multibyte text: every bound is a char index.
#[test]
fn l7_selection_bounds_are_char_indices_on_multibyte_text() {
    let text = "\u{65e5}\u{672c}\u{8a9e}abc";
    let mut app = seed(text, 0);
    app.set_cursor(0, false);
    app.extend_selection(3);
    assert_eq!(
        app.selected_text().as_deref(),
        Some("\u{65e5}\u{672c}\u{8a9e}"),
        "selection took the wrong slice of a multibyte buffer"
    );
    assert_eq!(app.delete_selection().as_deref(), Some("\u{65e5}\u{672c}\u{8a9e}"));
    assert_eq!(app.input, "abc");
}

/// L7 x D3. Deleting a selection that ends at the buffer end must not truncate
/// or duplicate the trailing character.
#[test]
fn l7_delete_selection_to_end_of_multibyte_buffer() {
    let mut app = seed("ab\u{65e5}\u{672c}", 4);
    app.set_cursor(2, false);
    app.extend_selection(2);
    assert_eq!(app.delete_selection().as_deref(), Some("\u{65e5}\u{672c}"));
    assert_eq!(app.input, "ab");
}

/// L7 x D3. A kill that spans a multi-byte boundary must cut on character
/// boundaries, never mid-codepoint.
#[test]
fn l7_kill_range_never_splits_a_codepoint() {
    let mut app = seed("a\u{65e5}\u{672c}\u{8a9e}b", 5);
    app.kill_range(1, 4, "test");
    assert_eq!(app.input, "ab", "kill_range cut inside a codepoint");
    assert!(app.yank());
    assert_eq!(app.input, "a\u{65e5}\u{672c}\u{8a9e}b");
}

/// L7 x D1. CRLF paste plus word-delete plus transpose: three interacting
/// normalisations on one buffer.
#[test]
fn l7_crlf_paste_then_edits_stay_consistent() {
    let mut app = App::new();
    app.insert_str("one\r\ntwo\rthree");
    assert_eq!(app.input, "one\ntwo\nthree", "CRLF/CR were not normalised");
    app.move_cursor_home();
    app.delete_word_left();
    assert!(app.cursor_pos <= app.input.chars().count());
    app.transpose_chars();
    assert!(app.cursor_pos <= app.input.chars().count());
}

/// L4 x D2. An edit that shrinks the buffer must invalidate the selection
/// anchor. `kill_range` (and `backspace`/`delete_char`) move the cursor but
/// leave `sel_anchor` behind, so the TUI keeps believing in a selection.
///
/// Found by the `cursor_never_escapes_the_buffer` property test with the
/// minimal input `ops = [insert('a'), extend_selection(2), delete_word_left]`.
#[test]
fn l4_edits_that_shrink_the_buffer_clear_the_selection_anchor() {
    let mut app = seed("", 0);
    app.insert_char('a');
    app.extend_selection(2);

    app.delete_word_left();

    assert_eq!(app.input, "");
    assert!(
        app.sel_anchor.is_none(),
        "sel_anchor {:?} survived an edit that emptied the buffer",
        app.sel_anchor
    );
    assert!(!app.has_selection(), "a stale anchor makes the TUI think text is selected");
    assert_eq!(app.selected_text(), None);
}

/// L7 x D6. The stale anchor has a user-visible consequence: Ctrl+C copies the
/// empty "selection" instead of clearing the input line, and Ctrl+X deletes
/// nothing at all.
#[test]
fn l7_stale_anchor_does_not_silently_swallow_a_keypress() {
    let mut app = seed("", 0);
    app.insert_char('a');
    app.extend_selection(2);
    app.delete_word_left();

    assert!(app.input.is_empty());
    assert_eq!(
        app.delete_selection(),
        None,
        "Backspace with a phantom selection must not become a no-op"
    );
    assert!(!app.has_selection());
}

// ── L3: invariants that must hold across arbitrary op sequences ────────────

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    /// L3 x D10. Whatever sequence of edits runs, the cursor stays a valid char
    /// index into the buffer. This is the invariant every other method assumes.
    #[test]
    fn cursor_never_escapes_the_buffer(
        seed_text in prop::collection::vec(any::<char>(), 0..24),
        ops in prop::collection::vec(0u8..22, 1..48),
    ) {
        let text: String = seed_text.into_iter().collect();
        let mut app = App::new();
        app.insert_str(&text);
        for op in ops {
            apply(&mut app, &op_for(op));
            let len = app.input.chars().count();
            prop_assert!(
                app.cursor_pos <= len,
                "cursor_pos {} > buffer length {} after op {} (input {:?})",
                app.cursor_pos, len, op, app.input
            );
            prop_assert!(
                app.sel_anchor.map(|a| a <= len).unwrap_or(true),
                "sel_anchor escaped the buffer after op {} (input {:?})",
                op, app.input
            );
        }
    }

    /// L3 x D1. `input_lines_count` is clamped so a runaway paste cannot make
    /// the input box swallow the viewport.
    #[test]
    fn input_lines_count_stays_clamped(
        text in prop::collection::vec(any::<char>(), 0..80),
    ) {
        let s: String = text.into_iter().collect();
        let app = seed(&s, 0);
        let n = app.input_lines_count();
        prop_assert!((1..=14).contains(&n), "input_lines_count returned {}", n);
    }

    /// L3 x D1. `input_visual_lines_count` divides by the available width; a
    /// zero or absurd width must not divide by zero or return zero rows.
    #[test]
    fn visual_lines_never_diverges(
        text in prop::collection::vec(any::<char>(), 0..60),
        width in 0usize..40,
    ) {
        let s: String = text.into_iter().collect();
        let app = seed(&s, 0);
        let n = app.input_visual_lines_count(width);
        prop_assert!(n >= 1, "width {} produced {} visual lines", width, n);
    }

    /// L3 x D10. `cursor_coordinates` must report a line that exists in the
    /// buffer, for every cursor position.
    #[test]
    fn cursor_coordinates_stay_within_the_buffer(
        text in prop::collection::vec(any::<char>(), 0..40),
        pos in 0usize..40,
    ) {
        let s: String = text.into_iter().collect();
        let len = s.chars().count();
        let app = seed(&s, pos.min(len));
        let (line, _col) = app.cursor_coordinates();
        let lines = s.split('\n').count();
        prop_assert!(line < lines, "line {} of {} reported", line, lines);
    }
}

fn op_for(n: u8) -> Op {
    match n {
        0 => Op::Insert('a'),
        1 => Op::Insert('\u{65e5}'),
        2 => Op::InsertStr("xy".into()),
        3 => Op::Backspace,
        4 => Op::Delete,
        5 => Op::DeleteWordLeft,
        6 => Op::DeleteWordRight,
        7 => Op::DeleteToStart,
        8 => Op::DeleteToEnd,
        9 => Op::MoveLeft,
        10 => Op::MoveRight,
        11 => Op::MoveUp,
        12 => Op::MoveDown,
        13 => Op::Home,
        14 => Op::End,
        15 => Op::TransposeChars,
        16 => Op::TransposeWords,
        17 => Op::Yank,
        18 => Op::YankPop,
        19 => Op::Undo,
        20 => Op::ExtendSelection(2),
        _ => Op::SetCursor(3),
    }
}
