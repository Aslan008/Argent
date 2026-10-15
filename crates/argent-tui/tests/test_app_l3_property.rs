use proptest::prelude::*;
use argent_tui::app::App;

#[derive(Debug, Clone)]
enum TextAction {
    InsertChar(char),
    InsertStr(String),
    Backspace,
    DeleteChar,
    DeleteWordLeft,
    DeleteWordRight,
    DeleteToStart,
    DeleteToEnd,
    MoveLeft,
    MoveRight,
    MoveHome,
    MoveEnd,
    MoveUp,
    MoveDown,
    /// Explicit newline: `any::<char>()` draws '\n' with probability ~1e-6, so
    /// without this arm multi-line input (and its clamp invariant) is never
    /// actually exercised.
    Newline,
}

fn text_action_strategy() -> impl Strategy<Value = TextAction> {
    prop_oneof![
        any::<char>().prop_map(TextAction::InsertChar),
        Just(TextAction::Newline),
        "\\PC*".prop_map(TextAction::InsertStr),
        Just(TextAction::Backspace),
        Just(TextAction::DeleteChar),
        Just(TextAction::DeleteWordLeft),
        Just(TextAction::DeleteWordRight),
        Just(TextAction::DeleteToStart),
        Just(TextAction::DeleteToEnd),
        Just(TextAction::MoveLeft),
        Just(TextAction::MoveRight),
        Just(TextAction::MoveHome),
        Just(TextAction::MoveEnd),
        Just(TextAction::MoveUp),
        Just(TextAction::MoveDown),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_cursor_and_utf8_invariants_hold(actions in prop::collection::vec(text_action_strategy(), 1..100)) {
        let mut app = App::new();

        for action in actions {
            match action {
                TextAction::InsertChar(c) => app.insert_char(c),
                TextAction::Newline => app.insert_char('\n'),
                TextAction::InsertStr(s) => app.insert_str(&s),
                TextAction::Backspace => app.backspace(),
                TextAction::DeleteChar => app.delete_char(),
                TextAction::DeleteWordLeft => app.delete_word_left(),
                TextAction::DeleteWordRight => app.delete_word_right(),
                TextAction::DeleteToStart => app.delete_to_start(),
                TextAction::DeleteToEnd => app.delete_to_end(),
                TextAction::MoveLeft => app.move_cursor_left(),
                TextAction::MoveRight => app.move_cursor_right(),
                TextAction::MoveHome => app.move_cursor_home(),
                TextAction::MoveEnd => app.move_cursor_end(),
                TextAction::MoveUp => { let _ = app.move_cursor_up(); },
                TextAction::MoveDown => { let _ = app.move_cursor_down(); },
            }

            // Invariant 1: Cursor pos never exceeds character count
            let char_count = app.input.chars().count();
            prop_assert!(app.cursor_pos <= char_count, "cursor_pos {} exceeds char count {}", app.cursor_pos, char_count);

            // Invariant 2: Byte offset corresponding to cursor is ALWAYS a valid UTF-8 boundary
            let byte_offset = app.input.char_indices().nth(app.cursor_pos).map(|(i, _)| i).unwrap_or(app.input.len());
            prop_assert!(app.input.is_char_boundary(byte_offset), "byte offset {} is not a valid UTF-8 boundary in {:?}", byte_offset, app.input);

            // Invariant 3: Line count clamp invariant [1..=14] (App::input_lines_count
            // clamps to 14, matching the input-area cap in ui.rs draw_input)
            let line_count = app.input_lines_count();
            prop_assert!((1..=14).contains(&line_count), "line count {} is out of [1, 14]", line_count);

            // Invariant 4: cursor_coordinates returns valid line and column
            let (line, _col) = app.cursor_coordinates();
            let total_lines = app.input.split('\n').count();
            prop_assert!(line < total_lines, "line {} must be strictly less than total lines {}", line, total_lines);
        }
    }
}
