use argent_tui::app::App;

#[test]
fn test_boundary_empty_string() {
    let mut app = App::new();
    // All of these should be no-ops and never panic on empty input
    app.backspace();
    app.delete_char();
    app.delete_word_left();
    app.delete_word_right();
    app.move_cursor_left();
    app.move_cursor_right();
    app.move_cursor_home();
    app.move_cursor_end();
    assert!(!app.move_cursor_up());
    assert!(!app.move_cursor_down());
    assert_eq!(app.cursor_pos, 0);
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_coordinates(), (0, 0));
    assert_eq!(app.input_lines_count(), 1);
    assert!(app.submit_input().is_none());
}

#[test]
fn test_boundary_single_char() {
    let mut app = App::new();
    app.insert_char('A');
    assert_eq!(app.cursor_pos, 1);
    app.backspace();
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);

    app.insert_char('Z');
    app.move_cursor_home();
    app.delete_char();
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);
}

#[test]
fn test_boundary_whitespace_only() {
    let mut app = App::new();
    app.insert_str("   \t  ");
    assert!(app.submit_input().is_none()); // Trimmed to empty -> returns None
    assert_eq!(app.input, "   \t  "); // submit_input leaves input untouched if empty

    app.delete_word_left();
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);
}

#[test]
fn test_boundary_utf8_cyrillic_multibyte() {
    let mut app = App::new();
    // 2-byte characters
    app.insert_str("Привет мир");
    assert_eq!(app.cursor_pos, 10);
    assert_eq!(app.input.len(), 19); // 10 chars, 19 bytes (9 cyrillic * 2 + 1 space)

    // Delete word left
    app.delete_word_left();
    assert_eq!(app.input, "Привет ");
    assert_eq!(app.cursor_pos, 7);

    // Delete space + word left
    app.delete_word_left();
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);
}

#[test]
fn test_boundary_utf8_emoji_4byte() {
    let mut app = App::new();
    // 4-byte characters
    app.insert_str("✨ Rocket 🚀 Fire 🔥");
    assert_eq!(app.cursor_pos, 17);

    app.delete_word_left();
    assert_eq!(app.input, "✨ Rocket 🚀 Fire ");

    app.delete_word_left();
    assert_eq!(app.input, "✨ Rocket 🚀 ");

    app.delete_word_left();
    assert_eq!(app.input, "✨ Rocket ");

    app.delete_word_left();
    assert_eq!(app.input, "✨ ");

    app.delete_word_left();
    assert_eq!(app.input, "");
    assert_eq!(app.cursor_pos, 0);
}

#[test]
fn test_boundary_crlf_normalization() {
    let mut app = App::new();
    app.insert_str("Line 1\r\nLine 2\rLine 3\nLine 4");
    assert!(!app.input.contains('\r'), "CRLF must be normalized to standard LF");
    assert_eq!(app.input, "Line 1\nLine 2\nLine 3\nLine 4");
    assert_eq!(app.input_lines_count(), 4);
}

#[test]
fn test_boundary_multiline_empty_lines() {
    let mut app = App::new();
    app.insert_str("\n\n\n");
    assert_eq!(app.input_lines_count(), 4);
    assert_eq!(app.cursor_pos, 3);

    // Navigate up through empty lines
    assert!(app.move_cursor_up());
    assert_eq!(app.cursor_pos, 2);
    assert!(app.move_cursor_up());
    assert_eq!(app.cursor_pos, 1);
    assert!(app.move_cursor_up());
    assert_eq!(app.cursor_pos, 0);
    assert!(!app.move_cursor_up()); // Top reached

    // Navigate down
    assert!(app.move_cursor_down());
    assert_eq!(app.cursor_pos, 1);
    assert!(app.move_cursor_down());
    assert_eq!(app.cursor_pos, 2);
    assert!(app.move_cursor_down());
    assert_eq!(app.cursor_pos, 3);
    assert!(!app.move_cursor_down()); // Bottom reached
}

#[test]
fn test_boundary_ultra_large_input() {
    let mut app = App::new();
    let large = "a".repeat(10_000);
    app.insert_str(&large);
    assert_eq!(app.input.len(), 10_000);
    assert_eq!(app.cursor_pos, 10_000);

    app.delete_to_start();
    assert_eq!(app.input.len(), 0);
    assert_eq!(app.cursor_pos, 0);
}
