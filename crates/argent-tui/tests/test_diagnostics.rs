//! The diagnostics file has to be readable, bounded and harmless.
//!
//! It exists because the freeze could not be described: "it happened at some point, the
//! wheel scrolled the wrong thing" is not enough to find a cause. A line a minute, plus
//! one for every resize, focus change and wheel-as-keys event, turns the next occurrence
//! into data — and writing it must never be able to make things worse.
//!
//! The contract is tested at the public writer, with files under the crate's own
//! `target/` directory: `%TEMP%` is not writable for this test process, and a
//! diagnostics test that cannot create its own file would prove nothing.

use std::fs;
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("diag-tests");
    fs::create_dir_all(&dir).expect("target доступен для записи");
    dir.join(name)
}

#[test]
fn a_line_is_written_and_appended() {
    let path = scratch("integration.log");
    let _ = fs::remove_file(&path);

    argent_tui::diag_to(&path, "alive tick=7 busy=false").expect("строка пишется");
    argent_tui::diag_to(&path, "resize 120x40").expect("и вторая тоже");

    let text = fs::read_to_string(&path).expect("файл читается");
    let _ = fs::remove_file(&path);

    assert!(text.contains("alive tick=7 busy=false"), "{text}");
    assert!(text.contains("resize 120x40"), "дописывается, а не перезаписывается: {text}");
    assert!(text.contains("t="), "у каждой строки есть метка времени: {text}");
}

#[test]
fn a_folder_is_created_when_it_is_missing() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target").join("diag-tests").join("nested");
    let path = dir.join("deeper").join("log.txt");
    let _ = fs::remove_dir_all(&dir);

    argent_tui::diag_to(&path, "hello").expect("каталог создаётся сам");

    assert!(path.exists(), "файл появился вместе с каталогом");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn an_unwritable_target_reports_an_error_instead_of_panicking() {
    let dir = scratch("blocked-dir");
    fs::create_dir_all(&dir).expect("каталог создан");

    // A directory where a file should be: the writer returns the error, quietly.
    assert!(argent_tui::diag_to(&dir, "x").is_err());

    let _ = fs::remove_dir(&dir);
}
