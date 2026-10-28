---
name: argent-tui-change
description: Use when changing the Argent terminal UI (crates/argent-tui) — layout, folding, cards, scrolling, panels, keys, colours. Covers the correct way to run the Rust suite on Windows, how to see the result instead of asking the person, and how to deploy a rebuilt extension.
version: 1.0.0
---

# Changing the Argent TUI

The UI is a Rust extension (`crates/argent-tui`) that Python loads as `argent_tui`.
Logic is easy to verify here; the *look* is not, unless you follow the steps below.
Two rounds of "the fold does not work as I expected" came from skipping them.

## 1. Run the suite the right way

Never call `cargo test` directly:

```powershell
cmd /c "crates\argent-tui\test.bat > scratch\tui_test.log 2>&1"
Write-Output "EXIT=$LASTEXITCODE"
Select-String -Path scratch\tui_test.log -Pattern "test result: FAILED|targets? failed"
```

Why this exact form:

- Four targets (`test_bridge_e2e`, `test_cli_parity`, `test_events_l1_l4`,
  `test_history_replay`) link libpython. Without `python311.dll` on PATH they die with
  `0xc0000135 STATUS_DLL_NOT_FOUND` before a single test runs. `test.bat` derives the
  interpreter directory from `.venv\pyvenv.cfg` and puts it on PATH.
- `test.bat` passes `--no-fail-fast`. cargo otherwise stops at the first failing binary,
  and the rest of the suite never runs — that is how five broken suites read as green.
- PowerShell turns cargo's stderr into a failed command; the exit code of a pipeline
  lies. Read `EXIT` and the log file, never `$?`.

## 2. Prove behaviour with headless render tests

`TestBackend` + `draw_ui` + assertions on the produced rows is the primary proof, and it
works without a terminal. Templates: `tests/test_turn_fold.rs`, `tests/test_card_scroll.rs`,
`tests/test_thinking_folds.rs`.

```rust
let mut terminal = Terminal::new(TestBackend::new(100, 40)).unwrap();
terminal.draw(|f| draw_ui(f, app)).unwrap();
let buf = terminal.backend().buffer().clone();
```

Assert on what a person would read (`row.contains("Ходы агента")`), not on internal
structures. When a test pins a product rule, say the rule in a comment — the next agent
needs to know whether breaking it is a bug or a decision.

## 3. Remember which look you are testing

`App::new()` draws `UiStyle::Classic`; the shipped look is **premium** (`argent/config.py`
defaults to it, and the bridge always sends the real value). A test that claims to check
what the person sees must say `app.ui.set_style(UiStyle::Premium)` — tool cards, folds and
the card scroll window only exist in that path.

## 4. Look at it before shipping

Deploy (ask the person first — the `.pyd` is locked by the running app):

```
/tui-build          # build, stage, install, report the build time
```

Then, *after the person restarts the window*, actually look:

1. `list_desktop_windows` — find the Argent/terminal window.
2. `capture_desktop_window` with that title (or `region='bottom:300'` for the transcript).
3. `view_image` on the saved PNG.

Two traps: a screenshot of the **old** running build proves nothing — check the build time
(`doctor`, or the pyd mtime) before believing the frame; and a captured frame is the
person's screen, so capture the window, not the whole desktop, unless asked.

## 5. Facts worth knowing before editing

- `App.messages` is a flat `Vec<MessageItem>`; there is no tree. Per-item view state that
  the enum does not carry belongs in `App` and is keyed by index — see `card_scroll`,
  `card_max`, `card_rects` (`app.rs`) for the pattern. Adding a field to every variant
  rewrites every constructor and every test.
- Turn folding lives in `fold_finished_turn` (`app.rs`) with `fold_groups`/`fold_open`;
  the header is `view.rs::fold_header`. One line per turn, named «Ходы агента»; a running
  turn never folds; `apply_history` folds a restored dialogue.
- `wheel()` gives the wheel to whatever is under the pointer, in this order: modals,
  tool cards (`card_rects`), panels (`todos_hover`, `queue_hover`), then the chat.
- `replace_in_file` warns when an anchored edit lands in the middle of a `///` run — read
  the warning, it means the doc comment now documents the wrong item.
- The working tree is huge (hundreds of modified files); `git diff` is useless for
  self-review. Diff only the paths you touched: `git diff -- crates/argent-tui/src/app.rs`.
