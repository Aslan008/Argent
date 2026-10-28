# Changelog

All notable changes to Argent are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` (MIT), this changelog, `SECURITY.md` and `CONTRIBUTING.md`.
- API keys are stored in the OS credential store (`src/agent/secret_store.py`:
  Windows Credential Manager / macOS Keychain / Secret Service); a legacy
  plaintext key is migrated on first read and removed from the config file. With
  no usable vault the key falls back to the config file, so nothing locks users
  out.
- `tests/test_packaging.py` and `tests/test_dependency_lists.py` guard the new
  package layout: every module in `argent/` imports, and nothing installable is
  left at the repository root.
- CI: lint gate, Python 3.11/3.12 matrix, coverage report, wheel build that is
  installed and imported outside the source tree, and a Linux smoke job (see
  `.github/workflows/ci.yml`).

### Changed
- **`/vibe` is a switch, not a one-shot.** A finished turn no longer turns the mode
  off by itself: it stays on until you type `/vibe off`, and what it does is approve
  commands (a run the agent did not finish still picks itself up). Before, the very
  first completed turn ran `set_vibe(False)`, so `/vibe` lasted exactly one task.
  `src/bridge/repl_bridge.py` (`_maybe_auto_continue`); tests updated in
  `tests/test_vibe_forced_stop.py`, `tests/test_tui_fixes_regression.py`.
- **The TUI behaves like a window, not a printout.** Hovering explains a control
  (model, thinking, a tool card, the `todo` list); the context chip shows a token
  breakdown (system prompt / tool schemas / history) on hover; tool cards are inset
  from the answer text and carry `▸`/`▾` when they can be opened; reasoning blocks
  do the same. An approval is a framed, tinted banner with a title and key hints
  instead of a paragraph above the prompt.
- **Ctrl+C stops the work first.** While a turn runs, `Ctrl+C` now cancels it even
  if the prompt has a draft or a selection (the draft is kept); clearing the input
  first was why a busy agent looked like it ignored Ctrl+C. The UI says
  `Останавливаю…` immediately, a second press exits, and the agent checks the stop
  flag on every streamed chunk (including the fallback stream) so the request is
  abandoned at the next token.
- **One package.** The agent moved from top-level modules (`main.py`, `agent.py`,
  `config.py`, `logger.py`, `tools/`, …) into `argent/` (`argent/main.py`,
  `argent/tools/`, …). Run it with `python -m argent.main` or `python -m
  argent.run_tui`; the `argent` console command is unchanged. `src/` remains a
  separate package because its data paths (`src/leeth/data/...`) are addressed at
  run time. This removes the top-level `logger`, `config`, `session`, `tools`
  names that could shadow — or be shadowed by — an unrelated installed package.
- **One answer header per run.** The `промежуточный ответ` chip is gone: it
  labelled the assistant's narration before a tool call as an incomplete answer,
  appeared over tool calls, and could be printed twice for one turn. An answer
  now says `✦ Argent` once; a re-sent segment continues under the same header,
  and whitespace-only content draws no header at all.
- **Spend in the toolbar's right corner.** The session total (`sum 12.3k tok ·
  $0.0123`, from the provider's own usage counts) is pinned to the right edge in
  the premium TUI, so a long model/provider/cwd half can no longer push it off
  the row. Nothing is shown before the first request.
- **Ignored settings are named.** When the current provider/model is known to
  ignore a control — KoboldCPP has no thinking or depth field, Ollama takes a
  level only for `gpt-oss` — the corner shows `n/a: thinking, depth`. Cloud
  providers stay silent because their per-model metadata is not available
  offline (`argent/model_capabilities.py`).
- `logger.py` creates `~/.argent/logs` lazily on first log record instead of at
  import time, and degrades to a null handler when the directory exists but
  cannot be written, so importing Argent no longer fails in a read-only home.
- Real-window UI Automation tests are marked `integration` and excluded from the
  default run; a 300-second per-test timeout guards the suite.
- `pytest.ini`, `requirements-ci.txt`, `pyproject.toml` (dev extras) gained
  `pytest-timeout`, `pytest-cov` and `ruff`; the license is declared as an SPDX
  string with `license-files`, so wheel builds are warning-free.

### Fixed
- Lazy tools failed with `No module named 'tools.search_ops'` after the package
  move: a session started before the move keeps the loader table in memory, and
  `run_deep_research` still pointed at the removed top-level `deep_research`
  module in the code itself. All loader paths now live under `argent.`, and
  `tests/test_tool_registry_consistency.py` resolves every lazy loader, so the
  next stale path fails in CI instead of at the model's first call.

### Removed
- Local-only `refs/backup/pre-scrub-master` reference (old pre-scrub history with
  multi-gigabyte ONNX/PT blobs). See `reports/pre_scrub_backup_removed.md`.
- Stray `package.json` copied from another (unrelated) desktop application.

## [0.1.0] - 2026-03-13

Initial public iteration: terminal agent, provider abstraction (Ollama, Z.ai,
OpenRouter, KoboldCPP, Anthropic), tool layer, context trimming and compaction,
checkpoint/rewind, rooms, automations, Telegram bridge, Rust TUI, Android client.
