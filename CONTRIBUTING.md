# Contributing

Thanks for looking. Argent is a personal project, but it is built to be read:
the comments explain *why* a decision was made, and the tests are the contract.

## Before you start

- Python **3.11+** (CI runs 3.12), Windows for the full feature set (desktop
  control, ConPTY); Linux/macOS support the terminal agent and the test suite.
- `pip install -r requirements.txt` installs the runtime plus the optional RAG /
  browser / research stacks. For a light install: `pip install -e .`.
- Rust tests need a stable toolchain and `maturin develop -m
  crates/argent-tui/Cargo.toml`.

## Checks to run before opening a pull request

```bash
python -m pytest -q            # unit suite (no browser, no real desktop)
python -m pytest -m integration  # only if your change touches the browser/desktop
cargo test --manifest-path crates/argent-tui/Cargo.toml
```

If you add a module for the agent, put it under `argent/` (tools go in
`argent/tools/`). The package is discovered by name, so there is no list to
update — and `tests/test_dependency_lists.py` fails if a module is left at the
repository root, where an installed copy would not contain it.

## House rules

- **Safety is the feature.** A tool that can change the machine goes through
  `approval.py`. A new shell-capable path needs a risk grade and a test that the
  destructive case asks (or is refused, when unattended).
- **A new tool needs a schema and a test.** `tools/schemas.py` is the catalog;
  tools that are lazily imported also need a `_REGISTRY` entry.
- **Do not commit generated data.** Weights, `.jsonl` trajectories, logs,
  session dumps and exports are ignored on purpose; see the privacy section of
  `.gitignore`.
- **Explain the failure in the comment.** "Why this exists" beats "what it does";
  the next reader is usually debugging the case that made you write it.
- Commit messages: Conventional Commits where practical (`fix:`, `feat:`,
  `perf:`), the body in Russian or English — both are used in the history.

## Security

Do not open a public issue for a vulnerability; see [SECURITY.md](SECURITY.md).
