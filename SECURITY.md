# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Report it privately to
the maintainer (Aslan008) through GitHub's *Security → Report a vulnerability*
flow, or by direct message if you already have a contact.

Include: affected version or commit, a minimal reproduction, and the impact you
believe it has (for example: an approval bypass, a secret written to a log, a
path escape from the project directory).

You can expect an acknowledgement within a few days. This is a personal project
maintained by one person: fixes are best-effort, and there is no bug bounty.

## Supported versions

Only the `master` branch is supported. There are no maintained release branches
yet; the version in `version.py` is informational until the first tagged release.

## Security model (what is and is not a vulnerability)

Argent runs with the privileges of the user who started it and is *not* a
sandbox. These are deliberate design decisions, not bugs:

- **The approval gate is the boundary for shell commands.** `run_command`,
  `start_background_command` and desktop actions go through `approval.py`.
  File tools (`read_file`, `write_file`, `replace_in_file`) are not gated: the
  agent can read and write files the user can, anywhere the user can.
- **Unattended runs refuse gated actions.** Scheduled tasks (`/tasks`) and
  `arg` runs with nobody watching do not auto-approve anything; a refused action
  is recorded in the run report.
- **Project-supplied code is untrusted.** Hooks (`/hooks trust`) and permission
  files (`/permissions trust`) do not take effect for `allow` until the user has
  read them and trusted them; `deny`/`ask` rules from a repository apply at once
  because they only remove power.
- **Secrets are redacted in logs, tool output and the screen.** API keys are
  stored in the OS credential store (Windows Credential Manager, macOS Keychain,
  Secret Service) through `src/agent/secret_store.py`; only when no vault is
  available does a key fall back to `~/.argent_coder_config.json` (user-only
  permissions). A legacy key in the config is migrated into the vault on first
  read and removed from the file.
- **Web content, skills, MCP servers and model output are untrusted input.** A
  prompt injection that convinces the model to call a safe tool is possible; the
  gate limits what the consequence can be, and `deny` rules are absolute.

Reports that only demonstrate "the model was persuaded to run a command the user
approved" are out of scope. Reports that show a **bypass of the gate**, a secret
reaching disk/network when it should not, or code execution from a project file
without trust are in scope and very welcome.
