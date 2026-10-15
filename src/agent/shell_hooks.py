"""Hooks you write as a line of JSON, not as a Python plugin.

`hook_manager` loads Python plugins: powerful, and a program to write for "run the
linter after an edit". This module covers the everyday case with configuration — a
shell command bound to an event:

    {"hooks": {
        "PreToolUse":       [{"matcher": "run_command", "if": "git push*",
                              "command": "echo pushing needs a review first >&2 & exit 2"}],
        "PostToolUse":      [{"matcher": "write_file|replace_in_file", "if": "*.py",
                              "command": "ruff check \"$ARGENT_FILE\"", "timeout": 60}],
        "UserPromptSubmit": [{"command": "git status --short"}],
        "Stop":             [{"command": "notify-send Argent done"}],
        "SessionStart":     [{"command": "git log --oneline -3"}]}}

in `.argent/hooks.json` (this project) and `~/.argent/hooks.json` (everywhere).

How a hook talks back — the same contract other agents use, so a script written for
one works in the other:

* It gets the event as JSON on stdin and in ARGENT_* environment variables
  (ARGENT_EVENT, ARGENT_TOOL, ARGENT_FILE, ARGENT_COMMAND, ARGENT_PROJECT_DIR).
  Values are NEVER pasted into the command line: a file name chosen by a model is
  data, and splicing it into a shell string would let it be code.
* Exit 0 is "fine". Exit 2 is "no": PreToolUse stops the tool call and the model is
  told why (stderr), UserPromptSubmit drops the message. Any other exit code is a
  broken hook, which is logged and does NOT stop the work — a typo in a lint script
  must not freeze the agent.
* PostToolUse output (stdout on success, stderr on exit 2) is appended to the tool's
  result, so the model sees the lint errors it just caused. UserPromptSubmit stdout
  is added to the prompt as context. Stop and SessionStart output is shown to the
  person.

A project's own `hooks.json` is code that runs on your machine the moment you open
the folder, and a cloned repository can carry one. So it does not run until you have
read and trusted it (`/hooks trust`); an edited file is untrusted again, because the
trust belongs to a specific content. The global file is yours and always runs.
"""

import fnmatch
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from logger import get_logger

log = get_logger("hooks")

EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")
TOOL_EVENTS = ("PreToolUse", "PostToolUse")

DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
#: What of a hook's output goes back to the model / the person.
MAX_OUTPUT_CHARS = 4000
#: A tool argument longer than this is cut before being handed to a hook on stdin.
MAX_PAYLOAD_VALUE = 20000

#: Arguments whose value names the thing a call is about (for `if` patterns and ARGENT_FILE).
_FILE_KEYS = ("file_path", "path", "source", "destination", "directory", "dir_path")
_COMMAND_KEYS = ("command", "cmd")


@dataclass(frozen=True)
class HookSpec:
    event: str
    command: str
    matcher: str = ""          # tool-name glob(s) separated by |; empty or * = every tool
    when: str = ""             # glob over the call's file path or command line
    timeout: int = DEFAULT_TIMEOUT
    source: str = "project"    # "global" | "project"

    def describe(self) -> str:
        bits = [self.event]
        if self.matcher:
            bits.append(f"[{self.matcher}]")
        if self.when:
            bits.append(f"if {self.when}")
        return " ".join(bits) + f" → {self.command}"


@dataclass
class HookOutcome:
    spec: HookSpec
    code: Optional[int]        # None: could not run or timed out
    stdout: str = ""
    stderr: str = ""
    problem: str = ""          # "timed out", "could not start: ..."

    @property
    def blocks(self) -> bool:
        return self.code == 2


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + f"\n…[{len(text) - limit} more characters cut]"


def subject(args: dict) -> Tuple[str, str]:
    """(file, command) a tool call is about."""
    file = next((str(args[k]) for k in _FILE_KEYS if isinstance(args.get(k), str)), "")
    command = next((str(args[k]) for k in _COMMAND_KEYS if isinstance(args.get(k), str)), "")
    return file, command


def glob_match(value: str, pattern: str) -> bool:
    if os.name == "nt":
        value, pattern = value.lower().replace("\\", "/"), pattern.lower().replace("\\", "/")
    return fnmatch.fnmatchcase(value, pattern)


def command_parts(command: str) -> List[str]:
    """The whole line and each of its chained commands (so `cd x && git push` is a `git push`)."""
    parts = [command.strip()]
    try:
        from approval import _split_unquoted_chain_commands
        parts += [p.strip() for p in _split_unquoted_chain_commands(command)]
    except Exception:
        pass
    return [p for p in parts if p]


def _matches(spec: HookSpec, tool: str, args: dict) -> bool:
    if spec.matcher and spec.matcher != "*":
        if not any(fnmatch.fnmatchcase(tool, part.strip()) for part in spec.matcher.split("|") if part.strip()):
            return False
    if not spec.when:
        return True
    file, command = subject(args)
    candidates = ([file] if file else []) + (command_parts(command) if command else [])
    return any(glob_match(c, spec.when) or glob_match(os.path.basename(c), spec.when) for c in candidates)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trusted() -> dict:
    try:
        import config
        return dict(config.get_trusted_files() or {})
    except Exception:
        return {}


def trust_file(path: Path) -> str:
    """Remember the file's current content as trusted; returns its digest."""
    import config
    digest = file_digest(path)
    trusted = _trusted()
    trusted[str(path.resolve())] = digest
    config.set_trusted_files(trusted)
    return digest


def is_trusted(path: Path) -> bool:
    try:
        return _trusted().get(str(path.resolve())) == file_digest(path)
    except OSError:
        return False


def parse_specs(data, source: str) -> Tuple[List[HookSpec], List[str]]:
    """(specs, problems) from a decoded hooks.json."""
    specs, problems = [], []
    table = data.get("hooks", data) if isinstance(data, dict) else None
    if not isinstance(table, dict):
        return [], ["expected an object with a \"hooks\" table"]
    for event, entries in table.items():
        if event not in EVENTS:
            problems.append(f"unknown event '{event}' (known: {', '.join(EVENTS)})")
            continue
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            problems.append(f"{event}: expected a list of hooks")
            continue
        for i, entry in enumerate(entries, 1):
            if not isinstance(entry, dict) or not str(entry.get("command", "")).strip():
                problems.append(f"{event} #{i}: a hook needs a \"command\"")
                continue
            try:
                timeout = int(entry.get("timeout", DEFAULT_TIMEOUT))
            except (TypeError, ValueError):
                timeout = DEFAULT_TIMEOUT
            specs.append(HookSpec(
                event=event, command=str(entry["command"]).strip(),
                matcher=str(entry.get("matcher", "") or ""),
                when=str(entry.get("if", "") or ""),
                timeout=max(1, min(timeout, MAX_TIMEOUT)), source=source))
    return specs, problems


@dataclass
class _Loaded:
    key: tuple = ()
    specs: List[HookSpec] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    untrusted: Optional[Path] = None


class ShellHooks:
    """The configured hooks of the current project plus the global ones."""

    def __init__(self, project_dir=None, home=None):
        self._project_dir = Path(project_dir) if project_dir else None
        self._home = Path(home) if home else None
        self._loaded = _Loaded(key=None)

    # -- where the files are ------------------------------------------------------------

    def project_root(self) -> Path:
        if self._project_dir:
            return self._project_dir
        from project_paths import project_root_or_cwd
        return project_root_or_cwd()

    def files(self) -> List[Tuple[str, Path]]:
        home = self._home or Path.home()
        return [("global", home / ".argent" / "hooks.json"),
                ("project", self.project_root() / ".argent" / "hooks.json")]

    # -- loading ------------------------------------------------------------------------

    def _fresh(self) -> _Loaded:
        files = self.files()
        key = []
        for source, path in files:
            try:
                st = path.stat()
                key.append((str(path), st.st_mtime_ns, st.st_size, is_trusted(path) if source == "project" else True))
            except OSError:
                key.append((str(path), None))
        key = tuple(key)
        if self._loaded.key == key:
            return self._loaded
        loaded = _Loaded(key=key)
        for source, path in files:
            if not path.is_file():
                continue
            if source == "project" and not is_trusted(path):
                loaded.untrusted = path
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                loaded.notes.append(f"{path}: {e}")
                continue
            specs, problems = parse_specs(data, source)
            loaded.specs += specs
            loaded.notes += [f"{path}: {p}" for p in problems]
        self._loaded = loaded
        return loaded

    def specs(self, event: Optional[str] = None) -> List[HookSpec]:
        specs = self._fresh().specs
        return [s for s in specs if event is None or s.event == event]

    def status(self) -> dict:
        loaded = self._fresh()
        return {"specs": list(loaded.specs), "notes": list(loaded.notes), "untrusted": loaded.untrusted}

    def has_any(self) -> bool:
        loaded = self._fresh()
        return bool(loaded.specs)

    # -- running ------------------------------------------------------------------------

    def _run(self, spec: HookSpec, payload: dict, env_extra: dict) -> HookOutcome:
        from src.agent.shell import run_text

        env = os.environ.copy()
        env.update({k: str(v) for k, v in env_extra.items()})
        env["ARGENT_EVENT"] = spec.event
        env["ARGENT_PROJECT_DIR"] = str(self.project_root())
        env.setdefault("PYTHONUTF8", "1")
        try:
            done = run_text(spec.command, shell=True, capture_output=True, input=json.dumps(payload, default=str),
                            timeout=spec.timeout, cwd=str(self.project_root()), env=env,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return HookOutcome(spec, None, problem=f"timed out after {spec.timeout}s")
        except OSError as e:
            return HookOutcome(spec, None, problem=f"could not start: {e}")
        return HookOutcome(spec, done.returncode, done.stdout or "", done.stderr or "")

    @staticmethod
    def _payload(event: str, tool: str = "", args: Optional[dict] = None, **extra) -> dict:
        args = args or {}
        clipped = {k: (v[:MAX_PAYLOAD_VALUE] if isinstance(v, str) else v) for k, v in args.items()}
        payload = {"event": event, "cwd": os.getcwd()}
        if tool:
            payload["tool_name"] = tool
            payload["tool_input"] = clipped
        payload.update(extra)
        return payload

    def _tool_env(self, tool: str, args: dict) -> dict:
        file, command = subject(args)
        return {"ARGENT_TOOL": tool, "ARGENT_FILE": file, "ARGENT_COMMAND": command}

    def _record(self, outcome: HookOutcome) -> None:
        if outcome.problem:
            log.warning("hook %s: %s", outcome.spec.describe(), outcome.problem)
        elif outcome.code not in (0, 2):
            log.warning("hook %s exited with %s: %s", outcome.spec.describe(), outcome.code,
                        _clip(outcome.stderr, 300))

    # -- the events ---------------------------------------------------------------------

    def pre_tool(self, tool: str, args: dict) -> Optional[str]:
        """Why a hook refuses this call, or None."""
        for spec in self.specs("PreToolUse"):
            if not _matches(spec, tool, args):
                continue
            out = self._run(spec, self._payload("PreToolUse", tool, args), self._tool_env(tool, args))
            self._record(out)
            if out.blocks:
                reason = _clip(out.stderr or out.stdout) or "no reason given"
                return f"Blocked by a hook ({spec.source}): {reason}"
        return None

    def post_tool(self, tool: str, args: dict, result) -> str:
        """Text to append to the tool's result (hook output), possibly empty."""
        notes = []
        for spec in self.specs("PostToolUse"):
            if not _matches(spec, tool, args):
                continue
            payload = self._payload("PostToolUse", tool, args, tool_response=str(result)[:MAX_PAYLOAD_VALUE])
            out = self._run(spec, payload, self._tool_env(tool, args))
            self._record(out)
            text = _clip(out.stderr if out.blocks else out.stdout)
            if out.blocks and not text:
                text = _clip(out.stdout)
            if text:
                label = "hook reports a problem" if out.blocks else "hook output"
                notes.append(f"[{label}: {spec.command}]\n{text}")
        return "\n\n".join(notes)

    def user_prompt(self, prompt: str) -> Tuple[Optional[str], str]:
        """(prompt with the hooks' context added, or None if one dropped it; a note for the person)."""
        extra, note = [], []
        for spec in self.specs("UserPromptSubmit"):
            out = self._run(spec, self._payload("UserPromptSubmit", prompt=prompt), {})
            self._record(out)
            if out.blocks:
                return None, _clip(out.stderr or out.stdout) or "the message was stopped by a hook"
            if out.stdout.strip() and out.code == 0:
                extra.append(_clip(out.stdout))
        if extra:
            prompt = prompt + "\n\n[Context from hooks]\n" + "\n".join(extra)
        return prompt, "\n".join(note)

    def notify(self, event: str, **extra) -> str:
        """Run a Stop / SessionStart hook; what they printed, for the person."""
        printed = []
        for spec in self.specs(event):
            out = self._run(spec, self._payload(event, **extra), {})
            self._record(out)
            text = _clip(out.stdout if out.code == 0 else (out.stderr or out.stdout))
            if text:
                printed.append(text)
        return "\n".join(printed)


SHELL_HOOKS = ShellHooks()


# -- what a frontend calls: never raise, a hooks file must not take the session down -------

def prompt_submitted(prompt: str) -> Tuple[Optional[str], str]:
    """(prompt to send, or None if a hook stopped it; note for the person)."""
    try:
        return SHELL_HOOKS.user_prompt(prompt)
    except Exception as e:
        log.warning("UserPromptSubmit hooks failed: %s", e)
        return prompt, ""


def turn_stopped() -> str:
    try:
        return SHELL_HOOKS.notify("Stop")
    except Exception as e:
        log.warning("Stop hooks failed: %s", e)
        return ""


def session_started() -> str:
    try:
        return SHELL_HOOKS.notify("SessionStart")
    except Exception as e:
        log.warning("SessionStart hooks failed: %s", e)
        return ""
