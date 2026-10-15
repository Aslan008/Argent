"""Standing permission rules: what the agent may always do, must always ask about, or never do.

The approval gate answers one call at a time. Its only memory is "always allow `git`
this session" — gone on exit, keyed on a program name, and blind to what the program
is asked to do (`git status` and `git push --force` are both `git`). People who work
with Argent every day end up pressing `y` on the same forty commands, or switching
the gate to auto and giving up the protection altogether.

Rules fix both ends:

    {"allow": ["run_command(git status*)", "run_command(npm test*)"],
     "ask":   ["run_command(git push*)"],
     "deny":  ["write_file(.env*)", "read_file(*.pem)", "run_command(rm -rf*)"]}

in `.argent/permissions.json` (this project) and `~/.argent/permissions.json` (all).

* **deny** beats everything: the call does not run, in any mode (auto, vibe, plan),
  and the model is told it was the person's rule so it does not look for a way round.
* **ask** beats allow and beats the shortcuts (a session grant, auto mode): the person
  is asked even when they would not be otherwise.
* **allow** approves a *shell command* without asking — never a destructive one, and
  never for an unattended run, which has its own, narrower allowlist. Allowing
  `write_file(...)` would mean nothing: file edits are not gated by approval at all.

How a pattern is read. `Tool(pattern)`; `Tool` alone means every call of that tool.
For shell commands `*` matches within ONE command and never across `; & | < > $`:
"git log*" allows `git log --oneline` and not `git log; rm -rf x`. Deny and ask rules
look at each chained command as well as the whole line, so `cd x && rm -rf y` is
caught by `rm -rf*`. For files the pattern is a glob over the project-relative path
(or just the file name when the pattern has no slash).

A project's rules file can come from a repository you cloned, and an `allow` in it
would let the repository decide what runs on your machine without asking. So the
project file is honoured for deny and ask (they only take power away) at once, and
for allow only after `/permissions trust` — see src.agent.shell_hooks for the same
reasoning about hooks.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from logger import get_logger
from src.agent.shell_hooks import (
    command_parts, file_digest, glob_match, is_trusted, subject, trust_file,
)

log = get_logger("permissions")

KINDS = ("deny", "ask", "allow")
COMMAND_TOOLS = frozenset({"run_command", "start_background_command", "run_admin_command"})

_RULE_RE = re.compile(r"^\s*([A-Za-z_*][\w*|]*)\s*(?:\((.*)\))?\s*$", re.DOTALL)


@dataclass(frozen=True)
class Rule:
    kind: str
    tool: str                    # glob(s) separated by |
    pattern: Optional[str]       # None: any call of the tool
    source: str                  # "global" | "project"

    @property
    def raw(self) -> str:
        return f"{self.tool}({self.pattern})" if self.pattern is not None else self.tool

    def covers_tool(self, tool: str) -> bool:
        import fnmatch
        return any(fnmatch.fnmatchcase(tool, part.strip()) for part in self.tool.split("|") if part.strip())


def parse_rule(text: str, kind: str, source: str = "project") -> Rule:
    match = _RULE_RE.match(text or "")
    if not match:
        raise ValueError(f"cannot read the rule {text!r}: write Tool(pattern), e.g. run_command(git status*)")
    tool, pattern = match.group(1), match.group(2)
    if pattern is not None and not pattern.strip():
        pattern = None
    return Rule(kind, tool, pattern.strip() if pattern is not None else None, source)


def _relative(path: str, root: Path) -> str:
    try:
        p = Path(path)
        if p.is_absolute():
            p = p.resolve().relative_to(root.resolve())
        return os.path.normpath(str(p)).replace("\\", "/")
    except (ValueError, OSError):
        return path.replace("\\", "/")


def _command_matches(command: str, pattern: str, whole_line_only: bool) -> bool:
    from src.automation.runner import command_allowed
    if command_allowed(command, [pattern]):
        return True
    if whole_line_only:
        return False
    return any(command_allowed(part, [pattern]) for part in command_parts(command))


def matches(rule: Rule, tool: str, args: dict, root: Path) -> bool:
    if not rule.covers_tool(tool):
        return False
    if rule.pattern is None or rule.pattern == "*":
        return True
    file, command = subject(args or {})
    if tool in COMMAND_TOOLS or command and not file:
        # An allow must cover the line exactly as it will run; deny/ask also look inside chains.
        return bool(command) and _command_matches(command, rule.pattern, whole_line_only=rule.kind == "allow")
    if file:
        rel = _relative(file, root)
        return glob_match(rel, rule.pattern) or ("/" not in rule.pattern and glob_match(rel.rsplit("/", 1)[-1], rule.pattern))
    return False


def parse_file_data(data, source: str) -> Tuple[List[Rule], List[str]]:
    rules, problems = [], []
    if not isinstance(data, dict):
        return [], ["expected an object with \"allow\" / \"ask\" / \"deny\" lists"]
    for kind in KINDS:
        entries = data.get(kind, [])
        if not isinstance(entries, list):
            problems.append(f"\"{kind}\" must be a list")
            continue
        for entry in entries:
            try:
                rules.append(parse_rule(str(entry), kind, source))
            except ValueError as e:
                problems.append(str(e))
    for key in data:
        if key not in KINDS:
            problems.append(f"unknown key '{key}' (use allow, ask, deny)")
    return rules, problems


class Permissions:
    def __init__(self, project_dir=None, home=None):
        self._project_dir = Path(project_dir) if project_dir else None
        self._home = Path(home) if home else None
        self._key = None
        self._rules: List[Rule] = []
        self._notes: List[str] = []
        self._untrusted: Optional[Path] = None
        self._fixed: Optional[List[Rule]] = None

    @classmethod
    def fixed(cls, rules: List[Rule], project_dir) -> "Permissions":
        """Rules handed over by a parent process: no files are read, nothing is trusted or edited."""
        perms = cls(project_dir=project_dir)
        perms._fixed = list(rules)
        return perms

    def project_root(self) -> Path:
        if self._project_dir:
            return self._project_dir
        from project_paths import project_root_or_cwd
        return project_root_or_cwd()

    def files(self) -> List[Tuple[str, Path]]:
        home = self._home or Path.home()
        return [("global", home / ".argent" / "permissions.json"),
                ("project", self.project_root() / ".argent" / "permissions.json")]

    def _load(self) -> None:
        if self._fixed is not None:
            self._rules = self._fixed
            return
        key = []
        for source, path in self.files():
            try:
                st = path.stat()
                key.append((str(path), st.st_mtime_ns, st.st_size, is_trusted(path) if source == "project" else True))
            except OSError:
                key.append((str(path), None))
        key = tuple(key)
        if key == self._key:
            return
        rules, notes, untrusted = [], [], None
        for source, path in self.files():
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                notes.append(f"{path}: {e}")
                continue
            parsed, problems = parse_file_data(data, source)
            notes += [f"{path}: {p}" for p in problems]
            if source == "project" and not is_trusted(path):
                untrusted = path
                parsed = [r for r in parsed if r.kind != "allow"]      # take-power-away rules still apply
            rules += parsed
        self._key, self._rules, self._notes, self._untrusted = key, rules, notes, untrusted

    # -- queries ------------------------------------------------------------------------

    def rules(self, kind: Optional[str] = None) -> List[Rule]:
        self._load()
        return [r for r in self._rules if kind is None or r.kind == kind]

    def notes(self) -> List[str]:
        self._load()
        return list(self._notes)

    def untrusted(self) -> Optional[Path]:
        self._load()
        return self._untrusted

    def first(self, kind: str, tool: str, args: dict) -> Optional[Rule]:
        root = self.project_root()
        return next((r for r in self.rules(kind) if matches(r, tool, args, root)), None)

    def denied(self, tool: str, args: dict) -> Optional[str]:
        """The message for the model when a deny rule forbids this call, else None."""
        rule = self.first("deny", tool, args)
        if rule is None:
            return None
        return (f"Denied by the user's permission rule {rule.raw} ({rule.source}). This is the person's standing "
                "rule, not an error: do not look for another way to do the same thing. If it is really needed, "
                "say what and why, and let them decide.")

    def asked(self, tool: str, args: dict) -> Optional[Rule]:
        return self.first("ask", tool, args)

    def command_rule(self, kind: str, command: str) -> Optional[Rule]:
        """The first rule of `kind` covering a shell command (what the approval gate knows)."""
        root = self.project_root()
        return next((r for r in self.rules(kind)
                     if matches(r, "run_command", {"command": command}, root)), None)

    # -- editing ------------------------------------------------------------------------

    def add(self, kind: str, text: str, scope: str = "project") -> Rule:
        rule = parse_rule(text, kind, scope)
        path = dict(self.files())[scope]
        data = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise ValueError(f"{path} is not valid JSON — fix or delete it first")
        entries = list(data.get(kind, []))
        if rule.raw not in entries:
            entries.append(rule.raw)
        data[kind] = entries
        self._write(path, data, scope)
        return rule

    def remove(self, text: str, scope: Optional[str] = None) -> int:
        removed = 0
        for source, path in self.files():
            if scope and source != scope or not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            changed = False
            for kind in KINDS:
                entries = data.get(kind, [])
                keep = [e for e in entries if e.strip() != text.strip()]
                if len(keep) != len(entries):
                    data[kind], changed = keep, True
                    removed += len(entries) - len(keep)
            if changed:
                self._write(path, data, source)
        return removed

    def _write(self, path: Path, data: dict, scope: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.is_file()
        was_trusted = scope == "project" and existed and is_trusted(path)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # A rule added through the command is the person's own, so a file they just created, or one they
        # already trusted, stays trusted. A file that arrived with the repository and was never read does
        # NOT become trusted because someone appended one line to it.
        if scope == "project" and (was_trusted or not existed):
            trust_file(path)
        self._key = None

    def trust(self) -> Optional[Path]:
        path = dict(self.files())["project"]
        if not path.is_file():
            return None
        trust_file(path)
        self._key = None
        return path


PERMISSIONS = Permissions()
