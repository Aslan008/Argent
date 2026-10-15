"""What a node is told to ignore, and what it may reach: MCP servers and skills.

A room's agent node can be given skills and MCP servers; `ignore` rules take things
out of what it can see or use. The rules are plain text, so a room stays data:

  mcp:<server>               every tool of the server is off limits
  mcp:<server>.<tool>        one tool (a pattern: ``mcp:unity.delete_*``, ``mcp:*.write*``)
  skill:<name>               the skill does not exist for this node
  skill:<name>#<heading>     the skill is read without that section (up to the next
                             heading of the same or a higher level)

The rules are ENFORCED where the tools live (`call_mcp_tool`, `list_mcp_tools`,
`read_skill`, `list_skills` ask this module), not only written into a prompt: a model
that is told "do not use X" and a model that cannot reach X are different things.

A run installs the node's policy for the time the node works (`use(...)`); several
nested policies add up — the strictest wins.
"""

import fnmatch
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Rule:
    kind: str                 # "mcp" or "skill"
    name: str                 # server / skill (a pattern)
    sub: Optional[str] = None  # tool pattern (mcp) or heading (skill)


class RuleError(ValueError):
    pass


def parse_rule(text: str) -> Rule:
    """One `ignore` entry as a Rule. Raises RuleError for what is not understood."""
    raw = str(text or "").strip()
    if raw.startswith("mcp:"):
        body = raw[4:].strip()
        if not body:
            raise RuleError(f"'{raw}': name the server (mcp:<server>[.<tool>])")
        server, _, tool = body.partition(".")
        return Rule("mcp", server, tool or None)
    if raw.startswith("skill:"):
        body = raw[6:].strip()
        name, _, heading = body.partition("#")
        if not name.strip():
            raise RuleError(f"'{raw}': name the skill (skill:<name>[#<heading>])")
        return Rule("skill", name.strip(), heading.strip() or None)
    raise RuleError(f"'{raw}': a rule starts with mcp: or skill:")


def _match(pattern: str, value: str) -> bool:
    return fnmatch.fnmatchcase((value or "").lower(), (pattern or "").lower())


class Policy:
    """The rules one node works under."""

    def __init__(self, ignore=(), mcp_servers=None):
        self.rules = [parse_rule(r) for r in (ignore or ())]
        # None: any server. A list: only those (names or patterns).
        self.mcp_servers = None if mcp_servers is None else [str(s) for s in mcp_servers]

    # ── MCP ──────────────────────────────────────────────────────────────────
    def mcp_server_allowed(self, server: str) -> bool:
        if self.mcp_servers is not None and not any(_match(p, server) for p in self.mcp_servers):
            return False
        return not any(r.kind == "mcp" and r.sub is None and _match(r.name, server) for r in self.rules)

    def mcp_tool_allowed(self, server: str, tool: str) -> bool:
        if not self.mcp_server_allowed(server):
            return False
        return not any(
            r.kind == "mcp" and r.sub is not None and _match(r.name, server) and _match(r.sub, tool)
            for r in self.rules)

    # ── skills ───────────────────────────────────────────────────────────────
    def skill_allowed(self, name: str) -> bool:
        return not any(r.kind == "skill" and r.sub is None and _match(r.name, name) for r in self.rules)

    def skill_headings(self, name: str) -> list:
        return [r.sub for r in self.rules if r.kind == "skill" and r.sub and _match(r.name, name)]

    def cut_skill(self, name: str, text: str) -> str:
        """`text` without the sections the rules ignore for skill `name`."""
        headings = self.skill_headings(name)
        if not headings or not text:
            return text
        out, skipping, skip_level, fenced = [], False, 0, False
        for line in text.splitlines():
            if _FENCE.match(line):
                fenced = not fenced
            m = None if fenced else _HEADING.match(line)
            if m:
                level, title = len(m.group(1)), m.group(2)
                if skipping and level <= skip_level:
                    skipping = False
                if not skipping and any(_match(h, title) for h in headings):
                    skipping, skip_level = True, level
                    continue
            if not skipping:
                out.append(line)
        return "\n".join(out).strip("\n")

    # ── words ────────────────────────────────────────────────────────────────
    def describe(self) -> str:
        """The rules as a few lines for a worker's prompt (they are enforced anyway)."""
        lines = []
        if self.mcp_servers is not None:
            lines.append("MCP servers you may use: " + (", ".join(self.mcp_servers) or "none"))
        for r in self.rules:
            if r.kind == "mcp":
                lines.append(f"Do not use MCP {'server ' + r.name if r.sub is None else 'tool ' + r.name + '.' + r.sub}.")
            elif r.sub is None:
                lines.append(f"Do not use the skill '{r.name}'.")
            else:
                lines.append(f"Ignore the section '{r.sub}' of the skill '{r.name}'.")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return bool(self.rules) or self.mcp_servers is not None


# ── the active policy ────────────────────────────────────────────────────────────

_lock = threading.RLock()
#: (scope, policy) pairs — a policy applies to the scope it was installed in and the ones nested in it,
#: so one parallel branch's rules are not applied to its sibling (see run_scope.py).
_stack: list = []


@contextmanager
def use(policy: Optional["Policy"]):
    """Install `policy` for the duration of the block (a node's work)."""
    if not policy:
        yield
        return
    import run_scope
    entry = (run_scope.current(), policy)
    with _lock:
        _stack.append(entry)
    try:
        yield
    finally:
        with _lock:
            for i in range(len(_stack) - 1, -1, -1):
                if _stack[i] is entry:
                    del _stack[i]
                    break


def _active() -> list:
    import run_scope
    with _lock:
        return [p for owner, p in _stack if run_scope.visible(owner)]


def mcp_tool_blocked(server: str, tool: str = "") -> Optional[str]:
    """A reason when the active policy forbids the call, else None."""
    for p in _active():
        if tool:
            if not p.mcp_tool_allowed(server, tool):
                return f"MCP tool '{server}.{tool}' is ignored by this room's rules and cannot be used here."
        elif not p.mcp_server_allowed(server):
            return f"MCP server '{server}' is ignored by this room's rules and cannot be used here."
    return None


def skill_blocked(name: str) -> Optional[str]:
    for p in _active():
        if not p.skill_allowed(name):
            return f"Skill '{name}' is ignored by this room's rules and cannot be used here."
    return None


def cut_skill(name: str, text: str) -> str:
    """`text` as the active policy lets this node read skill `name`."""
    for p in _active():
        text = p.cut_skill(name, text)
    return text


def validate_rules(rules) -> list:
    """Problems with a node's `ignore` entries, as readable text."""
    errors = []
    for r in rules or ():
        try:
            parse_rule(r)
        except RuleError as e:
            errors.append(str(e))
    return errors
