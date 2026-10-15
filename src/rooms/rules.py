"""Room-wide rules, applied where a step runs.

A room may say once, for all of its steps: no tools at all, nothing that changes the world,
these tools are off. The rules travel with the run state (`State.rules`), so a room called by
another room works under the stricter of the two, and they are checked in two places:

  * when a room is saved (`static_problems`: a step that names a switched-off tool is an error
    you can see in the editor), and
  * when a step is about to run (`Rules.node_block`, `Rules.agent_tools`): a step the rules
    forbid does not run — it leaves an explanation instead of a result, and the run goes on.

A model that is told "do not write files" and a model that has no write tool are different
things; this module is the second.
"""

from typing import Optional

#: Tools that change files, run commands or act on the outside world. Plan mode's list, plus the
#: ones that are mutating in a room's sense (a shell can do anything; memory writes to disk).
_EXTRA_WRITE_TOOLS = frozenset({"run_command", "memory"})

#: Leeth modes that only read / compute (the rest press buttons, type into forms or run tasks).
_LEETH_PURE = frozenset({"filter", "classify", "scan"})
_LEETH_TOOL = {"delegate": "delegate_to_leeth", "filter": "leeth_map", "classify": "leeth_map",
               "scan": "leeth_scan_page", "fill": "leeth_fill_form", "click": "leeth_click_many"}


def write_tools() -> frozenset:
    from src.agent.plan_mode import MUTATING_TOOLS
    return frozenset(MUTATING_TOOLS) | _EXTRA_WRITE_TOOLS


def merge(parent: dict, own) -> dict:
    """The rules for a room run: what the caller already imposed, tightened by the room's own
    (`own` is a RoomRules or None). Restrictions add up; model parameters and the hint of the
    room itself win over the caller's."""
    base = dict(parent or {})
    base.setdefault("deny_tools", [])
    base.setdefault("ignore", [])
    if own is None:
        return base
    data = own.model_dump() if hasattr(own, "model_dump") else dict(own)
    out = dict(base)
    out["no_tools"] = bool(base.get("no_tools") or data.get("no_tools"))
    out["read_only"] = bool(base.get("read_only") or data.get("read_only"))
    out["deny_tools"] = sorted(set(base.get("deny_tools") or []) | set(data.get("deny_tools") or []))
    out["ignore"] = list(dict.fromkeys(list(base.get("ignore") or []) + list(data.get("ignore") or [])))
    out["memory"] = data.get("memory") or base.get("memory") or "isolated"
    out["memory_chars"] = data.get("memory_chars") or base.get("memory_chars") or 6000
    out["sampling"] = {**(base.get("sampling") or {}), **(data.get("sampling") or {})}
    out["hint"] = data.get("hint") or None
    return out


class Rules:
    """A read-only view of the rules dict a run carries."""

    def __init__(self, data: Optional[dict] = None):
        self.data = dict(data or {})

    @property
    def no_tools(self) -> bool:
        return bool(self.data.get("no_tools"))

    @property
    def read_only(self) -> bool:
        return bool(self.data.get("read_only"))

    @property
    def deny_tools(self) -> list:
        return list(self.data.get("deny_tools") or [])

    @property
    def ignore(self) -> list:
        return list(self.data.get("ignore") or [])

    @property
    def sampling(self) -> dict:
        return dict(self.data.get("sampling") or {})

    @property
    def memory(self) -> str:
        return self.data.get("memory") or "isolated"

    @property
    def memory_chars(self) -> int:
        return int(self.data.get("memory_chars") or 6000)

    @property
    def hint(self) -> Optional[str]:
        return self.data.get("hint") or None

    def restricts(self) -> bool:
        return bool(self.no_tools or self.read_only or self.deny_tools)

    # ── questions ─────────────────────────────────────────────────────────────
    def tool_block(self, tool: str) -> Optional[str]:
        """Why `tool` may not be called under these rules, or None."""
        if self.no_tools:
            return "the room's rules switch all tools off (no_tools)"
        if tool in self.deny_tools:
            return f"the room's rules switch the tool '{tool}' off (deny_tools)"
        if self.read_only and tool in write_tools():
            return f"the room is read-only, and '{tool}' changes things (read_only)"
        return None

    def node_block(self, node) -> Optional[str]:
        """Why this step may not run under these rules, or None. Only steps that act through a tool
        (a tool, an MCP call, a plugin, a Leeth action) can be blocked; thinking, notifying,
        asking and pausing never are."""
        t = node.type
        if t == "tool":
            return self.tool_block(node.tool)
        if t == "mcp":
            tool = "list_mcp_tools" if (node.mode or "call") == "list" else "call_mcp_tool"
            return self.tool_block(tool)
        if t == "plugin":
            if self.no_tools:
                return "the room's rules switch all tools off (no_tools)"
            if self.read_only:
                return "the room is read-only, and a plugin may change anything (read_only)"
            if "plugin" in self.deny_tools:
                return "the room's rules switch plugins off (deny_tools)"
            return None
        if t == "leeth":
            mode = node.mode or "delegate"
            tool = _LEETH_TOOL.get(mode, "delegate_to_leeth")
            if self.no_tools:
                return "the room's rules switch all tools off (no_tools)"
            if tool in self.deny_tools or "leeth" in self.deny_tools:
                return f"the room's rules switch Leeth ({mode}) off (deny_tools)"
            if self.read_only and mode not in _LEETH_PURE:
                return f"the room is read-only, and Leeth '{mode}' acts (read_only)"
            return None
        return None

    def agent_tools(self, tools) -> Optional[list]:
        """The tools an agent step may have: its own list (None = everything) minus what the rules
        switch off. Unchanged (None stays None) when the rules restrict nothing."""
        if not self.restricts():
            return None if tools is None else list(tools)
        if self.no_tools:
            return []
        if tools is None:
            from tools.schemas import AVAILABLE_TOOLS
            tools = list(AVAILABLE_TOOLS)
        return [t for t in tools if self.tool_block(t) is None]


def _view(room) -> Optional["Rules"]:
    rules = getattr(room, "rules", None)
    if rules is None:
        return None
    view = Rules(rules.model_dump())
    return view if view.restricts() else None


def static_problems(room) -> list:
    """Steps of a room that its own rules forbid (an error when the room is saved: the editor shows
    which step, instead of the step silently doing nothing at run time)."""
    view = _view(room)
    if view is None:
        return []
    out = []
    for n in room.nodes:
        if n.disabled:
            continue
        why = view.node_block(n)
        if why:
            what = {"tool": f"tool '{n.tool}'", "mcp": f"MCP call {n.server}.{n.tool or '(list)'}",
                    "plugin": f"plugin command '{n.command}'", "leeth": f"Leeth ({n.mode or 'delegate'})"}.get(n.type, n.type)
            out.append(f"node '{n.id}': {what} cannot run — {why}")
    return out


def static_warnings(room) -> list:
    """Things the rules will quietly change: an agent that lists tools the room switches off."""
    view = _view(room)
    if view is None:
        return []
    out = []
    for n in room.nodes:
        if n.type == "agent" and n.tools and not n.disabled:
            gone = [t for t in n.tools if view.tool_block(t)]
            if gone:
                out.append(f"agent '{n.id}': {', '.join(gone)} {'is' if len(gone) == 1 else 'are'} switched off by the "
                           "room's rules and will be taken away from it")
    return out
