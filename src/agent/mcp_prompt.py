"""The MCP section of the system prompt: a map, not a catalog.

This used to inline every tool of every server with its description and a
call example. Measured against the user's Unity server — 140 tools — that one
section was 27k characters, ~6.8k tokens, and it rode along in EVERY request:
the whole system prompt went from 9.6k to 37k characters. A turn uses one or
two of those tools, so the other 138 were paid for and never read.

So the prompt now carries only what cannot be looked up later — which servers
exist, how big they are, and what they smell like — and the model fetches
signatures on demand with `list_mcp_tools`. The names alone are enough to know
that "bake the navmesh" is something this session can actually do.

Kept deliberately stable across turns: the section is the prefix of every
request, so churn here invalidates the provider's KV cache. Tool names are
sorted rather than left in server order for that reason.
"""

# Below this a server is small enough to list in full, which saves the model a
# discovery round-trip. Above it, a sample is enough to convey what it covers.
FULL_LIST_LIMIT = 15
SAMPLE_SIZE = 10


def model_access_warning():
    """Why a perfectly healthy server can still be invisible to the model.

    'Server running, 140 tools' and 'the model can use it' are two independent
    states, and both look like ON. When they disagree the section is dropped
    wholesale — correct, because a list of servers with no way to act on them is
    worse than nothing — but silently, which cost a real debugging session.
    """
    from argent.config import get_disabled_tools, get_mcp_servers

    if not get_mcp_servers():
        return None
    if "call_mcp_tool" in (get_disabled_tools() or []):
        return ("MCP-серверы настроены, но инструмент `call_mcp_tool` отключён — "
                "модель их не видит и не может вызвать. Включите: /tools")
    return None


def _endpoint(cfg: dict) -> str:
    if cfg.get("url"):
        return cfg["url"]
    return f"{cfg.get('command', '?')} {' '.join(cfg.get('args', []))}".strip()


def _server_line(cfg: dict, names, error: str = None) -> str:
    name = cfg["name"]
    stype = cfg.get("type", "stdio")
    if error:
        # Stated plainly: a server that is configured but unreachable explains a
        # failure the model would otherwise blame on itself and retry.
        return f"- `{name}` ({stype}) — UNREACHABLE ({error}). Do not call it."
    if not names:
        # Its own state, not a synonym for "down": the server answered and offered
        # nothing. Calling it will not help; the server's own log might.
        return (f"- `{name}` ({stype}) — connected, but reports NO TOOLS. "
                f"Check the server itself before calling it.")
    if len(names) <= FULL_LIST_LIMIT:
        return f"- `{name}` ({stype}, {len(names)} tools): {', '.join(names)}"
    sample = ", ".join(names[:SAMPLE_SIZE])
    return (f"- `{name}` ({stype}, {len(names)} tools) — e.g. {sample}, "
            f"and {len(names) - SAMPLE_SIZE} more")


def build_mcp_section(mcp_servers, client=None) -> str:
    """One line per WORKING server. `client` is injectable for tests.

    A server that stopped answering is dropped from the section after its first
    failed listing: it cannot be called anyway, and the line cost every single
    request forever. It stays in `list_mcp_tools` and in `/mcp`, which is where
    a person looks when something is missing from the prompt.
    """
    if client is None:
        from argent.mcp_client import mcp_client as client

    lines, dropped = [], []
    for cfg in mcp_servers:
        names, error = [], None
        try:
            tools = client.list_tools(cfg["name"])
            broken = [t for t in tools if "error" in t]
            names = sorted(t["name"] for t in tools if "name" in t and "error" not in t)
            if not names and broken:
                error = str(broken[0].get("error", "unknown error"))[:80]
        except Exception as e:
            error = str(e)[:80]
        if error:
            dropped.append(cfg["name"])
            continue
        lines.append(_server_line(cfg, names, None))

    if not lines:
        return ""

    head = [
        "## MCP SERVERS (external tools)",
        "These servers are connected to this session. Their tool signatures are NOT "
        "listed here — one server can expose hundreds, and carrying them all would "
        "cost more than the whole rest of these instructions.",
        "- `list_mcp_tools(server_name, filter)` — look up the tools you need, by topic.",
        "- `call_mcp_tool(server_name, tool_name, arguments)` — then run one.",
        "When a request is clearly the job of one of these servers, use it without "
        "being told to.",
        "",
    ]
    return "\n".join(head + lines + [""])
