"""The `memory` tool: read, search and write the long-term notes. See src/memory."""

from logger import get_logger

log = get_logger("tools")

_ACTIONS = ("list", "read", "search", "write", "forget")


def memory(action: str = "list", name: str = "", query: str = "", description: str = "", body: str = "",
           type: str = "project", scope: str = "project") -> str:
    from src.memory import hub
    from src.memory.store import MemoryRefused, TYPES

    action = (action or "list").strip().lower()
    if not hub.enabled():
        return "Error: long-term memory is switched off (/memory on)."
    if action not in _ACTIONS:
        return f"Error: action must be one of: {', '.join(_ACTIONS)}."
    try:
        if action == "list":
            text = hub.prompt_section(6000)
            return text or "The memory has no notes yet."
        if action == "read":
            if not name.strip():
                return "Error: name is required for read."
            node = hub.find(name)
            if node is None:
                return f"Error: no note '{name}'. Use action='list' or action='search'."
            return (f"[{node.scope}] {node.name} ({node.type}, written by {node.source}, {node.updated or 'no date'})\n"
                    f"{node.description}\n\n{node.body}")
        if action == "search":
            if not query.strip():
                return "Error: query is required for search."
            found = hub.search(query, limit=5)
            if not found:
                return "Nothing found."
            lines = []
            for node in found:
                store = hub.store_for(node.scope)
                lines.append(f"- [{node.scope}] {node.name} — {node.description}\n    {store.snippet(node, query)}")
            return "\n".join(lines)
        if action == "write":
            store = hub.store_for(scope)
            if type not in TYPES:
                return f"Error: type must be one of: {', '.join(TYPES)}."
            result = store.put(name, description, body, type=type, source="agent")
            verb = "created" if result.created else ("updated" if result.changed else "unchanged")
            note = f" ({'; '.join(result.warnings)})" if result.warnings else ""
            return f"Note {store.scope}:{result.node.name} {verb}{note}."
        # forget
        if not name.strip():
            return "Error: name is required for forget."
        scope_prefix, bare = hub.split_scope(name)
        stores = [hub.store_for(scope_prefix)] if scope_prefix else hub.stores()
        for store in stores:
            if store.forget(bare, by="agent"):
                return f"Note {store.scope}:{bare} moved to the trash (the user can bring it back with /memory restore)."
        return f"Error: no note '{name}'."
    except MemoryRefused as e:
        return f"Error: {e}"
    except OSError as e:
        log.warning("memory tool failed: %s", e)
        return f"Error: the memory folder is not writable ({e})."
