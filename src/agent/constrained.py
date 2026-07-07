"""
Grammar-constrained agent steps for small local models.

Instead of parsing (and repairing) free-form output after the fact, tiny
models on Ollama get a JSON schema passed to the decoder via the `format`
parameter: the model physically cannot emit an invalid step. Every step is
one of:

    {"reply": "<text for the user>"}
    {"tool": {"name": "<tool>", "arguments": {...}}}

StepStreamExtractor keeps the streaming UX alive: while the constrained JSON
arrives chunk by chunk, the contents of the "reply" string are decoded and
emitted incrementally, so the user still sees text appear token by token.
"""

import json


def build_step_schema() -> dict:
    """JSON schema for one agent step, passed to Ollama's `format`."""
    return {
        "anyOf": [
            {
                "type": "object",
                "properties": {"reply": {"type": "string"}},
                "required": ["reply"],
            },
            {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["name", "arguments"],
                    }
                },
                "required": ["tool"],
            },
        ]
    }


def build_tool_catalog(tool_schemas: list) -> str:
    """Compact in-context tool catalog for models without a native tool channel.

    One line per tool: name(param, optional?) — first sentence of description.
    Placed near the end of the prompt, where small models actually attend.
    """
    lines = ["## AVAILABLE TOOLS (use EXACT names and parameter keys in your tool steps)"]
    for s in tool_schemas:
        fn = s.get("function", {})
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters", {}) or {}
        props = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])
        params = ", ".join(p if p in required else f"{p}?" for p in props)
        desc = (fn.get("description") or "").split(". ")[0].strip()[:120]
        lines.append(f"- {name}({params}) — {desc}")

    has_write = any((s.get("function", {}) or {}).get("name") == "write_file" for s in tool_schemas)
    if has_write:
        lines.append(
            "\nTIP — to write a file WITHOUT any escaping, use a verbatim block instead of a "
            "JSON tool call (content is saved exactly as written — no \\n, no doubled \\\\):\n"
            "```write_file C:/path/to/file.ext\n<file content, exactly as it should be on disk>\n```\n"
            "Use forward slashes in paths to avoid backslash escaping."
        )
    return "\n".join(lines)


_REPLY_KEY = '"reply"'
_TOOL_KEY = '"tool"'

_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/", "b": "\b",
    "f": "\f", "n": "\n", "r": "\r", "t": "\t",
}


class StepStreamExtractor:
    """Incrementally extracts the "reply" string from streaming step JSON.

    mode is None until the step branch is known, then "reply" or "tool".
    feed() returns decoded reply text that is safe to show immediately;
    finalize() parses the full buffer into the final step dict.
    """

    def __init__(self):
        self.buffer = ""
        self.mode = None          # None | "reply" | "tool"
        self._in_string = False   # currently inside the reply string value
        self._string_done = False
        self._pending = ""        # incomplete escape sequence tail

    @property
    def size(self) -> int:
        return len(self.buffer)

    def feed(self, chunk: str) -> str:
        """Consume a raw chunk, return reply text ready for display ("" if none)."""
        if not chunk:
            return ""
        self.buffer += chunk

        if self.mode is None:
            self._detect_mode()
        if self.mode != "reply" or self._string_done:
            return ""
        return self._drain_reply()

    def _detect_mode(self):
        reply_at = self.buffer.find(_REPLY_KEY)
        tool_at = self.buffer.find(_TOOL_KEY)
        if reply_at == -1 and tool_at == -1:
            return
        if reply_at != -1 and (tool_at == -1 or reply_at < tool_at):
            # Enter reply mode once the opening quote of the value arrives.
            after = self.buffer[reply_at + len(_REPLY_KEY):]
            quote = after.find('"')
            if quote != -1:
                self.mode = "reply"
                self._in_string = True
                # Everything after the opening quote is string payload.
                self._payload_start = reply_at + len(_REPLY_KEY) + quote + 1
                self._consumed = self._payload_start
        else:
            self.mode = "tool"

    def _drain_reply(self) -> str:
        """Decode available reply-string characters, handling JSON escapes."""
        out = []
        raw = self._pending + self.buffer[self._consumed:]
        self._pending = ""
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\":
                if i + 1 >= len(raw):
                    self._pending = raw[i:]
                    i = len(raw)
                    break
                esc = raw[i + 1]
                if esc == "u":
                    if i + 6 > len(raw):
                        self._pending = raw[i:]
                        i = len(raw)
                        break
                    try:
                        out.append(chr(int(raw[i + 2:i + 6], 16)))
                    except ValueError:
                        out.append(raw[i:i + 6])
                    i += 6
                else:
                    out.append(_ESCAPES.get(esc, esc))
                    i += 2
            elif ch == '"':
                # Unescaped quote: the reply string is finished.
                self._string_done = True
                self._in_string = False
                i += 1
                break
            else:
                out.append(ch)
                i += 1
        self._consumed = len(self.buffer) - (len(raw) - i)
        return "".join(out)

    def finalize(self) -> dict | None:
        """Parse the complete buffer into {"reply": ...} or {"tool": {...}}."""
        text = self.buffer.strip()
        if not text:
            return None
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                import json5
                parsed = json5.loads(text)
            except Exception:
                pass
        if not isinstance(parsed, dict):
            return None

        tool = parsed.get("tool")
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            args = tool.get("arguments")
            return {"tool": {"name": tool["name"], "arguments": args if isinstance(args, dict) else {}}}
        # Some models flatten the tool branch: {"name": ..., "arguments": ...}
        if isinstance(parsed.get("name"), str) and "arguments" in parsed:
            args = parsed.get("arguments")
            return {"tool": {"name": parsed["name"], "arguments": args if isinstance(args, dict) else {}}}
        if isinstance(parsed.get("reply"), str):
            return {"reply": parsed["reply"]}
        return None
