"""
Salvage of a truncated file-writing tool call.

When a model writes a large file via write_file/append_to_file and the
generation hits the length limit, the tool-call JSON is cut off mid-string.
Instead of throwing that work away and asking the model to "start over in
chunks" (which a weak model rarely manages), we extract the partial content
that WAS generated, write it to the file, and ask the model to continue with
append_to_file. Repeated truncation then becomes incremental progress, so a
file of any size can be produced even on a small context window.
"""

import re

# Tool calls whose content can be salvaged and continued via append.
SALVAGEABLE_TOOLS = ("write_file", "append_to_file", "write_obsidian_note")

_PATH_RE = re.compile(r'"(file_path|filename|filepath|path|file|note_path)"\s*:\s*"([^"]*)"')
# Grab everything after the opening quote of "content": "  (greedy to EOF).
_CONTENT_RE = re.compile(r'"content"\s*:\s*"([\s\S]*)$')


def _decode_partial(raw: str) -> str:
    """Undo JSON string escapes in a partial (unterminated) content value."""
    # Drop a dangling backslash that would belong to an incomplete escape.
    if raw.endswith("\\") and not raw.endswith("\\\\"):
        raw = raw[:-1]
    return (
        raw.replace("\\n", "\n").replace("\\t", "\t")
           .replace("\\r", "\r").replace('\\"', '"').replace("\\\\", "\\")
    )


def extract_partial_write(arguments):
    """From a truncated tool call's arguments, return (file_path, partial_content)
    or None if it isn't a recoverable content write.

    `arguments` may be the raw partial JSON string (the usual truncation case)
    or an already-parsed dict."""
    if isinstance(arguments, dict):
        path = next((arguments[k] for k in
                     ("file_path", "filename", "filepath", "path", "file", "note_path")
                     if arguments.get(k)), None)
        if path and "content" in arguments:
            return str(path), str(arguments.get("content") or "")
        return None

    s = str(arguments)
    pm = _PATH_RE.search(s)
    cm = _CONTENT_RE.search(s)
    if not pm or not cm:
        return None
    file_path = pm.group(2)
    if not file_path:
        return None
    return file_path, _decode_partial(cm.group(1))


def trim_to_last_line(content: str) -> str:
    """Keep only complete lines: drop a half-generated trailing line so the
    file never ends mid-token. If there's only one line, keep it as-is."""
    idx = content.rfind("\n")
    if idx == -1:
        return content
    return content[:idx + 1]


def last_lines(content: str, n: int = 2) -> str:
    """The final n non-empty lines, to orient the model on where to continue."""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])
