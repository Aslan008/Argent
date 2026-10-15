"""Make the model's arguments the types the schema promised.

Models emit JSON, and they emit it slightly wrong all the time: `"5"` for an
integer, `"false"` for a boolean, a bare string where an array was declared.
Python does not care until it does, and then the failure is rarely honest:

* `write_file(overwrite="false")` — measured: a 400-line file that
  `overwrite=False` protects is silently destroyed, because a non-empty string
  is truthy and `not overwrite` becomes False. The guard exists precisely to
  prevent that write, and the wrong type turns it off.
* `grep_search(max_results="3")` — measured: reports "No matches found" for a
  pattern that has three matches. The model then believes the code does not
  contain what it was looking for.
* `read_file(start_line="5")` — leaks "'>' not supported between instances of
  'str' and 'int'" at a model that has no idea what that means.

Coercing here rather than in twenty tools is the point: this is where the
model's output meets Python, and every present and future tool crosses it.

Nothing is invented. A value that cannot be converted is passed through
untouched, so the tool still gets to reject it — better a tool's own error
message than a guess made in this file.
"""

import json

from logger import get_logger

log = get_logger("tools")

_TRUE = {"true", "yes", "y", "1", "on", "да"}
_FALSE = {"false", "no", "n", "0", "off", "нет", "none", "null"}


def _schema_types(tool_name: str) -> dict:
    """{parameter: declared type} for one tool, or {} if it has no schema."""
    from tools.schemas import TOOL_SCHEMAS
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function") or {}
        if fn.get("name") != tool_name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        return {name: spec.get("type") for name, spec in props.items() if spec.get("type")}
    return {}


def _to_bool(value):
    if isinstance(value, bool):
        return value, False
    if isinstance(value, (int, float)):
        return bool(value), True
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True, True
        if text in _FALSE:
            return False, True
    return value, False


def _to_number(value, want_int: bool):
    if isinstance(value, bool):
        return value, False           # True is not 1 here; that is a real mistake
    if isinstance(value, int) and want_int:
        return value, False
    if isinstance(value, (int, float)) and not want_int:
        return value, False
    if isinstance(value, float) and want_int:
        return (int(value), True) if value.is_integer() else (value, False)
    if isinstance(value, str):
        text = value.strip()
        try:
            return (int(text), True) if want_int else (float(text), True)
        except ValueError:
            # "10.0" for an integer field is still unambiguous.
            try:
                as_float = float(text)
            except ValueError:
                return value, False
            if want_int and as_float.is_integer():
                return int(as_float), True
    return value, False


def _to_array(value):
    if isinstance(value, list):
        return value, False
    if isinstance(value, tuple):
        return list(value), True
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed, True
            except json.JSONDecodeError:
                pass
        if "\n" in text:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                return lines, True
        # One item where a list was declared: the model meant a list of one.
        return ([value], True) if text else (value, False)
    if isinstance(value, (int, float, bool, dict)):
        return [value], True
    return value, False


def _to_object(value):
    if isinstance(value, dict):
        return value, False
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed, True
        except json.JSONDecodeError:
            pass
    return value, False


def _to_string(value, key: str = ""):
    if isinstance(value, str):
        return value, False
    if isinstance(value, bool):
        return ("true" if value else "false"), True
    if isinstance(value, (int, float)):
        return str(value), True
    if isinstance(value, (list, dict)) and key.endswith("_json"):
        # A parameter named *_json declares a string that CONTAINS json, which
        # invites the model to send the object itself. Measured:
        # multi_replace_in_file then answers "the JSON object must be str,
        # bytes or bytearray, not list" — true, useless, and the edit is lost.
        try:
            return json.dumps(value, ensure_ascii=False), True
        except (TypeError, ValueError):
            return value, False
    return value, False


_CONVERTERS = {
    "boolean": _to_bool,
    "integer": lambda v: _to_number(v, True),
    "number": lambda v: _to_number(v, False),
    "array": _to_array,
    "object": _to_object,
    "string": _to_string,
}


TOOL_ARG_ALIASES: dict = {
    "replace_in_file": {
        "old_string": "target_text",
        "old_str": "target_text",
        "old_text": "target_text",
        "target": "target_text",
        "new_string": "replacement_text",
        "new_str": "replacement_text",
        "new_text": "replacement_text",
        "replacement": "replacement_text",
        "replace": "replacement_text",
        "path": "file_path",
        "start": "start_line",
        "from_line": "start_line",
        "start_lineno": "start_line",
        "end": "end_line",
        "to_line": "end_line",
        "end_lineno": "end_line",
    },
    "read_file": {
        "path": "file_path",
        "file": "file_path",
        "filename": "file_path",
    },
    "write_file": {
        "path": "file_path",
        "file": "file_path",
        "filename": "file_path",
        "text": "content",
    },
    "delete_file": {
        "path": "file_path",
        "file": "file_path",
    },
    "get_file_outline": {
        "path": "file_path",
        "file": "file_path",
    },
    "read_file_chunk": {
        "path": "file_path",
        "file": "file_path",
    },
    "multi_replace_in_file_chunk": {
        "path": "file_path",
        "file": "file_path",
    },
    "run_command": {
        "cmd": "command",
    },
    "start_background_command": {
        "cmd": "command",
    },
    "read_background_command": {
        "id": "task_id",
        "process_id": "task_id",
    },
    "stop_background_command": {
        "id": "task_id",
        "process_id": "task_id",
    },
    # grep_search accepts `query` and `path` itself (it treats query/pattern/symbol
    # and path/directory as the same inputs), so only the spellings it does NOT
    # know are renamed. Renaming query->pattern as well made the coercion layer
    # rewrite a call that was already valid.
    "grep_search": {
        "regex": "pattern",
        "dir": "directory",
    },
    # browser_action carries every verb's payload in `text`; these are the names a
    # model reaches for first (eval_js code, an upload path, a wait timeout).
    "browser_action": {
        "expression": "text",
        "script": "text",
        "javascript": "text",
        "js": "text",
        "code": "text",
        "file_path": "text",
        "filepath": "text",
        "path": "text",
        "timeout": "amount",
        "seconds": "amount",
    },
    "browser_open": {
        "show": "headed",
        "visible": "headed",
        "gui": "headed",
        "display": "headed",
        "window": "headed",
        "link": "url",
        "target_url": "url",
    },
}


def remap_tool_aliases(tool_name: str, arguments: dict) -> tuple:
    """Remap common argument synonyms (e.g. old_string -> target_text)."""
    if not isinstance(arguments, dict) or not arguments:
        return arguments, []

    aliases = TOOL_ARG_ALIASES.get(tool_name)
    has_headless = tool_name == "browser_open" and "headless" in arguments

    if not aliases and not has_headless:
        return arguments, []

    if aliases and not any(k in arguments for k in aliases) and not has_headless:
        return arguments, []

    out = dict(arguments)
    remapped = []

    # Models often send headless=False to mean headed=True, or headless=True for headed=False
    if has_headless:
        val, _ = _to_bool(out.pop("headless"))
        if "headed" not in out:
            out["headed"] = not bool(val)
            remapped.append(f"headless={val!r}->headed={out['headed']!r}")
        else:
            remapped.append(f"headless={val!r} dropped (headed={out['headed']!r} already present)")

    if aliases:
        for old_key, new_key in aliases.items():
            if old_key in out and new_key not in out:
                out[new_key] = out.pop(old_key)
                remapped.append(f"{old_key}->{new_key}")
    return out, remapped


def coerce_arguments(tool_name: str, arguments: dict) -> tuple:
    """(arguments with declared types applied, list of what was changed)."""
    if not isinstance(arguments, dict) or not arguments:
        return arguments, []

    # First remap any well-known parameter aliases (e.g. old_string -> target_text)
    arguments, changed = remap_tool_aliases(tool_name, arguments)

    types = _schema_types(tool_name)
    if not types:
        return arguments, changed

    out = {}
    for key, value in arguments.items():
        declared = types.get(key)
        convert = _CONVERTERS.get(declared)
        if convert is None:
            out[key] = value
            continue
        # The string converter needs the name: only a *_json parameter should
        # accept an object and serialise it.
        new_value, did = (_to_string(value, key) if declared == "string"
                          else convert(value))
        out[key] = new_value
        if did:
            changed.append(f"{key}={value!r}->{new_value!r}")
    return out, changed


def coerce_and_log(tool_name: str, arguments: dict) -> dict:
    """coerce_arguments, with a line on disk when anything had to be fixed.

    Logged because every one of these is a model getting the schema wrong. It
    works now, but a tool whose arguments are always repaired is a tool whose
    description needs rewriting, and that is invisible without the record.
    """
    coerced, changed = coerce_arguments(tool_name, arguments)
    if changed:
        log.info("coerced %s argument(s) for %s: %s",
                 len(changed), tool_name, "; ".join(changed)[:300])
    return coerced
