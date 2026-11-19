import json
import re
from argent.tools import get_available_tools

# Parameter name aliases for known tools.
#
# The canonical name for "the file or directory this tool works on" is `path`. These
# tables exist because models reach for `filename`, `file`, `dir`, `directory` or the
# old `file_path`; what they normalise TO is the schema's own name, which is `path`
# everywhere. If the target were still `file_path`, a call that arrived as `filename`
# would be translated into a parameter the tools no longer declare.
_PATH_ALIASES = {
    "filename": "path",
    "file_path": "path",
    "filepath": "path",
    "file": "path",
    "dir_path": "path",
    "dirpath": "path",
    "directory": "path",
    "directory_path": "path",
    "dir": "path",
}

_PARAM_ALIASES = {
    "write_file": dict(_PATH_ALIASES),
    "read_file": dict(_PATH_ALIASES),
    "delete_file": dict(_PATH_ALIASES),
    "replace_in_file": dict(_PATH_ALIASES),
    "replace_python_function": dict(_PATH_ALIASES),
    "multi_replace_in_file": {
        "changes": "changes_json",
        "replacements": "changes_json",
        "edits": "changes_json",
    },
    "create_directory": dict(_PATH_ALIASES),
    "list_directory": dict(_PATH_ALIASES),
    "get_file_outline": dict(_PATH_ALIASES),
    "view_image": dict(_PATH_ALIASES),
    "analyze_project": dict(_PATH_ALIASES),
    "run_deep_linter": dict(_PATH_ALIASES),
    "grep_search": {
        **_PATH_ALIASES,
        "search_path": "path",
        "searchpath": "path",
        "query": "pattern",
        "regex": "pattern",
        "search_query": "pattern",
        # grep / ripgrep flag names a model may reach for
        "whole_words": "whole_word",
        "word_regexp": "whole_word",
        "match_whole_word": "whole_word",
        "fixed_strings": "literal",
        "fixed_string": "literal",
        "match_case": "case_sensitive",
    },
    "search_files": dict(_PATH_ALIASES),
    "move_file": {
        "src": "source",
        "source_path": "source",
        "dst": "destination",
        "dest": "destination",
        "destination_path": "destination",
    },
    "copy_file": {
        "src": "source",
        "source_path": "source",
        "dst": "destination",
        "dest": "destination",
        "destination_path": "destination",
    },
}

def normalize_tool_params(parsed: dict) -> dict:
    """Normalize parameter names to match expected tool schemas."""
    if not isinstance(parsed, dict):
        return parsed
    tool_name = parsed.get("name", "")
    arguments = parsed.get("arguments")
    if not isinstance(arguments, dict):
        return parsed
    aliases = _PARAM_ALIASES.get(tool_name)
    if aliases:
        normalized = {}
        for key, value in arguments.items():
            canonical = aliases.get(key, key)
            if canonical in normalized:
                # If canonical already holds a non-empty value, an alias cannot overwrite it
                if normalized[canonical] and (key != canonical or value is None or value == ""):
                    continue
                # If current has value and existing was empty/None, overwrite
                if value is not None and value != "":
                    normalized[canonical] = value
            else:
                normalized[canonical] = value
        parsed["arguments"] = normalized
    return parsed

def extract_balanced_json(text: str, start_pos: int) -> str | None:
    """Extract a balanced JSON object from text starting at start_pos.
    Uses brace-counting to handle nested objects correctly.
    Returns the complete JSON string or None if no balanced object found."""
    if start_pos >= len(text) or text[start_pos] != '{':
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start_pos, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start_pos:i + 1]
    return None

def decode_json_escapes(s: str) -> str:
    r"""Decode JSON string escapes (\n \t \r \" \\ \/ \b \f \uXXXX) from a raw
    string extracted from malformed JSON.

    Deliberately does NOT use codecs 'unicode_escape': that codec is latin-1
    based and mojibakes any non-ASCII content (e.g. Cyrillic), silently
    corrupting recovered file content. This decodes char-by-char so real UTF-8
    text passes through untouched."""
    simple = {'n': '\n', 't': '\t', 'r': '\r', '"': '"',
              '\\': '\\', '/': '/', 'b': '\b', 'f': '\f'}
    out = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch != '\\' or i + 1 >= n:
            out.append(ch)
            i += 1
            continue
        nxt = s[i + 1]
        if nxt in simple:
            out.append(simple[nxt])
            i += 2
            continue
        if nxt == 'u' and i + 6 <= n:
            try:
                cp = int(s[i + 2:i + 6], 16)
            except ValueError:
                out.append(nxt)
                i += 2
                continue
            # Combine a UTF-16 surrogate pair (an emoji) into one codepoint.
            if 0xD800 <= cp <= 0xDBFF and s[i + 6:i + 8] == '\\u':
                try:
                    lo = int(s[i + 8:i + 12], 16)
                    if 0xDC00 <= lo <= 0xDFFF:
                        out.append(chr(0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)))
                        i += 12
                        continue
                except ValueError:
                    pass
            out.append(chr(cp))
            i += 6
            continue
        out.append(nxt)  # unknown escape — keep the char literally
        i += 2
    # A \ud83d whose partner never arrived (truncated output, a split stream
    # chunk, a malformed low escape) becomes a lone surrogate here, and a lone
    # surrogate cannot be encoded as UTF-8. Left in, it poisons the history and
    # every LATER request dies on encoding — for a character nobody sees.
    from argent.text_safety import repair_surrogates
    return repair_surrogates(''.join(out))

# The body is GREEDY and the closing fence must be a line that is exactly ```
# (optional trailing spaces): so a file whose content itself contains ``` code
# fences is captured whole, up to the LAST bare fence, instead of being
# silently truncated at the first inner fence. MULTILINE anchors the close to a
# line boundary; DOTALL lets the body span newlines.
_FENCED_WRITE_RE = re.compile(
    r"```write_file[ \t]+(?P<path>[^\n`]+?)[ \t]*\n(?P<body>.*)\n```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)


def parse_fenced_write(content: str) -> dict | None:
    r"""Detect a verbatim write_file block that avoids ALL JSON escaping:

        ```write_file <path>
        <raw file content — any newlines/quotes/backslashes, written as-is>
        ```

    The content is taken verbatim, so the model never has to escape \n, \\ or
    quotes (the token-wasting, error-prone part of JSON tool args). The info
    string must be exactly ``write_file <path>``, so it can't misfire on ordinary
    ```code``` blocks. Returns {"parsed": {...}, "match_str": <block>} or None.
    """
    if not content or "```write_file" not in content:
        return None
    m = _FENCED_WRITE_RE.search(content)
    if not m:
        return None
    path = m.group("path").strip().strip('"').strip("'").strip("`").strip()
    if not path:
        return None
    return {
        "parsed": {"name": "write_file",
                   "arguments": {"path": path, "content": m.group("body")}},
        "match_str": m.group(0),
    }


# --- FORMAT 0: a tool call written as XML-ish tags -------------------------------
# `<create_artifact><parameter name="filename">x</parameter></create_artifact>` is the
# shape a model falls back to when it misses the tool-call channel. Rendered as text,
# it was taken for a final answer: the turn ended and the work was never done.
_XML_PARAM_RE = re.compile(
    r'<\s*parameter\s+name\s*=\s*"([^"]+)"\s*>(.*?)<\s*/\s*parameter\s*>',
    re.DOTALL | re.IGNORECASE)
_XML_ATTR_RE = re.compile(r'([A-Za-z_][\w\-.]*)\s*=\s*"([^"]*)"', re.DOTALL)


def _xml_attrs(text: str) -> dict:
    return {m.group(1): m.group(2) for m in _XML_ATTR_RE.finditer(text or "")}


def parse_xml_tool_call(content: str, tool_names=None) -> dict | None:
    """A call written as `<tool>…</tool>` or `<tool a="1"/>`, read back into a call.

    Only tags whose name is a real tool are considered, so ordinary markup in an
    answer (`<div>`, `<b>`) is never mistaken for one.
    """
    if not content or "<" not in content:
        return None
    names = sorted(get_available_tools() if tool_names is None else tool_names,
                   key=len, reverse=True)
    for name in names:
        tag = re.escape(name)
        block = re.search(r'<\s*' + tag + r'\b([^>]*)>(.*?)<\s*/\s*' + tag + r'\s*>',
                          content, re.DOTALL | re.IGNORECASE)
        if block:
            args = {m.group(1): m.group(2).strip() for m in _XML_PARAM_RE.finditer(block.group(2))}
            if not args:
                args = _xml_attrs(block.group(1))
            if args:
                return {"parsed": {"name": name, "arguments": args}, "match_str": block.group(0)}
            continue
        empty = re.search(r'<\s*' + tag + r'\b([^>]*)/>', content, re.DOTALL | re.IGNORECASE)
        if empty:
            args = _xml_attrs(empty.group(1))
            if args:
                return {"parsed": {"name": name, "arguments": args}, "match_str": empty.group(0)}
    return None


def looks_like_tool_call_text(content: str, tool_names=None) -> bool:
    """Whether this text is a tool call the model wrote instead of making one.

    Used where a turn decides it has its final answer: such text is not an answer,
    and accepting it ends the run with nothing done.
    """
    if not content or "<" not in content:
        return False
    names = get_available_tools() if tool_names is None else tool_names
    lowered = content.lower()
    for name in names:
        tag = name.lower()
        if f"<{tag}" in lowered or f"< {tag}" in lowered:
            return True
    return False


def parse_raw_tool_call(content: str) -> dict | None:
    """Parse a raw tool call from AI-generated content.
    Returns {"parsed": {"name": ..., "arguments": {...}}, "match_str": ...} or None."""
    # Verbatim write_file block first — unambiguous and escaping-free.
    fenced = parse_fenced_write(content)
    if fenced:
        return fenced

    # XML-ish tool tags: unambiguous too, and the shape a model reaches for when the
    # native channel fails. Checked before JSON, which is easy to fake in prose.
    xml_call = parse_xml_tool_call(content)
    if xml_call:
        return xml_call

    clean = content.strip()
    
    # --- FORMAT 1: Markdown code blocks ---
    md_match = re.search(r'```(?:json)?\s*(\{.+)', clean, re.DOTALL)
    if md_match:
        # Extract balanced JSON from inside the code block
        inner_start = md_match.start(1)
        json_candidate = extract_balanced_json(clean, inner_start)
        if json_candidate:
            # Find the closing ``` to determine the full match string
            end_pos = inner_start + len(json_candidate)
            closing = clean.find('```', end_pos)
            if closing != -1:
                match_str = clean[md_match.start():closing + 3]
            else:
                match_str = clean[md_match.start():end_pos]
            
            result = try_parse_json_tool(json_candidate)
            if result:
                result["match_str"] = match_str
                return result

    # --- FORMAT 2, 3, 4: Find any JSON object in the content ---
    # Look for the first { that could be the start of a tool call JSON
    brace_pos = clean.find('{')
    if brace_pos != -1:
        json_candidate = extract_balanced_json(clean, brace_pos)
        if json_candidate:
            result = try_parse_json_tool(json_candidate)
            if result:
                result["match_str"] = json_candidate
                return result

    # --- LAST RESORT: Regex extraction for heavily malformed JSON ---
    current_tools = get_available_tools()
    for tool_name in current_tools:
        if f'"{tool_name}"' in clean:
            result = try_recover_malformed_tool(clean, tool_name)
            if result:
                return result
    
    return None

def repair_json_strings(json_str: str) -> str:
    """Escapes literal newlines and tabs inside JSON string values."""
    chars = []
    in_string = False
    escape = False
    for char in json_str:
        if char == '"' and not escape:
            in_string = not in_string
            chars.append(char)
        elif char == '\\' and in_string and not escape:
            escape = True
            chars.append(char)
        else:
            if in_string:
                if char == '\n':
                    chars.append('\\n')
                elif char == '\t':
                    chars.append('\\t')
                elif char == '\r':
                    chars.append('\\r')
                else:
                    chars.append(char)
            else:
                chars.append(char)
            escape = False
    return "".join(chars)

def fix_common_json_errors(json_str: str) -> str:
    """Aggressive auto-fixing for common small model JSON errors."""
    s = json_str.strip()
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r',\s*\}', '}', s)
    s = re.sub(r',\s*\]', ']', s)
    
    # Count braces strictly OUTSIDE string literals to avoid corrupting valid code
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False
    for ch in s:
        if ch == '"' and not escape:
            in_string = not in_string
        elif ch == '\\' and in_string and not escape:
            escape = True
            continue
        elif not in_string:
            if ch == '{':
                open_braces += 1
            elif ch == '}':
                open_braces = max(0, open_braces - 1)
            elif ch == '[':
                open_brackets += 1
            elif ch == ']':
                open_brackets = max(0, open_brackets - 1)
        escape = False

    if open_brackets > 0:
        s += ']' * open_brackets
    if open_braces > 0:
        s += '}' * open_braces
    return s

def try_parse_json_tool(json_str: str) -> dict | None:
    """Try to parse a JSON string as a tool call."""
    # First repair control chars inside JSON strings
    repaired_json = repair_json_strings(json_str)
    try:
        parsed = json.loads(repaired_json)
    except json.JSONDecodeError:
        try:
            # Fall back to aggressive structural error fixing only if raw parse failed
            fixed_json = fix_common_json_errors(repaired_json)
            parsed = json.loads(fixed_json)
        except json.JSONDecodeError:
            return None
    
    if not isinstance(parsed, dict):
        return None
    
    # Format: {"name": ..., "arguments": {...}} or {"function": ..., "arguments": {...}}
    tool_id = parsed.get("name") or parsed.get("function")
    if tool_id and isinstance(tool_id, str) and "arguments" in parsed:
        parsed["name"] = tool_id # Canonicalize
        parsed = normalize_tool_params(parsed)
        return {"parsed": parsed}
    
    # Format: {"tool_name": {params}} (shorthand)
    current_tools = get_available_tools()
    for key, value in parsed.items():
        if key in current_tools and isinstance(value, dict):
            tool_parsed = {
                "name": key,
                "arguments": value
            }
            tool_parsed = normalize_tool_params(tool_parsed)
            return {"parsed": tool_parsed}
    
    return None

def try_recover_malformed_tool(content: str, tool_name: str) -> dict | None:
    """Last-resort recovery for malformed JSON (e.g., write_file with unescaped newlines)."""
    CONTENT_TOOLS = ["write_file", "replace_in_file"]
    if tool_name not in CONTENT_TOOLS:
        return None

    fp_match = re.search(
        r'"(?:file_path|filename|filepath|path|file)"\s*:\s*"([^"]+)"', content
    )
    ct_match = re.search(r'"content"\s*:\s*"([\s\S]*)', content)
    
    if not fp_match or not ct_match:
        return None
    
    recovered = ct_match.group(1)
    # Strip trailing patterns
    recovered = re.sub(r'"\s*\}\s*\}?\s*$', '', recovered)
    recovered = recovered.rstrip().rstrip('"')
    
    # Decode escapes
    recovered = decode_json_escapes(recovered)
    
    parsed = {
        "name": tool_name,
        "arguments": {
            "path": fp_match.group(1),
            "content": recovered
        }
    }
    
    return {"parsed": parsed, "match_str": content}
