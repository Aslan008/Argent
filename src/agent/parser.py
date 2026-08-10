import json
import re
import os
from tools import get_available_tools

# Parameter name aliases for known tools
_PARAM_ALIASES = {
    "write_file": {
        "filename": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filepath": "file_path",
    },
    "read_file": {
        "filename": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filepath": "file_path",
    },
    "delete_file": {
        "filename": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filepath": "file_path",
    },
    "replace_in_file": {
        "filename": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filepath": "file_path",
    },
    "replace_python_function": {
        "filename": "file_path",
        "path": "file_path",
        "file": "file_path",
        "filepath": "file_path",
    },
    "create_directory": {
        "path": "dir_path",
        "directory": "dir_path",
        "dir": "dir_path",
        "directory_path": "dir_path",
    },
    "list_directory": {
        "path": "dir_path",
        "directory": "dir_path",
        "dir": "dir_path",
        "directory_path": "dir_path",
    },
    "grep_search": {
        "dir": "directory",
        "path": "directory",
        "directory_path": "directory",
    },
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
    tool_name = parsed.get("name", "")
    arguments = parsed.get("arguments", {})
    aliases = _PARAM_ALIASES.get(tool_name)
    if aliases:
        normalized = {}
        for key, value in arguments.items():
            canonical = aliases.get(key, key)
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
            # Combine a UTF-16 surrogate pair (😀) into one codepoint.
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
    from text_safety import repair_surrogates
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
                   "arguments": {"file_path": path, "content": m.group("body")}},
        "match_str": m.group(0),
    }


def parse_raw_tool_call(content: str) -> dict | None:
    """Parse a raw tool call from AI-generated content.
    Returns {"parsed": {"name": ..., "arguments": {...}}, "match_str": ...} or None."""
    # Verbatim write_file block first — unambiguous and escaping-free.
    fenced = parse_fenced_write(content)
    if fenced:
        return fenced

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
    
    # Append missing closers
    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')
    if open_brackets > 0:
        s += ']' * open_brackets
    if open_braces > 0:
        s += '}' * open_braces
    return s

def try_parse_json_tool(json_str: str) -> dict | None:
    """Try to parse a JSON string as a tool call."""
    try:
        # First repair control chars inside JSON strings
        repaired_json = repair_json_strings(json_str)
        repaired_json = fix_common_json_errors(repaired_json)
        parsed = json.loads(repaired_json)
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
    CONTENT_TOOLS = ["write_file", "replace_python_function", "replace_in_file"]
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
            "file_path": fp_match.group(1),
            "content": recovered
        }
    }
    
    return {"parsed": parsed, "match_str": content}
