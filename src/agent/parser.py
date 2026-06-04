import json
import re
import codecs
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
    """Decode JSON escape sequences in a raw string extracted from malformed JSON."""
    try:
        return codecs.decode(s, 'unicode_escape')
    except Exception:
        pass
    result = s
    result = result.replace('\\n', '\n')
    result = result.replace('\\t', '\t')
    result = result.replace('\\r', '\r')
    result = result.replace('\\"', '"')
    result = result.replace('\\\\', '\\')
    return result

def parse_raw_tool_call(content: str) -> dict | None:
    """Parse a raw tool call from AI-generated content.
    Returns {"parsed": {"name": ..., "arguments": {...}}, "match_str": ...} or None."""
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
    CONTENT_TOOLS = ["write_file", "write_obsidian_note", "replace_python_function", "replace_in_file"]
    if tool_name not in CONTENT_TOOLS:
        return None
    
    fp_match = re.search(
        r'"(?:file_path|filename|filepath|path|file|note_path)"\s*:\s*"([^"]+)"', content
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
    
    path_param = "file_path"
    if tool_name == "write_obsidian_note":
        path_param = "note_path"
        
    parsed = {
        "name": tool_name,
        "arguments": {
            path_param: fp_match.group(1),
            "content": recovered
        }
    }
    
    return {"parsed": parsed, "match_str": content}
