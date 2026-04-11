"""
Tool Call Recovery Layer for Argent.
Helps small/local models that produce malformed or misspelled tool calls.
"""

import json
import re
import difflib
from typing import Optional, Dict, Any

from logger import get_logger

log = get_logger("tool_recovery")


def fuzzy_match_tool(name: str, available_tools: dict, cutoff: float = 0.6) -> Optional[str]:
    """Find the closest matching tool name using fuzzy string matching."""
    if name in available_tools:
        return name
    
    matches = difflib.get_close_matches(name, available_tools.keys(), n=1, cutoff=cutoff)
    if matches:
        log.info("Fuzzy matched tool '%s' -> '%s'", name, matches[0])
        return matches[0]
    
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    for tool_name in available_tools:
        if tool_name.lower().replace("-", "_") == name_lower:
            log.info("Normalized tool '%s' -> '%s'", name, tool_name)
            return tool_name
    
    return None


def recover_json_arguments(raw_args: str) -> Optional[Dict[str, Any]]:
    """Attempt to recover a valid JSON dict from malformed tool arguments.
    Handles common mistakes: missing quotes, trailing commas, single quotes, etc.
    """
    if not raw_args:
        return {}
    
    if isinstance(raw_args, dict):
        return raw_args
    
    text = str(raw_args).strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    try:
        fixed = text.replace("'", '"')
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    try:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            fragment = text[start:end + 1]
            fragment = fragment.replace("'", '"')
            fragment = re.sub(r',\s*}', '}', fragment)
            return json.loads(fragment)
    except json.JSONDecodeError:
        pass
    
    args = {}
    kv_pattern = re.findall(r'"?(\w+)"?\s*[:=]\s*"([^"]*)"', text)
    for key, value in kv_pattern:
        args[key] = value
    
    if args:
        log.info("Recovered %d args via regex extraction", len(args))
        return args
    
    log.warning("Failed to recover JSON from: %s", text[:100])
    return None


def recover_tool_call(raw_tool: dict, available_tools: dict) -> Optional[dict]:
    """Full recovery pipeline for a single tool call.
    Returns a corrected tool call dict or None if unrecoverable.
    """
    func = raw_tool.get("function", raw_tool)
    name = func.get("name", "")
    raw_args = func.get("arguments", {})
    
    matched_name = fuzzy_match_tool(name, available_tools)
    if not matched_name:
        log.warning("Tool '%s' not found and no fuzzy match", name)
        return None
    
    if isinstance(raw_args, str):
        recovered_args = recover_json_arguments(raw_args)
        if recovered_args is None:
            recovered_args = {}
    elif isinstance(raw_args, dict):
        recovered_args = raw_args
    else:
        recovered_args = {}
    
    return {
        "function": {
            "name": matched_name,
            "arguments": recovered_args,
        }
    }
