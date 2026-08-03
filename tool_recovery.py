"""
Tool Call Recovery Layer for Argent.
Helps small/local models that produce malformed or misspelled tool calls.
"""

import json
import re
import difflib
import inspect
from typing import Optional, Dict, Any

from logger import get_logger

log = get_logger("tool_recovery")


# Similarity floor for the last-resort fuzzy pass.
#
# Measured against this registry: genuine typos score 0.73-0.97
# ('read_fil'->'read_file' 0.94, 'runCommand'->'run_command' 0.86), while
# names of tools we simply do NOT have score 0.76-0.80 against an unrelated
# neighbour ('edit_file'->'read_file' 0.78, 'create_file'->'read_file' 0.80,
# 'list_files'->'list_skills' 0.76). The two ranges OVERLAP, so no threshold
# separates them cleanly — 0.6 was low enough to silently turn a write into a
# read. 0.8 keeps the clear typos and rejects most foreign names; the residual
# risk is handled by reporting every substitution back to the model (see
# recover_tool_call), so a wrong guess is visible rather than silent.
FUZZY_CUTOFF = 0.8


def _accepts_args(func, args) -> bool:
    """Whether `func` could have been the intended target of a call carrying
    these argument names.

    Lexical distance cannot tell a reader from a writer: 'create_file' and
    'append_to_file' both sit ~0.8 from 'read_file', so any threshold that
    accepts real typos also accepts those. The arguments can: a call carrying
    `content` was never meant for a tool that has no `content` parameter. This
    turns the dangerous class of substitution — one that drops the payload and
    reports success — into a plain "no such tool" the model can react to.
    """
    if not args:
        return True
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True                                  # builtins etc. — don't guess
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True                                  # **kwargs takes anything
    return all(k in params for k in args)


def fuzzy_match_tool(name: str, available_tools: dict, cutoff: float = FUZZY_CUTOFF,
                     args=None) -> Optional[str]:
    """Resolve a tool name, tolerating the mistakes small models make.

    Order matters: exact, then normalization (case/separator differences are a
    formatting artefact and resolve with certainty), and only then approximate
    matching. Running fuzzy first let a close-but-wrong neighbour win over an
    exact normalized match.

    `args`, when given, rejects fuzzy candidates whose signature cannot accept
    the call's arguments.
    """
    if name in available_tools:
        return name

    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    for tool_name in available_tools:
        if tool_name.lower().replace("-", "_") == name_lower:
            log.info("Normalized tool '%s' -> '%s'", name, tool_name)
            return tool_name

    # Several candidates, so a signature-incompatible best match can be skipped
    # in favour of a slightly more distant but plausible one.
    for candidate in difflib.get_close_matches(name, available_tools.keys(), n=5, cutoff=cutoff):
        if _accepts_args(available_tools[candidate], args):
            log.info("Fuzzy matched tool '%s' -> '%s'", name, candidate)
            return candidate
        log.info("Rejected fuzzy match '%s' -> '%s': signature cannot take %s",
                 name, candidate, sorted(args or ()))

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

    Returns a corrected tool call dict, or None if unrecoverable. When the name
    had to be changed the result carries ``renamed_from`` so the caller can tell
    the model what actually ran: a substitution the model never learns about is
    indistinguishable from success, and one wrong guess ('append_to_file' ->
    'read_file') then swallows the write entirely.
    """
    func = raw_tool.get("function", raw_tool)
    name = func.get("name", "")
    raw_args = func.get("arguments", {})
    
    # Arguments are recovered BEFORE the name: they are what tells a misspelled
    # reader from a misspelled writer when the names alone are ambiguous.
    if isinstance(raw_args, str):
        recovered_args = recover_json_arguments(raw_args)
        if recovered_args is None:
            recovered_args = {}
    elif isinstance(raw_args, dict):
        recovered_args = raw_args
    else:
        recovered_args = {}

    matched_name = fuzzy_match_tool(name, available_tools, args=recovered_args)
    if not matched_name:
        log.warning("Tool '%s' not found and no fuzzy match", name)
        return None

    recovered = {
        "function": {
            "name": matched_name,
            "arguments": recovered_args,
        }
    }
    if matched_name != name:
        recovered["renamed_from"] = name
    return recovered
