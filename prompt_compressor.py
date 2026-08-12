"""
Smart Prompt Compression for Argent.
Adapts system prompt and tool results based on model size.
Small models get shorter prompts to stay within context limits.
"""

import re
import threading

from config import get_model_size_category
from logger import get_logger

log = get_logger("agent")

# A line that names a tool but is not an instruction to call one. Headings and
# blank lines carry no tool dependency, so they must never be dropped on their
# own account — only when the whole section around them empties out.
_HEADING = re.compile(r"^\s*#{1,6}\s")
_WORD = re.compile(r"[a-z0-9_-]{2,}")


def drop_unavailable_tool_lines(prompt: str, available: set) -> str:
    """Remove prompt lines that instruct the model to use tools it was not given.

    The system prompt and the tool schemas are assembled independently, so they
    drift: sections kept telling the model to call `create_artifact`,
    `list_directory` or `create_plugin` on turns where those schemas were never
    sent — either disabled in config or outside the mode's toolset. The model
    then emits a call for a name that does not exist and the recovery layer has
    to guess at it.

    Filtering by line rather than by section keeps a rule whose neighbours are
    still valid, and a heading left with no instructions under it is dropped
    with them.
    """
    if not available:
        # If no tools are available, we must process the prompt to strip ALL tool lines,
        # not just short-circuit and return the whole prompt!
        pass

    kept, section, section_has_content = [], None, False

    def flush():
        # A heading only earns its tokens if something survived beneath it.
        if section is not None and section_has_content:
            kept.extend(section)

    for line in prompt.split("\n"):
        if _HEADING.match(line):
            flush()
            section, section_has_content = [line], False
            continue

        named = {w for w in _WORD.findall(line) if w in _ALL_TOOL_NAMES()}
        if named and not named <= available:
            continue

        target = section if section is not None else kept
        target.append(line)
        if section is not None and line.strip():
            section_has_content = True

    flush()
    return "\n".join(kept)


_TOOL_NAME_CACHE = None
_CACHE_LOCK = threading.Lock()


def _ALL_TOOL_NAMES() -> set:
    """Every tool name Argent knows, so the filter can tell a tool reference
    from an ordinary word. Imported lazily — tools imports config, and config
    must not depend on this module at import time."""
    global _TOOL_NAME_CACHE
    if _TOOL_NAME_CACHE is None:
        with _CACHE_LOCK:
            if _TOOL_NAME_CACHE is None:
                try:
                    from tools.schemas import TOOL_SCHEMAS
                    _TOOL_NAME_CACHE = {t["function"]["name"] for t in TOOL_SCHEMAS} | {"semantic_search"}
                except ImportError as e:
                    log.error(f"Failed to load tools schemas due to import error: {e}")
                    raise
                except (KeyError, TypeError) as e:
                    log.error(f"Malformed TOOL_SCHEMAS: {e}")
                    raise
                except Exception as e:
                    log.error(f"Unexpected error loading tools schemas: {e}")
                    raise
    return _TOOL_NAME_CACHE

_MINI_SYSTEM_SUFFIX = """
## RULES (SHORT)
- Use tools via JSON. Follow schemas exactly.
- If the task is unclear or ambiguous, use ask_user_questions to clarify BEFORE writing any code.
- Ask about: tech stack, design preferences, data format, target platform.
- One tool call per response unless independent.
- Verify results after execution.
- Be concise. No unnecessary explanations.
"""

# Heavy sections a weak model gains little from; stripped for tiny models.
# Names must match the real headings emitted by build_system_prompt — note
# that several of these are already omitted there for small tiers, so this is
# a safety net rather than the primary mechanism.
_TINY_SECTIONS_TO_STRIP = [
    "## 2. PLUGIN DEVELOPMENT",
    "## 4. PLANNING MODE & ARTIFACTS",
    "## 5. UI & TERMINOLOGY STANDARDS",
]


def _strip_section(prompt: str, heading: str) -> str:
    # Find the heading case-insensitively, and ensure we only drop that section
    pattern = re.compile(r"^" + re.escape(heading) + r"\b.*$", re.IGNORECASE | re.MULTILINE)
    while True:
        match = pattern.search(prompt)
        if not match:
            break
        idx = match.start()
        # Find the next section (any heading level)
        next_match = re.search(r"^#+\s", prompt[match.end():], re.MULTILINE)
        if next_match:
            nxt = match.end() + next_match.start()
        else:
            nxt = -1
        prompt = prompt[:idx] + (prompt[nxt:] if nxt != -1 else "")
    return prompt


def compress_system_prompt(full_prompt: str, model_name: str, category: str = None) -> str:
    """Compress the system prompt for small models.
    - tiny (<3B): strip heavy sections and append a short rules suffix
    - small (3-7B): append a short clarification reminder
    - medium/large/cloud: no compression

    ``category`` lets the caller pass the tier it already resolved. Re-deriving
    it here meant one prompt could be BUILT for one tier and COMPRESSED for
    another whenever the two lookups disagreed — which is exactly what happened
    when a caller's tier was stubbed but this module's was not.
    """
    category = category or get_model_size_category(model_name)

    if category not in ("tiny", "small", "medium", "large", "cloud"):
        log.warning(f"Unknown model category '{category}', falling back to full prompt")
        return full_prompt

    if category in ("medium", "large", "cloud"):
        return full_prompt

    prompt = full_prompt

    if category == "tiny":
        for section in _TINY_SECTIONS_TO_STRIP:
            prompt = _strip_section(prompt, section)
        prompt = prompt.rstrip() + "\n" + _MINI_SYSTEM_SUFFIX

    elif category == "small":
        prompt = prompt.rstrip() + "\n\n- IMPORTANT: If the task is unclear, use ask_user_questions to clarify before making assumptions.\n"

    return prompt


def compress_tool_result(result: str, model_name: str,
                         max_lines: int = None, max_chars: int = None) -> str:
    """Compress tool output to prevent context window explosion."""
    if not isinstance(result, str):
        try:
            result = str(result)
        except Exception:
            try:
                result = repr(result)
            except Exception:
                result = f"<Unrepresentable object: {type(result).__name__}>"

    category = get_model_size_category(model_name)

    if max_lines is None:
        if category == "tiny": max_lines = 40
        elif category == "small": max_lines = 100
        elif category == "medium": max_lines = 500
        else: max_lines = 2000 # cloud and large
    if max_chars is None:
        # A generous per-line width, so output that already fits the line budget
        # with normal-width lines also fits here — the char budget is a safety
        # net for pathologically wide lines, not a second, tighter limit.
        max_chars = max_lines * 120

    max_lines = max(1, int(max_lines))
    max_chars = max(1, int(max_chars))

    lines = result.splitlines()
    if len(lines) > max_lines:
        if max_lines == 1:
            head = [lines[0]] if lines else []
            tail = []
        else:
            half = max_lines // 2
            head = lines[:half]
            tail = lines[-half:] if half > 0 else []
        separator = f"\n... [{len(lines) - len(head) - len(tail)} lines truncated to protect context window] ...\n"
        result = "\n".join(head) + separator + "\n".join(tail)

    # Second pass: even within the line budget, a handful of huge lines (or the
    # wide head/tail we just kept) can still overflow. Trim by characters.
    # str slicing is per code point, so this never splits a multi-byte char.
    if len(result) > max_chars:
        removed = len(result) - max_chars
        separator = f"\n... [{removed} characters truncated to protect context window] ...\n"
        if len(separator) >= max_chars:
            result = result[:max_chars]
        else:
            half = (max_chars - len(separator)) // 2
            tail_len = max_chars - len(separator) - half
            result = result[:half] + separator + (result[-tail_len:] if tail_len > 0 else "")

    return result


def get_adaptive_context_window(model_name: str, base_window: int) -> int:
    """Suggest optimal context window based on model size."""
    if base_window <= 0:
        base_window = 4096
        
    category = get_model_size_category(model_name)
    
    recommendations = {
        "tiny": 2048,
        "small": 4096,
        "medium": 8192,
        "large": base_window,
        "cloud": base_window,
    }
    
    return min(recommendations.get(category, base_window), base_window)
