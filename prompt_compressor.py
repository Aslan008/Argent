"""
Smart Prompt Compression for Argent.
Adapts system prompt and tool results based on model size.
Small models get shorter prompts to stay within context limits.
"""

from config import get_model_size_category

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
    idx = prompt.find(heading)
    if idx == -1:
        return prompt
    nxt = prompt.find("\n## ", idx + len(heading))
    return prompt[:idx] + (prompt[nxt + 1:] if nxt != -1 else "")


def compress_system_prompt(full_prompt: str, model_name: str) -> str:
    """Compress the system prompt for small models.
    - tiny (<3B): strip heavy sections and append a short rules suffix
    - small (3-7B): append a short clarification reminder
    - medium/large/cloud: no compression
    """
    category = get_model_size_category(model_name)

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


def compress_tool_result(result: str, model_name: str, max_lines: int = None) -> str:
    """Compress tool output to prevent context window explosion.
    Truncates long outputs, keeping the beginning and the end.
    Applies to all models, but thresholds vary by model size.
    """
    category = get_model_size_category(model_name)
    
    if max_lines is None:
        if category == "tiny": max_lines = 40
        elif category == "small": max_lines = 100
        elif category == "medium": max_lines = 500
        else: max_lines = 2000 # cloud and large
    
    lines = result.splitlines()
    if len(lines) <= max_lines:
        return result
    
    half = max_lines // 2
    head = lines[:half]
    tail = lines[-half:]
    
    separator = f"\n... [{len(lines) - max_lines} lines truncated to protect context window] ...\n"
    
    return "\n".join(head) + separator + "\n".join(tail)


def get_adaptive_context_window(model_name: str, base_window: int) -> int:
    """Suggest optimal context window based on model size."""
    category = get_model_size_category(model_name)
    
    recommendations = {
        "tiny": 2048,
        "small": 4096,
        "medium": 8192,
        "large": base_window,
        "cloud": base_window,
    }
    
    return min(recommendations.get(category, base_window), base_window)
