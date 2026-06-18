"""Helpers for staying inside a model's real context window.

Two failure modes this addresses:
- Proactive: history must be trimmed into the room left *after* the system
  prompt, the tool schemas and the model's own response — not against the whole
  window — or the assembled request overflows a provider with a hard limit
  (e.g. KoboldCPP rejects it with HTTP 400 exceed_context_size_error).
- Reactive: when a provider still reports an overflow, parse the real window it
  reports so the agent can shrink its budget to fit and retry.
"""

import re

# Substrings that mark a "your prompt is too big for the context window" error
# across providers (KoboldCPP, llama.cpp, vLLM, OpenAI-compatible, ...).
_CTX_KEYWORDS = (
    "context size",
    "context length",
    "context window",
    "exceed_context",
    "maximum context",
    "too many tokens",
    "n_ctx",
)

# Patterns to pull the real window size (in tokens) out of such an error.
_CTX_PATTERNS = (
    r"context size \((\d+)",
    r"['\"]?n_ctx['\"]?\s*[:=]\s*(\d+)",
    r"maximum context length is (\d+)",
    r"context (?:size|length|window)[^\d]{0,40}?(\d+)\s*tokens",
)


def is_context_overflow(text: str) -> bool:
    """True if `text` looks like a context-window-exceeded error."""
    t = (text or "").lower()
    return any(k in t for k in _CTX_KEYWORDS)


def parse_context_limit(text: str):
    """Extract the model's real context window (tokens) from an error, or None."""
    s = text or ""
    for pattern in _CTX_PATTERNS:
        m = re.search(pattern, s)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def effective_history_budget(window: int, sys_tokens: int,
                             response_reserve: int = 1024,
                             tools_reserve: int = 2048,
                             floor: int = 2048) -> int:
    """Tokens left for trimmable history after reserving everything else.

    The model's window must hold the system prompt, the tool schemas sent
    alongside, the conversation history and the response. Reserve all of those
    (plus a margin for drift between our token estimate and the model's real
    tokenizer) and return what history may use.
    """
    margin = int(max(0, window) * 0.08)
    return max(floor, window - max(0, sys_tokens) - response_reserve - tools_reserve - margin)
