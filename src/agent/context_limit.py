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


# Patterns to pull the size of the REQUEST that was refused (tokens) out of such an error.
_PROMPT_PATTERNS = (
    r"request \((\d+) tokens\)",
    r"['\"]?n_prompt_tokens['\"]?\s*[:=]\s*(\d+)",
    r"resulted in (\d+) tokens",
    r"you requested (\d+) tokens",
    r"prompt is too long: (\d+) tokens",
)


def parse_prompt_tokens(text: str):
    """How many tokens the refused request had, as the provider counted them, or None."""
    s = text or ""
    for pattern in _PROMPT_PATTERNS:
        m = re.search(pattern, s)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


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


_PROBE_CACHE: dict = {}          # api root -> (checked at, window or None)
_PROBE_TTL_S = 300.0


def _api_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def probe_server_context(base_url: str, timeout: float = 1.0, now: float | None = None):
    """The context window a local OpenAI-compatible server was STARTED with, or None if it does not say.

    KoboldCPP answers /api/extra/true_max_context_length (and /api/v1/config/max_context_length);
    llama.cpp's llama-server answers /props with n_ctx. Asked before the first request, the window no
    longer has to be learnt from an HTTP 400 at the end of a long turn. Cached for a few minutes per
    server, failures too, so a server that does not answer costs one short timeout, not one per turn."""
    import time
    root = _api_root(base_url)
    if not root:
        return None
    now = time.time() if now is None else now
    hit = _PROBE_CACHE.get(root)
    if hit and now - hit[0] < _PROBE_TTL_S:
        return hit[1]
    window = None
    try:
        import requests
        for path, pick in (
            ("/api/extra/true_max_context_length", lambda j: j.get("value")),
            ("/api/v1/config/max_context_length", lambda j: j.get("value")),
            ("/props", lambda j: (j.get("default_generation_settings") or {}).get("n_ctx") or j.get("n_ctx")),
        ):
            try:
                resp = requests.get(root + path, timeout=timeout)
            except requests.RequestException:
                break                       # the server is not there at all: no point asking the rest
            if resp.status_code != 200:
                continue
            try:
                value = pick(resp.json() or {})
            except (ValueError, AttributeError):
                continue
            if isinstance(value, (int, float)) and value >= 256:
                window = int(value)
                break
    except Exception:
        window = None
    _PROBE_CACHE[root] = (now, window)
    return window


def effective_history_budget(window: int, sys_tokens: int,
                             response_reserve: int = 1024,
                             tools_reserve: int = 2048,
                             floor: int | None = None) -> int:
    """Tokens left for trimmable history after reserving everything else.

    The model's window must hold the system prompt, the tool schemas sent
    alongside, the conversation history and the response. Reserve all of those
    (plus a margin for drift between our token estimate and the model's real
    tokenizer) and return what history may use.
    """
    window = max(0, window)
    sys_tokens = max(0, sys_tokens)
    margin = int(window * 0.08)
    calc_budget = window - sys_tokens - response_reserve - tools_reserve - margin

    if floor is not None:
        # An explicit floor requested by caller must be strictly respected
        return max(floor, calc_budget)

    # On small context models (< 8k), default floor cannot exceed a reasonable fraction
    # of the remaining room, or system + history overflows the window.
    remaining_hard = max(256, window - sys_tokens - response_reserve - margin)
    dynamic_floor = min(2048, max(256, int(window * 0.25)))
    budget = max(dynamic_floor, calc_budget)
    return min(budget, remaining_hard)
