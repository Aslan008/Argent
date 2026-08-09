"""How big a context this model can actually take.

The context window used to be picked from a fixed list — 2048 / 4096 / 8192 —
that had nothing to do with the model in front of you. Choose too small and the
conversation is trimmed away for no reason; too large and the request
overflows, which is what happens by default when you move a chat from a cloud
model to a local one.

Ollama knows the real number: /api/show returns e.g. qwen35.context_length =
262144. It is offered as a REFERENCE, not applied: that is the model's
architectural maximum, and loading a 9B at 262k would ask for memory the
machine does not have. The user still chooses — but now from numbers that mean
something for this model instead of a list someone hard-coded.
"""

from logger import get_logger

log = get_logger("ui")

# Cold Ollama can take a while to answer; the caller is a menu, so failing fast
# and falling back to the plain list beats a menu that hangs.
_TIMEOUT_SECONDS = 6


def detect_context_length(model: str, provider: str = "ollama") -> int | None:
    """The model's maximum context in tokens, or None if it cannot be known."""
    if not model or provider != "ollama":
        # Only Ollama exposes this locally. Cloud providers publish it per
        # model on their own sites, and guessing from a name would be fiction.
        return None
    try:
        import os

        import requests

        # Same resolution the rest of the codebase uses for the local server.
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = f"http://{host}"
        resp = requests.post(f"{host}/api/show",
                             json={"model": model}, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        info = (resp.json() or {}).get("model_info") or {}
    except Exception as e:
        log.info("context length for %r unavailable: %s", model, e)
        return None

    for key, value in info.items():
        if key.endswith("context_length") and isinstance(value, int) and value > 0:
            return value
    return None


def context_choices(model: str, provider: str = "ollama", current: int = None) -> list:
    """Menu entries for the context window, anchored to this model's real limit.

    Fractions of the maximum rather than round numbers: what matters is how
    much of the model you are using, and 8192 means something completely
    different on a 262k model than on an 8k one.
    """
    maximum = detect_context_length(model, provider)
    if not maximum:
        return None                       # caller keeps its static list

    steps = []
    for fraction, label in ((1 / 8, "экономно"), (1 / 4, "умеренно"),
                            (1 / 2, "щедро"), (1, "максимум модели")):
        value = int(maximum * fraction)
        # Below this a coding conversation cannot hold one file plus the
        # prompt, so offering it would only look like a choice.
        if value < 2048:
            continue
        steps.append((value, label))

    seen, entries = set(), []
    for value, label in steps:
        if value in seen:
            continue
        seen.add(value)
        note = " — текущее" if current == value else ""
        entries.append(f"{value} ({label}){note}")
    return entries
