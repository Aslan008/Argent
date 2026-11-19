"""Which UI settings the current provider or model is known to ignore.

The toolbar shows `think` and a reasoning depth because they are useful on the
providers that read them. On a provider that never sends the setting, showing it
as if it worked is a lie the user pays for: they change a value and nothing
happens. This module reports the cases we can *prove from the code*, not a guess
for every model:

* **KoboldCPP** is an OpenAI-compatible local server; the client has no
  reasoning fields it understands, so neither the toggle nor the depth reaches
  anything.
* **Ollama** takes `think` as a boolean, and only the `gpt-oss` family accepts a
  level (`OllamaProvider._THINK_LEVEL_FAMILIES`). For every other Ollama model
  the depth is ignored. Whether the model can think at all is asked of the
  server once (`_think_capable`); when that probe said no, the toggle is inert
  too.

Cloud providers are deliberately absent: their capabilities live in per-model
metadata we do not have offline, and a confident wrong badge is worse than none.
"""

from typing import List

#: The internal setting name -> the short word the toolbar shows.
_LABELS = {"thinking": "thinking", "effort": "depth"}


def inert_settings(provider: str, model: str) -> List[str]:
    """Settings this provider/model is known to ignore, in a stable order."""
    provider = (provider or "").strip().lower()
    model_l = (model or "").strip().lower()
    inert: List[str] = []

    if provider == "koboldcpp":
        inert = ["thinking", "effort"]
    elif provider == "ollama":
        if "gpt-oss" not in model_l:
            inert.append("effort")
        try:
            from argent.providers import OllamaProvider

            # The probe's verdict is authoritative and cached; "unknown" means
            # the model has not been asked yet, and is not reported.
            if OllamaProvider._think_capable.get(model) is False:
                inert.append("thinking")
        except Exception:
            pass

    return [name for name in ("thinking", "effort") if name in inert]


def inert_label(provider: str, model: str) -> str:
    """`n/a: thinking, depth`, or an empty string when nothing is known."""
    words = [_LABELS[name] for name in inert_settings(provider, model)]
    return f"n/a: {', '.join(words)}" if words else ""
