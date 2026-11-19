"""Feature flags for work that must not change the default experience.

Everything here is **off unless explicitly enabled** (environment variable or
`features` object in the user's config). The rest of the code asks
`features.enabled("replay")` and does nothing when it is false: no files, no
sinks, no prompt changes.
"""

from __future__ import annotations

import os
from typing import Dict

#: Names accepted by :func:`enabled`. Keep this list small and explicit.
FEATURES = ("replay", "bench", "skills_v2", "lessons")

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _coerce(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    # A typo must not silently enable a feature.
    return False


def _from_config(name: str):
    try:
        from argent.config import load_config

        data = load_config() or {}
        flags = data.get("features")
        if isinstance(flags, dict) and name in flags:
            return _coerce(flags[name])
    except Exception:
        pass
    return None


def enabled(name: str) -> bool:
    """True only when the feature was explicitly switched on.

    Precedence: environment variable > config `features.<name>` > off.
    """
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; known: {', '.join(FEATURES)}")
    env_value = os.environ.get(f"ARGENT_FEATURE_{name.upper()}")
    if env_value is not None:
        return _coerce(env_value)
    value = _from_config(name)
    return bool(value) if value is not None else False


def set_enabled(name: str, value: bool) -> None:
    """Persist a feature switch in the user's config (used by a future CLI)."""
    if name not in FEATURES:
        raise KeyError(f"unknown feature {name!r}; known: {', '.join(FEATURES)}")
    from argent.config import load_config, save_config

    config = load_config() or {}
    flags = config.get("features")
    if not isinstance(flags, dict):
        flags = {}
    flags[name] = bool(value)
    config["features"] = flags
    save_config(config)


def snapshot() -> Dict[str, bool]:
    """All flags, for `/status` or a diagnostic command."""
    return {name: enabled(name) for name in FEATURES}
