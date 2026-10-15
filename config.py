import json
import os
import re
from pathlib import Path

# We store configuration in the user's home directory
CONFIG_FILE = Path.home() / ".argent_coder_config.json"

DEFAULT_MODEL = "llama3.1"

_CONFIG_CACHE = None
# What the cache was read from: the file's identity (see _file_signature), or None
# for "no file". Named for the mtime it started as.
_CONFIG_MTIME = None


def quarantine_config(reason: str) -> Path | None:
    """Move an unreadable config aside and say so. Returns the new path.

    An unreadable file used to be swallowed silently: the session started on
    defaults — no provider, no API keys, no MCP servers, no knowledge bases —
    and the very next setting change wrote those defaults back over the file,
    destroying settings that were still perfectly recoverable by hand.
    """
    import sys
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = CONFIG_FILE.with_name(f"{CONFIG_FILE.name}.broken-{stamp}")
    try:
        CONFIG_FILE.replace(backup)
    except OSError as e:
        print(f"[Argent] Конфиг нечитаем ({reason}), и отложить его не удалось: {e}",
              file=sys.stderr)
        return None
    print(f"[Argent] Конфиг нечитаем ({reason}). Он сохранён как {backup.name}; "
          f"Argent продолжает на настройках по умолчанию.", file=sys.stderr)
    return backup


def _file_signature():
    """Identity of the config file as it is now: (mtime_ns, size, inode), or None.

    The modification time alone is not enough. Two writes inside one timestamp
    tick (the filesystem's clock is coarser than the nanoseconds it reports) look
    identical, so a process kept serving settings another process had already
    replaced. An atomic save also changes the file's identity (os.replace makes a
    new inode), and a rewrite that changes any value usually changes the size.
    """
    try:
        st = CONFIG_FILE.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def load_config() -> dict:
    """Load configuration from disk with caching and change verification.

    Checks the file's identity (see _file_signature) so concurrent Argent
    processes always see the freshest settings and never clobber each other with
    stale in-memory caches.
    """
    global _CONFIG_CACHE, _CONFIG_MTIME
    mtime = _file_signature()

    # "No file" is a state the cache can describe too (mtime None on both sides).
    # Requiring a real mtime meant that until the first save a new dict was built
    # on every call, so a setting changed in memory vanished on the next read —
    # which is exactly what a read-only home directory looks like, and what a
    # test with no config file saw.
    if _CONFIG_CACHE is not None and mtime == _CONFIG_MTIME:
        return _CONFIG_CACHE

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            quarantine_config(str(e))
        else:
            if isinstance(loaded, dict):
                _CONFIG_CACHE = loaded
                _CONFIG_MTIME = mtime
                return _CONFIG_CACHE
            # Valid JSON of the wrong shape is worse than broken JSON: nothing
            # raises here, and every _get() after it fails with AttributeError.
            quarantine_config(f"ожидался объект, получен {type(loaded).__name__}")
    _CONFIG_CACHE = {"model": DEFAULT_MODEL}
    # A quarantined file is gone from where it was: describe what is on disk now.
    _CONFIG_MTIME = mtime if CONFIG_FILE.exists() else None
    return _CONFIG_CACHE


def save_config(config: dict):
    """Save configuration to disk and update cache.

    Written atomically (temp + os.replace) so a crash mid-write or a second
    Argent instance racing on the same file can't corrupt the config.
    """
    global _CONFIG_CACHE, _CONFIG_MTIME
    _CONFIG_CACHE = config
    from atomic_io import atomic_write_text
    atomic_write_text(CONFIG_FILE, json.dumps(config, indent=4))
    _CONFIG_MTIME = _file_signature()


#: Model parameters a room's step may set for the time it works (see generation_overrides).
GENERATION_KEYS = ("temperature", "top_p", "top_k", "min_p", "repeat_penalty", "reasoning_effort", "thinking_budget")
#: (scope, layer) pairs; a layer applies to the scope it was installed in and the ones nested in it
#: (see run_scope.py), so parallel branches do not read each other's model parameters.
_GENERATION_STACK: list = []


def _get(key: str, default=None):
    if _GENERATION_STACK and key in GENERATION_KEYS:
        import run_scope
        for owner, layer in reversed(list(_GENERATION_STACK)):
            if key in layer and run_scope.visible(owner):
                return layer[key]
    return load_config().get(key, default)


class generation_overrides:
    """`with generation_overrides({"temperature": 0.1, "top_p": 0.8}):` — for the duration of the
    block the model parameters read through config (temperature, top_p, top_k, min_p, repeat_penalty,
    reasoning_effort, thinking_budget) are these values instead of the saved ones; nothing is written
    to the config file. Nested blocks stack; unknown keys are ignored."""

    def __init__(self, values):
        self._layer = {k: v for k, v in (values or {}).items() if k in GENERATION_KEYS and v is not None}
        self._entry = None

    def __enter__(self):
        if self._layer:
            import run_scope
            self._entry = (run_scope.current(), self._layer)
            _GENERATION_STACK.append(self._entry)
        return self

    def __exit__(self, *exc):
        entry, self._entry = self._entry, None
        if entry is not None:
            # by identity: an equal layer installed by another branch must stay
            for i in range(len(_GENERATION_STACK) - 1, -1, -1):
                if _GENERATION_STACK[i] is entry:
                    del _GENERATION_STACK[i]
                    break
        return False


def _set(key: str, value):
    config = load_config()
    config[key] = value
    save_config(config)


def _api_key(stored_key: str, *env_names: str):
    """A key saved with /key wins; otherwise the first of these environment
    variables that is set.

    Only the Claude key used to fall back to the environment, so an OpenRouter or
    DeepSeek user who followed their provider's own documentation (export
    OPENROUTER_API_KEY=...) was told no key was configured, and a key kept out of
    a plain-text config file — the better place for it — was not an option."""
    value = _get(stored_key)
    if value:
        return value
    for name in env_names:
        from_env = os.environ.get(name)
        if from_env and from_env.strip():
            return from_env.strip()
    return value


# ─── Model & provider ────────────────────────────────────────────────────────

ZAI_ENDPOINT_GENERAL = "https://api.z.ai/api/paas/v4/"
ZAI_ENDPOINT_CODING = "https://api.z.ai/api/coding/paas/v4/"
DEFAULT_KOBOLDCPP_URL = "http://localhost:5001/v1"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_XKIRO_URL = "https://api.xkiro.com/v1"
DEFAULT_WORMSOFT_URL = "https://ai.wormsoft.ru/api/gpt"
DEFAULT_DEEPSEEK_URL = "https://api.deepseek.com"
DEFAULT_CLAUDE_URL = "https://api.anthropic.com"


def get_provider_models() -> dict:
    """Dictionary mapping provider names to their last selected models."""
    return _get("provider_models", {})


def get_model_for_provider(provider_name: str) -> str | None:
    """Return the saved model for a specific provider if known."""
    return get_provider_models().get(provider_name)


def set_model_for_provider(provider_name: str, model_name: str):
    """Save the model for a specific provider."""
    models_map = dict(get_provider_models())
    models_map[provider_name] = model_name
    _set("provider_models", models_map)


def get_current_model(provider_name: str = None) -> str:
    prov = provider_name or get_provider()
    saved = get_model_for_provider(prov)
    if saved:
        return saved
    return _get("model", DEFAULT_MODEL)


def set_current_model(model_name: str, provider_name: str = None):
    prov = provider_name or get_provider()
    config = load_config()
    config["model"] = model_name
    provider_models = dict(config.get("provider_models", {}))
    provider_models[prov] = model_name
    config["provider_models"] = provider_models
    save_config(config)


def get_provider() -> str:
    """Currently selected API provider (e.g. 'ollama', 'zai', 'koboldcpp')."""
    return _get("provider", "ollama")


def set_provider(provider_name: str):
    config = load_config()
    config["provider"] = provider_name
    provider_models = config.get("provider_models", {})
    if provider_name in provider_models:
        config["model"] = provider_models[provider_name]
    save_config(config)


def get_zai_api_key() -> str | None:
    return _api_key("zai_api_key", "ZAI_API_KEY")


def set_zai_api_key(api_key: str):
    _set("zai_api_key", api_key)


def get_zai_endpoint() -> str:
    return _get("zai_endpoint", ZAI_ENDPOINT_GENERAL)


def set_zai_endpoint(endpoint: str):
    _set("zai_endpoint", endpoint)


def get_koboldcpp_url() -> str:
    return _get("koboldcpp_url", DEFAULT_KOBOLDCPP_URL)


def set_koboldcpp_url(url: str):
    _set("koboldcpp_url", url)


def get_openrouter_api_key() -> str | None:
    return _api_key("openrouter_api_key", "OPENROUTER_API_KEY")


def set_openrouter_api_key(api_key: str):
    _set("openrouter_api_key", api_key)


def get_openrouter_url() -> str:
    return _get("openrouter_url", DEFAULT_OPENROUTER_URL)


def set_openrouter_url(url: str):
    _set("openrouter_url", url)


def get_xkiro_api_key() -> str | None:
    return _get("xkiro_api_key")


def set_xkiro_api_key(api_key: str):
    _set("xkiro_api_key", api_key)


def get_xkiro_url() -> str:
    return _get("xkiro_url", DEFAULT_XKIRO_URL)


def set_xkiro_url(url: str):
    _set("xkiro_url", url)


def get_wormsoft_api_key() -> str | None:
    return _get("wormsoft_api_key")


def set_wormsoft_api_key(api_key: str):
    _set("wormsoft_api_key", api_key)


def get_wormsoft_url() -> str:
    url = _get("wormsoft_url", DEFAULT_WORMSOFT_URL)
    if url in (
        "https://ai.wormsoft.ru/v1",
        "https://ai.wormsoft.ru/v1/",
        "https://ai.wormsoft.ru/api/v1",
        "https://ai.wormsoft.ru/api/v1/",
    ):
        url = DEFAULT_WORMSOFT_URL
        _set("wormsoft_url", url)
    return url


def set_wormsoft_url(url: str):
    _set("wormsoft_url", url)


def get_deepseek_api_key() -> str | None:
    return _api_key("deepseek_api_key", "DEEPSEEK_API_KEY")


def set_deepseek_api_key(api_key: str):
    _set("deepseek_api_key", api_key)


def get_deepseek_url() -> str:
    return _get("deepseek_url", DEFAULT_DEEPSEEK_URL)


def set_deepseek_url(url: str):
    _set("deepseek_url", url)


def get_claude_api_key() -> str | None:
    return _api_key("claude_api_key", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY")


def set_claude_api_key(api_key: str):
    _set("claude_api_key", api_key)


def get_claude_workspace_id() -> str | None:
    """The Anthropic workspace to use. An API key that is not scoped to one workspace (an
    organisation-level key) is refused by every call without it ("this request must include the
    anthropic-workspace-id header")."""
    return _api_key("claude_workspace_id", "ANTHROPIC_WORKSPACE_ID")


def set_claude_workspace_id(workspace_id: str | None):
    _set("claude_workspace_id", (workspace_id or "").strip() or None)


def get_claude_url() -> str:
    return _get("claude_url", DEFAULT_CLAUDE_URL)


def set_claude_url(url: str):
    _set("claude_url", url)


def get_temperature() -> float:
    val = _get("temperature", 0.3)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.3


def set_temperature(val: float):
    try:
        v = float(val)
        v = max(0.0, min(2.0, v))
        _set("temperature", round(v, 4))
    except (ValueError, TypeError):
        _set("temperature", 0.3)


def get_top_p() -> float | None:
    """Nucleus sampling probability threshold (0.0–1.0). None means model default."""
    val = _get("top_p")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def set_top_p(val: float | None):
    """Set top_p sampling threshold (0.0–1.0), or None to reset."""
    if val is None:
        _set("top_p", None)
        return
    try:
        v = float(val)
        v = max(0.0001, min(1.0, v))
        _set("top_p", round(v, 4))
    except (ValueError, TypeError):
        _set("top_p", None)


def get_top_k() -> int | None:
    """Top-K vocabulary filtering threshold. None means model default."""
    val = _get("top_k")
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            pass
    return None


def set_top_k(val: int | None):
    """Set top_k sampling threshold (integer >= 1), or None to reset."""
    if val is None:
        _set("top_k", None)
        return
    try:
        v = int(val)
        if v < 1:
            _set("top_k", None)
        else:
            _set("top_k", v)
    except (ValueError, TypeError):
        _set("top_k", None)


def get_min_p() -> float | None:
    """Min-P sampling threshold relative to top token probability (0.0–1.0). None means model default."""
    val = _get("min_p")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def set_min_p(val: float | None):
    """Set min_p sampling threshold (0.0–1.0), or None to reset."""
    if val is None:
        _set("min_p", None)
        return
    try:
        v = float(val)
        v = max(0.0, min(1.0, v))
        _set("min_p", round(v, 4))
    except (ValueError, TypeError):
        _set("min_p", None)


def get_repeat_penalty() -> float | None:
    """Repetition penalty multiplier (e.g. 1.05–1.2). None means model default."""
    val = _get("repeat_penalty")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    return None


def set_repeat_penalty(val: float | None):
    """Set repetition penalty multiplier, or None to reset."""
    if val is None:
        _set("repeat_penalty", None)
        return
    try:
        v = float(val)
        v = max(0.5, min(2.5, v))
        _set("repeat_penalty", round(v, 4))
    except (ValueError, TypeError):
        _set("repeat_penalty", None)


def get_sampling_params() -> dict:
    """Return all active sampling parameters as a dict."""
    return {
        "temperature": get_temperature(),
        "top_p": get_top_p(),
        "top_k": get_top_k(),
        "min_p": get_min_p(),
        "repeat_penalty": get_repeat_penalty(),
    }


SAMPLING_PRESETS = {
    "code": {
        "temperature": 0.2,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
        "top_p": None,
        "top_k": None,
    },
    "precise": {
        "temperature": 0.0,
        "min_p": None,
        "repeat_penalty": None,
        "top_p": None,
        "top_k": None,
    },
    "creative": {
        "temperature": 0.8,
        "min_p": None,
        "repeat_penalty": 1.1,
        "top_p": 0.95,
        "top_k": 50,
    },
    "default": {
        "temperature": 0.3,
        "min_p": None,
        "repeat_penalty": None,
        "top_p": None,
        "top_k": None,
    },
}


def apply_sampling_preset(preset_name: str) -> dict:
    """Apply a named sampling preset ('code', 'precise', 'creative', 'default')."""
    key = preset_name.lower().strip()
    if key in ("coding", "coder"):
        key = "code"
    elif key in ("deterministic", "exact", "math"):
        key = "precise"
    elif key in ("chat", "story"):
        key = "creative"
    elif key in ("reset", "defaults"):
        key = "default"

    if key not in SAMPLING_PRESETS:
        raise ValueError(f"Неизвестный пресет '{preset_name}'. Доступные: {list(SAMPLING_PRESETS.keys())}")

    preset = SAMPLING_PRESETS[key]
    set_temperature(preset["temperature"])
    set_top_p(preset["top_p"])
    set_top_k(preset["top_k"])
    set_min_p(preset["min_p"])
    set_repeat_penalty(preset["repeat_penalty"])
    return preset


def reset_sampling_params():
    """Reset all sampling parameters to model defaults (temperature resets to 0.3)."""
    apply_sampling_preset("default")


def get_context_window() -> int:
    """Configured context window size in tokens (default 32768)."""
    return _get("context_window", 32768)


def set_context_window(size: int):
    _set("context_window", size)


def get_critic_auto() -> bool:
    """Whether an independent critic reviews the staged diff before /commit."""
    return _get("critic_auto", False)


def set_critic_auto(val: bool):
    _set("critic_auto", val)


def get_rooms_spawn() -> bool:
    """Whether the rooms triage may author NEW rooms (spawn_room). Off by
    default: an AI-proposed room still passes the validator + human approval
    before it can run, but the whole capability is opt-in."""
    return _get("rooms_spawn", False)


def set_rooms_spawn(val: bool):
    _set("rooms_spawn", val)


def get_rooms_hint() -> bool:
    """Whether a one-line nudge to the matching room is added to the model's request when the person's
    message clearly fits one (src/rooms/hint.py). On by default: it costs nothing unless a room fits."""
    return bool(_get("rooms_hint", True))


def set_rooms_hint(val: bool):
    _set("rooms_hint", bool(val))


def get_critic_model() -> str:
    """Model the critic runs on. Empty = the current main model."""
    return _get("critic_model", "")


def set_critic_model(name: str):
    _set("critic_model", name)


def get_critic_provider() -> str:
    """Provider for the critic model. Empty = the current main provider."""
    return _get("critic_provider", "")


def set_critic_provider(name: str):
    _set("critic_provider", name)


def get_command_guard() -> str:
    """Command risk-gate strictness: 'off' | 'warn' | 'block'. Default 'warn'.

    off   = legacy destructive-command confirmation only.
    warn  = confirm risky commands, showing the reason.
    block = refuse catastrophic commands outright (no prompt).
    """
    return _get("command_guard", "warn")


def set_command_guard(level: str):
    _set("command_guard", level)


def get_loop_guard_mode() -> str:
    """LoopGuard mode: 'auto' | 'on' | 'off'. Default 'auto'.

    auto = enabled for Cloud, Standard, and Tiny models; disabled for MoE models.
    on   = forced enabled for all models.
    off  = disabled completely.
    """
    return _get("loop_guard_mode", "auto")


def set_loop_guard_mode(mode: str):
    """Set LoopGuard mode: 'auto' | 'on' | 'off'."""
    mode_clean = (mode or "auto").lower().strip()
    if mode_clean not in ("auto", "on", "off"):
        raise ValueError(f"Invalid loop_guard mode '{mode}'. Valid: 'auto', 'on', 'off'")
    _set("loop_guard_mode", mode_clean)


def get_max_generation_tokens() -> int:
    """Max tokens for generation. Default 8192 to prevent truncation in local APIs."""
    return _get("max_generation_tokens", 8192)


def set_max_generation_tokens(tokens: int):
    _set("max_generation_tokens", tokens)


def get_auxiliary_model() -> str | None:
    """Optional lightweight model for service tasks (summarization, /commit).
    None means service tasks reuse the main model."""
    return _get("auxiliary_model")


def set_auxiliary_model(model: str | None):
    if model is None:
        config = load_config()
        config.pop("auxiliary_model", None)
        save_config(config)
    else:
        _set("auxiliary_model", model)


def get_auxiliary_provider() -> str | None:
    """Provider for the auxiliary model. None means reuse the main provider."""
    return _get("auxiliary_provider")


def set_auxiliary_provider(provider: str | None):
    if provider is None:
        config = load_config()
        config.pop("auxiliary_provider", None)
        save_config(config)
    else:
        _set("auxiliary_provider", provider)


def get_compress_fallback_models() -> list:
    """Free OpenRouter models to move to, in this order, when the model that compresses the
    context is out of quota or down. Empty: any free model the catalogue offers."""
    value = _get("compress_fallback_models", [])
    return [str(m).strip() for m in value if str(m).strip()] if isinstance(value, list) else []


def set_compress_fallback_models(models: list | None):
    _set("compress_fallback_models", [str(m).strip() for m in (models or []) if str(m).strip()])


def get_compress_on_failure() -> str:
    """What to do when the compression model is out of quota or stays down: `auto` moves to
    the next free model by itself, `ask` lets the user choose (when someone is there to ask)."""
    value = str(_get("compress_on_failure", "auto") or "auto").strip().lower()
    return value if value in ("auto", "ask") else "auto"


def set_compress_on_failure(mode: str):
    _set("compress_on_failure", "ask" if str(mode).strip().lower() == "ask" else "auto")


def get_compress_retries() -> int:
    """How many times the same model is asked before it is given up on (it is only busy)."""
    try:
        return max(1, min(6, int(_get("compress_retries", 3))))
    except (TypeError, ValueError):
        return 3


def set_compress_retries(count: int):
    _set("compress_retries", max(1, min(6, int(count))))


def is_summary_precompute_enabled() -> bool:
    """Whether the summary a compaction will need is written ahead of time, in the background, once the history
    is close to its limit (src/agent/summary_precompute.py). On by default: it only shortens the wait."""
    return bool(_get("summary_precompute", True))


def set_summary_precompute_enabled(enabled: bool):
    _set("summary_precompute", bool(enabled))


def get_network_patience() -> float:
    """Seconds a turn waits for a lost connection (or a busy provider) before giving up; 0 = do not wait.

    The wait costs nothing but time: the dialog is kept, the unfinished request is simply sent again once the
    connection is back (see src/agent/netstate.py)."""
    try:
        return max(0.0, min(86400.0, float(_get("network_patience", 900))))
    except (TypeError, ValueError):
        return 900.0


def set_network_patience(seconds: float):
    _set("network_patience", max(0.0, min(86400.0, float(seconds))))


def get_strip_reasoning() -> bool:
    """Whether reasoning (thinking) blocks are stripped from history before LLM calls."""
    return _get("strip_reasoning", True)


def set_strip_reasoning(enabled: bool):
    _set("strip_reasoning", enabled)

def get_thinking_mode() -> str:
    """Thinking generation mode: 'auto' (default), 'on' (forced), or 'off' (disabled)."""
    return _get("thinking_mode", "auto")


def set_thinking_mode(mode: str):
    """Set thinking mode: 'auto', 'on', or 'off'."""
    _set("thinking_mode", mode)


def get_reasoning_effort() -> str | None:
    """Reasoning effort depth: 'low', 'medium', 'high', 'xhigh', or None."""
    return _get("reasoning_effort", None)


def set_reasoning_effort(effort: str | None):
    """Set reasoning effort depth: 'low', 'medium', 'high', 'xhigh', or None for default."""
    _set("reasoning_effort", effort)


def get_thinking_budget() -> int | None:
    """Token budget for model thinking, or None for unrestricted."""
    return _get("thinking_budget", None)


def set_thinking_budget(budget: int | None):
    """Set token budget for model thinking, or None for unrestricted."""
    _set("thinking_budget", budget)


def get_thinking_rethink() -> bool:
    """Whether models are instructed to perform an explicit <rethink> verification pass."""
    return _get("thinking_rethink", False)


def set_thinking_rethink(enabled: bool):
    """Enable or disable explicit <rethink> verification protocol."""
    _set("thinking_rethink", bool(enabled))


def get_thinking_display_label() -> str:
    """Return a compact human-readable thinking badge, e.g. 'ON/high', 'OFF', 'AUTO+rethink'."""
    mode = (get_thinking_mode() or "auto").upper()
    effort = get_reasoning_effort()
    rethink = get_thinking_rethink()
    suffix = "+rethink" if rethink else ""
    if mode == "OFF":
        return f"OFF{suffix}" if rethink else "OFF"
    if effort:
        return f"{mode}/{effort}{suffix}"
    return f"{mode}{suffix}"


#: Single flattened thinking level surfaced by the TUI. The underlying config
#: keeps `thinking_mode` (auto/on/off) and `reasoning_effort` separate; this maps
#: them onto one cycle so the frontend can show and change one value.
THINKING_LEVELS = ("off", "auto", "low", "medium", "high", "xhigh")


def get_thinking_level() -> str:
    """Flatten thinking_mode + reasoning_effort into one of THINKING_LEVELS.

    `off` wins over everything; an explicit effort wins over `auto`.
    """
    mode = (get_thinking_mode() or "auto").strip().lower()
    effort = (get_reasoning_effort() or "").strip().lower()
    if mode == "off":
        return "off"
    if effort in THINKING_LEVELS:
        return effort
    return "auto"


def set_thinking_level(level: str) -> str:
    """Apply one of THINKING_LEVELS. Raises ValueError on anything else.

    Setting an effort implies thinking_mode='on' — otherwise the effort would be
    silently ignored by the provider.
    """
    val = (level or "").strip().lower()
    if val not in THINKING_LEVELS:
        raise ValueError(
            f"Invalid thinking level '{level}'. Choose from: {', '.join(THINKING_LEVELS)}."
        )
    if val == "off":
        set_thinking_mode("off")
        set_reasoning_effort(None)
    elif val == "auto":
        set_thinking_mode("auto")
        set_reasoning_effort(None)
    else:
        set_thinking_mode("on")
        set_reasoning_effort(val)
    return val


def get_thinking_display_mode() -> str:
    """How thinking is rendered in terminal: 'summary' (Claude Code compact) or 'full' (tree)."""
    return _get("thinking_display_mode", "summary")


def set_thinking_display_mode(mode: str):
    """Set thinking display mode: 'summary' or 'full'."""
    val = mode.strip().lower() if mode else "summary"
    if val not in ("summary", "full"):
        raise ValueError(f"Invalid thinking display mode '{mode}'. Must be 'summary' or 'full'.")
    _set("thinking_display_mode", val)




# ─── Model size classification ───────────────────────────────────────────────

VALID_MODEL_CATEGORIES = ("tiny", "small", "medium", "large", "cloud", "moe")


def get_model_category_override() -> str | None:
    """The user's manual model category override, or None for auto-detection."""
    val = _get("model_category_override")
    if val and val in VALID_MODEL_CATEGORIES:
        return val
    return None


def set_model_category_override(category: str | None):
    """Set or clear the manual model category override.
    Pass None to revert to auto-detection."""
    config = load_config()
    if category is None:
        config.pop("model_category_override", None)
    else:
        if category not in VALID_MODEL_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Valid: {VALID_MODEL_CATEGORIES}")
        config["model_category_override"] = category
    save_config(config)


def get_moe_specs(model_name: str) -> tuple[float, float] | None:
    """Extract (total_params_b, active_params_b) from a MoE model name if detectable."""
    if not model_name:
        return None
    name = model_name.lower()

    # 1. Explicit aXb or aXm format: :35a3b, 35a3b, 35A3B, 8b-a1b, qwen2-57b-a14b, deepseek-v2:16b-a2.4b, granite-3.0-3b-a800m
    m = re.search(r'(?:^|[-_:])(?:(\d+(?:\.\d+)?)([bm])?)?[aA](\d+(?:\.\d+)?)([bm])', name)
    if m:
        total_val, total_unit, active_val, active_unit = m.group(1), m.group(2), m.group(3), m.group(4)
        active_b = float(active_val) if active_unit == 'b' else float(active_val) / 1000.0
        if total_val:
            total_b = float(total_val) if (total_unit == 'b' or not total_unit) else float(total_val) / 1000.0
        else:
            # Search for total param count strictly before the '[aA]' marker to prevent matching active_b itself
            prefix = name[:m.start() + m.group(0).lower().find('a')]
            m_total = re.search(r'(\d+(?:\.\d+)?)([bm])', prefix)
            if m_total:
                t_val, t_unit = m_total.group(1), m_total.group(2)
                total_b = float(t_val) if t_unit == 'b' else float(t_val) / 1000.0
            else:
                total_b = active_b * 8.0
        if active_b > 0:
            if total_b < active_b:
                total_b = active_b * 4.0
            return (round(total_b, 1), round(active_b, 2))

    # 2. Explicit NxMb format: 8x7b, 8x22b, 4x7b (e.g. Mixtral)
    m2 = re.search(r'(\d+)x(\d+(?:\.\d+)?)b', name)
    if m2:
        num_experts = int(m2.group(1))
        expert_size = float(m2.group(2))
        if num_experts >= 2 and expert_size > 0:
            active_b = min(expert_size * 2.0, expert_size * num_experts)
            total_b = expert_size * num_experts
            return (round(total_b, 1), round(active_b, 1))

    # 3. Known MoE architecture presets (when explicit params are not in the name)
    # DeepSeek-R1 distilled dense models (e.g. deepseek-r1:1.5b, 7b, 8b, 14b, 32b, 70b, distill-qwen/llama)
    # are DENSE models, NOT MoE. Only the full frontier 671B model is MoE.
    is_r1_dense = "deepseek-r1" in name and any(
        k in name for k in ("distill", ":1.5b", "-1.5b", ":7b", "-7b", ":8b", "-8b", ":14b", "-14b", ":32b", "-32b", ":70b", "-70b", "qwen", "llama")
    )
    if not is_r1_dense:
        if re.search(r'\bdeepseek-(?:v3|r1)\b', name) or "deepseek-v3" in name or ("deepseek-r1" in name and (":671b" in name or name.endswith("deepseek-r1"))):
            return (671.0, 37.0)

    if re.search(r'\bdeepseek-v2(?:\.5)?\b', name) or "deepseek-v2" in name:
        if "lite" in name:
            return (16.0, 2.4)
        return (236.0, 21.0)
    if "phi-3.5-moe" in name:
        return (42.0, 6.6)
    if "dbrx" in name:
        return (132.0, 36.0)
    if ("snowflake-arctic" in name or re.search(r'\barctic-(?:instruct|base|l|medium)\b', name)) and not ("embed" in name):
        return (480.0, 17.0)
    if "jamba" in name:
        if "large" in name:
            return (398.0, 94.0)
        return (52.0, 12.0)
    if "grok-1" in name:
        return (314.0, 86.0)
    if "olmoe" in name:
        m_ol = re.search(r'olmoe(?:-(\d+(?:\.\d+)?)b-(\d+(?:\.\d+)?)b)?', name)
        if m_ol and m_ol.group(1) and m_ol.group(2):
            return (float(m_ol.group(2)), float(m_ol.group(1)))
        return (7.0, 1.0)
    if "jetmoe" in name:
        m_jet = re.search(r'jetmoe(?:-(\d+(?:\.\d+)?)b)?', name)
        total = float(m_jet.group(1)) if m_jet and m_jet.group(1) else 8.0
        return (total, round(total / 4.0, 1))

    # 4. Generic named MoE models with size indicated (e.g. model-moe-14b, moe-32b, qwen2.5-32b-instruct-moe)
    m3 = (
        re.search(r'(?:^|[-_:])moe[-_:]?(?:.*?[-_:])?(\d+(?:\.\d+)?)b', name)
        or re.search(r'(\d+(?:\.\d+)?)b(?:.*?[-_:])?moe\b', name)
    )
    if m3:
        total_b = float(m3.group(1))
        if total_b > 0:
            active_b = round(total_b / 4.0, 1)
            return (total_b, active_b)

    return None


def is_moe_model(model_name: str) -> bool:
    """Check if model uses a Mixture of Experts architecture."""
    if not model_name:
        return False
    if get_moe_specs(model_name) is not None:
        return True
    name = model_name.lower()
    # Recognize named MoE models even when param weights are omitted, avoiding false positives like embeddings
    if re.search(r'(?:^|[-_:])moe(?:[-_:]|$)', name):
        if not any(x in name for x in ("embed", "tokenizer", "dataset")):
            return True
    return False


def _classify_by_size(size_b: float) -> str:
    """Classify model by parameter count in billions."""
    if size_b < 3:
        return "tiny"
    elif size_b < 7:
        return "small"
    elif size_b < 13:
        return "medium"
    else:
        return "large"


def get_model_size_category(model_name: str) -> str:
    """Detect model size category from model name.
    Returns: 'tiny' (<3B), 'small' (3-7B), 'medium' (7-13B), 'large' (>13B), 'cloud' (z.ai), 'moe'.

    Priority: manual override > MoE active params > total params > keyword fallback.
    """
    if not model_name:
        return "medium"

    # Priority 1: Manual user override
    override = get_model_category_override()
    if override:
        return override

    name = model_name.lower()

    # Cloud provider detection. The explicit ':cloud' / '-cloud' tag counts too:
    # aggregators name models like "minimax-m3:cloud", and such a model must
    # never fall through to the keyword heuristic below.
    if any(k in name for k in ["glm-", "gpt-", "claude-", "gemini-"]):
        return "cloud"
    if re.search(r"[:\-_]cloud\b", name):
        return "cloud"

    # Priority 2: MoE active parameter detection
    moe_specs = get_moe_specs(name)
    if moe_specs:
        total_b, active_b = moe_specs
        return _classify_by_size(active_b)

    # Priority 3: Total parameter count (existing logic)
    size_patterns = [
        (r':(\d+(?:\.\d+)?)b', 1.0),
        (r'[-_](\d+(?:\.\d+)?)b', 1.0),
        (r':(\d+(?:\.\d+)?)x', 1.0),
    ]

    for pattern, multiplier in size_patterns:
        match = re.search(pattern, name)
        if match:
            size_b = float(match.group(1)) * multiplier
            return _classify_by_size(size_b)

    # Priority 4: Keyword fallback.
    # Matched on token boundaries, NOT as bare substrings: "mini" inside
    # "minimax" (a large cloud model) used to classify it as tiny, which
    # silently handed a capable model the crippled small-model prompt, the slim
    # toolset and a tiny history budget. Size markers like "1b" still match
    # anywhere, since they appear glued to the name ("qwen2.5-1.5b").
    tiny_sizes = ["0.5b", "1b", "1.5b", "2b"]
    if any(k in name for k in tiny_sizes):
        return "tiny"
    # "tiny" as a name prefix is reliable (tinyllama, tinydolphin), so a
    # following letter is fine. "mini"/"nano"/"micro" are NOT: minimax and
    # ministral are big models, so those need a boundary on both sides
    # (phi-3-mini matches, minimax-m3 does not).
    if re.search(r"(?:^|[\s\-_:./])tiny", name):
        return "tiny"
    if re.search(r"(?:^|[\s\-_:./])(mini|nano|micro)(?:$|[\s\-_:./0-9])", name):
        return "tiny"

    return "medium"


# ─── Paths & integrations ────────────────────────────────────────────────────

def get_hooks_dir() -> str:
    # Default to local ./plugins folder (evaluated at call time)
    return _get("hooks_dir", str(Path.cwd() / "plugins"))


def set_hooks_dir(path: str):
    _set("hooks_dir", path)


def get_skills_dir() -> str:
    return _get("skills_dir", str(Path.cwd() / "skills"))


def set_skills_dir(path: str):
    _set("skills_dir", path)


def get_visuals_dir() -> str:
    return _get("visuals_dir", str(Path.cwd() / "visuals"))


def set_visuals_dir(path: str):
    _set("visuals_dir", path)


# ─── Feature toggles ─────────────────────────────────────────────────────────

def get_disabled_tools() -> list[str]:
    return _get("disabled_tools", [])


def set_disabled_tools(tools_list: list[str]):
    _set("disabled_tools", tools_list)


def get_verbose_status() -> bool:
    """Whether verbose status indicators (spinners) are enabled."""
    return _get("verbose_status", True)


def set_verbose_status(enabled: bool):
    _set("verbose_status", enabled)


def get_show_tool_results() -> bool:
    """Whether a tool's full output is printed after every call.

    Off by default. A tool result is written for the model — read_file returns
    the whole file, grep_search every match — and printing it pushed the
    conversation off the screen. The compact line keeps the shape of the
    result; /debug or this setting brings back the full panel.
    """
    return _get("show_tool_results", False)


def set_show_tool_results(enabled: bool):
    _set("show_tool_results", enabled)


def get_debug_mode() -> bool:
    """Whether debug mode (detailed logs in chat) is enabled."""
    return _get("debug_mode", False)


def set_debug_mode(enabled: bool):
    _set("debug_mode", enabled)


def get_autonomous_plugins_enabled() -> bool:
    """Whether the AI is allowed to autonomously create plugins."""
    return _get("autonomous_plugins", False)


def set_autonomous_plugins_enabled(enabled: bool):
    _set("autonomous_plugins", enabled)


def get_disabled_plugins() -> list[str]:
    """Globally disabled plugins (by filenames)."""
    return _get("disabled_plugins", [])


def set_disabled_plugins(plugins_list: list[str]):
    _set("disabled_plugins", plugins_list)


def get_trusted_files() -> dict:
    """{absolute path: sha256} of the project config files (hooks.json, permissions.json) the user has read and trusted."""
    return _get("trusted_files", {})


def set_trusted_files(files: dict):
    _set("trusted_files", files)


# ─── RAG / embeddings ────────────────────────────────────────────────────────

def get_embedding_provider() -> str:
    """Configured embedding provider: 'sentence_transformers' or 'ollama'."""
    return _get("embedding_provider", "sentence_transformers")


def set_embedding_provider(provider: str):
    _set("embedding_provider", provider)


def get_ollama_embedding_model() -> str:
    return _get("ollama_embedding_model", "nomic-embed-text")


def get_reranker_model() -> str:
    """Cross-encoder used to rerank research results.

    Empty means the default English model (92MB). Set it to a multilingual one
    (see rerank.MULTILINGUAL_MODEL, ~471MB) when you research in languages other
    than English: the English model scores non-English passages so low that even
    highly relevant ones are buried under mediocre English results.
    """
    return _get("reranker_model", "") or ""


def set_reranker_model(model: str):
    _set("reranker_model", (model or "").strip())


def get_search_languages() -> list:
    """Languages the model should write search queries in.

    Default ["en"]: technical documentation, issues and answers are
    overwhelmingly English, and a translated query silently loses most good
    sources. Add "ru" when local-language sources matter (regional services,
    Habr, non-technical research) — but pair it with a multilingual reranker,
    or those results get scored near-zero and dropped anyway.
    """
    value = _get("search_languages", ["en"])
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    return [v for v in (value or []) if v] or ["en"]


def set_search_languages(languages):
    if isinstance(languages, str):
        languages = [l.strip() for l in languages.split(",")]
    _set("search_languages", [l for l in (languages or []) if l] or ["en"])


def get_brave_api_key() -> str:
    """API key for the optional Brave Search engine. Empty by default: web
    search stays keyless and zero-setup, and adding the key just widens the
    federation with a second independent index."""
    return _api_key("brave_api_key", "BRAVE_API_KEY") or ""


def set_brave_api_key(key: str):
    _set("brave_api_key", (key or "").strip())


def get_stream_read_timeout() -> float:
    """Seconds without a single token before the generation is declared dead.

    Not a limit on how long a model may think — it measures the GAP between
    tokens, so a slow generation is untouched. It exists because a dropped
    connection produced 948 seconds of spinner and had to be killed by hand;
    nothing distinguishes "still working" from "the socket is gone" except
    silence, and forever is the wrong amount of silence to accept.
    """
    try:
        return max(30.0, float(_get("stream_read_timeout", 180)))
    except (TypeError, ValueError):
        return 180.0


def set_stream_read_timeout(seconds: float):
    _set("stream_read_timeout", max(30.0, float(seconds)))


def get_ollama_api_key() -> str:
    """Key for Ollama's hosted web search (ollama.com/settings/keys).

    Separate from the local Ollama server, which needs no key: this one buys a
    second independent index. Measured on real queries, 80% of what it returns
    is absent from the other five engines.

    Note the key is stored in plain JSON like the rest of the config, and it
    also authenticates Ollama's CLOUD MODELS — so it is worth more than a
    search-only key. Treat the config file accordingly.
    """
    return _api_key("ollama_api_key", "OLLAMA_API_KEY") or ""


def set_ollama_api_key(key: str):
    _set("ollama_api_key", (key or "").strip())


def get_search_engine() -> str:
    """Default search engine: 'all', 'duckduckgo', 'brave', 'ollama', 'github', 'stackoverflow', 'wikipedia'."""
    return _get("search_engine", "all") or "all"


def set_search_engine(engine: str):
    """Set default search engine."""
    _set("search_engine", (engine or "all").strip().lower())


def get_mcp_call_timeout() -> float:
    """How long one MCP tool call may take before Argent gives up.

    Was a hard-coded 600. Ten minutes of a frozen terminal with no output and
    no way to cancel is never the right default: an editor that has not
    answered in two minutes is stuck, not busy, and the model can retry far
    cheaper than you can wait. Raise it for genuinely long operations (a build,
    a bake) via config.
    """
    try:
        return max(5.0, float(_get("mcp_call_timeout", 120)))
    except (TypeError, ValueError):
        return 120.0


def set_mcp_call_timeout(seconds: float):
    _set("mcp_call_timeout", max(5.0, float(seconds)))


def get_desktop_notifications() -> bool:
    """Whether a finished automation may raise a desktop notification.

    On by default: a scheduled run you have to remember to check is just a log
    file. Runs stay quiet unless they found something new, broke, or hit an
    action that needs you — see src/automation/notify.py.
    """
    return bool(_get("desktop_notifications", True))


def set_desktop_notifications(enabled: bool):
    _set("desktop_notifications", bool(enabled))


def get_embedding_batch_size() -> int:
    """How many texts go into one native Ollama /api/embed request. Batching is
    far more efficient than concurrent single-text calls; 32 keeps a request
    modest on a small GPU. Clamped to >= 1."""
    try:
        return max(1, int(_get("embedding_batch_size", 32)))
    except (TypeError, ValueError):
        return 32


def get_embedding_concurrency() -> int:
    """Parallel Ollama embedding requests during indexing. Defaults to 4 —
    conservative for a weak local box (10 concurrent requests could OOM a
    small GPU or time out). Raise it in config on a capable machine.
    Clamped to >= 1."""
    try:
        return max(1, int(_get("embedding_concurrency", 4)))
    except (TypeError, ValueError):
        return 4


def set_ollama_embedding_model(model: str):
    _set("ollama_embedding_model", model)


def get_auto_rag() -> bool:
    """Whether RAG semantic search is automatically enabled on startup."""
    return _get("auto_rag", False)


def set_auto_rag(enabled: bool):
    _set("auto_rag", enabled)

def get_auto_kb() -> bool:
    """Whether external Knowledge Bases are automatically loaded on startup."""
    return _get("auto_kb", True)

def set_auto_kb(enabled: bool):
    _set("auto_kb", enabled)


def get_auto_retrieve() -> bool:
    """Whether to automatically run semantic_search on each user query and
    inject the top results into context (proactive RAG), so a weak model
    doesn't have to remember to call the tool. Off by default."""
    return _get("auto_retrieve", False)


def set_auto_retrieve(enabled: bool):
    _set("auto_retrieve", enabled)


# ─── Browser automation ──────────────────────────────────────────────────────

def get_browser_mode() -> str:
    """Browser mode: 'isolated' (Playwright Chromium) or 'user' (CDP to real browser)."""
    return _get("browser_mode", "isolated")


def set_browser_mode(mode: str):
    _set("browser_mode", mode)


def get_browser_state_mode() -> str:
    """State extraction mode: 'auto' (tier-based), 'a11y', or 'dom'."""
    return _get("browser_state_mode", "auto")


def set_browser_state_mode(mode: str):
    if mode not in ("auto", "a11y", "dom"):
        raise ValueError(f"Invalid browser_state_mode: {mode!r}. Must be 'auto', 'a11y', or 'dom'.")
    _set("browser_state_mode", mode)


def get_effective_browser_state_mode() -> str:
    """Resolve 'auto' to 'a11y' or 'dom' based on current model tier.

    tiny/small models get 'dom' (compact, saves tokens);
    medium/large/cloud get 'a11y' (richer semantics).
    """
    mode = get_browser_state_mode()
    if mode == "auto":
        category = get_model_size_category(get_current_model())
        if category in ("tiny", "small"):
            return "dom"
        return "a11y"
    return mode


def get_browser_name() -> str:
    """Which browser to use in 'user' mode: 'auto', 'chrome', 'yandex', 'edge', 'brave'."""
    return _get("browser_name", "auto")


def set_browser_name(name: str):
    _set("browser_name", name)


# ─── MCP servers ─────────────────────────────────────────────────────────────

def get_mcp_servers() -> list[dict]:
    return _get("mcp_servers", [])


def set_mcp_servers(servers: list[dict]):
    _set("mcp_servers", servers)


def add_mcp_server(name: str, server_type: str = "stdio", url: str = None,
                   command: str = None, args: list = None, env: dict = None):
    """Add or update an MCP server in config."""
    servers = [s for s in get_mcp_servers() if s["name"] != name]
    entry = {"name": name, "type": server_type}
    if url:
        entry["url"] = url
    if command:
        entry["command"] = command
    if args:
        entry["args"] = args
    if env:
        entry["env"] = env
    servers.append(entry)
    set_mcp_servers(servers)


def remove_mcp_server(name: str):
    servers = [s for s in get_mcp_servers() if s["name"] != name]
    set_mcp_servers(servers)


# ─── External knowledge bases ────────────────────────────────────────────────

def get_external_kbs() -> list[dict]:
    return _get("external_kbs", [])


def set_external_kbs(kbs: list[dict]):
    _set("external_kbs", kbs)


def add_external_kb(kb_id: str, name: str, path: str):
    kbs = [kb for kb in get_external_kbs() if kb["id"] != kb_id]
    kbs.append({"id": kb_id, "name": name, "path": path, "enabled": True})
    set_external_kbs(kbs)


def remove_external_kb(kb_id: str):
    kbs = [kb for kb in get_external_kbs() if kb["id"] != kb_id]
    set_external_kbs(kbs)


def toggle_external_kb(kb_id: str) -> bool:
    """Toggle the enabled status of an external Knowledge Base. Returns the new status."""
    kbs = get_external_kbs()
    new_status = False
    for kb in kbs:
        if kb["id"] == kb_id:
            kb["enabled"] = not kb.get("enabled", True)
            new_status = kb["enabled"]
            break
    set_external_kbs(kbs)
    return new_status


# ─── Telegram bot ─────────────────────────────────────────────────────────────

def get_telegram_token() -> str | None:
    return _get("telegram_token", None)


def set_telegram_token(token: str | None):
    _set("telegram_token", token.strip() if token else None)


def get_telegram_chat_id() -> str | None:
    val = _get("telegram_chat_id", None)
    return str(val) if val is not None else None


def set_telegram_chat_id(chat_id: str | int | None):
    _set("telegram_chat_id", str(chat_id).strip() if chat_id is not None else None)


def get_telegram_enabled() -> bool:
    return bool(_get("telegram_enabled", False))


def set_telegram_enabled(enabled: bool):
    _set("telegram_enabled", bool(enabled))


def get_telegram_mode() -> str:
    """Mode for Telegram bot: 'shared' (companion, shared agent + console mirror) or 'isolated' (standalone)."""
    mode = _get("telegram_mode", "shared")
    return mode if mode in ("shared", "isolated") else "shared"


def set_telegram_mode(mode: str):
    if mode not in ("shared", "isolated"):
        raise ValueError(f"Unknown Telegram mode: {mode}. Must be 'shared' or 'isolated'.")
    _set("telegram_mode", mode)


def get_sound_notification() -> bool:
    """Whether sound notifications are enabled after turn completion or error."""
    val = _get("sound_notification", True)
    if isinstance(val, str):
        return val.strip().lower() not in ("false", "0", "off", "no", "disable")
    return bool(val)


def set_sound_notification(enabled: bool):
    """Enable or disable sound notifications."""
    _set("sound_notification", bool(enabled))


DEFAULT_SOUND_EVENTS = {
    "done": "system",
    "prompt": "system",
    "error": "system",
}


def get_sound_events_config() -> dict[str, str]:
    """Return dictionary mapping events (done, prompt, error) to their sound targets."""
    raw = _get("sound_events", {})
    res = dict(DEFAULT_SOUND_EVENTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in res and isinstance(v, str):
                res[k] = v.strip()
    return res


def get_sound_event(event: str) -> str:
    """Return sound target ('system', 'synth', 'chord', 'mute', or WAV file path) for event."""
    norm_event = str(event).strip().lower() if event else "done"
    if norm_event in ("confirm", "attention"):
        norm_event = "prompt"
    cfg = get_sound_events_config()
    return cfg.get(norm_event, "system")


def set_sound_event(event: str, value: str):
    """Set sound target for a specific event ('done', 'prompt', 'error') or 'all'."""
    norm_event = str(event).strip().lower() if event else "done"
    norm_val = str(value).strip() if value else "system"

    valid_events = ("done", "prompt", "error", "all", "*", "confirm", "attention")
    if norm_event not in valid_events:
        raise ValueError(
            f"Unknown sound event: '{event}'. Must be one of: done, prompt, error, all"
        )

    current = get_sound_events_config()

    if norm_event in ("all", "*"):
        for k in ("done", "prompt", "error"):
            current[k] = norm_val
    else:
        if norm_event in ("confirm", "attention"):
            norm_event = "prompt"
        current[norm_event] = norm_val

    _set("sound_events", current)


# ── Turn Snapshots & Argent Brain ────────────────────────────────────

def get_turn_snapshot_mode() -> str:
    """Mode for turn snapshots after LLM responses: 'compact', 'verbose', or 'off'.
    Defaults to 'compact'."""
    val = _get("turn_snapshot_mode", "compact")
    if not isinstance(val, str) or val.strip().lower() not in ("compact", "verbose", "off"):
        return "compact"
    return val.strip().lower()


def set_turn_snapshot_mode(mode: str):
    """Set mode for turn snapshots: 'compact', 'verbose', or 'off'."""
    norm = str(mode or "").strip().lower()
    if norm not in ("compact", "verbose", "off"):
        raise ValueError(
            f"Invalid snapshot mode: '{mode}'. Must be 'compact', 'verbose', or 'off'."
        )
    _set("turn_snapshot_mode", norm)


def is_secret_redaction_enabled() -> bool:
    """Whether keys and tokens are hidden in command output, logs and the screen (on by default)."""
    return bool(_get("redact_secrets", True))


def set_secret_redaction_enabled(enabled: bool):
    _set("redact_secrets", bool(enabled))


def is_local_governor_enabled() -> bool:
    """Whether requests to local model servers go through the shared lanes (on by default)."""
    return bool(_get("local_governor", True))


def set_local_governor_enabled(enabled: bool):
    _set("local_governor", bool(enabled))


def get_local_parallelism() -> int:
    """How many requests one local model server may work on at once. 1 = strictly one at a time (the safe
    default on a laptop); a server started with OLLAMA_NUM_PARALLEL>1 and enough VRAM can take more."""
    try:
        return min(8, max(1, int(_get("local_parallelism", 1))))
    except (TypeError, ValueError):
        return 1


def set_local_parallelism(value: int):
    _set("local_parallelism", min(8, max(1, int(value))))


def is_lower_server_priority_enabled() -> bool:
    """Whether Argent lowers the OS priority of the local model server while it uses it (off by default:
    another program's priority is not ours to change unasked)."""
    return bool(_get("lower_model_server_priority", False))


def set_lower_server_priority_enabled(enabled: bool):
    _set("lower_model_server_priority", bool(enabled))


def get_advisor_model() -> str:
    """Model of the Ultracode advisor (a stronger second opinion). Empty = the main model stands in."""
    return _get("advisor_model", "")


def set_advisor_model(name: str):
    _set("advisor_model", name)


def get_advisor_provider() -> str:
    """Provider of the advisor model. Empty = the main provider."""
    return _get("advisor_provider", "")


def set_advisor_provider(name: str):
    _set("advisor_provider", name)


def is_advisor_cloud_consented() -> bool:
    """The person agreed that plans and patches go to a CLOUD advisor while the main model is local."""
    return bool(_get("advisor_cloud_consent", False))


def set_advisor_cloud_consented(enabled: bool):
    _set("advisor_cloud_consent", bool(enabled))


def get_ultracode_level():
    """'auto' (from calibration) or a fixed ladder level 0..3."""
    value = _get("ultracode_level", "auto")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, int) and 0 <= value <= 3:
        return value
    return "auto"


def set_ultracode_level(value):
    _set("ultracode_level", value if value == "auto" else min(3, max(0, int(value))))


def is_ultracode_keyword_enabled() -> bool:
    """Whether the word `ultracode` in a message switches the mode on for the session (like /ultracode on)."""
    return bool(_get("workflow_keyword_trigger", True))


def set_ultracode_keyword_enabled(enabled: bool):
    _set("workflow_keyword_trigger", bool(enabled))


def get_ultracode_max_minutes() -> int:
    """Wall-clock budget of one Ultracode run; past it the run stops with a report and can be resumed."""
    try:
        return min(24 * 60, max(5, int(_get("ultracode_max_minutes", 120))))
    except (TypeError, ValueError):
        return 120


def set_ultracode_max_minutes(minutes: int):
    _set("ultracode_max_minutes", min(24 * 60, max(5, int(minutes))))


def is_memory_enabled() -> bool:
    """Long-term memory notes (src/memory): the index in the prompt and the `memory` tool. On by default."""
    return bool(_get("memory_enabled", True))


def set_memory_enabled(enabled: bool):
    _set("memory_enabled", bool(enabled))


def get_dream_model() -> str:
    """Model that consolidates the memory while the PC is idle. Empty = the main model."""
    return _get("dream_model", "")


def set_dream_model(name: str):
    _set("dream_model", name)


def get_dream_provider() -> str:
    """Provider of the consolidation model. Empty = the main provider."""
    return _get("dream_provider", "")


def set_dream_provider(name: str):
    _set("dream_provider", name)


def is_memory_cloud_consented() -> bool:
    """The person agreed that (redacted) session excerpts go to a CLOUD model for the consolidation."""
    return bool(_get("memory_cloud_consent", False))


def set_memory_cloud_consented(enabled: bool):
    _set("memory_cloud_consent", bool(enabled))


def is_dream_global_enabled() -> bool:
    """Whether the consolidation may also read and change the GLOBAL notes (read in every project). Off by
    default: what a project's sessions taught should not reach other projects unless the person wants it."""
    return bool(_get("dream_global", False))


def set_dream_global_enabled(enabled: bool):
    _set("dream_global", bool(enabled))


def is_dream_on_battery_allowed() -> bool:
    """Whether the consolidation may run while the laptop is on battery. Off by default."""
    return bool(_get("dream_on_battery", False))


def set_dream_on_battery_allowed(enabled: bool):
    _set("dream_on_battery", bool(enabled))


def get_desktop_allowed_apps() -> list:
    """Programs (process names, e.g. notepad.exe) Argent may act in through desktop control. Empty at the start."""
    value = _get("desktop_allowed_apps", [])
    if not isinstance(value, list):
        return []
    return [str(v).strip().lower() for v in value if str(v).strip()]


def set_desktop_allowed_apps(apps: list):
    cleaned = []
    for app in apps or []:
        name = str(app).strip().lower()
        if name and name not in cleaned:
            cleaned.append(name)
    _set("desktop_allowed_apps", cleaned)


def get_desktop_stop_chord() -> str:
    """The key chord that ends desktop control at once, wherever the focus is."""
    value = str(_get("desktop_stop_chord", "") or "").strip().lower()
    return value or "ctrl+alt+shift+f12"


def set_desktop_stop_chord(chord: str):
    _set("desktop_stop_chord", str(chord or "").strip().lower())


def is_limit_server_threads_enabled() -> bool:
    """Whether requests ask a local Ollama to leave a couple of CPU cores free (num_thread). Off by default."""
    return bool(_get("limit_server_threads", False))


def set_limit_server_threads_enabled(enabled: bool):
    _set("limit_server_threads", bool(enabled))


def is_local_agent_enabled() -> bool:
    """Whether local agent execution (delegate_local tool) is enabled."""
    return bool(_get("local_agent_enabled", False))


def set_local_agent_enabled(enabled: bool):
    """Enable or disable local agent execution."""
    _set("local_agent_enabled", bool(enabled))


def get_local_model_tier() -> int:
    """Maximum allowed local model tier (0: Cloud, 1: <=0.5B, 2: <=1.5B, 3: <=3B)."""
    try:
        return max(0, min(3, int(_get("local_model_tier", 2))))
    except (TypeError, ValueError):
        return 2


def set_local_model_tier(tier: int | str):
    """Set maximum allowed local model tier (0/'cloud', 1, 2, or 3)."""
    if str(tier).strip().lower() in ("0", "cloud", "t0", "c"):
        val = 0
    else:
        try:
            val = max(0, min(3, int(tier)))
        except (TypeError, ValueError):
            val = 2
    _set("local_model_tier", val)


def get_local_model_path() -> str | None:
    """User-specified local model path overriding tier defaults."""
    return _get("local_model_path", None)


def set_local_model_path(path: str | None):
    """Set custom local model path (None to reset to tier defaults)."""
    _set("local_model_path", (path or "").strip() if path else None)


def get_local_agent_cloud_provider() -> str:
    """Provider for Cloud tier (T0). Defaults to auxiliary provider or 'openrouter'."""
    prov = _get("local_agent_cloud_provider")
    if prov:
        return str(prov).strip().lower()
    return _get("auxiliary_provider") or "openrouter"


def set_local_agent_cloud_provider(provider: str | None):
    """Set cloud provider for delegated Cloud tier."""
    _set("local_agent_cloud_provider", (provider or "").strip().lower() if provider else None)


def get_local_agent_cloud_model() -> str:
    """Model for Cloud tier (T0). Defaults to auxiliary model or 'google/gemini-2.0-flash-exp:free'."""
    m = _get("local_agent_cloud_model")
    if m:
        return str(m).strip()
    return _get("auxiliary_model") or "google/gemini-2.0-flash-exp:free"


def set_local_agent_cloud_model(model: str | None):
    """Set cloud model for delegated Cloud tier."""
    _set("local_agent_cloud_model", (model or "").strip() if model else None)


def is_local_agent_cloud_fallback() -> bool:
    """Whether to fallback to Cloud tier (T0) if local RAM/GPU resources are insufficient."""
    return bool(_get("local_agent_cloud_fallback", True))


def set_local_agent_cloud_fallback(enabled: bool):
    """Enable or disable automatic cloud fallback on local resource exhaustion."""
    _set("local_agent_cloud_fallback", bool(enabled))


def get_local_agent_timeout() -> float:
    """Timeout in seconds for delegated model inference."""
    try:
        return max(5.0, float(_get("local_agent_timeout", 30.0)))
    except (TypeError, ValueError):
        return 30.0


def set_local_agent_timeout(seconds: float):
    """Set timeout in seconds for delegated model inference."""
    _set("local_agent_timeout", max(5.0, float(seconds)))


# ── Needle 3 Coprocessor ────────────────────────────────────────────

def get_needle_enabled() -> bool:
    """Whether Needle 3 coprocessor is enabled for fast local tool dispatch."""
    return bool(_get("needle_enabled", False))


def set_needle_enabled(enabled: bool):
    """Enable or disable Needle 3 coprocessor."""
    _set("needle_enabled", bool(enabled))


def get_needle_fast_path() -> bool:
    """Whether Needle 3 Fast-Path mode is active (instant tool interception)."""
    return bool(_get("needle_fast_path", False))


def set_needle_fast_path(enabled: bool):
    """Enable or disable Needle 3 Fast-Path mode for routine tools."""
    _set("needle_fast_path", bool(enabled))


# ── Fast-Path Dispatch & MicroModels ────────────────────────────────

def get_fast_mode() -> bool:
    """Whether fast-path tool dispatch is enabled."""
    return get_fast_dispatch_backend() != "off"


def set_fast_mode(enabled: bool):
    """Enable or disable fast-path tool dispatch."""
    if enabled:
        prev = _get("fast_dispatch_backend_prev", "leeth")
        set_fast_dispatch_backend(prev if prev != "off" else "leeth")
    else:
        cur = get_fast_dispatch_backend()
        if cur != "off":
            _set("fast_dispatch_backend_prev", cur)
        set_fast_dispatch_backend("off")


def get_micromodel_enabled() -> bool:
    """Whether custom MicroModel intent router is active."""
    return get_fast_dispatch_backend() == "micromodel"


def set_micromodel_enabled(enabled: bool):
    """Enable or disable custom MicroModel intent router."""
    if enabled:
        set_fast_dispatch_backend("micromodel")
    else:
        set_fast_dispatch_backend("off")


def get_speculative_lookahead_enabled() -> bool:
    """Whether Speculative CoT Lookahead Pre-fetching is active."""
    return _get("speculative_lookahead_enabled", True) and get_micromodel_enabled()


def set_speculative_lookahead_enabled(enabled: bool):
    """Enable or disable Speculative CoT Lookahead Pre-fetching."""
    _set("speculative_lookahead_enabled", bool(enabled))


# ── Leeth System One Decision Model ──────────────────────────────────

def get_leeth_enabled() -> bool:
    """Whether Leeth System One non-autoregressive decision model is active."""
    val = _get("leeth_enabled", None)
    if val is not None:
        return bool(val)
    try:
        from src.leeth.runtime import leeth
        return leeth.is_available
    except Exception:
        return False


def set_leeth_enabled(enabled: bool):
    """Enable or disable Leeth System One decision engine."""
    _set("leeth_enabled", bool(enabled))


def get_fast_dispatch_backend() -> str:
    """Fast-path dispatch backend: 'leeth', 'micromodel', 'needle', or 'off'.
    
    Default is 'leeth' if leeth is enabled/available, otherwise 'off'.
    """
    val = _get("fast_dispatch_backend", None)
    if val is not None:
        return str(val).lower().strip()
    if get_leeth_enabled():
        return "leeth"
    return "off"


def set_fast_dispatch_backend(backend: str):
    """Set fast-path dispatch backend ('leeth', 'micromodel', 'needle', 'off')."""
    b = str(backend).lower().strip()
    if b not in ("leeth", "micromodel", "needle", "off"):
        raise ValueError(f"Invalid fast dispatch backend '{backend}'. Choose from: 'leeth', 'micromodel', 'needle', 'off'")
    _set("fast_dispatch_backend", b)


LEETH_BACKENDS = ("legacy", "leeth2")


def get_leeth_backend() -> str:
    """Which model answers Leeth's typed questions: 'legacy' (400M + mmBERT router, the default) or 'leeth2'
    (single calibrated mmBERT-based model, src/leeth2). The env var LEETH_BACKEND overrides the stored setting."""
    import os

    val = os.environ.get("LEETH_BACKEND") or _get("leeth_backend", "legacy")
    val = str(val).lower().strip()
    return val if val in LEETH_BACKENDS else "legacy"


def set_leeth_backend(backend: str):
    b = str(backend).lower().strip()
    if b not in LEETH_BACKENDS:
        raise ValueError(f"Invalid Leeth backend '{backend}'. Choose from: {', '.join(LEETH_BACKENDS)}")
    _set("leeth_backend", b)


def get_leeth_device() -> str:
    """Get preferred device for Leeth inference ('auto', 'cuda', 'cpu', 'onnx')."""
    return _get("leeth_device", "auto")


def set_leeth_device(device: str):
    """Set preferred device for Leeth inference ('auto', 'cuda', 'cpu', 'onnx')."""
    dev = device.lower().strip()
    if dev not in ("auto", "cuda", "cpu", "onnx"):
        raise ValueError(f"Invalid device '{device}'. Choose from: 'auto', 'cuda', 'cpu', 'onnx'")
    _set("leeth_device", dev)


def get_leeth_review_mode() -> str:
    """Review mode for Leeth fast-path proposals: 'off', 'agent' (LLM reviewer), or 'human' (terminal prompt).

    Default is 'agent': every fast-path proposal is reviewed by the main agent
    before execution (active learning — speed is traded for training signal).
    """
    val = _get("leeth_review_mode", "agent")
    return val if val in ("off", "agent", "human") else "agent"


def set_leeth_review_mode(mode: str):
    """Set Leeth proposal review mode ('off', 'agent', 'human')."""
    mode = mode.lower().strip()
    if mode not in ("off", "agent", "human"):
        raise ValueError(f"Invalid review mode '{mode}'. Choose from: 'off', 'agent', 'human'")
    _set("leeth_review_mode", mode)


def get_leeth_feedback_enabled() -> bool:
    """Whether Leeth decisions are logged for active-learning fine-tuning."""
    return bool(_get("leeth_feedback_enabled", True))


def set_leeth_feedback_enabled(enabled: bool):
    """Enable or disable Leeth decision logging."""
    _set("leeth_feedback_enabled", bool(enabled))


# ── Interface style (the Rust TUI) ───────────────────────────────────

UI_STYLES = ("premium", "classic")


def get_ui_style() -> str:
    """How the TUI looks: 'premium' (designed surfaces) or 'classic' (the REPL's look).

    ARGENT_UI_STYLE overrides the saved value for one run. The terminal REPL has no
    premium look and ignores this.
    """
    env = (os.environ.get("ARGENT_UI_STYLE") or "").strip().lower()
    if env in UI_STYLES:
        return env
    val = str(_get("ui_style", "premium") or "").strip().lower()
    return val if val in UI_STYLES else "premium"


def set_ui_style(style: str):
    """Save the TUI's style ('premium' or 'classic')."""
    style = str(style).strip().lower()
    if style not in UI_STYLES:
        raise ValueError(f"Invalid UI style '{style}'. Choose from: 'premium', 'classic'")
    _set("ui_style", style)
