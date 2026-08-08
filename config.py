import json
import re
from pathlib import Path

# We store configuration in the user's home directory
CONFIG_FILE = Path.home() / ".argent_coder_config.json"

DEFAULT_MODEL = "llama3.1"

_CONFIG_CACHE = None


def load_config() -> dict:
    """Load configuration from disk with caching."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                _CONFIG_CACHE = json.load(f)
                return _CONFIG_CACHE
        except json.JSONDecodeError:
            pass
    _CONFIG_CACHE = {"model": DEFAULT_MODEL}
    return _CONFIG_CACHE


def save_config(config: dict):
    """Save configuration to disk and update cache.

    Written atomically (temp + os.replace) so a crash mid-write or a second
    Argent instance racing on the same file can't corrupt the config.
    """
    global _CONFIG_CACHE
    _CONFIG_CACHE = config
    from atomic_io import atomic_write_text
    atomic_write_text(CONFIG_FILE, json.dumps(config, indent=4))


def _get(key: str, default=None):
    return load_config().get(key, default)


def _set(key: str, value):
    config = load_config()
    config[key] = value
    save_config(config)


# ─── Model & provider ────────────────────────────────────────────────────────

ZAI_ENDPOINT_GENERAL = "https://api.z.ai/api/paas/v4/"
ZAI_ENDPOINT_CODING = "https://api.z.ai/api/coding/paas/v4/"
DEFAULT_KOBOLDCPP_URL = "http://localhost:5001/v1"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


def get_current_model() -> str:
    return _get("model", DEFAULT_MODEL)


def set_current_model(model_name: str):
    _set("model", model_name)


def get_provider() -> str:
    """Currently selected API provider (e.g. 'ollama', 'zai', 'koboldcpp')."""
    return _get("provider", "ollama")


def set_provider(provider_name: str):
    _set("provider", provider_name)


def get_zai_api_key() -> str | None:
    return _get("zai_api_key")


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
    return _get("openrouter_api_key")


def set_openrouter_api_key(api_key: str):
    _set("openrouter_api_key", api_key)


def get_openrouter_url() -> str:
    return _get("openrouter_url", DEFAULT_OPENROUTER_URL)


def set_openrouter_url(url: str):
    _set("openrouter_url", url)


def get_temperature() -> float:
    return _get("temperature", 0.3)


def set_temperature(val: float):
    _set("temperature", val)


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


def get_strip_reasoning() -> bool:
    """Whether reasoning (thinking) blocks are stripped from history before LLM calls."""
    return _get("strip_reasoning", True)


def set_strip_reasoning(enabled: bool):
    _set("strip_reasoning", enabled)


# ─── Model size classification ───────────────────────────────────────────────

VALID_MODEL_CATEGORIES = ("tiny", "small", "medium", "large", "cloud")


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
    Returns: 'tiny' (<3B), 'small' (3-7B), 'medium' (7-13B), 'large' (>13B), 'cloud' (z.ai).

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
    # Models like "LFM2.5-8B-A1B" have 8B total but only 1B active params.
    # The "A<N>B" suffix indicates active parameters — classify by those.
    moe_match = re.search(r'[-_]a(\d+(?:\.\d+)?)b', name)
    if moe_match:
        active_b = float(moe_match.group(1))
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

def get_obsidian_vault() -> str | None:
    return _get("obsidian_vault")


def set_obsidian_vault(path: str):
    _set("obsidian_vault", path)


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
    return _get("brave_api_key", "") or ""


def set_brave_api_key(key: str):
    _set("brave_api_key", (key or "").strip())


def get_ollama_api_key() -> str:
    """Key for Ollama's hosted web search (ollama.com/settings/keys).

    Separate from the local Ollama server, which needs no key: this one buys a
    second independent index. Measured on real queries, 80% of what it returns
    is absent from the other five engines.

    Note the key is stored in plain JSON like the rest of the config, and it
    also authenticates Ollama's CLOUD MODELS — so it is worth more than a
    search-only key. Treat the config file accordingly.
    """
    return _get("ollama_api_key", "") or ""


def set_ollama_api_key(key: str):
    _set("ollama_api_key", (key or "").strip())


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
