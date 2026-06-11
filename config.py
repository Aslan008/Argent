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
    """Save configuration to disk and update cache."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = config
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


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


def get_temperature() -> float:
    return _get("temperature", 0.3)


def set_temperature(val: float):
    _set("temperature", val)


def get_context_window() -> int:
    """Configured context window size in tokens (default 32768)."""
    return _get("context_window", 32768)


def set_context_window(size: int):
    _set("context_window", size)


def get_max_generation_tokens() -> int:
    """Max tokens for generation. Default 8192 to prevent truncation in local APIs."""
    return _get("max_generation_tokens", 8192)


def set_max_generation_tokens(tokens: int):
    _set("max_generation_tokens", tokens)


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

    # Cloud provider detection
    if any(k in name for k in ["glm-", "gpt-", "claude-", "gemini-"]):
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

    # Priority 4: Keyword fallback
    tiny_keywords = ["0.5b", "1b", "1.5b", "2b", "tiny", "mini", "nano", "micro"]
    if any(k in name for k in tiny_keywords):
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


def set_ollama_embedding_model(model: str):
    _set("ollama_embedding_model", model)


def get_auto_rag() -> bool:
    """Whether RAG semantic search is automatically enabled on startup."""
    return _get("auto_rag", False)


def set_auto_rag(enabled: bool):
    _set("auto_rag", enabled)


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
