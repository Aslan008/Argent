import json
import os
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

def get_current_model() -> str:
    """Get currently selected model from config."""
    config = load_config()
    return config.get("model", DEFAULT_MODEL)

def set_current_model(model_name: str):
    """Save newly selected model to config."""
    config = load_config()
    config["model"] = model_name
    save_config(config)

def get_provider() -> str:
    """Get currently selected API provider (e.g., 'ollama', 'zai')."""
    config = load_config()
    return config.get("provider", "ollama")

def set_provider(provider_name: str):
    """Save newly selected API provider."""
    config = load_config()
    config["provider"] = provider_name
    save_config(config)

def get_zai_api_key() -> str | None:
    """Get the Z.ai API key from config."""
    config = load_config()
    return config.get("zai_api_key")

def set_zai_api_key(api_key: str):
    """Save the Z.ai API key to config."""
    config = load_config()
    config["zai_api_key"] = api_key
    save_config(config)

ZAI_ENDPOINT_GENERAL = "https://api.z.ai/api/paas/v4/"
ZAI_ENDPOINT_CODING = "https://api.z.ai/api/coding/paas/v4/"

def get_zai_endpoint() -> str:
    """Get the configured Z.ai API endpoint. Defaults to General API."""
    config = load_config()
    return config.get("zai_endpoint", ZAI_ENDPOINT_GENERAL)

def set_zai_endpoint(endpoint: str):
    """Save the Z.ai API endpoint to config."""
    config = load_config()
    config["zai_endpoint"] = endpoint
    save_config(config)

DEFAULT_KOBOLDCPP_URL = "http://localhost:5001/v1"

def get_koboldcpp_url() -> str:
    """Get the configured KoboldCPP API URL."""
    config = load_config()
    return config.get("koboldcpp_url", DEFAULT_KOBOLDCPP_URL)

def set_koboldcpp_url(url: str):
    """Save the KoboldCPP API URL to config."""
    config = load_config()
    config["koboldcpp_url"] = url
    save_config(config)


def get_obsidian_vault() -> str | None:
    """Get the configured Obsidian vault path."""
    config = load_config()
    return config.get("obsidian_vault")

def set_obsidian_vault(path: str):
    """Save the Obsidian vault path to config."""
    config = load_config()
    config["obsidian_vault"] = path
    save_config(config)

def get_disabled_tools() -> list[str]:
    """Get the list of globally disabled tools."""
    config = load_config()
    return config.get("disabled_tools", [])

def set_disabled_tools(tools_list: list[str]):
    """Save the list of disabled tools to config."""
    config = load_config()
    config["disabled_tools"] = tools_list
    save_config(config)

def get_verbose_status() -> bool:
    """Check if verbose status indicators (spinners) are enabled."""
    config = load_config()
    return config.get("verbose_status", True)

def set_verbose_status(enabled: bool):
    """Enable or disable verbose status indicators."""
    config = load_config()
    config["verbose_status"] = enabled
    save_config(config)

def get_hooks_dir() -> str:
    """Get the configured hooks directory path."""
    config = load_config()
    # Default to local ./plugins folder
    default = str(Path.cwd() / "plugins")
    return config.get("hooks_dir", default)

def set_hooks_dir(path: str):
    """Save the custom hooks directory path to config."""
    config = load_config()
    config["hooks_dir"] = path
    save_config(config)

def get_skills_dir() -> str:
    """Get the configured skills directory path."""
    config = load_config()
    # Default to local ./skills folder
    default = str(Path.cwd() / "skills")
    return config.get("skills_dir", default)

def set_skills_dir(path: str):
    """Save the custom skills directory path to config."""
    config = load_config()
    config["skills_dir"] = path
    save_config(config)

def get_visuals_dir() -> str:
    """Get the configured visuals directory path."""
    config = load_config()
    # Default to local ./visuals folder
    default = str(Path.cwd() / "visuals")
    return config.get("visuals_dir", default)

def set_visuals_dir(path: str):
    """Save the custom visuals directory path to config."""
    config = load_config()
    config["visuals_dir"] = path
    save_config(config)

def get_autonomous_plugins_enabled() -> bool:
    """Check if AI is allowed to autonomously create plugins."""
    config = load_config()
    return config.get("autonomous_plugins", False)

def set_autonomous_plugins_enabled(enabled: bool):
    """Save preference for autonomous plugin creation."""
    config = load_config()
    config["autonomous_plugins"] = enabled
    save_config(config)
def get_disabled_plugins() -> list[str]:
    """Get the list of globally disabled plugins (by filenames)."""
    config = load_config()
    return config.get("disabled_plugins", [])

def set_disabled_plugins(plugins_list: list[str]):
    """Save the list of disabled plugins to config."""
    config = load_config()
    config["disabled_plugins"] = plugins_list
    save_config(config)

def get_context_window() -> int:
    """Get the configured context window size."""
    config = load_config()
    # Default to 8192 if not set
    return config.get("context_window", 8192)

def set_context_window(size: int):
    """Save the context window size to config."""
    config = load_config()
    config["context_window"] = size
    save_config(config)

def get_embedding_provider() -> str:
    """Get the configured embedding provider. 'sentence_transformers' or 'ollama'."""
    config = load_config()
    return config.get("embedding_provider", "sentence_transformers")

def set_embedding_provider(provider: str):
    """Save the embedding provider. Valid: 'sentence_transformers', 'ollama'."""
    config = load_config()
    config["embedding_provider"] = provider
    save_config(config)

def get_ollama_embedding_model() -> str:
    """Get the Ollama model to use for embeddings."""
    config = load_config()
    return config.get("ollama_embedding_model", "nomic-embed-text")

def set_ollama_embedding_model(model: str):
    """Save the Ollama embedding model."""
    config = load_config()
    config["ollama_embedding_model"] = model
    save_config(config)

def get_mcp_servers() -> list[dict]:
    """Get the list of configured MCP servers."""
    config = load_config()
    return config.get("mcp_servers", [])

def set_mcp_servers(servers: list[dict]):
    """Save the MCP servers configuration."""
    config = load_config()
    config["mcp_servers"] = servers
    save_config(config)

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
    """Remove an MCP server from config."""
    servers = [s for s in get_mcp_servers() if s["name"] != name]
    set_mcp_servers(servers)

import re

def get_model_size_category(model_name: str) -> str:
    """Detect model size category from model name.
    Returns: 'tiny' (<3B), 'small' (3-7B), 'medium' (7-13B), 'large' (>13B), 'cloud' (z.ai).
    """
    if not model_name:
        return "medium"
    
    name = model_name.lower()
    
    if any(k in name for k in ["glm-", "gpt-", "claude-", "gemini-"]):
        return "cloud"
    
    size_patterns = [
        (r':(\d+(?:\.\d+)?)b', 1.0),
        (r'[-_](\d+(?:\.\d+)?)b', 1.0),
        (r':(\d+(?:\.\d+)?)x', 1.0),
    ]
    
    for pattern, multiplier in size_patterns:
        match = re.search(pattern, name)
        if match:
            size_b = float(match.group(1)) * multiplier
            if size_b < 3:
                return "tiny"
            elif size_b < 7:
                return "small"
            elif size_b < 13:
                return "medium"
            else:
                return "large"
    
    tiny_keywords = ["0.5b", "1b", "1.5b", "2b", "tiny", "mini", "nano", "micro"]
    if any(k in name for k in tiny_keywords):
        return "tiny"
    
    return "medium"

def get_strip_reasoning() -> bool:
    """Check if reasoning (thinking) blocks should be stripped from history before LLM calls."""
    config = load_config()
    return config.get("strip_reasoning", True)

def set_strip_reasoning(enabled: bool):
    """Save user preference for stripping reasoning blocks from history."""
    config = load_config()
    config["strip_reasoning"] = enabled
    save_config(config)

def get_temperature() -> float:
    """Get the configured temperature for LLM generation. Defaults to 0.3."""
    config = load_config()
    return config.get("temperature", 0.3)

def set_temperature(val: float):
    """Save the configured temperature for LLM generation."""
    config = load_config()
    config["temperature"] = val
    save_config(config)


def get_browser_mode() -> str:
    """Get browser mode: 'isolated' (Playwright Chromium) or 'user' (CDP to real browser).
    Default: 'isolated'.
    """
    config = load_config()
    return config.get("browser_mode", "isolated")

def set_browser_mode(mode: str):
    """Save browser mode. Valid: 'isolated', 'user'."""
    config = load_config()
    config["browser_mode"] = mode
    save_config(config)

def get_browser_name() -> str:
    """Get which browser to use in 'user' mode: 'auto', 'chrome', 'yandex', 'edge', 'brave'.
    Default: 'auto' (first detected).
    """
    config = load_config()
    return config.get("browser_name", "auto")

def set_browser_name(name: str):
    """Save which browser to use. Valid: 'auto', 'chrome', 'yandex', 'edge', 'brave'."""
    config = load_config()
    config["browser_name"] = name
    save_config(config)
