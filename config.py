import json
import os
from pathlib import Path

# We store configuration in the user's home directory
CONFIG_FILE = Path.home() / ".argent_coder_config.json"

DEFAULT_MODEL = "llama3.1"

def load_config() -> dict:
    """Load configuration from disk."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"model": DEFAULT_MODEL}

def save_config(config: dict):
    """Save configuration to disk."""
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
