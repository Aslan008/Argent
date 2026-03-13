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
