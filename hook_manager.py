import os
import sys
import importlib.util
from pathlib import Path
from ui import print_system, print_error

def get_global_hooks_dir() -> Path:
    """Returns the path to the global ~/.argent/hooks directory."""
    # We use ~/.argent/hooks/ for global plugins
    hook_dir = Path.home() / ".argent" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    return hook_dir

class HookManager:
    """Manages dynamic loading and execution of custom user plugins (hooks)."""
    
    def __init__(self):
        self.hooks_dir = get_global_hooks_dir()
        self._plugins = []
        self._load_plugins()

    def _load_plugins(self):
        """Discovers and loads all .py files in the hooks directory."""
        if not self.hooks_dir.exists():
            return
            
        sys.path.insert(0, str(self.hooks_dir))
        
        loaded_count = 0
        for item in self.hooks_dir.iterdir():
            if item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                plugin_name = item.stem
                try:
                    spec = importlib.util.spec_from_file_location(plugin_name, str(item))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self._plugins.append(module)
                        loaded_count += 1
                except Exception as e:
                    print_error(f"Failed to load plugin '{item.name}': {e}")
                    
        # Remove from sys.path to avoid polluting the host app imports unnecessarily
        if str(self.hooks_dir) in sys.path:
            sys.path.remove(str(self.hooks_dir))
            
        if loaded_count > 0:
            print_system(f"Loaded {loaded_count} global plugin(s) from {self.hooks_dir}")

    def call_hook(self, event_name: str, *args, **kwargs):
        """
        Calls a specific event on all loaded plugins that implement it.
        Returns a list of results from all plugins that handled the event.
        For boolean events (like on_tool_call), you may want to parse the results.
        For modifier events (like pre_prompt), you'll want to use `call_modifier_hook`.
        """
        results = []
        for plugin in self._plugins:
            if hasattr(plugin, event_name) and callable(getattr(plugin, event_name)):
                try:
                    func = getattr(plugin, event_name)
                    res = func(*args, **kwargs)
                    results.append(res)
                except Exception as e:
                    print_error(f"Plugin '{plugin.__name__}' failed during '{event_name}': {e}")
        return results

    def call_modifier_hook(self, event_name: str, initial_value, *args, **kwargs):
        """
        Passes a value sequentially through all plugins that implement the event.
        Useful for modifying strings (like user prompts).
        """
        current_value = initial_value
        for plugin in self._plugins:
            if hasattr(plugin, event_name) and callable(getattr(plugin, event_name)):
                try:
                    func = getattr(plugin, event_name)
                    # The event signature must be: func(current_value, *args, **kwargs) -> modified_value
                    modified_value = func(current_value, *args, **kwargs)
                    if modified_value is not None:
                        current_value = modified_value
                except Exception as e:
                    print_error(f"Plugin '{plugin.__name__}' failed during '{event_name}': {e}")
        return current_value

    def get_custom_commands(self) -> dict:
        """
        Discovers all functions in plugins that start with 'command_'.
        Returns a dictionary mapping command strings (e.g. 'deploy') to the function.
        """
        commands = {}
        for plugin in self._plugins:
            for attr_name in dir(plugin):
                if attr_name.startswith("command_"):
                    func = getattr(plugin, attr_name)
                    if callable(func):
                        # command_deploy -> deploy
                        cmd_name = attr_name[len("command_"):]
                        commands[cmd_name] = func
        return commands

# Singleton instance
hook_manager = HookManager()
