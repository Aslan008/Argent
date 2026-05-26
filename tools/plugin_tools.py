from pathlib import Path
from config import get_hooks_dir
from hook_manager import hook_manager
from tools._helpers import _validate_code_syntax
from logger import get_logger

log = get_logger("tools")

def create_plugin(name: str, code: str) -> str:
    """Creates a new Python plugin in the configured plugins directory.
    Automatically handles directory path, .py extension, and syntax validation.
    """
    try:
        hooks_dir = Path(get_hooks_dir()).expanduser().resolve()
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        if not name.endswith(".py"):
            name += ".py"
            
        file_path = hooks_dir / name
        
        if "from ui import console" not in code and "import ui" not in code:
            code = "from ui import console\n" + code
            
        temp_path = hooks_dir / f"_temp_{name}"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        error = _validate_code_syntax(str(temp_path))
        if error:
            temp_path.unlink()
            return f"Failed to create plugin due to syntax error:\n{error}"
            
        if file_path.exists():
            file_path.unlink()
        temp_path.rename(file_path)
        
        hook_manager.reload_plugins()
        
        return f"Successfully created plugin '{name}' in {hooks_dir}. It is now active."
    except Exception as e:
        return f"Error creating plugin: {e}"


def delete_plugin(name: str) -> str:
    """Deletes a plugin from the plugins directory and reloads the hook manager."""
    try:
        hooks_dir = Path(get_hooks_dir()).expanduser().resolve()
        
        if not name.endswith(".py"):
            name += ".py"
            
        file_path = hooks_dir / name
        
        if file_path.exists():
            file_path.unlink()
            hook_manager.reload_plugins()
            return f"Plugin '{name}' deleted successfully."
        else:
            return f"Plugin '{name}' not found in {hooks_dir}."
    except Exception as e:
        return f"Error deleting plugin: {e}"
