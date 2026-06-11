import os
from pathlib import Path

def get_obsidian_vault():
    return r"C:\Obsidian Folder\Obsidianist"

def test_write_obsidian_note(note_path, content, tags=None, aliases=None, overwrite=False):
    vault_path = get_obsidian_vault()
    if not vault_path:
        return "Error: Obsidian vault path is not configured."
        
    try:
        base_path = Path(vault_path).expanduser().resolve()
        print(f"Base path: {base_path}")
        if not base_path.exists():
            return f"Error: Obsidian vault directory '{vault_path}' does not exist."
            
        if not note_path.endswith('.md'):
            note_path += '.md'
            
        full_path = (base_path / note_path).resolve()
        print(f"Full path: {full_path}")
        
        print(f"Full path starts with base path: {str(full_path).startswith(str(base_path))}")
        
        if not str(full_path).startswith(str(base_path)):
            return f"Error: Invalid path '{note_path}' attempts to write outside the Obsidian vault."
            
        if full_path.exists() and not overwrite:
            return f"Error: Note '{note_path}' already exists."
            
        # Check if we can create the parent
        print(f"Parent: {full_path.parent}")
        # We won't actually create it in the test script to avoid side effects if not needed, 
        # but let's check if parents can be made.
        
        return "Checks passed (theoretically)"
        
    except Exception as e:
        return f"Error: {e}"

print(test_write_obsidian_note("Templates/Unique_Technical_Template.md", "some content"))
