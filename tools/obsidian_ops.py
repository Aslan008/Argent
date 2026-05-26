import os
import yaml
from pathlib import Path
from config import get_obsidian_vault
from logger import get_logger

log = get_logger("tools")

def write_obsidian_note(note_path: str, content: str, tags: list = None, aliases: list = None, overwrite: bool = False) -> str:
    """Create or overwrite an Obsidian note with correct YAML frontmatter."""
    vault_path = get_obsidian_vault()
    if not vault_path:
        return "Error: Obsidian vault path is not configured. Please use the `/obsidian <path>` command to set it."
        
    try:
        base_path = Path(vault_path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Obsidian vault directory '{vault_path}' does not exist."
            
        if not note_path.endswith('.md'):
            note_path += '.md'
            
        full_path = (base_path / note_path).resolve()
        
        try:
            common = os.path.commonpath([str(full_path), str(base_path)])
            if common != str(base_path):
                return f"Error: Invalid path '{note_path}' attempts to write outside the Obsidian vault."
        except ValueError:
            return f"Error: Invalid path '{note_path}'."
            
        if full_path.exists() and not overwrite:
            return f"Error: Note '{note_path}' already exists. Use overwrite=True if you meant to replace it, or use replace_in_file for localized edits."
            
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        yaml_lines = ["---"]
        has_frontmatter = False
        
        if aliases and isinstance(aliases, list) and len(aliases) > 0:
            yaml_lines.append("aliases:")
            for alias in aliases:
                yaml_lines.append(f"  - {alias}")
            has_frontmatter = True
            
        if tags and isinstance(tags, list) and len(tags) > 0:
            yaml_lines.append("tags:")
            for tag in tags:
                clean_tag = str(tag).lstrip('#').replace(' ', '_')
                yaml_lines.append(f"  - {clean_tag}")
            has_frontmatter = True
            
        yaml_lines.append("---")
        
        if '\\n' in content or '\\t' in content:
            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            
        final_content = ""
        if has_frontmatter:
            final_content = "\n".join(yaml_lines) + "\n\n" + content
        else:
            final_content = content
            
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(final_content)
            
        return f"Successfully created Obsidian note.\nIMPORTANT: The absolute path to this note is '{full_path}'.\nYou MUST use this absolute path if you need to read, replace_in_file, or delete this note!"
        
    except Exception as e:
        return f"Error writing Obsidian note '{note_path}': {e}"

def search_obsidian_notes(query: str = None, tag: str = None) -> str:
    """Search for notes in the Obsidian vault by tag or text content."""
    vault_path = get_obsidian_vault()
    if not vault_path:
        return "Error: Obsidian vault path is not configured. Please use the `/obsidian <path>` command to set it."
        
    base_path = Path(vault_path).expanduser().resolve()
    if not base_path.exists():
        return f"Error: Obsidian vault directory '{vault_path}' does not exist."
        
    if not query and not tag:
        return "Error: You must provide either a 'query' or a 'tag' to search for."
        
    results = []
    target_tag = tag.lstrip('#').lower() if tag else None
    
    for md_file in base_path.rglob("*.md"):
        if any(part.startswith('.') for part in md_file.parts):
            continue
            
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            match_found = False
            snippet = ""
            
            if target_tag:
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        frontmatter_str = parts[1]
                        try:
                            fm = yaml.safe_load(frontmatter_str) or {}
                            tags_in_fm = fm.get('tags', [])
                            if isinstance(tags_in_fm, str):
                                tags_in_fm = [t.strip() for t in tags_in_fm.split(',')]
                            if tags_in_fm and isinstance(tags_in_fm, list):
                                if any(target_tag == str(t).lstrip('#').lower() for t in tags_in_fm):
                                    match_found = True
                        except Exception:
                            pass
                
                if not match_found and f"#{target_tag}" in content.lower():
                    match_found = True
            else:
                match_found = True
                
            if match_found and query:
                q_lower = query.lower()
                idx = content.lower().find(q_lower)
                if idx != -1:
                    match_found = True
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(query) + 40)
                    snippet = "... " + content[start:end].replace('\n', ' ') + " ..."
                else:
                    match_found = False
                    
            if match_found:
                rel_path = md_file.relative_to(base_path)
                res_str = f"- **{rel_path}**"
                if snippet:
                    res_str += f"\n  Snippet: {snippet}"
                results.append(res_str)
                
        except Exception:
            pass
            
    if not results:
        conditions = []
        if query: conditions.append(f"query='{query}'")
        if tag: conditions.append(f"tag='{tag}'")
        return f"No notes found matching " + " and ".join(conditions) + "."
        
    truncated = ""
    if len(results) > 20:
        truncated = f"\n...and {len(results) - 20} more matches."
        results = results[:20]
        
    return f"Found {len(results)} matches:\n" + "\n".join(results) + truncated

def update_obsidian_properties(note_path: str, add_tags: list = None, remove_tags: list = None, add_aliases: list = None, remove_aliases: list = None, properties: dict = None) -> str:
    """Safely update Obsidian note YAML frontmatter properties."""
    vault_path = get_obsidian_vault()
    if not vault_path:
        return "Error: Obsidian vault path is not configured. Please use the `/obsidian <path>` command to set it."
        
    try:
        base_path = Path(vault_path).expanduser().resolve()
        if not note_path.endswith('.md'):
            note_path += '.md'
            
        full_path = (base_path / note_path).resolve()
        
        if not full_path.exists():
            return f"Error: Note '{note_path}' does not exist."
            
        if not str(full_path).startswith(str(base_path)):
            return f"Error: Invalid path attempts to write outside vault."
            
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        fm = {}
        body = content
        
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1]) or {}
                    body = parts[2]
                except yaml.YAMLError as yerr:
                    return f"Error parsing existing YAML in '{note_path}': {yerr}"
        
        if not isinstance(fm, dict):
            fm = {}
            
        def update_list(field, add_items, remove_items, clean_hash=False):
            current = fm.get(field, [])
            if isinstance(current, str):
                current = [i.strip() for i in current.split(',')]
            if not isinstance(current, list):
                current = []
                
            if add_items and isinstance(add_items, list):
                for item in add_items:
                    clean_item = str(item).lstrip('#').replace(' ', '_') if clean_hash else str(item)
                    if clean_item not in current:
                        current.append(clean_item)
                        
            if remove_items and isinstance(remove_items, list):
                for item in remove_items:
                    clean_item = str(item).lstrip('#').replace(' ', '_') if clean_hash else str(item)
                    if clean_item in current:
                        current.remove(clean_item)
                        
            if current:
                fm[field] = current
            elif field in fm:
                del fm[field]
                
        update_list('tags', add_tags, remove_tags, clean_hash=True)
        update_list('aliases', add_aliases, remove_aliases)
        
        if properties and isinstance(properties, dict):
            for k, v in properties.items():
                if v is None:
                    if k in fm:
                        del fm[k]
                else:
                    fm[k] = v
                    
        if fm:
            new_fm_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False)
            body_trimmed = body.lstrip('\n')
            final_content = f"---\n{new_fm_str}---\n"
            if body_trimmed:
                final_content += "\n" + body_trimmed
        else:
            final_content = body.lstrip('\n')
            
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        return f"Successfully updated properties for '{note_path}'."
        
    except Exception as e:
        return f"Error updating properties for '{note_path}': {e}"
