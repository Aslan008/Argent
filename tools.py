import os
import json
import subprocess
from pathlib import Path
import questionary
from ddgs import DDGS
import requests
import difflib
from rich.syntax import Syntax
from rich.panel import Panel
import queue
import threading
import ctypes
import time
from bs4 import BeautifulSoup
import webbrowser
from datetime import datetime
from ui import console
from config import get_obsidian_vault, get_hooks_dir
import yaml
import py_compile
from deep_research import run_deep_research
from project_manager import ProjectManager
from hook_manager import hook_manager
from skill_manager import skill_manager
from intelligence import intel
from mcp_client import mcp_client
from file_tracker import snapshot
from logger import get_logger
from memory_manager import memory

log = get_logger("tools")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def __resolve_path(file_path: str) -> Path:
    """Resolve file path locally first. If missing, check if it exists in the configured Obsidian Vault."""
    path = Path(file_path).expanduser().resolve()
    if path.exists():
        return path
        
    vault_str = get_obsidian_vault()
    if vault_str:
        vault_path = Path(vault_str).expanduser().resolve()
        try:
            possible_vault_file = (vault_path / file_path).resolve()
            if possible_vault_file.exists() and str(possible_vault_file).startswith(str(vault_path)):
                return possible_vault_file
        except Exception:
            pass
            
    return path

def _is_plugin_path_restricted(file_path: str) -> str | None:
    """Checks if the path is inside the plugins directory and returns an error if restricted."""
    try:
        from config import get_hooks_dir
        abs_path = os.path.abspath(file_path)
        hooks_dir = os.path.abspath(get_hooks_dir())
        if abs_path.startswith(hooks_dir):
            return (
                f"ERROR: Direct modification of files in the plugins directory is restricted. "
                f"You MUST use the `create_plugin` or `delete_plugin` tools for all plugin-related tasks. "
                f"These tools ensure mandatory syntax validation and automatic system reloading."
            )
    except Exception:
        pass
    return None

def _validate_code_syntax(file_path: str) -> str | None:
    """Quietly checks if the written Python or C# file has syntax errors.
    Returns the error string if failed, or None if passed."""
    if not file_path:
        return None
    file_path = str(file_path).strip()
    
    # 1. Python Validation
    if file_path.endswith('.py'):
        try:
            py_compile.compile(file_path, doraise=True)
            return None
        except py_compile.PyCompileError as e:
            return f"SyntaxError in your Python code:\n{e.msg}\n\nPlease fix this syntax error using the `replace_in_file` tool."
        except Exception as e:
            return f"Validation Error: {e}"
            
    # 2. C# (Unity) Validation via Headless .NET
    if file_path.endswith('.cs'):
        path_obj = Path(file_path).resolve()
        csproj_file = None
        for p in path_obj.parents:
            cs_files = list(p.glob("*.csproj"))
            if cs_files:
                csproj_file = cs_files[0]
                break
                
        if csproj_file:
            try:
                import subprocess
                result = subprocess.run(["dotnet", "build", str(csproj_file), "-v", "q", "/nologo"], capture_output=True, text=True, timeout=15)
                if result.returncode != 0:
                    return f"C# Compiler Error:\n{result.stdout}\n\nPlease fix this compiler error using the `replace_in_file` tool."
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
                
    return None

def _print_diff(old_text, new_text, filename):
    """Show a beautiful unified diff in the console."""
    diff = list(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}"
    ))
    if diff:
        diff_str = "".join(diff)
        syntax = Syntax(diff_str, "diff", theme="monokai", background_color="default")
        console.print(Panel(syntax, title=f"Changes in {filename}", border_style="green"))

# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------

# --- Project Brain Tools ---

def add_project_task(description: str) -> str:
    """Add a task to the current project plan."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project. Use /project to start one."
    task_id = pm.add_task(description)
    return f"Task {task_id} added: '{description}'. If you have no more tasks to add, stop calling tools and reply 'DONE'."

def complete_project_task(task_id: int, summary: str) -> str:
    """Mark a project task as completed. You MUST provide a summary of what you did."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    # Phase guard: only allow completion during execution
    status = pm.data.get("status", "")
    if status != "executing":
        return f"Error: Cannot complete tasks during '{status}' phase. You should only call add_project_task now."
    pm.complete_task(task_id, summary)
    return f"Task {task_id} completed. Summary saved."

def plan_work_changes(strategy: str, files_to_edit: str, files_to_create: str) -> str:
    """Submit the investigation plan for an existing project. Call this ONCE during the Phase 1 investigation."""
    pm = ProjectManager()
    if not pm.active or pm.data.get("mode") != "work":
        return "Error: No active /work session."
        
    edit_list = [f.strip() for f in files_to_edit.split(',') if f.strip()] if files_to_edit else []
    create_list = [f.strip() for f in files_to_create.split(',') if f.strip()] if files_to_create else []
    
    # Check if we need confirmation for new files
    if create_list and not pm.data.get("work_auto_mode", False):
        print(f"\n[bold yellow]Agent requesting to create NEW files for /work:[/bold yellow] {', '.join(create_list)}")
        approved = questionary.confirm("Do you want to allow these files to be created?").ask()
        if not approved:
            return f"Error: User denied creation of {', '.join(create_list)}. Revise your plan to ONLY modify existing files, without creating these new ones. Call plan_work_changes again."
        
    pm.data["work_strategy"] = strategy
    pm.data["files_to_edit"] = edit_list
    pm.data["files_to_create"] = create_list
    pm.set_status("work_planning")
    return "Plan accepted. Moving to task generation phase."

def add_work_task(description: str) -> str:
    """Add a micro-task for the current /work session."""
    pm = ProjectManager()
    if not pm.active or pm.data.get("mode") != "work":
        return "Error: No active /work session."
    task_id = pm.add_task(description)
    return f"Work task {task_id} added: '{description}'. If you have no more tasks to add, stop calling tools and reply 'DONE'."

def list_project_tasks() -> str:
    """View the current project status with all tasks and their summaries."""
    pm = ProjectManager()
    if not pm.active:
        return "No active project."
    data = pm.data
    result = f"Project: {data['objective']}\nProgress: {pm.get_progress_display()}\n\n"
    for t in data["tasks"]:
        marker = "[DONE]" if t["status"] == "completed" else "[ ]"
        result += f"{t['id']}. {marker} {t['description']}\n"
        if t.get("result_summary"):
            result += f"   Result: {t['result_summary']}\n"
    return result.strip()

def write_project_spec(spec: str) -> str:
    """Write the detailed technical specification for the current project. Call this ONCE during the specification phase."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    pm.set_spec(spec)
    return "Project specification saved successfully."

def write_project_architecture(architecture: str, files: str) -> str:
    """Write the high-level architecture map for the project. List ALL files, their purpose, and dependencies between them."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    # Parse the comma-separated file list
    file_list = [f.strip() for f in files.split(',') if f.strip()]
    pm.set_architecture(architecture, file_list)
    return f"Architecture saved. Files to detail: {', '.join(file_list) if file_list else 'NONE - provide the files parameter!'}"

def write_file_spec(filename: str, spec: str) -> str:
    """Write a detailed specification for a single file. Include: path, imports, classes, methods (with params and types), fields, and logic description."""
    pm = ProjectManager()
    if not pm.active:
        return "Error: No active project."
    pm.set_file_spec(filename, spec)
    pending = pm.get_pending_spec_files()
    if pending:
        return f"Spec for '{filename}' saved. Remaining files without spec: {', '.join(pending)}"
    else:
        return f"Spec for '{filename}' saved. All files now have detailed specs!"

def read_file(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """Read the contents of a file. Optionally read a specific range of lines (1-indexed)."""
    try:
        path = __resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        with open(path, "r", encoding="utf-8") as f:
            if start_line is not None or end_line is not None:
                lines = f.readlines()
                total = len(lines)
                s = max(1, start_line or 1) - 1
                e = min(total, end_line or total)
                selected = lines[s:e]
                header = f"[Lines {s+1}-{e} of {total}]\n"
                return header + "".join(selected)
            else:
                content = f.read()
                line_count = content.count('\n') + 1
                if line_count > 500:
                    return f"[File has {line_count} lines. Showing first 500. Use start_line/end_line to read specific sections.]\n" + "\n".join(content.splitlines()[:500])
                return content
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

def delete_file(file_path: str) -> str:
    """Delete a file from the file system."""
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = __resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        print(f"\n[bold yellow]Agent requesting to delete file:[/bold yellow] {file_path}")
        approved = questionary.confirm("Do you want to allow this file to be deleted?").ask()
        
        if not approved:
            return f"Deletion aborted by user. The file '{file_path}' was NOT deleted."
            
        path.unlink()
        return f"Successfully deleted '{file_path}'."
    except Exception as e:
        return f"Error deleting file '{file_path}': {e}"

def write_file(file_path: str, content: str) -> str:
    """Write or overwrite content to a file. Creates directories if needed."""
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            snapshot(str(path))
        if '\\n' in content or '\\t' in content:
            content = content.replace('\\\\', '\\').replace('\\n', '\n').replace('\\t', '\t')
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            return f"File '{file_path}' written successfully, BUT COMPILATION FAILED:\n\n{validation_error}"
            
        # Synchronous RAG update
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        log.info("write_file: %s (%d chars)", file_path, len(content))
        memory.add_file_modified(file_path)
        memory.add_completed(f"Wrote {file_path} ({len(content)} chars)")
        return f"Successfully wrote to '{file_path}'."
    except Exception as e:
        log.error("write_file error %s: %s", file_path, e)
        return f"Error writing file '{file_path}': {e}"

def write_obsidian_note(note_path: str, content: str, tags: list = None, aliases: list = None, overwrite: bool = False) -> str:
    """Create or overwrite an Obsidian note with correct YAML frontmatter."""
    vault_path = get_obsidian_vault()
    if not vault_path:
        return "Error: Obsidian vault path is not configured. Please use the `/obsidian <path>` command to set it."
        
    try:
        base_path = Path(vault_path).expanduser().resolve()
        if not base_path.exists():
            return f"Error: Obsidian vault directory '{vault_path}' does not exist."
            
        # Ensure the note path ends with .md
        if not note_path.endswith('.md'):
            note_path += '.md'
            
        full_path = (base_path / note_path).resolve()
        
        # Security check to ensure we don't write outside the vault
        if not str(full_path).startswith(str(base_path)):
            return f"Error: Invalid path '{note_path}' attempts to write outside the Obsidian vault."
            
        if full_path.exists() and not overwrite:
            return f"Error: Note '{note_path}' already exists. Use overwrite=True if you meant to replace it, or use replace_in_file for localized edits."
            
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build YAML frontmatter
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
                # Remove '#' and replace spaces with underscores
                clean_tag = str(tag).lstrip('#').replace(' ', '_')
                yaml_lines.append(f"  - {clean_tag}")
            has_frontmatter = True
            
        yaml_lines.append("---")
        
        # Decode literal escapes like in write_file
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
        # Skip hidden directories like .obsidian
        if any(part.startswith('.') for part in md_file.parts):
            continue
            
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            match_found = False
            snippet = ""
            
            # 1. Check tags if tag is provided
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
                match_found = True # No tag filter, file is valid for text search
                
            # 2. Check query if provided and file passed tag check
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
        
    # Limit max returned results to 20 to avoid giant context usage
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
                    
        # Write back
        if fm:
            new_fm_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False)
            final_content = f"---\n{new_fm_str}---\n" + body.lstrip('\n')
        else:
            final_content = body.lstrip('\n')
            
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
            
        return f"Successfully updated properties for '{note_path}'."
        
    except Exception as e:
        return f"Error updating properties for '{note_path}': {e}"

import ast

def replace_python_function(file_path: str, function_name: str, new_code: str) -> str:
    """Surgically replace a top-level function or class method in a Python file. 
    function_name can be 'my_func' or 'MyClass.my_method'.
    """
    try:
        path = __resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        snapshot(str(path))
        
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        # Parse AST to find the exact line numbers
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Error: The existing file '{file_path}' has a SyntaxError and cannot be parsed: {e}"

        target_node = None
        
        # Determine if we are looking for a Class.Method or a top-level function
        parts = function_name.split('.')
        if len(parts) == 1:
            # Top-level function
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                    target_node = node
                    break
        elif len(parts) == 2:
            # Class method
            class_name, method_name = parts
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                            target_node = child
                            break
                    if target_node:
                        break
        else:
            return f"Error: Invalid function_name format '{function_name}'. Use 'func' or 'Class.func'."

        if not target_node:
            return f"Error: Function/Method '{function_name}' not found in '{file_path}'."

        # AST nodes have 1-indexed lineno and end_lineno
        start_line = target_node.lineno - 1  # 0-indexed for our array
        # Account for decorators: use the lowest lineno of decorators if they exist
        if hasattr(target_node, "decorator_list") and target_node.decorator_list:
            start_line = getattr(target_node.decorator_list[0], 'lineno', target_node.lineno) - 1

        end_line = target_node.end_lineno # This is the line number *after* the function ends (or the exact end)

        lines = source.splitlines()
        
        # Re-construct the file
        prefix = lines[:start_line]
        suffix = lines[end_line:] if end_line is not None and end_line < len(lines) else []

        # Ensure new_code trailing spacing is appropriate
        new_lines = new_code.strip('\n').split('\n')
        
        # Calculate the original base indentation of the function we are replacing
        original_first_line = lines[start_line]
        base_indent_str = original_first_line[:len(original_first_line) - len(original_first_line.lstrip())]
        
        # Normalize the incoming code's indentation
        if new_lines:
            incoming_first_line = new_lines[0]
            incoming_base_indent_str = incoming_first_line[:len(incoming_first_line) - len(incoming_first_line.lstrip())]
            
            # Re-indent everything
            for i, line in enumerate(new_lines):
                if not line.strip():
                    new_lines[i] = "" # clean blank lines
                    continue
                    
                # Strip the incoming base indent, then add the original base indent
                if line.startswith(incoming_base_indent_str):
                    stripped_line = line[len(incoming_base_indent_str):]
                else:
                    stripped_line = line.lstrip() # Fallback
                
                new_lines[i] = base_indent_str + stripped_line

        final_lines = prefix + new_lines + suffix
        new_source = '\n'.join(final_lines) + '\n'

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_source)

        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            # For AST surgeries, usually the LLM messes up indentation.
            return f"Function replaced, BUT COMPILATION FAILED:\n\n{validation_error}\nHint: Check indentation (4 spaces per block). Use replace_python_function again to fix it."

        # Show visual diff
        _print_diff(source, new_source, file_path)

        # Synchronous RAG update
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        return f"Successfully replaced '{function_name}' in '{file_path}'."
        
    except Exception as e:
        import traceback
        return f"Error surgically replacing function: {e}\n{traceback.format_exc()}"

def replace_in_file(file_path: str, target_text: str, replacement_text: str) -> str:
    """Replace exactly matching text in a file with new text."""
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = __resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        if target_text not in content:
            hint = ""
            lines = target_text.strip().split('\n')
            if lines:
                first_line = lines[0].strip()
                if first_line and len(first_line) > 3:
                    idx = content.find(first_line)
                    if idx != -1:
                        start_idx = max(0, idx - 50)
                        end_idx = min(len(content), idx + len(first_line) + 300)
                        actual_snippet = content[start_idx:end_idx]
                        hint = f"\n\nHint: We found a partial match for your target_text. Here is the EXACT text from the file (including whitespaces/newlines):\n```\n{actual_snippet}\n```\nCopy the exact text from this snippet for your target_text."
                        
            return f"Error: The target text was not found in '{file_path}'. Make sure it matches exactly, including whitespace and indentation.{hint}"

        # Same decoding of literal escapes locally so standard match works
        target_text_processed = target_text
        if '\\n' in target_text_processed or '\\t' in target_text_processed:
            target_text_processed = target_text_processed.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            
        replacement_text_processed = replacement_text
        if '\\n' in replacement_text_processed or '\\t' in replacement_text_processed:
            replacement_text_processed = replacement_text_processed.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')

        # Count occurrences
        count = content.count(target_text_processed)
        if count > 1:
            return f"Error: The target text appears {count} times in '{file_path}'. Please provide a more specific, unique block of text to replace."

        if target_text_processed not in content:
            # Fallback hint mechanism
            start_snippet = target_text_processed[:30].strip()
            # Try to find something that looks like the start text
            lines = content.split('\n')
            hint = ""
            for i, line in enumerate(lines):
                if start_snippet in line:
                    context = '\n'.join(lines[max(0, i-2):min(len(lines), i+10)])
                    hint = f"\n\nHINT: I found something similar around line {i+1}. It looks exactly like this:\n```\n{context}\n```\n\nMake sure your `target_text` has the EXACT spacing and indentation shown in this snippet."
                    break
                    
            return f"Error: The target text was not found in '{file_path}'. Make sure you matched spacing and indentation perfectly.{hint}"

        new_content = content.replace(target_text_processed, replacement_text_processed)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            # Revert changes because it broke compilation
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Modification aborted because it broke code compilation. Changes reverted.\n\n{validation_error}"
            
        # Show visual diff
        _print_diff(content, new_content, file_path)

        # Synchronous RAG update
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        log.info("replace_in_file: %s (replaced %d chars)", file_path, len(target_text_processed))
        memory.add_file_modified(file_path)
        memory.add_completed(f"Edited {file_path}")
        return f"Successfully replaced text in '{file_path}'."
    except Exception as e:
        log.error("replace_in_file error %s: %s", file_path, e)
        return f"Error replacing text in '{file_path}': {e}"

def multi_replace_in_file(changes_json: str) -> str:
    """Apply multiple text replacements across one or multiple files using a JSON array string.
    The JSON array should contain objects with 'file_path', 'target_text', and 'replacement_text' keys.
    Example: '[{"file_path": "src/main.py", "target_text": "old_func()", "replacement_text": "new_func()"}, {"file_path": "src/utils.py", "target_text": "old_var", "replacement_text": "new_var"}]'
    """
    try:
        import json
        changes = json.loads(changes_json)
        if not isinstance(changes, list):
            return "Error: changes_json must be a JSON array of objects."
            
        report = []
        for change in changes:
            fp = change.get("file_path")
            
            restriction_error = _is_plugin_path_restricted(fp)
            if restriction_error:
                report.append(f"Skipping '{fp}': {restriction_error}")
                continue
                
            target = change.get("target_text", "")
            repl = change.get("replacement_text", "")
            if not fp:
                report.append("Skipping change: 'file_path' is missing.")
                continue
            res = replace_in_file(fp, target, repl)
            report.append(f"[{fp}]: {res}")
            
        return "Multi-replace execution finished:\n" + "\n".join(report)
    except Exception as e:
        return f"Error executing multi_replace: {e}"

def get_file_outline(file_path: str) -> str:
    """
    Analyzes a Python file and returns a structured outline of its contents,
    including classes, functions, and their methods.
    """
    try:
        path = __resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        snapshot(str(path))
        
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Error: The file '{file_path}' has a SyntaxError and cannot be parsed: {e}"

        outline = []
        
        def _get_indent(node):
            # Helper to determine indentation level for nested structures
            if hasattr(node, 'col_offset'):
                return '    ' * (node.col_offset // 4) # Assuming 4 spaces per indent
            return ''

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                outline.append(f"{_get_indent(node)}def {node.name}(...): (line {node.lineno})")
            elif isinstance(node, ast.AsyncFunctionDef):
                outline.append(f"{_get_indent(node)}async def {node.name}(...): (line {node.lineno})")
            elif isinstance(node, ast.ClassDef):
                bases = [b.id if isinstance(b, ast.Name) else '...' for b in node.bases]
                outline.append(f"{_get_indent(node)}class {node.name}({', '.join(bases)}): (line {node.lineno})")
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, ast.FunctionDef):
                        outline.append(f"{_get_indent(item)}    def {item.name}(...): (line {item.lineno})")
                    elif isinstance(item, ast.AsyncFunctionDef):
                        outline.append(f"{_get_indent(item)}    async def {item.name}(...): (line {item.lineno})")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.module if isinstance(node, ast.ImportFrom) else ''
                names = ', '.join([n.name for n in node.names])
                outline.append(f"{_get_indent(node)}import {module_name}{' from ' if module_name else ''}{names} (line {node.lineno})")
            elif isinstance(node, ast.Assign):
                # Simple top-level assignments
                targets = ', '.join([t.id for t in node.targets if isinstance(t, ast.Name)])
                if targets:
                    outline.append(f"{_get_indent(node)}Variable: {targets} (line {node.lineno})")

        if not outline:
            return f"No significant structures found in '{file_path}'."
            
        return f"Outline of '{file_path}':\n" + "\n".join(outline)

    except Exception as e:
        return f"Error getting file outline for '{file_path}': {e}"

def list_directory(dir_path: str) -> str:
    """List the contents of a directory."""
    try:
        path = __resolve_path(dir_path)
        if not path.exists():
            return f"Error: Directory '{dir_path}' does not exist."
        if not path.is_dir():
            return f"Error: '{dir_path}' is not a directory."
        
        items = list(path.iterdir())
        if not items:
            return f"Directory '{dir_path}' is empty."
        
        output = [f"Contents of {dir_path}:"]
        for item in items:
            type_str = "DIR" if item.is_dir() else "FILE"
            output.append(f"[{type_str}] {item.name}")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing directory '{dir_path}': {e}"

def search_files(directory: str = ".", pattern: str = "*", name_contains: str = None, content_contains: str = None, max_results: int = 50) -> str:
    """Recursively search for files matching criteria. Can filter by file pattern, name, and content."""
    try:
        start_path = __resolve_path(directory)
        if not start_path.exists():
            return f"Error: Directory '{directory}' does not exist."
        if not start_path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        results = []
        name_filter = name_contains.lower() if name_contains else None
        content_filter = content_contains.lower() if content_contains else None
        
        for file_path in start_path.rglob(pattern):
            if not file_path.is_file():
                continue
            
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if any(part in ['node_modules', '__pycache__', 'Library', 'Temp', 'obj', 'bin'] for part in file_path.parts):
                continue
            
            if name_filter and name_filter not in file_path.name.lower():
                continue
            
            snippet = None
            if content_filter:
                try:
                    if file_path.suffix.lower() in ['.exe', '.dll', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.mp3', '.mp4', '.wav', '.asset', '.meta', '.prefab', '.unity']:
                        continue
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if content_filter not in content.lower():
                        continue
                    idx = content.lower().find(content_filter)
                    if idx != -1:
                        start = max(0, idx - 40)
                        end = min(len(content), idx + len(content_filter) + 60)
                        snippet = content[start:end].replace('\
', ' ').strip()
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                except Exception:
                    continue
            
            results.append({'path': str(file_path), 'name': file_path.name, 'snippet': snippet})
            
            if len(results) >= max_results:
                break
        
        if not results:
            conditions = []
            if pattern != '*':
                conditions.append(f"pattern='{pattern}'")
            if name_contains:
                conditions.append(f"name contains '{name_contains}'")
            if content_contains:
                conditions.append(f"content contains '{content_contains}'")
            condition_str = " and ".join(conditions) if conditions else "any file"
            return f"No files found matching: {condition_str}"
        
        output = [f"Found {len(results)} file(s):"]
        output.append("-" * 60)
        
        for r in results:
            output.append(f"{r['path']}")
            if r['snippet']:
                output.append(f"  >> {r['snippet']}")
        
        if len(results) >= max_results:
            output.append(f"\
(Results limited to {max_results}. Use max_results parameter to see more.)")
        
        return "\
".join(output)
        
    except Exception as e:
        return f"Error searching files: {e}"

def run_command(command: str) -> str:
    """Execute a console command and return its output. Requires user confirmation. Streams output to console."""
    console.print(f"\n[bold yellow]Agent requesting to run command:[/bold yellow] {command}")
    approved = questionary.confirm("Do you want to allow this command to run?").ask()
    
    if not approved:
        return f"Execution aborted by user. The command '{command}' was NOT run."
        
    try:
        def decode_output(b: bytes) -> str:
            if not b:
                return ""
            try:
                return b.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return b.decode('cp866')
                except UnicodeDecodeError:
                    return b.decode('cp1251', errors='replace')
                    
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        
        output_lines = []
        for raw_line in iter(process.stdout.readline, b""):
            if not raw_line: 
                break
            decoded_line = decode_output(raw_line)
            output_lines.append(decoded_line)
            # Print to console in real-time, removing trailing newline since print adds one
            console.print(f"[dim]{decoded_line.rstrip()}[/dim]")
            
        process.stdout.close()
        process.wait()
        
        final_output = "".join(output_lines).strip()
        output = f"Exit code: {process.returncode}\n"
        
        if final_output:
            output += f"OUTPUT:\n{final_output}\n"
            
        log.info("run_command: %s (exit=%d)", command, process.returncode)
        
        # Special cases: some Windows programs always return non-zero exit codes on success
        cmd_lower = command.strip().lower()
        is_fire_and_forget = any(cmd_lower.startswith(p) for p in ["explorer", "start ", "start.", 'start"'])
        
        if process.returncode != 0 and not is_fire_and_forget:
            memory.add_error(f"Command '{command}' failed (exit={process.returncode})")
        else:
            memory.add_completed(f"Ran: {command}")
        return output.strip()
    except Exception as e:
        log.error("run_command error: %s: %s", command, e)
        return f"Error running command '{command}': {e}"

def run_admin_command(command: str) -> str:
    """Execute a PowerShell command with Administrator privileges (UAC prompt)."""
    console.print(f"\n[bold yellow]Agent requesting to run command as ADMINISTRATOR:[/bold yellow] {command}")
    approved = questionary.confirm("Do you want to allow this command to run with Admin privileges (UAC)?").ask()
    
    if not approved:
        return f"Execution aborted by user. The admin command '{command}' was NOT run."
        
    try:
        temp_out = Path("C:/Windows/Temp/argent_admin_out.txt")
        if temp_out.exists():
            temp_out.unlink()
            
        # Wrap command to output to temp file
        wrapped_command = f"{command} > '{temp_out}' 2>&1"
        
        # Execute via ShellExecuteW with 'runas' verb
        # UINT ShellExecuteW(HWND hwnd, LPCWSTR lpOperation, LPCWSTR lpFile, LPCWSTR lpParameters, LPCWSTR lpDirectory, INT nShowCmd);
        # 0 = SW_HIDE (hide the window)
        result = ctypes.windll.shell32.ShellExecuteW(
            None, 
            "runas", 
            "powershell.exe", 
            f"-Command \"{wrapped_command}\"", 
            None, 
            0
        )
        
        # result <= 32 means error in ShellExecute
        if result <= 32:
            return f"Error: UAC prompt was denied or execution failed. Error code: {result}"
            
        # Wait for file to become available or timeout
        timeout = 20
        start_time = time.time()
        while time.time() - start_time < timeout:
            if temp_out.exists():
                try:
                    with open(temp_out, "r", encoding="utf-8", errors="replace") as f:
                        out = f.read().strip()
                    temp_out.unlink()  # Cleanup
                    return f"Admin execution completed.\nOutput:\n{out}"
                except PermissionError:
                    pass # Still writing
            time.sleep(0.5)
            
        return "Admin execution started, but timed out waiting for output file. It may still be running in the background."
        
    except Exception as e:
        return f"Error running admin command '{command}': {e}"

def read_git_diff() -> str:
    """Read the current unstaged and staged git diff of the project."""
    try:
        # Check if it's a git repo
        is_git = subprocess.run("git rev-parse --is-inside-work-tree", shell=True, capture_output=True, text=True)
        if is_git.returncode != 0:
            return "Error: This directory is not a Git repository."
            
        # Get unstaged diff
        unstaged = subprocess.run("git diff", shell=True, capture_output=True, text=True).stdout
        # Get staged diff
        staged = subprocess.run("git diff --cached", shell=True, capture_output=True, text=True).stdout
        
        res = ""
        if staged:
            res += "=== STAGED CHANGES (READY TO COMMIT) ===\n" + staged + "\n"
        if unstaged:
            res += "=== UNSTAGED CHANGES ===\n" + unstaged + "\n"
            
        return res if res else "No changes detected in Git."
    except Exception as e:
        return f"Error reading git diff: {e}"

def ask_user_questions(questions: list) -> str:
    """Ask the user a series of structured questions.
    Expects a list of dimension objects:
    [{"type": "text"|"single_choice"|"multi_choice", "question": "...", "options": ["opt1", "opt2"]}]
    Returns a structured string with all answers.
    """
    from prompt_toolkit import prompt as ptk_prompt
    
    if not isinstance(questions, list):
        return "Error: 'questions' must be a JSON array of objects."
        
    responses = {}
    console.print("\n[bold cyan]🔍 Уточнение требований:[/bold cyan]")
    
    for q in questions:
        q_type = q.get("type", "text")
        q_text = q.get("question", "Question?")
        options = q.get("options", [])
        
        console.print(f"\n[bold yellow]{q_text}[/bold yellow]")
        
        if q_type == "text":
            console.print("[dim](Введите текст и нажмите Enter)[/dim]")
            try:
                answer = ptk_prompt("Ваш ответ ❯ ")
                responses[q_text] = answer.strip() if answer.strip() else "No answer"
            except (KeyboardInterrupt, EOFError):
                responses[q_text] = "Skipped"
                
        elif q_type in ("single_choice", "multi_choice"):
            display_options = options.copy()
            if "✏ Свой вариант..." not in display_options:
                display_options.append("✏ Свой вариант...")
                
            if q_type == "single_choice":
                console.print("[dim](Выберите один вариант стрелками ↑↓ и нажмите Enter)[/dim]")
                try:
                    selected = questionary.select("Выберите:", choices=display_options).ask()
                    if selected == "✏ Свой вариант...":
                        custom = ptk_prompt("Введите свой вариант ❯ ")
                        responses[q_text] = custom.strip() if custom.strip() else "No answer"
                    elif selected:
                        responses[q_text] = selected
                    else:
                        responses[q_text] = "Skipped"
                except (KeyboardInterrupt, EOFError):
                    responses[q_text] = "Skipped"
            else: # multi_choice
                console.print("[dim](Выделите пробелом нужные варианты и нажмите Enter)[/dim]")
                try:
                    selected = questionary.checkbox("Выберите варианты:", choices=display_options).ask()
                    if selected and "✏ Свой вариант..." in selected:
                        selected.remove("✏ Свой вариант...")
                        custom = ptk_prompt("Введите свой(и) вариант(ы) через запятую ❯ ")
                        if custom.strip():
                            selected.append(custom.strip())
                    
                    responses[q_text] = ", ".join(selected) if selected else "No answer"
                except (KeyboardInterrupt, EOFError):
                    responses[q_text] = "Skipped"
    
    # Save everything to memory facts
    import io
    summary = io.StringIO()
    for k, v in responses.items():
        summary.write(f"- {k}: {v}\n")
        memory.add_fact(f"User preference on '{k}': {v}")
        
    return f"User responses:\n{summary.getvalue()}"

ACTIVE_PROCESSES = {}
_pid_counter = 1

def start_background_command(command: str) -> str:
    """Launch a command in the background and return its PID."""
    console.print(f"\n[bold yellow]Agent requesting to start background command:[/bold yellow] {command}")
    approved = questionary.confirm("Do you want to allow this background process?").ask()
    
    if not approved:
        return f"Execution aborted by user. The command '{command}' was NOT started."
        
    global _pid_counter
    pid = str(_pid_counter)
    _pid_counter += 1
    
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0  # unbuffered
        )
        
        out_queue = queue.Queue()
        err_queue = queue.Queue()
        
        def reader(pipe, q):
            try:
                while True:
                    data = pipe.read(1024)
                    if not data:
                        break
                    q.put(data)
            except Exception:
                pass
                
        threading.Thread(target=reader, args=(process.stdout, out_queue), daemon=True).start()
        threading.Thread(target=reader, args=(process.stderr, err_queue), daemon=True).start()
        
        ACTIVE_PROCESSES[pid] = {
            "process": process,
            "out_queue": out_queue,
            "err_queue": err_queue,
            "command": command
        }
        
        return f"Started background process with PID: {pid}"
    except Exception as e:
        return f"Error starting background command '{command}': {e}"

def read_background_command(pid: str) -> str:
    """Read the latest output from a background process."""
    if pid not in ACTIVE_PROCESSES:
        return f"Error: No active process with PID {pid}."
        
    proc_info = ACTIVE_PROCESSES[pid]
    process = proc_info["process"]
    
    def decode_q(q):
        data = bytearray()
        while True:
            try:
                chunk = q.get_nowait()
                data.extend(chunk)
            except queue.Empty:
                break
        
        if not data:
            return ""
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                return data.decode('cp866')
            except UnicodeDecodeError:
                return data.decode('cp1251', errors='replace')
                
    stdout = decode_q(proc_info["out_queue"])
    stderr = decode_q(proc_info["err_queue"])
    
    retcode = process.poll()
    if retcode is not None:
        status = f"Process {pid} has EXITED with code {retcode}."
    else:
        status = f"Process {pid} is RUNNING."
        
    res = f"--- {status} ---\n"
    if stdout:
        res += f"STDOUT:\n{stdout}\n"
    if stderr:
        res += f"STDERR:\n{stderr}\n"
        
    if not stdout and not stderr:
        res += "No new output.\n"
        
    return res

def send_background_command(pid: str, input_string: str) -> str:
    """Send text to the standard input of a running background process."""
    if pid not in ACTIVE_PROCESSES:
        return f"Error: No active process with PID {pid}."
        
    process = ACTIVE_PROCESSES[pid]["process"]
    if process.poll() is not None:
        return f"Error: Process {pid} has already exited."
        
    try:
        print(f"\n[bold yellow]Agent sending input to PID {pid}:[/bold yellow] {input_string.strip()}")
        if not input_string.endswith('\n'):
            input_string += '\n'
        process.stdin.write(input_string.encode('utf-8'))
        process.stdin.flush()
        return f"Sent input to PID {pid}."
    except Exception as e:
        return f"Error sending input to PID {pid}: {e}"

def stop_background_command(pid: str) -> str:
    """Terminate a background process."""
    if pid not in ACTIVE_PROCESSES:
        return f"Error: No active process with PID {pid}."
        
    process = ACTIVE_PROCESSES[pid]["process"]
    try:
        process.terminate()
        del ACTIVE_PROCESSES[pid]
        return f"Terminated background process PID {pid}."
    except Exception as e:
        return f"Error terminating PID {pid}: {e}"

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return f"No results found for query: '{query}'"
            
        formatted_results = [f"Search results for: '{query}'\n"]
        for i, res in enumerate(results, 1):
            formatted_results.append(f"{i}. {res.get('title', 'No Title')}")
            formatted_results.append(f"   URL: {res.get('href', 'No URL')}")
            formatted_results.append(f"   Snippet: {res.get('body', 'No Snippet')}\n")
            
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error searching the web: {e}"

def read_webpage(url: str) -> str:
    """Read and extract text content from a webpage URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text(separator=' ', strip=True)
        
        # Limit to reasonable length to avoid context window explosion (e.g., first 10k chars)
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "... [Content Truncated]"
            
        return f"Content of {url}:\n\n{text}"
    except Exception as e:
        return f"Error reading webpage '{url}': {e}"

def create_svg_image(svg_code: str, filename: str = None) -> str:
    """
    Creates an SVG image file from the provided SVG code and opens it in the default web browser.
    Excellent for diagrams, UI mockups, and visual explanations.
    """
    try:
        from config import get_visuals_dir
        visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
        visuals_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"svg_{timestamp}.svg"
        
        if not filename.endswith(".svg"):
            filename += ".svg"
            
        file_path = visuals_dir / filename
        
        # Ensure it has the correct XML declaration if missing
        if "<?xml" not in svg_code:
            svg_code = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_code
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
            
        # Open in browser
        webbrowser.open(f"file:///{file_path}")
        
        return f"Successfully created SVG image at '{file_path}'. It should now be open in your browser."
    except Exception as e:
        return f"Error creating SVG image: {e}"



def create_plugin(name: str, code: str) -> str:
    """
    Creates a new Python plugin in the configured plugins directory.
    Automatically handles directory path, .py extension, and syntax validation.
    """
    try:
        hooks_dir = Path(get_hooks_dir()).expanduser().resolve()
        hooks_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure correct extension
        if not name.endswith(".py"):
            name += ".py"
            
        file_path = hooks_dir / name
        
        # Validation: Check if it uses console from ui
        if "from ui import console" not in code and "import ui" not in code:
            code = "from ui import console\n" + code
            
        # Write temporary file for syntax check
        temp_path = hooks_dir / f"_temp_{name}"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(code)
            
        error = _validate_code_syntax(str(temp_path))
        if error:
            temp_path.unlink()
            return f"Failed to create plugin due to syntax error:\n{error}"
            
        # Move temp to final destination
        if file_path.exists():
            file_path.unlink()
        temp_path.rename(file_path)
        
        # Reload plugins immediately
        hook_manager.reload_plugins()
        
        return f"Successfully created plugin '{name}' in {hooks_dir}. It is now active."
    except Exception as e:
        return f"Error creating plugin: {e}"


def delete_plugin(name: str) -> str:
    """
    Deletes a plugin from the plugins directory and reloads the hook manager.
    """
    try:
        hooks_dir = Path(get_hooks_dir()).expanduser().resolve()
        
        # Ensure correct extension
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


def list_skills() -> str:
    """Lists all available markdown-based skills."""
    skills = skill_manager.list_skills()
    if not skills:
        return "No skills found. You can create one via `create_skill`."
    
    output = "Available Skills:\n"
    for s in skills:
        output += f"  - {s['name']}: {s['description']}\n"
    return output


def read_skill(name: str) -> str:
    """Reads the full instructions of a specific skill."""
    content = skill_manager.read_skill(name)
    if content:
        return f"Instructions for skill '{name}':\n\n{content}"
    return f"Skill '{name}' not found."


def create_skill(name: str, instructions: str, description: str = "") -> str:
    """Creates or updates a markdown-based skill."""
    result = skill_manager.create_skill(name, instructions, description)
    return result


def delete_skill(name: str) -> str:
    """Deletes a skill."""
    result = skill_manager.delete_skill(name)
    return result

def find_definition(file_path: str, line: int, column: int) -> str:
    """Find the definition of a symbol at the given line and column."""
    results = intel.find_definitions(file_path, line, column)
    if not results:
        return "No definitions found."
    if "error" in results[0]:
        return f"Error finding definitions: {results[0]['error']}"
    
    out = "Found definitions:\n"
    for d in results:
        out += f"- {d['name']} ({d['type']}) in {d['file_path']}:{d['line']}:{d['column']}\n"
        out += f"  {d['description']}\n"
    return out

def find_references(file_path: str, line: int, column: int) -> str:
    """Find all references to a symbol at the given line and column."""
    results = intel.find_references(file_path, line, column)
    if not results:
        return "No references found."
    if "error" in results[0]:
        return f"Error finding references: {results[0]['error']}"
    
    out = "Found references:\n"
    for r in results:
        out += f"- {r['name']} in {r['file_path']}:{r['line']}:{r['column']}\n"
    return out

def git_checkpoint(message: str) -> str:
    """Create a temporary git commit (checkpoint) to save state before an experiment."""
    try:
        # Check if it's a git repo
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        if res.returncode != 0:
            return "Error: Not a git repository. Checkpoints require git."
        
        subprocess.run(["git", "add", "."], check=True)
        # Check if there are changes to commit
        res = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if res.returncode == 0:
            return "No changes to checkpoint."
            
        subprocess.run(["git", "commit", "-m", f"Argent Checkpoint: {message}"], check=True)
        return f"Checkpoint created: '{message}'"
    except Exception as e:
        return f"Error creating checkpoint: {e}"

def git_rollback() -> str:
    """Roll back the last checkpoint (git reset --hard HEAD~1). Use this if an experiment failed."""
    try:
        # Check if it's a git repo
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        if res.returncode != 0:
            return "Error: Not a git repository."
        
        # Get the last commit message
        res = subprocess.run(["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True)
        last_msg = res.stdout.strip()
        
        if not last_msg.startswith("Argent Checkpoint:"):
            return f"Error: The last commit ('{last_msg}') was not an Argent Checkpoint. Rollback aborted for safety."
            
        print(f"\n[bold red]Rolling back last checkpoint:[/bold red] {last_msg}")
        approved = questionary.confirm("Are you sure you want to revert ALL changes to the last checkpoint?").ask()
        if not approved:
            return "Rollback aborted by user."
            
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        return f"Successfully rolled back: {last_msg}"
    except Exception as e:
        return f"Error rolling back: {e}"

def call_mcp_tool(server_name: str, tool_name: str, arguments_json: str) -> str:
    """Call a standardized tool from an MCP server. arguments_json must be a valid JSON string."""
    try:
        args = json.loads(arguments_json)
        return mcp_client.call_tool(server_name, tool_name, args)
    except json.JSONDecodeError:
        return "Error: arguments_json must be a valid JSON string."
    except Exception as e:
        return f"Error calling MCP tool: {e}"

def run_subagent(role: str, task: str, tools_json: str = None) -> str:
    """Spawn a specialized sub-agent for an isolated task. role can be 'Coder', 'Researcher', 'Reviewer', 'DocWriter'. tools_json is an optional JSON list of tools to allow."""
    from orchestrator import spawn_subagent
    tools = None
    if tools_json:
        try:
            tools = json.loads(tools_json)
        except Exception:
            pass
    return spawn_subagent(role, task, tools)

# ---------------------------------------------------------------------------
# Tool Mapping & Schemas
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "add_project_task": add_project_task,
    "complete_project_task": complete_project_task,
    "list_project_tasks": list_project_tasks,
    "write_project_spec": write_project_spec,
    "write_project_architecture": write_project_architecture,
    "write_file_spec": write_file_spec,
    "plan_work_changes": plan_work_changes,
    "add_work_task": add_work_task,
    "read_file": read_file,
    "write_file": write_file,
    "write_obsidian_note": write_obsidian_note,
    "search_obsidian_notes": search_obsidian_notes,
    "update_obsidian_properties": update_obsidian_properties,
    "run_deep_research": run_deep_research,
    "replace_in_file": replace_in_file,
    "replace_python_function": replace_python_function,
    "delete_file": delete_file,
    "list_directory": list_directory,
    "search_files": search_files,
    "run_command": run_command,
    "run_admin_command": run_admin_command,
    "start_background_command": start_background_command,
    "read_background_command": read_background_command,
    "send_background_command": send_background_command,
    "stop_background_command": stop_background_command,
    "search_web": search_web,
    "read_webpage": read_webpage,
    "get_file_outline": get_file_outline,
    "multi_replace_in_file": multi_replace_in_file,
    "read_git_diff": read_git_diff,
    "create_plugin": create_plugin,
    "delete_plugin": delete_plugin,
    "list_skills": list_skills,
    "read_skill": read_skill,
    "create_skill": create_skill,
    "delete_skill": delete_skill,
    "find_definition": find_definition,
    "find_references": find_references,
    "git_checkpoint": git_checkpoint,
    "git_rollback": git_rollback,
    "call_mcp_tool": call_mcp_tool,
    "run_subagent": run_subagent,
    "create_svg_image": create_svg_image,
    "ask_user_questions": ask_user_questions
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_project_task",
            "description": "Add a task to the current project plan. Use this to break down the project into steps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A concrete description of what this task should accomplish."
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_work_changes",
            "description": "Submit your investigation and proposed changes for an existing codebase. Call this ONCE during Phase 1 investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "description": "High-level description of what you will change and how you will solve the problem."
                    },
                    "files_to_edit": {
                        "type": "string",
                        "description": "Comma-separated list of EXISTING files you need to modify."
                    },
                    "files_to_create": {
                        "type": "string",
                        "description": "Comma-separated list of NEW files you need to create."
                    }
                },
                "required": ["strategy", "files_to_edit", "files_to_create"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_work_task",
            "description": "Add a micro-task to the project plan during Phase 2. Make tasks small, like modifying a single method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the task."
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_project_task",
            "description": "Mark a project task as completed. You MUST provide a summary of what you did.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "The ID of the task to complete."
                    },
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of what was accomplished. Mention files created or modified."
                    }
                },
                "required": ["task_id", "summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_project_tasks",
            "description": "View the current project status with all tasks and their completion summaries.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": "Call a standardized tool via the Model Context Protocol (MCP). Use this to interact with external services like GitHub, Slack, or Google Search via a standard interface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_name": { "type": "string", "description": "The name of the MCP server (e.g., 'github')." },
                    "tool_name": { "type": "string", "description": "The name of the tool to call." },
                    "arguments_json": { "type": "string", "description": "JSON string of arguments for the tool." }
                },
                "required": ["server_name", "tool_name", "arguments_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_subagent",
            "description": "Spawn a specialized sub-agent to handle a specific part of a complex task in isolation. This is great for research, code reviews, or implementing small isolated modules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": { "type": "string", "description": "The role of the sub-agent: 'Coder', 'Researcher', 'Reviewer', 'DocWriter'." },
                    "task": { "type": "string", "description": "The specific task instructions for the sub-agent." },
                    "tools_json": { "type": "string", "description": "Optional JSON list of tool names allowed for this sub-agent (e.g., '[\"read_file\", \"write_file\"]')." }
                },
                "required": ["role", "task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_project_spec",
            "description": "Write the detailed technical specification for the current project. Describe all files, classes, fields (with types), methods (with parameters), and relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "string",
                        "description": "The full technical specification text describing all files, classes, fields, methods, and their relationships."
                    }
                },
                "required": ["spec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_project_architecture",
            "description": "Write the high-level architecture map for the project. List ALL files needed, their purpose, and dependencies between them. Do NOT describe implementation details — just the structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "architecture": {
                        "type": "string",
                        "description": "The architecture map: list of all files, their purpose, and which files they depend on."
                    },
                    "files": {
                        "type": "string",
                        "description": "Comma-separated list of ALL project file paths to be created. Example: 'src/main.py, src/calculator.py, src/utils.py'. Use full paths including folders."
                    }
                },
                "required": ["architecture", "files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file_spec",
            "description": "Write a detailed specification for ONE specific file. Include: file path, all imports, class/function names, method signatures with parameter types and return types, field names with types, and a 1-2 sentence logic description for each method.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The file path exactly as it appears in the architecture (e.g., 'src/converter.py')."
                    },
                    "spec": {
                        "type": "string",
                        "description": "The detailed specification: imports, classes, methods (name, params, return type, logic), fields (name, type, default)."
                    }
                },
                "required": ["filename", "spec"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads file content. Files over 500 lines are auto-truncated; use start_line/end_line to read specific sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The absolute or relative path to the file to read."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional. First line to read (1-indexed). Omit to start from the beginning."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional. Last line to read (1-indexed, inclusive). Omit to read to the end."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes content to a file, replacing its current contents. Creates intermediate directories if missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The string content to write into the file."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_obsidian_note",
            "description": "Creates or entirely overwrites a markdown note in the user's Obsidian vault. Automatically formats YAML frontmatter for tags and aliases. WARNING: Do not use this to append or modify just a small part of an existing note unless you intend to OVERWRITE the entire note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Relative path of the note inside the vault (e.g., 'Ideas/Game Concept.md'). Extension .md is added automatically if missing."
                    },
                    "content": {
                        "type": "string",
                        "description": "The main text content of the note (Markdown formatted)."
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags WITHOUT the '#' symbol (e.g., ['npc', 'boss'])."
                    },
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of alternative titles (aliases) for the note."
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "If true, overwrites the note if it already exists. Default is false to prevent accidental data loss."
                    }
                },
                "required": ["note_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_obsidian_notes",
            "description": "Searches for markdown notes in the Obsidian vault by text content or tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional text to search for within the note contents (case-insensitive)."
                    },
                    "tag": {
                        "type": "string",
                        "description": "Optional tag to search for, either in the YAML frontmatter or inline as #tag (e.g., 'idea', 'boss')."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_svg_image",
            "description": "Create an SVG image from code and automatically open it in the web browser for the user to see. Use this to explain complex concepts, show UI designs, or create architecture diagrams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "svg_code": {
                        "type": "string",
                        "description": "The complete SVG XML code."
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (e.g., 'architecture_diagram.svg')."
                    }
                },
                "required": ["svg_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_plugin",
            "description": "Creates a new Python plugin (hook/command) in the './plugins/' directory. Automatically handles imports, syntax validation, and reloads Argent to active the new command immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The filename of the plugin (e.g., 'weather_plugin' or 'weather_plugin.py')."
                    },
                    "code": {
                        "type": "string",
                        "description": "The full Python code for the plugin. Remember to use 'command_NAME' for slash-commands and 'from ui import console' for output."
                    }
                },
                "required": ["name", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_plugin",
            "description": "Deletes an existing plugin from the './plugins/' directory and reloads Argent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The filename of the plugin to delete (e.g., 'weather_plugin.py')."
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "Lists all available markdown-based skills. Use this to discover specialized instructions you or the user have created.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": "Reads the full instructions of a specific skill. Use this to follow complex workflows or expert guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to read (e.g., 'SQL_Expert')."
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Creates a new markdown-based skill or updates an existing one. Use this to persist complex workflows or expert personas for future use.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill (e.g., 'Code_Reviewer')."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "The detailed instructions that Argent must follow when this skill is active."
                    },
                    "description": {
                        "type": "string",
                        "description": "A short one-sentence summary of what this skill does."
                    }
                },
                "required": ["name", "instructions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_skill",
            "description": "Deletes an existing skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to delete."
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_obsidian_properties",
            "description": "Safely updates tags, aliases, and custom properties in the YAML frontmatter of an existing Obsidian note. WARNING: This tool DOES NOT modify the main text body of the note. If the user asks to rewrite, expand, or add examples to an Obsidian note, DO NOT USE THIS TOOL. Use `replace_in_file` instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {
                        "type": "string",
                        "description": "Relative path of the note inside the vault to update."
                    },
                    "add_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags to add WITHOUT the '#' symbol."
                    },
                    "remove_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags to remove WITHOUT the '#' symbol."
                    },
                    "add_aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of aliases to add."
                    },
                    "remove_aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of aliases to remove."
                    },
                    "properties": {
                        "type": "object",
                        "description": "Dictionary of any key-value pairs to set in YAML frontmatter. To delete a key, set its value to null."
                    }
                },
                "required": ["note_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": "Replaces a specific text block in a file with new content. Use this to edit existing files without rewriting them entirely. The target text must be a unique, exact match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to edit."
                    },
                    "target_text": {
                        "type": "string",
                        "description": "The exact text block to be replaced. Must match exactly, including indentation and newlines."
                    },
                    "replacement_text": {
                        "type": "string",
                        "description": "The new text to insert in place of the target_text."
                    }
                },
                "required": ["file_path", "target_text", "replacement_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replace_python_function",
            "description": "Surgically replace an entire top-level function or class method in a Python file. Extremely reliable. Solves indentation and matching issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the python file."
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Name of the function. Use 'my_func' for top-level, or 'MyClass.my_method' for class methods."
                    },
                    "new_code": {
                        "type": "string",
                        "description": "The complete replacement code for the function, INCLUDING the 'def' line and full body. Indentation of the new code will be auto-corrected if it's a class method."
                    }
                },
                "required": ["file_path", "function_name", "new_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Deletes a file from the file system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to delete."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists all files and subdirectories within a given directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "The path to the directory to list."
                    }
                },
                "required": ["dir_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Recursively search for files matching specific criteria. Use this to find files by pattern (e.g., *.cs), by name, or by content without requiring user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "The root directory to start the search from. Default is '.'."
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern for file matching (e.g., '*.py', '**/*.cs'). Default is '*'."
                    },
                    "name_contains": {
                        "type": "string",
                        "description": "Case-insensitive substring that must be in the filename."
                    },
                    "content_contains": {
                        "type": "string",
                        "description": "Case-insensitive substring that must be inside the file's content."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of files to return. Default is 50."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Runs a CLI shell command on the user's system and returns the stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in the shell."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_admin_command",
            "description": "Runs a PowerShell command with Administrator privileges. This triggers a Windows UAC prompt for the user. Use this only when you explicitly need elevated permissions (e.g., editing registry, setting global system variables, installing system-wide services).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command string to execute as Administrator."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web using DuckDuckGo to find up-to-date information, documentation, or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return. Default is 5."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_webpage",
            "description": "Reads and extracts the main text content from a specific webpage URL. Use this to dive deeper into results found via search_web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the webpage to read."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_background_command",
            "description": "Starts a command in the background (e.g. dev servers, infinite loops) and returns a PID. Use this instead of run_command for long-running processes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in the background."
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_background_command",
            "description": "Reads newer output (stdout/stderr) from a currently running background process by PID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "The Process ID (PID) to read from."
                    }
                },
                "required": ["pid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_background_command",
            "description": "Sends string input to the stdin of a running background process. Use this to interact with REPLs or commands waiting for input.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "The Process ID (PID)."
                    },
                    "input_string": {
                        "type": "string",
                        "description": "The text to send to the command. Must include newline if you want to submit it."
                    }
                },
                "required": ["pid", "input_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_background_command",
            "description": "Terminates a background process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "string",
                        "description": "The Process ID (PID) to terminate."
                    }
                },
                "required": ["pid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_outline",
            "description": "Get the structural outline (classes and methods) of a Python file without reading the entire file body. Ideal for exploring large projects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multi_replace_in_file",
            "description": "Perform multiple target/replacement edits across one or several files in a single tool call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "changes_json": {
                        "type": "string",
                        "description": "A serialized JSON array of objects. Example: '[{\"file_path\": \"app.py\", \"target_text\": \"old\", \"replacement_text\": \"new\"}]'"
                    }
                },
                "required": ["changes_json"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_git_diff",
            "description": "Read the current unstaged and staged Git differences in the project. Use this to understand what has changed compared to the last commit.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_deep_research",
            "description": "Starts an autonomous Deep Research Sub-Agent that deeply researches an objective using search engines, reads top web pages, extracts data, and returns a massive synthesized technical report. Use this instead of search_web for broad topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The specific research objective or question (e.g., 'Best Unity DOTS optimizations for CPU spikes')."
                    }
                },
                "required": ["objective"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_definition",
            "description": "Find the definition of a class, function, or variable at a specific line and column. Use this instead of grep for precise navigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "The path to the file." },
                    "line": { "type": "integer", "description": "Line number (1-indexed)." },
                    "column": { "type": "integer", "description": "Column number (0-indexed)." }
                },
                "required": ["file_path", "line", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Find all usages (references) of a symbol at a specific line and column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "The path to the file." },
                    "line": { "type": "integer", "description": "Line number (1-indexed)." },
                    "column": { "type": "integer", "description": "Column number (0-indexed)." }
                },
                "required": ["file_path", "line", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkpoint",
            "description": "Create a temporary git commit to save current progress. Use this before making risky changes or running experiments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "Brief description of why you are checkpointing." }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_rollback",
            "description": "Revert all changes to the last 'Argent Checkpoint'. Use this if an experiment failed or logic is broken beyond simple repair.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
        {
        "type": "function",
        "function": {
            "name": "ask_user_questions",
            "description": "Ask the user a series of structured questions. Use this when you need clarification, preferences, or decisions on multiple points before proceeding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": "A list of question objects.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["text", "single_choice", "multi_choice"],
                                    "description": "The type of question."
                                },
                                "question": {
                                    "type": "string",
                                    "description": "The question text."
                                },
                                "options": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Required for single_choice/multi_choice. A list of string options."
                                }
                            },
                            "required": ["type", "question"]
                        }
                    }
                },
                "required": ["questions"]
            }
        }
    }
]

# Dynamically add the semantic search tool to the list of available tools ONLY if RAG is enabled.
try:
    from rag_engine import semantic_search, is_rag_enabled
    # We use a getter function that returns the tools, rather than a static list, 
    # to account for runtime changes (like enabling RAG mid-session).
except ImportError:
    pass

def get_available_tools() -> dict:
    """Returns the dictionary of Python functions the LLM can call."""
    from config import get_disabled_tools
    disabled = get_disabled_tools()
    
    tools = {k: v for k, v in AVAILABLE_TOOLS.items() if k not in disabled}
    try:
        from rag_engine import semantic_search, is_rag_enabled
        if is_rag_enabled() and "semantic_search" not in disabled:
            tools["semantic_search"] = semantic_search
    except ImportError:
        pass
    return tools

def get_tool_schemas() -> list[dict]:
    """Returns the JSON schemas for the available tools, dynamically adding RAG if enabled."""
    from config import get_disabled_tools
    disabled = get_disabled_tools()
    
    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] not in disabled]
    try:
        from rag_engine import is_rag_enabled
        if is_rag_enabled() and "semantic_search" not in disabled:
            schemas.append({
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Searches the project's codebase conceptually using AI embeddings. Returns relevant code snippets regardless of exact keywords. Use this when you need to understand where a feature is implemented.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A natural language query describing what code you want to find (e.g., 'where does the player take damage?')."
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of snippets to return (default is 5, recommend keeping under 10)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
    except ImportError:
        pass
    return schemas
