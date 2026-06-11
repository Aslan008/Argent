import ast
import shutil
from pathlib import Path
from file_tracker import snapshot
from memory_manager import memory
from tools._helpers import _resolve_path, _is_plugin_path_restricted, _validate_code_syntax, _print_diff, _shift_indent
from logger import get_logger

log = get_logger("tools")

def read_file(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """Read the contents of a file. Optionally read a specific range of lines (1-indexed)."""
    try:
        path = _resolve_path(file_path)
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
                lines = []
                for i, line in enumerate(f):
                    if i >= 500:
                        return f"[File exceeds 500 lines. Showing first 500. Use start_line/end_line to read specific sections.]\n" + "".join(lines)
                    lines.append(line)
                return "".join(lines)
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

def delete_file(file_path: str) -> str:
    """Delete a file from the file system."""
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        from approval import request_approval
        approved = request_approval(f"удалить файл '{file_path}'", destructive=True)

        if not approved:
            return f"Deletion aborted by user. The file '{file_path}' was NOT deleted."
        
        snapshot(str(path))
        path.unlink()
        log.info("delete_file: %s", file_path)
        memory.add_completed(f"Deleted {file_path}")
        return f"Successfully deleted '{file_path}'."
    except Exception as e:
        log.error("delete_file error %s: %s", file_path, e)
        return f"Error deleting file '{file_path}': {e}"

def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """Write or overwrite content to a file. Creates directories if needed."""
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = _resolve_path(file_path)
        
        if path.exists() and path.is_file() and not overwrite:
            try:
                file_size_bytes = path.stat().st_size
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    existing_lines = sum(1 for _ in f)
                if existing_lines > 150 or file_size_bytes > 10000:
                    return f"Error: File '{file_path}' already exists (Size: {existing_lines} lines, {file_size_bytes} bytes). To prevent truncation, you MUST use 'replace_in_file' to edit it. If you truly intend to DESTROY and completely rewrite this file from scratch, call write_file again with the argument 'overwrite': true."
            except Exception:
                pass
                
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            snapshot(str(path))
        if '\\n' in content or '\\t' in content:
            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            
        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            return f"File '{file_path}' written successfully, BUT COMPILATION FAILED:\n\n{validation_error}"
            
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

def append_to_file(file_path: str, content: str) -> str:
    """Append content to an existing file or create a new one. Ideal for taking notes."""
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = _resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            snapshot(str(path))
            
        if '\\n' in content or '\\t' in content:
            content = content.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            
        with open(path, "a", encoding="utf-8") as f:
            if not content.startswith('\n'):
                f.write('\n')
            f.write(content)
            
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        log.info("append_to_file: %s (%d chars)", file_path, len(content))
        memory.add_file_modified(file_path)
        memory.add_completed(f"Appended to {file_path} ({len(content)} chars)")
        return f"Successfully appended content to '{file_path}'."
    except Exception as e:
        log.error("append_to_file error %s: %s", file_path, e)
        return f"Error appending to file '{file_path}': {e}"

def replace_python_function(file_path: str, function_name: str, new_code: str) -> str:
    """Surgically replace a top-level function or class method in a Python file. 
    function_name can be 'my_func' or 'MyClass.my_method'.
    """
    try:
        path = _resolve_path(file_path)
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
            return f"Error: The existing file '{file_path}' has a SyntaxError and cannot be parsed: {e}"

        target_node = None
        
        parts = function_name.split('.')
        if len(parts) == 1:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[0]:
                    target_node = node
                    break
        elif len(parts) == 2:
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

        start_line = target_node.lineno - 1
        if hasattr(target_node, "decorator_list") and target_node.decorator_list:
            start_line = getattr(target_node.decorator_list[0], 'lineno', target_node.lineno) - 1

        end_line = target_node.end_lineno

        lines = source.splitlines()
        
        prefix = lines[:start_line]
        suffix = lines[end_line:] if end_line is not None and end_line < len(lines) else []

        new_lines = new_code.strip('\n').split('\n')
        
        original_first_line = lines[start_line]
        base_indent_str = original_first_line[:len(original_first_line) - len(original_first_line.lstrip())]
        
        if new_lines:
            incoming_first_line = new_lines[0]
            incoming_base_indent_str = incoming_first_line[:len(incoming_first_line) - len(incoming_first_line.lstrip())]
            
            for i, line in enumerate(new_lines):
                if not line.strip():
                    new_lines[i] = ""
                    continue
                    
                if line.startswith(incoming_base_indent_str):
                    stripped_line = line[len(incoming_base_indent_str):]
                else:
                    stripped_line = line.lstrip()
                
                new_lines[i] = base_indent_str + stripped_line

        final_lines = prefix + new_lines + suffix
        new_source = '\n'.join(final_lines) + '\n'

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_source)

        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            return f"Function replaced, BUT COMPILATION FAILED:\n\n{validation_error}\nHint: Check indentation (4 spaces per block). Use replace_python_function again to fix it."

        _print_diff(source, new_source, file_path)

        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        log.info("replace_python_function: %s (%s)", file_path, function_name)
        memory.add_file_modified(file_path)
        memory.add_completed(f"Replaced {function_name} in {file_path}")
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
        from tools._helpers import _build_match_hint
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        target_text_processed = target_text
        if '\\n' in target_text_processed or '\\t' in target_text_processed:
            target_text_processed = target_text_processed.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            
        replacement_text_processed = replacement_text
        if '\\n' in replacement_text_processed or '\\t' in replacement_text_processed:
            replacement_text_processed = replacement_text_processed.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')

        fuzzy_note = ""
        if target_text_processed in content:
            count = content.count(target_text_processed)
            if count > 1:
                return f"Error: The target text appears {count} times in '{file_path}'. Please provide a more specific, unique block of text to replace."
            new_content = content.replace(target_text_processed, replacement_text_processed)
        else:
            # --- SMART EDIT: FUZZY FALLBACK ---
            # Small models reproduce code blocks with broken whitespace; a
            # single unambiguous whitespace-insensitive match is applied
            # automatically (re-indented), saving a full model round-trip.
            target_lines = [line.strip() for line in target_text_processed.splitlines() if line.strip()]
            content_lines = content.splitlines()
            matches = []
            if target_lines:
                for i in range(len(content_lines)):
                    t_idx = 0
                    c_idx = i
                    match_start = c_idx
                    while c_idx < len(content_lines) and t_idx < len(target_lines):
                        if not content_lines[c_idx].strip():
                            c_idx += 1
                            continue
                        if content_lines[c_idx].strip() == target_lines[t_idx]:
                            t_idx += 1
                            c_idx += 1
                        else:
                            break

                    if t_idx == len(target_lines):
                        matches.append((match_start, c_idx))

            if len(matches) > 1:
                return f"Error: The target text is ambiguous (found {len(matches)} fuzzy matches). Provide more context."
            if not matches:
                hint = _build_match_hint(target_text_processed, content)
                return f"Error: The target text was not found in '{file_path}'. Make sure it matches exactly, including whitespace and indentation.{hint}"

            start_line, end_line = matches[0]
            # Re-base the replacement onto the file's actual indentation.
            file_first = content_lines[start_line]
            file_indent = file_first[:len(file_first) - len(file_first.lstrip())]
            model_first = next((l for l in target_text_processed.splitlines() if l.strip()), "")
            model_indent = model_first[:len(model_first) - len(model_first.lstrip())]
            adjusted_replacement = _shift_indent(replacement_text_processed, model_indent, file_indent)
            new_content = "\n".join(
                content_lines[:start_line] + adjusted_replacement.splitlines() + content_lines[end_line:]
            )
            if content.endswith("\n") and not new_content.endswith("\n"):
                new_content += "\n"
            fuzzy_note = " (fuzzy match: whitespace/indentation differences in target_text were corrected automatically)"

        snapshot(str(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Modification aborted because it broke code compilation. Changes reverted.\n\n{validation_error}"
            
        _print_diff(content, new_content, file_path)

        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        log.info("replace_in_file: %s (replaced %d chars)%s", file_path, len(target_text_processed), fuzzy_note)
        memory.add_file_modified(file_path)
        memory.add_completed(f"Edited {file_path}")
        return f"Successfully replaced text in '{file_path}'.{fuzzy_note}"
    except Exception as e:
        log.error("replace_in_file error %s: %s", file_path, e)
        return f"Error replacing text in '{file_path}': {e}"

def multi_replace_in_file_chunk(file_path: str, changes_json: str) -> str:
    """Surgically replace multiple chunks of text in a single file by specifying line ranges."""
    import json
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        from tools._helpers import _print_diff, _validate_code_syntax
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
            
        changes = json.loads(changes_json)
        if not isinstance(changes, list):
            return "Error: changes_json must be a JSON array of objects."
            
        snapshot(str(path))
        
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
            
        lines = [line + "\n" for line in lines]
        original_content = "".join(lines)
            
        try:
            changes.sort(key=lambda x: x.get("start_line", 1), reverse=True)
        except TypeError:
            return "Error: Invalid chunk format. 'start_line' must be an integer."

        for change in changes:
            start_line = change.get("start_line")
            end_line = change.get("end_line")
            target = change.get("target_content", "")
            repl = change.get("replacement_content", "")
            
            if start_line is None or end_line is None:
                return "Error: Every chunk must contain 'start_line' and 'end_line'."
                
            s_idx = max(0, start_line - 1)
            e_idx = min(len(lines), end_line)
            
            if s_idx > e_idx or s_idx < 0:
                return f"Error: Invalid line range {start_line}-{end_line}."
                
            actual_target_lines = lines[s_idx:e_idx]
            actual_target = "".join(actual_target_lines)
            
            def normalize(t):
                return t.replace("\\n", "\n").replace("\\t", "\t").strip()
                
            norm_target = normalize(target)
            norm_actual = normalize(actual_target)
            
            if norm_target and norm_target != norm_actual:
                return f"Error: The target_content for lines {start_line}-{end_line} does not match the actual file content.\nExpected:\n{norm_target}\n\nActual:\n{norm_actual}"
                
            repl_processed = repl
            if '\\n' in repl_processed or '\\t' in repl_processed:
                repl_processed = repl_processed.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            
            if repl_processed and not repl_processed.endswith('\n'):
                repl_processed += '\n'
                
            lines[s_idx:e_idx] = [repl_processed] if repl_processed else []
            
        new_content = "".join(lines)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            with open(path, "w", encoding="utf-8") as f:
                f.write(original_content)
            return f"Modification aborted because it broke code compilation. Changes reverted.\n\n{validation_error}"
            
        _print_diff(original_content, new_content, file_path)
        
        log.info("multi_replace_in_file_chunk: %s", file_path)
        memory.add_file_modified(file_path)
        memory.add_completed(f"Edited {file_path} via chunk patcher")
        return f"Successfully applied chunk replacements to '{file_path}'."
        
    except json.JSONDecodeError:
        return "Error: changes_json is not valid JSON."
    except Exception as e:
        log.error("multi_replace_in_file_chunk error %s: %s", file_path, e)
        return f"Error replacing chunks in '{file_path}': {e}"

def multi_replace_in_file(changes_json: str) -> str:
    """Apply multiple text replacements across one or multiple files using a JSON array string."""
    import json
    try:
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
    """Analyzes a Python file and returns a structured outline of its contents."""
    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."
        
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Error: The file '{file_path}' has a SyntaxError and cannot be parsed: {e}"

        outline = []
        
        def _get_indent(node):
            if hasattr(node, 'col_offset'):
                return '    ' * (node.col_offset // 4)
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
                targets = ', '.join([t.id for t in node.targets if isinstance(t, ast.Name)])
                if targets:
                    outline.append(f"{_get_indent(node)}Variable: {targets} (line {node.lineno})")

        if not outline:
            return f"No significant structures found in '{file_path}'."
            
        return f"Outline of '{file_path}':\n" + "\n".join(outline)

    except Exception as e:
        return f"Error getting file outline for '{file_path}': {e}"

def create_directory(dir_path: str) -> str:
    """Create a directory and all parent directories if needed."""
    try:
        path = Path(dir_path).expanduser().resolve()
        if path.exists():
            return f"Directory '{dir_path}' already exists."
        path.mkdir(parents=True, exist_ok=True)
        log.info("create_directory: %s", dir_path)
        return f"Successfully created directory '{dir_path}'."
    except Exception as e:
        log.error("create_directory error %s: %s", dir_path, e)
        return f"Error creating directory '{dir_path}': {e}"

def move_file(source: str, destination: str) -> str:
    """Move or rename a file from source to destination."""
    restriction_error = _is_plugin_path_restricted(source)
    if restriction_error:
        return restriction_error
    try:
        src = _resolve_path(source)
        dst = Path(destination).expanduser().resolve()
        if not src.exists():
            return f"Error: Source '{source}' does not exist."
        if not src.is_file():
            return f"Error: '{source}' is not a file."
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return f"Error: Destination '{destination}' already exists."
        shutil.move(str(src), str(dst))
        log.info("move_file: %s -> %s", source, destination)
        memory.add_completed(f"Moved {source} -> {destination}")
        return f"Successfully moved '{source}' to '{destination}'."
    except Exception as e:
        log.error("move_file error %s: %s", source, e)
        return f"Error moving file '{source}': {e}"

def copy_file(source: str, destination: str) -> str:
    """Copy a file from source to destination. Creates parent directories if needed."""
    try:
        src = _resolve_path(source)
        dst = Path(destination).expanduser().resolve()
        if not src.exists():
            return f"Error: Source '{source}' does not exist."
        if not src.is_file():
            return f"Error: '{source}' is not a file."
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return f"Error: Destination '{destination}' already exists."
        shutil.copy2(str(src), str(dst))
        log.info("copy_file: %s -> %s", source, destination)
        memory.add_completed(f"Copied {source} -> {destination}")
        return f"Successfully copied '{source}' to '{destination}'."
    except Exception as e:
        log.error("copy_file error %s: %s", source, e)
        return f"Error copying file '{source}': {e}"

def list_directory(dir_path: str) -> str:
    """List the contents of a directory."""
    try:
        path = _resolve_path(dir_path)
        if not path.exists():
            return f"Error: Directory '{dir_path}' does not exist."
        if not path.is_dir():
            return f"Error: '{dir_path}' is not a directory."
        
        items = list(path.iterdir())
        if not items:
            return f"Directory '{dir_path}' is empty."
        
        dirs = []
        files = []
        for item in items:
            if item.is_dir():
                dirs.append(item)
            else:
                files.append(item)
        
        output = [f"Contents of {dir_path}:"]
        for item in sorted(dirs) + sorted(files):
            type_str = "DIR" if item.is_dir() else "FILE"
            line = f"[{type_str}] {item.name}"
            if item.is_file():
                try:
                    size = item.stat().st_size
                    if size < 1024:
                        line += f"  ({size} B)"
                    elif size < 1024 * 1024:
                        line += f"  ({size / 1024:.1f} KB)"
                    else:
                        line += f"  ({size / (1024 * 1024):.1f} MB)"
                except OSError:
                    pass
            output.append(line)
        return "\n".join(output)
    except Exception as e:
        return f"Error listing directory '{dir_path}': {e}"

def run_deep_linter(path: str = ".") -> str:
    """Run a deep static analysis (Pylint) on a file or directory."""
    import subprocess
    import os
    try:
        abs_path = str(_resolve_path(path))
        print(f"\n[bold yellow]Agent running deep linter on:[/bold yellow] {abs_path}")
        result = subprocess.run(
            ["pylint", abs_path, "-E", "--output-format=text"], 
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and not result.stdout.strip():
            return "Pylint: No errors found. The code is structurally sound."
        return f"Pylint Analysis Results:\n{result.stdout}\n{result.stderr}"
    except FileNotFoundError:
        return "Error: pylint is not installed. Run 'pip install pylint' first."
    except Exception as e:
        return f"Error running deep linter: {e}"
