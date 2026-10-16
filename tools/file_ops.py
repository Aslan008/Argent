import ast
import os
import shutil
from pathlib import Path
from typing import Any
from file_tracker import snapshot
from memory_manager import memory
from text_codec import read_text, sniff_file, write_text
from tools._helpers import (
    _resolve_path,
    _is_plugin_path_restricted,
    _is_unity_meta_restricted,
    _validate_code_syntax,
    _print_diff,
    _shift_indent,
    _maybe_unescape_content,
    _changed_region_preview,
    _unity_script_placement_error,
    _strip_read_line_numbers,
    _maybe_reload_plugin,
)
from logger import get_logger

log = get_logger("tools")

# A line budget bounds a source file but not a minified bundle, a base64 blob or
# a one-line JSON dump: those are a single line of any size, and the whole thing
# went into the context window. The character budget is the real bound; the line
# budget just keeps a narrow file from arriving as a wall.
#
# Both scale with what the model can hold. A 3B model on an 8k window drowns in
# 40k characters, while making a 128k cloud model read agent.py in four calls
# costs four round trips and four chances to lose the thread — and the files
# that matter most are the long ones.
_READ_BUDGET = {
    "tiny":  (400, 12_000),
    "small": (400, 12_000),
}
_READ_BUDGET_DEFAULT = (1500, 40_000)
# Kept as module constants because tests and callers refer to them.
MAX_READ_LINES, MAX_READ_CHARS = _READ_BUDGET_DEFAULT


def _read_budget() -> tuple:
    try:
        from config import get_current_model, get_model_size_category
        return _READ_BUDGET.get(get_model_size_category(get_current_model()),
                                _READ_BUDGET_DEFAULT)
    except Exception:
        return _READ_BUDGET_DEFAULT


def read_file(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """Read the contents of a file. Optionally read a specific range of lines (1-indexed)."""
    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        # Text or binary, and which encoding: a UTF-16 log (what PowerShell's `>` writes) has NUL bytes
        # but is text, so a bare "NUL in the first 4 KB" test called it binary.
        try:
            codec = sniff_file(path)
        except Exception as e:
            return f"Error reading file '{file_path}': {e}"
        if codec is None:
            return (
                f"Error: Cannot read '{file_path}': file appears to be binary. "
                f"Use an appropriate tool (e.g. view_image for images) instead."
            )

        if start_line is not None and start_line < 1:
            start_line = 1
        if end_line is not None and end_line < 1:
            return f"Error: 'end_line' ({end_line}) must be greater than or equal to 1."
        if start_line is not None and end_line is not None and start_line > end_line:
            return f"Error: 'start_line' ({start_line}) cannot be greater than 'end_line' ({end_line})."

        max_lines, max_chars = _read_budget()
        ranged = start_line is not None or end_line is not None
        first = max(1, start_line or 1)
        last = end_line if end_line is not None else None

        # Numbered, so a later reference to "line 412" is read off the output
        # instead of counted by a model that cannot count. It also makes
        # start_line/end_line usable on the follow-up read without guessing.
        numbered = []
        size = 0
        total = 0
        stopped_at_chars = False
        with open(path, "r", encoding=codec, errors="replace") as f:
            for number, line in enumerate(f, start=1):
                total = number
                if number < first:
                    continue
                if last is not None and number > last:
                    # Keep counting so the header can report the real total.
                    continue
                if not ranged and len(numbered) >= max_lines:
                    continue
                if size + len(line) > max_chars:
                    stopped_at_chars = True
                    break
                numbered.append(f"{number:>5}\t{line}")
                size += len(line)

        if total == 0:
            return ""

        if ranged and first > total:
            return f"Error: 'start_line' ({first}) exceeds total lines in file ({total})."

        body = "".join(numbered)
        if body and not body.endswith("\n"):
            body += "\n"

        shown_to = first + len(numbered) - 1

        if stopped_at_chars:
            return (f"[Stopped at {max_chars} characters — lines {first}-{shown_to}. "
                    f"The file has very long lines (minified, base64 or one-line JSON?). "
                    f"Use grep_search to find what you need, or start_line/end_line "
                    f"for a smaller range.]\n" + body)
        if ranged:
            return f"[Lines {first}-{shown_to} of {total}]\n" + body
        if total > max_lines:
            nav = (" or get_file_outline to jump straight to a symbol"
                   if path.suffix.lower() in (".py", ".cs") else "")
            return (f"[File exceeds {max_lines} lines ({total} total). Showing first "
                    f"{max_lines}. Use start_line/end_line to read specific "
                    f"sections{nav}.]\n" + body)
        return body
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"

def delete_file(file_path: str) -> str:
    """Delete a file from the file system."""
    restriction_error = _is_plugin_path_restricted(file_path) or _is_unity_meta_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        from approval import request_approval
        # The resolved path, not the string the model typed: a relative name can
        # resolve outside the working directory (see _resolve_path), and consent
        # to delete "notes.md" is not consent to delete some other notes.md.
        approved = request_approval(f"удалить файл '{path}'", destructive=True)

        if not approved:
            return f"Deletion aborted by user. The file '{file_path}' was NOT deleted."
        
        snapshot(str(path))
        try:
            import stat
            os.chmod(path, stat.S_IWRITE)
        except Exception:
            pass
        path.unlink()
        # Drop it from the vector index too, or semantic_search keeps serving
        # snippets of a file that no longer exists.
        try:
            from rag_engine import remove_file_index
            remove_file_index(str(path))
        except ImportError:
            pass
        log.info("delete_file: %s", file_path)
        memory.add_completed(f"Deleted {file_path}")
        _maybe_reload_plugin(str(path))
        return f"Successfully deleted '{file_path}'."
    except Exception as e:
        log.error("delete_file error %s: %s", file_path, e)
        return f"Error deleting file '{file_path}': {e}"

def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """Write or overwrite content to a file. Creates directories if needed."""
    restriction_error = _is_plugin_path_restricted(file_path) or _is_unity_meta_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = _resolve_path(file_path)

        # Only when CREATING a new file: refuse a Unity script placed where the
        # editor would never see it (inside the project but outside Assets/).
        if not path.exists():
            placement_error = _unity_script_placement_error(path)
            if placement_error:
                return placement_error

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
        existed = path.exists()
        prior_content = None
        codec = "utf-8"          # a new file is UTF-8; a rewritten one keeps the encoding it had
        if existed:
            snapshot(str(path))
            try:
                prior_content, codec = read_text(path)
            except Exception:
                prior_content, codec = None, "utf-8"
        content = _maybe_unescape_content(content)
        write_text(path, content, codec)

        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            # Safety net: never leave a file that doesn't compile on disk.
            # Restore the previous version, or remove a brand-new broken file.
            if existed:
                if prior_content is not None:
                    write_text(path, prior_content, codec)
                else:
                    from file_tracker import undo
                    undo(str(path))
                reverted = "The file was restored to its previous version."
            else:
                try:
                    path.unlink()
                except Exception:
                    pass
                reverted = "The new file was not created."
            return (f"Edit REJECTED: writing '{file_path}' would break compilation, so the change was "
                    f"NOT applied. {reverted}\n\n{validation_error}")
            
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        _maybe_reload_plugin(str(path))
        log.info("write_file: %s (%d chars)", file_path, len(content))
        memory.add_file_modified(file_path)
        memory.add_completed(f"Wrote {file_path} ({len(content)} chars)")
        return f"Successfully wrote to '{file_path}'."
    except Exception as e:
        log.error("write_file error %s: %s", file_path, e)
        return f"Error writing file '{file_path}': {e}"

def append_to_file(file_path: str, content: str) -> str:
    """Append content to an existing file or create a new one. Ideal for taking notes."""
    restriction_error = _is_plugin_path_restricted(file_path) or _is_unity_meta_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    try:
        path = _resolve_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            snapshot(str(path))
            
        content = _maybe_unescape_content(content)

        # Separate the append from what is already there, but only when it is
        # not already separated. The old code wrote \n unconditionally, so a
        # file that ended in a newline — nearly all of them — gained a blank
        # line per call, and a large file produced through repeated salvage
        # continuations ended up with one at every seam.
        needs_newline = False
        if path.exists() and path.stat().st_size and not content.startswith("\n"):
            with open(path, "rb") as probe:
                probe.seek(-1, os.SEEK_END)
                needs_newline = probe.read(1) not in (b"\n", b"\r")

        with open(path, "a", encoding="utf-8") as f:
            if needs_newline:
                f.write("\n")
            f.write(content)
            
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        _maybe_reload_plugin(str(path))
        log.info("append_to_file: %s (%d chars)", file_path, len(content))
        memory.add_file_modified(file_path)
        memory.add_completed(f"Appended to {file_path} ({len(content)} chars)")
        return f"Successfully appended content to '{file_path}'."
    except Exception as e:
        log.error("append_to_file error %s: %s", file_path, e)
        return f"Error appending to file '{file_path}': {e}"

def replace_python_function(file_path: str, function_name: str, new_code: str) -> str:
    """Surgically replace a function or method in a Python file.
    function_name is a dotted path: 'my_func', 'MyClass.my_method', or a nested
    target like 'Outer.Inner.method' or 'outer_func.inner_func'.
    """
    restriction_error = _is_plugin_path_restricted(file_path)
    if restriction_error:
        return restriction_error

    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        snapshot(str(path))
        
        source, codec = read_text(path)

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Error: The existing file '{file_path}' has a SyntaxError and cannot be parsed: {e}"

        parts = function_name.split('.')
        if not all(parts):
            return f"Error: Invalid function_name format '{function_name}'. Use 'func', 'Class.method' or 'Outer.Inner.method'."

        # Descend the dotted path one component at a time, so a method inside a
        # nested class (Outer.Inner.method) or a function nested in another
        # function is reachable — ast.iter_child_nodes only ever saw the top
        # level, which silently lost any target below the first tier.
        target_node = None
        scope = tree.body
        for depth, name in enumerate(parts):
            match = None
            for node in scope:
                if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        and node.name == name):
                    match = node
                    break
            if match is None:
                break
            if depth == len(parts) - 1:
                if isinstance(match, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    target_node = match
                break
            scope = match.body  # descend into the class/function body

        if not target_node:
            return f"Error: Function/Method '{function_name}' not found in '{file_path}'."

        start_line = target_node.lineno - 1
        if hasattr(target_node, "decorator_list") and target_node.decorator_list:
            start_line = getattr(target_node.decorator_list[0], 'lineno', target_node.lineno) - 1

        end_line = target_node.end_lineno

        lines = source.splitlines()
        
        prefix = lines[:start_line]
        suffix = lines[end_line:] if end_line is not None and end_line < len(lines) else []

        new_code = _maybe_unescape_content(new_code)
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

        write_text(path, new_source, codec)

        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            write_text(path, source, codec)
            return (f"Edit REJECTED: replacing '{function_name}' would break compilation, so the change "
                    f"was reverted.\n\n{validation_error}\nHint: check indentation (4 spaces per block), "
                    f"then call replace_python_function again with corrected code.")

        _print_diff(source, new_source, file_path)

        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        _maybe_reload_plugin(str(path))
        log.info("replace_python_function: %s (%s)", file_path, function_name)
        memory.add_file_modified(file_path)
        memory.add_completed(f"Replaced {function_name} in {file_path}")
        preview = _changed_region_preview(new_source, start_line, len(new_lines))
        preview_block = f"\n\nUpdated region:\n{preview}" if preview else ""
        return f"Successfully replaced '{function_name}' in '{file_path}'.{preview_block}"
        
    except Exception as e:
        import traceback
        return f"Error surgically replacing function: {e}\n{traceback.format_exc()}"

def replace_in_file(
    file_path: str = None,
    target_text: str = None,
    replacement_text: str = None,
    start_line: int | None = None,
    end_line: int | None = None,
    changes_json: Any = None,
    changes: Any = None,
) -> str:
    """Replace exactly matching text in a file with new text, or apply multiple changes.
    
    Modes:
    1. Single replacement: file_path + target_text + replacement_text (+ optional start_line/end_line).
    2. Multi-replacement: provide 'changes' or 'changes_json' as a list or JSON array of change objects
       [{file_path, target_text, replacement_text}] with atomic rollback on syntax/compile errors.
    """
    # Multi-replace delegation if changes or changes_json is provided
    multi_input = changes if changes is not None else changes_json
    if multi_input is None and target_text is None and file_path:
        if isinstance(file_path, (list, dict)):
            multi_input = file_path
        elif isinstance(file_path, str) and file_path.strip().startswith(("[", "{")) and '"target' in file_path:
            multi_input = file_path

    if multi_input is not None:
        if file_path:
            if isinstance(multi_input, list):
                for c in multi_input:
                    if isinstance(c, dict) and not c.get("file_path"):
                        c["file_path"] = file_path
            elif isinstance(multi_input, dict):
                if not multi_input.get("file_path"):
                    multi_input["file_path"] = file_path
                if "changes" in multi_input and isinstance(multi_input["changes"], list):
                    for c in multi_input["changes"]:
                        if isinstance(c, dict) and not c.get("file_path"):
                            c["file_path"] = file_path
            elif isinstance(multi_input, str) and multi_input.strip().startswith(("[", "{")):
                try:
                    import json
                    parsed = json.loads(multi_input)
                    if isinstance(parsed, list):
                        for c in parsed:
                            if isinstance(c, dict) and not c.get("file_path"):
                                c["file_path"] = file_path
                        multi_input = json.dumps(parsed)
                    elif isinstance(parsed, dict):
                        if not parsed.get("file_path"):
                            parsed["file_path"] = file_path
                        if "changes" in parsed and isinstance(parsed["changes"], list):
                            for c in parsed["changes"]:
                                if isinstance(c, dict) and not c.get("file_path"):
                                    c["file_path"] = file_path
                        multi_input = json.dumps(parsed)
                except Exception:
                    pass
        return multi_replace_in_file(multi_input)

    restriction_error = _is_plugin_path_restricted(file_path) or _is_unity_meta_restricted(file_path)
    if restriction_error:
        return restriction_error
        
    # An empty target matches between every character, so the file reports one
    # "occurrence" per position and the ambiguity hint tells the model to make
    # its target MORE distinctive — the opposite of the actual problem, which
    # is that there is no target. Usually it means the caller sent the wrong
    # shape and target_text defaulted to "".
    if not (target_text or "").strip():
        return (f"Error: 'target_text' is empty, so there is nothing to find in "
                f"'{file_path}'. Pass the exact existing text you want replaced. "
                f"(If you are passing `changes`, each entry must be "
                f"flat: {{\"file_path\", \"target_text\", \"replacement_text\"}}.)")

    try:
        from tools._helpers import _build_match_hint
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        raw_content, codec = read_text(path)

        has_crlf = "\r\n" in raw_content
        content = raw_content.replace("\r\n", "\n")
        content_lines = content.splitlines()

        target_text_processed = _strip_read_line_numbers(_maybe_unescape_content(target_text)).replace("\r\n", "\n")
        replacement_text_processed = _strip_read_line_numbers(
            _maybe_unescape_content(replacement_text)).replace("\r\n", "\n")

        fuzzy_note = ""
        anchor_line = 0            # 0-indexed start of the changed region
        new_line_count = 1         # lines the replacement spans, for the preview

        is_bounded = start_line is not None or end_line is not None
        if is_bounded:
            try:
                if start_line is not None:
                    start_line = int(start_line)
                if end_line is not None:
                    end_line = int(end_line)
            except (ValueError, TypeError):
                return "Error: 'start_line' and 'end_line' must be valid integers."

            total_lines = len(content_lines)
            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = total_lines
            if start_line < 1:
                start_line = 1
            if start_line > total_lines:
                return f"Error: 'start_line' ({start_line}) exceeds total lines in file ({total_lines})."
            if end_line < start_line:
                return f"Error: 'end_line' ({end_line}) cannot be less than 'start_line' ({start_line})."
            end_line = min(end_line, total_lines)

            start_char_idx = sum(len(line) + 1 for line in content_lines[:start_line - 1])
            end_char_idx = min(len(content), sum(len(line) + 1 for line in content_lines[:end_line]))
            window_text = content[start_char_idx:end_char_idx]

            if target_text_processed in window_text:
                count = window_text.count(target_text_processed)
                if count > 1:
                    locs = []
                    search_from = 0
                    for _ in range(count):
                        idx = window_text.find(target_text_processed, search_from)
                        if idx == -1:
                            break
                        line_offset = window_text.count("\n", 0, idx)
                        locs.append(f"line {start_line + line_offset}")
                        search_from = idx + 1
                    return (
                        f"Error: The target text appears {count} times within lines {start_line}-{end_line} "
                        f"in '{file_path}' (at {', '.join(locs[:5])}). Narrow down start_line/end_line, "
                        f"add distinctive nearby lines to target_text, or use write_file to rewrite it whole."
                    )
                pos = window_text.index(target_text_processed)
                abs_pos = start_char_idx + pos
                anchor_line = content.count("\n", 0, abs_pos)
                new_line_count = replacement_text_processed.count("\n") + 1
                new_content = (
                    content[:abs_pos]
                    + replacement_text_processed
                    + content[abs_pos + len(target_text_processed):]
                )
            else:
                # Fuzzy fallback strictly within the bounded range
                target_lines = [line.strip() for line in target_text_processed.splitlines() if line.strip()]
                matches = []
                if target_lines:
                    for i in range(start_line - 1, end_line):
                        if content_lines[i].strip() != target_lines[0]:
                            continue
                        match_start = i
                        t_idx = 1
                        c_idx = i + 1
                        while c_idx < min(len(content_lines), end_line) and t_idx < len(target_lines):
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
                    locs = [f"lines {s + 1}-{e}" for (s, e) in matches[:5]]
                    return (
                        f"Error: The target text matches {len(matches)} places within lines {start_line}-{end_line}: "
                        f"{'; '.join(locs)}. Narrow down your line range, add distinctive context, or use write_file to rewrite it whole."
                    )
                if not matches:
                    hint = _build_match_hint(target_text_processed, window_text)
                    return (
                        f"Error: The target text was not found within lines {start_line}-{end_line} in '{file_path}'. "
                        f"Make sure the line range covers the target text, or omit start_line/end_line to search the whole file.{hint}"
                    )

                m_start, m_end = matches[0]
                file_first = content_lines[m_start]
                file_indent = file_first[:len(file_first) - len(file_first.lstrip())]
                model_first = next((l for l in target_text_processed.splitlines() if l.strip()), "")
                model_indent = model_first[:len(model_first) - len(model_first.lstrip())]
                adjusted_replacement = _shift_indent(replacement_text_processed, model_indent, file_indent)
                adjusted_lines = adjusted_replacement.splitlines()
                new_content = "\n".join(
                    content_lines[:m_start] + adjusted_lines + content_lines[m_end:]
                )
                if content.endswith("\n") and not new_content.endswith("\n"):
                    new_content += "\n"
                anchor_line = m_start
                new_line_count = len(adjusted_lines) or 1
                fuzzy_note = f" (fuzzy match within lines {start_line}-{end_line}: whitespace/indentation differences in target_text were corrected automatically)"

        elif target_text_processed in content:
            count = content.count(target_text_processed)
            if count > 1:
                locs = []
                search_from = 0
                for _ in range(count):
                    idx = content.find(target_text_processed, search_from)
                    if idx == -1:
                        break
                    locs.append(f"line {content.count(chr(10), 0, idx) + 1}")
                    search_from = idx + 1
                return (
                    f"Error: The target text appears {count} times in '{file_path}' "
                    f"(at {', '.join(locs[:5])}). Add a distinctive nearby line (a unique "
                    f"comment, the method signature, or a preceding statement) so target_text "
                    f"is unique — or specify 'start_line' and 'end_line' to target a specific range — "
                    f"or, for a small file, use write_file to rewrite it whole."
                )
            pos = content.index(target_text_processed)
            anchor_line = content.count("\n", 0, pos)
            new_line_count = replacement_text_processed.count("\n") + 1
            new_content = content.replace(target_text_processed, replacement_text_processed)
        else:
            # --- SMART EDIT: FUZZY FALLBACK ---
            # Small models reproduce code blocks with broken whitespace; a
            # single unambiguous whitespace-insensitive match is applied
            # automatically (re-indented), saving a full model round-trip.
            target_lines = [line.strip() for line in target_text_processed.splitlines() if line.strip()]
            matches = []
            if target_lines:
                for i in range(len(content_lines)):
                    if content_lines[i].strip() != target_lines[0]:
                        continue
                    match_start = i
                    t_idx = 1
                    c_idx = i + 1
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
                locs = []
                for (s, e) in matches[:5]:
                    first = next((content_lines[k].strip() for k in range(s, min(e, len(content_lines)))
                                  if content_lines[k].strip()), "")
                    locs.append(f"lines {s + 1}-{e} (starts: \"{first[:50]}\")")
                return (
                    f"Error: The target text matches {len(matches)} places: {'; '.join(locs)}. "
                    f"They differ only in surrounding context, so pick ONE and add a distinctive "
                    f"nearby line (a unique comment, the method signature, or a preceding statement) "
                    f"to your target_text so it's unique — or specify 'start_line' and 'end_line' "
                    f"to target one specific occurrence — or, for a small file, use write_file to "
                    f"rewrite it whole."
                )
            if not matches:
                hint = _build_match_hint(target_text_processed, content)
                return f"Error: The target text was not found in '{file_path}'. Make sure it matches exactly, including whitespace and indentation.{hint}"

            m_start, m_end = matches[0]
            # Re-base the replacement onto the file's actual indentation.
            file_first = content_lines[m_start]
            file_indent = file_first[:len(file_first) - len(file_first.lstrip())]
            model_first = next((l for l in target_text_processed.splitlines() if l.strip()), "")
            model_indent = model_first[:len(model_first) - len(model_first.lstrip())]
            adjusted_replacement = _shift_indent(replacement_text_processed, model_indent, file_indent)
            adjusted_lines = adjusted_replacement.splitlines()
            new_content = "\n".join(
                content_lines[:m_start] + adjusted_lines + content_lines[m_end:]
            )
            if content.endswith("\n") and not new_content.endswith("\n"):
                new_content += "\n"
            anchor_line = m_start
            new_line_count = len(adjusted_lines) or 1
            fuzzy_note = " (fuzzy match: whitespace/indentation differences in target_text were corrected automatically)"

        write_content = new_content
        if has_crlf:
            write_content = write_content.replace("\r\n", "\n").replace("\n", "\r\n")
        else:
            write_content = write_content.replace("\r\n", "\n")

        snapshot(str(path))
        write_text(path, write_content, codec)
            
        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            write_text(path, raw_content, codec)
            return f"Modification aborted because it broke code compilation. Changes reverted.\n\n{validation_error}"
            
        _print_diff(content, new_content, file_path)

        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        _maybe_reload_plugin(str(path))
        log.info("replace_in_file: %s (replaced %d chars)%s", file_path, len(target_text_processed), fuzzy_note)
        memory.add_file_modified(file_path)
        memory.add_completed(f"Edited {file_path}")
        preview = _changed_region_preview(new_content, anchor_line, new_line_count)
        preview_block = f"\n\nUpdated region:\n{preview}" if preview else ""
        return f"Successfully replaced text in '{file_path}'.{fuzzy_note}{preview_block}"
    except Exception as e:
        log.error("replace_in_file error %s: %s", file_path, e)
        return f"Error replacing text in '{file_path}': {e}"

def multi_replace_in_file_chunk(file_path: str, changes_json: str) -> str:
    """Surgically replace multiple chunks of text in a single file by specifying line ranges."""
    import json
    restriction_error = _is_plugin_path_restricted(file_path) or _is_unity_meta_restricted(file_path)
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
        
        raw_content, codec = read_text(path)

        has_crlf = "\r\n" in raw_content
        has_trailing_newline = raw_content.endswith("\r\n") or raw_content.endswith("\n")
        lines = raw_content.splitlines(keepends=True)
        original_content = raw_content

        # Validate chunk format and non-overlapping invariant
        try:
            sorted_asc = sorted(changes, key=lambda x: x.get("start_line", 1))
        except TypeError:
            return "Error: Invalid chunk format. 'start_line' must be an integer."

        for i, cur in enumerate(sorted_asc):
            cur_start = cur.get("start_line")
            cur_end = cur.get("end_line")
            if cur_start is None or cur_end is None:
                return "Error: Every chunk must contain 'start_line' and 'end_line'."
            if not isinstance(cur_start, int) or not isinstance(cur_end, int):
                return "Error: Invalid chunk format. 'start_line' and 'end_line' must be integers."
            if cur_start < 1 or cur_end < cur_start:
                return f"Error: Invalid line range {cur_start}-{cur_end}."
            if i > 0:
                prev = sorted_asc[i - 1]
                if prev.get("end_line", 0) >= cur_start:
                    return (
                        f"Error: Replacement chunks overlap between line {prev.get('start_line')}-{prev.get('end_line')} "
                        f"and {cur_start}-{cur_end}. Chunks must specify non-overlapping ranges."
                    )

        # Apply replacements from bottom to top so line indices before start_line remain valid
        changes.sort(key=lambda x: x["start_line"], reverse=True)

        for change in changes:
            start_line = change["start_line"]
            end_line = change["end_line"]
            target = change.get("target_content", "")
            repl = change.get("replacement_content", "")

            s_idx = max(0, start_line - 1)
            e_idx = min(len(lines), end_line)

            actual_target_lines = lines[s_idx:e_idx]
            actual_target = "".join(actual_target_lines)

            def normalize(t):
                return _strip_read_line_numbers(_maybe_unescape_content(t)).replace("\r\n", "\n").strip()

            norm_target = normalize(target)
            norm_actual = normalize(actual_target)

            if norm_target and norm_target != norm_actual:
                return f"Error: The target_content for lines {start_line}-{end_line} does not match the actual file content.\nExpected:\n{norm_target}\n\nActual:\n{norm_actual}"

            repl_processed = _maybe_unescape_content(repl)
            if repl_processed:
                if has_crlf:
                    repl_processed = repl_processed.replace("\r\n", "\n").replace("\n", "\r\n")
                if not (repl_processed.endswith("\r\n") if has_crlf else repl_processed.endswith("\n")):
                    repl_processed += "\r\n" if has_crlf else "\n"

            lines[s_idx:e_idx] = [repl_processed] if repl_processed else []

        new_content = "".join(lines)
        if not has_trailing_newline:
            if new_content.endswith("\r\n"):
                new_content = new_content[:-2]
            elif new_content.endswith("\n"):
                new_content = new_content[:-1]

        write_text(path, new_content, codec)

        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            write_text(path, original_content, codec)
            return f"Modification aborted because it broke code compilation. Changes reverted.\n\n{validation_error}"

        _print_diff(original_content, new_content, file_path)
        
        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        _maybe_reload_plugin(str(path))
        log.info("multi_replace_in_file_chunk: %s", file_path)
        memory.add_file_modified(file_path)
        memory.add_completed(f"Edited {file_path} via chunk patcher")
        return f"Successfully applied chunk replacements to '{file_path}'."
        
    except json.JSONDecodeError:
        return "Error: changes_json is not valid JSON."
    except Exception as e:
        log.error("multi_replace_in_file_chunk error %s: %s", file_path, e)
        return f"Error replacing chunks in '{file_path}': {e}"


def _parse_multi_replace_changes(raw_input):
    """Parse changes_json with duplicate-key preservation and alias normalization."""
    import json
    ALIAS_MAP = {
        "path": "file_path", "file": "file_path", "filename": "file_path", "filepath": "file_path",
        "target": "target_text", "target_content": "target_text", "old_text": "target_text", "old_content": "target_text", "old": "target_text",
        "replacement": "replacement_text", "replacement_content": "replacement_text", "new_text": "replacement_text", "new_content": "replacement_text", "new": "replacement_text", "repl": "replacement_text"
    }

    def _object_pairs_hook(pairs):
        normalized = []
        for k, v in pairs:
            canonical = ALIAS_MAP.get(k, k)
            normalized.append((canonical, v))

        target_indices = [i for i, (k, v) in enumerate(normalized) if k == "target_text"]
        if len(target_indices) > 1:
            common_fp = next((v for k, v in normalized if k == "file_path" and v), None)
            chunks = []
            for idx, start_i in enumerate(target_indices):
                end_i = target_indices[idx + 1] if idx + 1 < len(target_indices) else len(normalized)
                chunk_pairs = normalized[start_i:end_i]
                d = {"file_path": common_fp} if common_fp else {}
                for ck, cv in chunk_pairs:
                    if ck == "replacement_text" and ck in d and d[ck] and (not cv or not str(cv).strip()):
                        continue
                    d[ck] = cv
                chunks.append(d)
            return {"__multi_unpacked__": chunks}

        res = {}
        for k, v in normalized:
            if k in res:
                existing = res[k]
                # Never let empty string or whitespace overwrite a non-empty string on duplicate keys!
                if isinstance(existing, str) and existing.strip() and (not isinstance(v, str) or not v.strip()):
                    log.warning("multi_replace_in_file: Preserved non-empty '%s' over duplicate empty/whitespace value.", k)
                    continue
                if (not existing) and v:
                    res[k] = v
                    continue
                res[k] = v
            else:
                res[k] = v
        return res

    if isinstance(raw_input, str):
        try:
            data = json.loads(raw_input, object_pairs_hook=_object_pairs_hook)
        except json.JSONDecodeError:
            return None, "Error: changes_json is not valid JSON."
    elif isinstance(raw_input, (list, dict)):
        try:
            s = json.dumps(raw_input)
            data = json.loads(s, object_pairs_hook=_object_pairs_hook)
        except Exception:
            data = raw_input
    else:
        return None, "Error: changes_json must be a JSON array of objects."

    items = []
    if isinstance(data, dict):
        if "__multi_unpacked__" in data:
            items.extend(data["__multi_unpacked__"])
        elif "changes" in data and isinstance(data["changes"], list):
            common_fp = data.get("file_path")
            for item in data["changes"]:
                if isinstance(item, dict):
                    if common_fp and not item.get("file_path"):
                        item["file_path"] = common_fp
                    if "__multi_unpacked__" in item:
                        for u in item["__multi_unpacked__"]:
                            if common_fp and not u.get("file_path"):
                                u["file_path"] = common_fp
                        items.extend(item["__multi_unpacked__"])
                    else:
                        items.append(item)
        else:
            return None, "Error: changes_json must be a JSON array of objects."
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "__multi_unpacked__" in item:
                items.extend(item["__multi_unpacked__"])
            else:
                items.append(item)
    else:
        return None, "Error: changes_json must be a JSON array of objects."

    return items, None


def _apply_atomic_multi_replace(file_path: str, file_edits: list) -> str:
    """Atomically apply multiple text replacements to a single file, sorted bottom-to-top."""
    try:
        from tools._helpers import _print_diff, _validate_code_syntax, _maybe_unescape_content, _strip_read_line_numbers, _shift_indent, _build_match_hint, _maybe_reload_plugin
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        raw_content, codec = read_text(path)

        has_crlf = "\r\n" in raw_content
        content = raw_content.replace("\r\n", "\n")
        content_lines = content.splitlines()

        located_edits = []

        for idx, change in file_edits:
            target = change.get("target_text")
            repl = change.get("replacement_text")
            if target is None:
                return f"Error in change #{idx + 1}: 'target_text' is missing."
            if repl is None:
                return f"Error in change #{idx + 1}: 'replacement_text' is missing. To delete code, pass an explicit empty string."

            target_proc = _strip_read_line_numbers(_maybe_unescape_content(target)).replace("\r\n", "\n")
            repl_proc = _strip_read_line_numbers(_maybe_unescape_content(repl)).replace("\r\n", "\n")

            if not target_proc:
                return f"Error in change #{idx + 1}: 'target_text' is empty."

            if target_proc in content:
                count = content.count(target_proc)
                if count > 1:
                    locs = []
                    search_from = 0
                    for _ in range(count):
                        idx_pos = content.find(target_proc, search_from)
                        if idx_pos == -1:
                            break
                        locs.append(f"line {content.count(chr(10), 0, idx_pos) + 1}")
                        search_from = idx_pos + 1
                    return (
                        f"Error: The target text in change #{idx + 1} appears {count} times in '{file_path}' "
                        f"(at {', '.join(locs[:5])}). Add surrounding context to make target_text unique."
                    )
                start_pos = content.index(target_proc)
                end_pos = start_pos + len(target_proc)
                located_edits.append((start_pos, end_pos, repl_proc, target_proc, idx, False))
            else:
                target_lines = [line.strip() for line in target_proc.splitlines() if line.strip()]
                matches = []
                if target_lines:
                    for i in range(len(content_lines)):
                        if content_lines[i].strip() != target_lines[0]:
                            continue
                        match_start = i
                        t_idx = 1
                        c_idx = i + 1
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
                    return f"Error: The target text in change #{idx + 1} matches {len(matches)} places ambiguously."
                if not matches:
                    hint = _build_match_hint(target_proc, content)
                    return f"Error: The target text in change #{idx + 1} was not found in '{file_path}'.{hint}"

                start_line, end_line = matches[0]
                start_pos = sum(len(content_lines[k]) + 1 for k in range(start_line))
                end_pos = sum(len(content_lines[k]) + 1 for k in range(end_line))
                if end_line == len(content_lines) and not content.endswith("\n"):
                    end_pos -= 1

                file_first = content_lines[start_line]
                file_indent = file_first[:len(file_first) - len(file_first.lstrip())]
                model_first = next((l for l in target_proc.splitlines() if l.strip()), "")
                model_indent = model_first[:len(model_first) - len(model_first.lstrip())]
                adjusted_replacement = _shift_indent(repl_proc, model_indent, file_indent)
                located_edits.append((start_pos, end_pos, adjusted_replacement, target_proc, idx, True))

        located_edits.sort(key=lambda x: x[0])
        for i in range(1, len(located_edits)):
            prev = located_edits[i - 1]
            curr = located_edits[i]
            if prev[1] > curr[0]:
                return f"Error: Replacement chunks #{prev[4] + 1} and #{curr[4] + 1} overlap in '{file_path}'. Chunks must specify non-overlapping regions."

        located_edits.sort(key=lambda x: x[0], reverse=True)
        new_content = content
        for start_pos, end_pos, repl_proc, _, _, _ in located_edits:
            new_content = new_content[:start_pos] + repl_proc + new_content[end_pos:]

        write_content = new_content
        if has_crlf:
            write_content = write_content.replace("\r\n", "\n").replace("\n", "\r\n")
        else:
            write_content = write_content.replace("\r\n", "\n")

        snapshot(str(path))
        write_text(path, write_content, codec)

        validation_error = _validate_code_syntax(str(path))
        if validation_error:
            write_text(path, raw_content, codec)
            return f"Modification aborted because it broke code compilation. Changes reverted.\n\n{validation_error}"

        _print_diff(content, new_content, file_path)

        try:
            from rag_engine import update_file_index
            update_file_index(str(path))
        except ImportError:
            pass

        _maybe_reload_plugin(str(path))
        memory.add_file_modified(file_path)
        memory.add_completed(f"Edited {file_path} ({len(located_edits)} edits)")
        fuzzy_count = sum(1 for e in located_edits if e[5])
        fuzzy_note = f" ({fuzzy_count} fuzzy-matched)" if fuzzy_count else ""
        return f"Successfully applied {len(located_edits)} non-overlapping replacements to '{file_path}'.{fuzzy_note}"
    except Exception as e:
        log.error("_apply_atomic_multi_replace error %s: %s", file_path, e)
        return f"Error replacing chunks in '{file_path}': {e}"


def multi_replace_in_file(changes_json) -> str:
    """Apply multiple text replacements across one or multiple files using a JSON array string or list of objects."""
    from collections import defaultdict
    try:
        changes, parse_err = _parse_multi_replace_changes(changes_json)
        if parse_err:
            return parse_err

        if not changes:
            return "Successfully: Multi-replace execution finished:\n"

        grouped_by_file = defaultdict(list)
        report = []

        for idx, change in enumerate(changes):
            if not isinstance(change, dict):
                report.append(f"Skipping change: item is not an object.")
                continue

            fp = change.get("file_path")
            if not fp:
                report.append("Skipping change: 'file_path' is missing.")
                continue

            restriction_error = _is_plugin_path_restricted(fp) or _is_unity_meta_restricted(fp)
            if restriction_error:
                report.append(f"Skipping '{fp}': {restriction_error}")
                continue

            grouped_by_file[fp].append((idx, change))

        for fp, file_edits in grouped_by_file.items():
            if len(file_edits) == 1:
                idx, change = file_edits[0]
                target = change.get("target_text")
                repl = change.get("replacement_text")
                if not (target or "").strip():
                    report.append(
                        f"[{fp}]: Error: 'target_text' is empty, so there is nothing to find in "
                        f"'{fp}'. Pass the exact existing text you want replaced. "
                        f"(If you are passing `changes`, each entry must be "
                        f"flat: {{\"file_path\", \"target_text\", \"replacement_text\"}}.)"
                    )
                    continue
                if repl is None:
                    report.append(f"[{fp}]: Error: 'replacement_text' is missing. To delete code, pass an explicit empty string.")
                    continue
                res = replace_in_file(fp, target, repl)
                report.append(f"[{fp}]: {res}")
            else:
                res = _apply_atomic_multi_replace(fp, file_edits)
                report.append(f"[{fp}]: {res}")

        return "Successfully: Multi-replace execution finished:\n" + "\n".join(report)
    except Exception as e:
        log.error("multi_replace_in_file error: %s", e)
        return f"Error executing multi_replace: {e}"

def get_file_outline(file_path: str) -> str:
    """Analyzes a Python file and returns a structured outline of its contents."""
    try:
        path = _resolve_path(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' does not exist."
        if not path.is_file():
            return f"Error: '{file_path}' is not a file."

        source, _codec = read_text(path)

        if path.suffix.lower() == ".cs":
            from tools._helpers import _csharp_outline
            cs_outline = _csharp_outline(source)
            if not cs_outline:
                return f"No classes or methods found in '{file_path}'."
            return f"Outline of '{file_path}':\n" + "\n".join(cs_outline)

        if path.suffix.lower() != ".py":
            return f"Error: Outline is only available for Python (.py) and C# (.cs) files, not '{path.suffix}'."

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Error: The file '{file_path}' has a SyntaxError and cannot be parsed: {e}"

        outline = []
        
        def _get_indent(node):
            if hasattr(node, 'col_offset'):
                return '    ' * (node.col_offset // 4)
            return ''

        def _walk_body(body, indent_level=0):
            """Recursively walk a class/function body, collecting defs and classes.

            Nested classes (Outer.Inner) and their methods were silently skipped
            by the old single-level loop — the model got an incomplete outline
            and could not locate methods inside nested classes."""
            prefix = '    ' * indent_level
            for item in body:
                if isinstance(item, ast.FunctionDef):
                    outline.append(f"{prefix}    def {item.name}(...): (line {item.lineno})")
                elif isinstance(item, ast.AsyncFunctionDef):
                    outline.append(f"{prefix}    async def {item.name}(...): (line {item.lineno})")
                elif isinstance(item, ast.ClassDef):
                    bases = [b.id if isinstance(b, ast.Name) else '...' for b in item.bases]
                    outline.append(f"{prefix}    class {item.name}({', '.join(bases)}): (line {item.lineno})")
                    _walk_body(item.body, indent_level + 1)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                outline.append(f"{_get_indent(node)}def {node.name}(...): (line {node.lineno})")
            elif isinstance(node, ast.AsyncFunctionDef):
                outline.append(f"{_get_indent(node)}async def {node.name}(...): (line {node.lineno})")
            elif isinstance(node, ast.ClassDef):
                bases = [b.id if isinstance(b, ast.Name) else '...' for b in node.bases]
                outline.append(f"{_get_indent(node)}class {node.name}({', '.join(bases)}): (line {node.lineno})")
                _walk_body(node.body, 1)
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
    restriction_error = (
        _is_plugin_path_restricted(source) or _is_plugin_path_restricted(destination)
        or _is_unity_meta_restricted(source) or _is_unity_meta_restricted(destination)
    )
    if restriction_error:
        return restriction_error
    try:
        src = _resolve_path(source)
        dst = Path(destination).expanduser().resolve()
        if not src.exists():
            return f"Error: Source '{source}' does not exist."
        if not src.is_file():
            return f"Error: '{source}' is not a file."
        if dst.exists() and dst.is_dir():
            dst = dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return f"Error: Destination '{dst}' already exists."
        snapshot(str(src))
        shutil.move(str(src), str(dst))
        try:
            from rag_engine import remove_file_index, update_file_index
            remove_file_index(str(src))
            update_file_index(str(dst))
        except ImportError:
            pass
        log.info("move_file: %s -> %s", source, destination)
        memory.add_completed(f"Moved {source} -> {destination}")
        return f"Successfully moved '{source}' to '{destination}'."
    except Exception as e:
        log.error("move_file error %s: %s", source, e)
        return f"Error moving file '{source}': {e}"

def copy_file(source: str, destination: str) -> str:
    """Copy a file from source to destination. Creates parent directories if needed."""
    restriction_error = (
        _is_plugin_path_restricted(source) or _is_plugin_path_restricted(destination)
        or _is_unity_meta_restricted(source) or _is_unity_meta_restricted(destination)
    )
    if restriction_error:
        return restriction_error
    try:
        src = _resolve_path(source)
        dst = Path(destination).expanduser().resolve()
        if not src.exists():
            return f"Error: Source '{source}' does not exist."
        if not src.is_file():
            return f"Error: '{source}' is not a file."
        if dst.exists() and dst.is_dir():
            dst = dst / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return f"Error: Destination '{dst}' already exists."
        shutil.copy2(str(src), str(dst))
        try:
            from rag_engine import update_file_index
            update_file_index(str(dst))
        except ImportError:
            pass
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
        
        all_sorted = sorted(dirs) + sorted(files)
        max_items = 200
        output = [f"Contents of {dir_path}:"]
        for item in all_sorted[:max_items]:
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
        if len(all_sorted) > max_items:
            output.append(f"... and {len(all_sorted) - max_items} more items. (Showing first {max_items}. Use search_files or specify a subdirectory to narrow down.)")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing directory '{dir_path}': {e}"

def run_deep_linter(path: str = ".") -> str:
    """Run a deep static analysis (Pylint) on a file or directory."""
    import subprocess
    try:
        abs_path = str(_resolve_path(path))
        from ui import console
        console.print(f"\n[bold yellow]Agent running deep linter on:[/bold yellow] {abs_path}")
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
