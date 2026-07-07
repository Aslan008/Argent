import os
import ast
import difflib
import py_compile
import subprocess
from pathlib import Path

from src.agent.shell import run_text
from rich.syntax import Syntax
from rich.panel import Panel
from ui import console
from config import get_obsidian_vault, get_hooks_dir
from logger import get_logger

log = get_logger("tools")

def _resolve_path(file_path: str) -> Path:
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
        abs_path = os.path.abspath(file_path)
        hooks_dir = os.path.abspath(get_hooks_dir())
        if abs_path.startswith(hooks_dir):
            basename = os.path.basename(file_path)
            if basename.endswith('.py'):
                return (
                    f"Error: Direct modification of files in the plugins directory is restricted. "
                    f"You MUST use the `create_plugin` or `delete_plugin` tools for all plugin-related tasks. "
                    f"These tools ensure mandatory syntax validation and automatic system reloading."
                )
            else:
                # Non-Python files (like .md, .txt) should not go in plugins/
                cwd = os.getcwd()
                return (
                    f"Error: The 'plugins/' directory is for Python plugins only. "
                    f"For documents and notes, write to the current project directory instead. "
                    f"Example: use file_path='{basename}' or file_path='{cwd}/{basename}'."
                )
    except Exception:
        pass
    return None


def _is_unity_meta_restricted(file_path: str) -> str | None:
    """Refuse edits to Unity .meta files — Unity generates and manages them, and
    hand-editing corrupts asset GUIDs/import settings."""
    if str(file_path).strip().lower().endswith(".meta"):
        return (
            "Error: Refusing to modify a Unity .meta file. Unity generates and manages "
            ".meta files automatically; editing them by hand corrupts asset GUIDs and "
            "import settings. Edit the asset itself (the .cs/.prefab/.asset), not its .meta — "
            "Unity will regenerate the .meta on its own."
        )
    return None


def _is_unity_project_file(file_path: str) -> bool:
    """True when a file lives inside a Unity project. Unity generates a .meta for
    every asset and keeps scripts under an Assets/ folder — either is a reliable,
    cheap signal."""
    try:
        if os.path.exists(str(file_path) + ".meta"):
            return True
        parts = {p.lower() for p in Path(file_path).resolve().parts}
        return "assets" in parts
    except Exception:
        return False


def _validate_code_syntax(file_path: str) -> str | None:
    """Quietly checks if the written Python or C# file has syntax errors.
    Returns the error string if failed, or None if passed."""
    if not file_path:
        return None
    file_path = str(file_path).strip()
    
    if file_path.endswith('.py'):
        try:
            py_compile.compile(file_path, doraise=True)
            try:
                import sys
                result = run_text([sys.executable, "-m", "flake8", "--select=F821,E999,F822,F831", file_path], capture_output=True, timeout=5)
                if result.returncode != 0 and result.stdout.strip():
                    return f"Syntax is correct, but LINTER DETECTED ERRORS:\n{result.stdout.strip()}\n\nPlease fix these errors using the `multi_replace_in_file_chunk` tool."
            except Exception:
                pass 
            return None
        except py_compile.PyCompileError as e:
            return f"SyntaxError in your Python code:\n{e.msg}\n\nPlease fix this syntax error using the `multi_replace_in_file_chunk` tool."
        except Exception as e:
            return f"Validation Error: {e}"
            
    if file_path.endswith('.cs'):
        # Unity compiles via its own pipeline; a plain `dotnet build` on a
        # Unity-generated .csproj is slow AND resolves Unity assemblies wrongly,
        # producing false compiler errors that would revert a valid edit. Skip
        # the build for anything inside a Unity project (fast + no false revert).
        if _is_unity_project_file(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "UnityEngine" in content or "UnityEditor" in content:
                    return None  # Skip standard dotnet build for Unity files to prevent false MSBuild reference errors
        except Exception:
            pass

        path_obj = Path(file_path).resolve()
        csproj_file = None
        for p in path_obj.parents:
            cs_files = list(p.glob("*.csproj"))
            if cs_files:
                csproj_file = cs_files[0]
                break
                
        if csproj_file:
            try:
                result = run_text(["dotnet", "build", str(csproj_file), "-v", "q", "/nologo"], capture_output=True, timeout=15)
                if result.returncode != 0:
                    return f"C# Compiler Error:\n{result.stdout}\n\nPlease fix this compiler error using the `multi_replace_in_file_chunk` tool."
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
                
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                json.load(f)
            return None
        except Exception as e:
            return f"JSON Syntax Error in your file:\n{e}\n\nPlease fix this syntax error using the `multi_replace_in_file_chunk` tool."

    if file_path.endswith(('.js', '.jsx')):
        try:
            result = run_text(["node", "--check", file_path], capture_output=True, timeout=5, shell=(os.name == 'nt'))
            if result.returncode != 0:
                return f"JavaScript Syntax Error:\n{result.stderr or result.stdout}\n\nPlease fix this syntax error using the `multi_replace_in_file_chunk` tool."
        except Exception as e:
            pass

    if file_path.endswith(('.ts', '.tsx')):
        try:
            result = run_text(["npx", "tsc", "--noEmit", "--skipLibCheck", file_path], capture_output=True, timeout=10, shell=(os.name == 'nt'))
            if result.returncode != 0:
                err_out = result.stderr or result.stdout
                if "error TS" in err_out or file_path in err_out:
                    return f"TypeScript Compiler Error:\n{err_out}\n\nPlease fix this compiler error using the `multi_replace_in_file_chunk` tool."
        except Exception:
            pass

    return None

_CS_TYPE_RE = None
_CS_METHOD_RE = None
_CS_METHOD_SKIP = {
    "return", "if", "for", "while", "switch", "using", "new", "throw", "else",
    "catch", "lock", "fixed", "foreach", "do", "yield", "await", "in", "is", "as",
}


def _csharp_outline(source: str) -> list:
    """Best-effort structural outline for a C# file (regex, no compiler).

    Unity scripts are routinely >500 lines; without this the model must read the
    whole file just to locate a method. Catches class/struct/interface/enum and
    method declarations with line numbers — conservative, so it may miss exotic
    forms but avoids flooding the outline with false positives.
    """
    import re
    global _CS_TYPE_RE, _CS_METHOD_RE
    if _CS_TYPE_RE is None:
        _CS_TYPE_RE = re.compile(
            r"^\s*(?:\[[^\]]*\]\s*)*"
            r"(?:(?:public|private|protected|internal|static|sealed|abstract|partial|readonly)\s+)*"
            r"\b(class|struct|interface|enum)\s+([A-Za-z_]\w*)"
        )
        # <returnType> <name>( ... ) then a body opener ({, =>) or line end.
        _CS_METHOD_RE = re.compile(
            r"^\s*(?:\[[^\]]*\]\s*)*"
            r"(?:(?:public|private|protected|internal|static|virtual|override|async|"
            r"sealed|abstract|extern|unsafe|new|partial)\s+)*"
            r"([\w<>\[\],\.\?]+)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:\{|=>|$)"
        )
    out = []
    for i, line in enumerate(source.splitlines(), 1):
        m = _CS_TYPE_RE.match(line)
        if m:
            out.append(f"{m.group(1)} {m.group(2)} (line {i})")
            continue
        mm = _CS_METHOD_RE.match(line)
        if mm and mm.group(1) not in _CS_METHOD_SKIP:
            out.append(f"    {mm.group(1)} {mm.group(2)}(...) (line {i})")
    return out


def _maybe_unescape_content(text: str) -> str:
    r"""Undo leaked JSON double-escaping (\n, \t, \\) — but ONLY when unambiguous.

    A small model sometimes emits file content with the escaping still literal,
    so the whole payload arrives as one physical line: "line1\nline2". But real
    source code legitimately contains a literal \n inside a string
    (Debug.Log("a\nb"), a regex like "\d+"), and rewriting THAT corrupts the
    file — which the C# validator does NOT catch, so the corruption reaches disk.
    The reliable fingerprint of a leaked-escaping payload is: NO real line breaks
    yet literal \n / \t markers present. If real newlines already exist the
    content is decoded — leave every literal \n untouched.
    """
    if not text or "\n" in text or "\r" in text:
        return text
    if "\\n" in text or "\\t" in text:
        return text.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    return text


def _changed_region_preview(new_content: str, start_line: int, new_line_count: int,
                            context: int = 3, max_lines: int = 30) -> str:
    """Numbered excerpt of the just-edited region so the model can verify the
    result WITHOUT a follow-up read_file. start_line is 0-indexed; the changed
    span is marked with an arrow."""
    lines = new_content.splitlines()
    if not lines:
        return ""
    span = max(1, new_line_count)
    start_line = max(0, min(start_line, len(lines) - 1))
    lo = max(0, start_line - context)
    hi = min(len(lines), start_line + span + context)
    truncated = False
    if hi - lo > max_lines:
        hi = lo + max_lines
        truncated = True
    width = len(str(hi))
    out = []
    for i in range(lo, hi):
        marker = "→" if start_line <= i < start_line + span else " "
        out.append(f"{marker} {str(i + 1).rjust(width)} | {lines[i]}")
    if truncated:
        out.append("    … (region truncated)")
    return "\n".join(out)


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

def _shift_indent(text: str, old_indent: str, new_indent: str) -> str:
    """Re-base the indentation of a block: every line carrying old_indent gets
    it swapped for new_indent; relative depth inside the block is preserved.
    Lines that don't carry old_indent are left untouched."""
    if old_indent == new_indent:
        return text
    out = []
    for line in text.splitlines():
        if not line.strip():
            out.append("")
        elif line.startswith(old_indent):
            out.append(new_indent + line[len(old_indent):])
        else:
            out.append(line)
    return "\n".join(out)


def _build_match_hint(target_text: str, content: str) -> str:
    """Build a helpful hint when target text is not found, showing the closest match."""
    lines = target_text.strip().split('\n')
    if not lines:
        return ""
    first_line = lines[0].strip()
    if not first_line or len(first_line) <= 3:
        return ""
    
    idx = content.find(first_line)
    if idx != -1:
        start_idx = max(0, idx - 50)
        end_idx = min(len(content), idx + len(first_line) + 300)
        actual_snippet = content[start_idx:end_idx]
        return f"\n\nHint: We found a partial match for your target_text. Here is the EXACT text from the file (including whitespaces/newlines):\n```\n{actual_snippet}\n```\nCopy the exact text from this snippet for your target_text."
    
    content_lines = content.split('\n')
    start_snippet = target_text[:30].strip()
    for i, line in enumerate(content_lines):
        if start_snippet in line:
            context = '\n'.join(content_lines[max(0, i-2):min(len(content_lines), i+10)])
            return f"\n\nHINT: Found something similar around line {i+1}:\n```\n{context}\n```\nMake sure your `target_text` has the EXACT spacing and indentation shown in this snippet."
    
    return ""
