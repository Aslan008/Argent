"""LSP-backed tools for Argent.

Provides ``check_code`` — instant diagnostics via Language Server Protocol —
and LSP-enhanced versions of ``find_definition`` / ``find_references`` that
work across all supported languages (Python, C#, TypeScript, Rust, Go, …),
falling back to the existing Jedi-based implementation for Python when LSP
is unavailable.
"""

import json
from logger import get_logger

log = get_logger("tools")


def check_code(file_path: str) -> str:
    """Check code for errors and warnings using the Language Server Protocol.

    Returns diagnostics (errors, warnings) for the file. Works with Python,
    C#, TypeScript, JavaScript, Rust, Go, Java, C++, Ruby, Kotlin, PHP, Dart.
    Requires the ``multilspy`` package (``pip install multilspy``).
    """
    from src.lsp.manager import lsp_manager

    if not lsp_manager.is_available():
        return (
            "LSP (multilspy) is not installed. Install it with: pip install multilspy\n"
            "Supported languages: Python, C#, TypeScript, JavaScript, Rust, Go, "
            "Java, C++, Ruby, Kotlin, PHP, Dart."
        )

    language = lsp_manager.detect_language(file_path)
    if language is None:
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else "?"
        return (
            f"LSP does not support '.{ext}' files. Supported extensions: "
            + ", ".join(sorted(lsp_manager.supported_extensions()))
        )

    diagnostics = lsp_manager.get_diagnostics(file_path)
    if diagnostics is None:
        return (
            f"LSP server for '{language}' could not be started. The language "
            f"server binary may need to be installed (multilspy downloads it "
            f"automatically on first use). Check /logs lsp for details."
        )

    return lsp_manager.format_diagnostics(file_path, diagnostics)


def lsp_definition(file_path: str, line: int, column: int) -> str:
    """Find the definition of a symbol at the given line and column using LSP.

    More accurate than grep-based search because it understands types and
    scope. Falls back to the existing Jedi-based tool for Python when LSP
    is unavailable.
    """
    from src.lsp.manager import lsp_manager

    # Try LSP first (cross-language, type-aware)
    if lsp_manager.is_available():
        results = lsp_manager.find_definition(file_path, line, column)
        if results:
            out = "Found definitions (LSP):\n"
            for d in results:
                out += f"- {d['file_path']}:{d['line']}:{d['column']}\n"
            return out.rstrip()

    # Fall back to Jedi (Python only)
    from intelligence import intel
    results = intel.find_definitions(file_path, line, column)
    if not results:
        return "No definitions found."
    if "error" in results[0]:
        return f"Error finding definitions: {results[0]['error']}"

    out = "Found definitions:\n"
    for d in results:
        out += f"- {d['name']} ({d['type']}) in {d['file_path']}:{d['line']}:{d['column']}\n"
        out += f"  {d['description']}\n"
    return out.rstrip()


def lsp_references(file_path: str, line: int, column: int) -> str:
    """Find all references to a symbol at the given line and column using LSP.

    More accurate than grep-based search because it understands types and
    scope. Falls back to the existing Jedi-based tool for Python when LSP
    is unavailable.
    """
    from src.lsp.manager import lsp_manager

    # Try LSP first (cross-language, type-aware)
    if lsp_manager.is_available():
        results = lsp_manager.find_references(file_path, line, column)
        if results:
            out = "Found references (LSP):\n"
            for r in results:
                out += f"- {r['file_path']}:{r['line']}:{r['column']}\n"
            return out.rstrip()

    # Fall back to Jedi (Python only)
    from intelligence import intel
    results = intel.find_references(file_path, line, column)
    if not results:
        return "No references found."
    if "error" in results[0]:
        return f"Error finding references: {results[0]['error']}"

    out = "Found references:\n"
    for r in results:
        out += f"- {r['name']} in {r['file_path']}:{r['line']}:{r['column']}\n"
    return out.rstrip()


def find_implementations(file_path: str, line: int, column: int) -> str:
    """Find implementations of an interface or abstract method using LSP.

    Given a symbol at a position (e.g. an interface method or abstract class),
    returns all concrete implementations. Works across all LSP-supported
    languages. Falls back to Jedi-based reference search for Python when the
    language server doesn't support textDocument/implementation.
    """
    from src.lsp.manager import lsp_manager

    # Try LSP first (cross-language)
    if lsp_manager.is_available():
        results = lsp_manager.find_implementations(file_path, line, column)
        if results:
            out = f"Found {len(results)} implementation(s) (LSP):\n"
            for r in results:
                out += f"- {r['file_path']}:{r['line']}:{r['column']}\n"
            return out.rstrip()

    # Fall back to Jedi (Python only) — find references that are definitions
    from intelligence import intel
    refs = intel.find_references(file_path, line, column)
    if not refs or "error" in refs[0]:
        return "No implementations found."

    # Filter for references that look like definitions (not call sites).
    # Jedi's description for definitions starts with "def " or "class ".
    impls = [r for r in refs if r.get("description", "").startswith(("def ", "class "))]
    if not impls:
        # No definition-like references; show all as best-effort
        impls = refs

    out = f"Found {len(impls)} implementation(s):\n"
    for r in impls:
        out += f"- {r['name']} in {r['file_path']}:{r['line']}\n"
    return out.rstrip()


def search_workspace_symbols(query: str) -> str:
    """Search for symbols (classes, functions, methods, etc.) across the
    entire workspace using LSP. Faster and more precise than grep because
    it understands code structure. Requires multilspy.

    Returns matching symbols with their file locations.
    """
    from src.lsp.manager import lsp_manager

    if not lsp_manager.is_available():
        return ("LSP (multilspy) is not installed. Install with: pip install multilspy")

    results = lsp_manager.get_workspace_symbols(query)
    if results is None:
        return ("LSP server could not process this request. The language "
                "server may not support workspace/symbol, or no server is running.")
    return lsp_manager.format_symbols(results)


def get_call_hierarchy(file_path: str, line: int, column: int,
                       direction: str = "incoming") -> str:
    """Get the call hierarchy for a function/method at the given position.

    ``direction``:
    - "incoming" — find all functions that CALL this function (callers).
    - "outgoing" — find all functions that this function CALLS (callees).

    Uses LSP (multilspy) when the language server supports it. Falls back to
    a Jedi-based implementation for Python when the server doesn't support
    call hierarchy (e.g. jedi-language-server).
    """
    from src.lsp.manager import lsp_manager

    if direction not in ("incoming", "outgoing"):
        direction = "incoming"

    # Try LSP first (cross-language)
    if lsp_manager.is_available():
        results = lsp_manager.get_call_hierarchy(file_path, line, column, direction)
        if results is not None:
            return lsp_manager.format_call_hierarchy(results, direction)

    # Fall back to Jedi (Python only)
    from intelligence import intel

    if direction == "incoming":
        # All references to this function are potential callers
        refs = intel.find_references(file_path, line, column)
        if not refs or "error" in refs[0]:
            return "No callers found."
        out = f"Callers ({len(refs)}):\n"
        for r in refs:
            out += f"  {r['name']} — {r['file_path']}:{r['line']}\n"
        return out.rstrip()
    else:
        # Outgoing: find all function calls within the function body using AST
        import ast
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
            calls = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.lineno <= line <= (node.end_lineno or node.lineno):
                        for child in ast.walk(node):
                            if isinstance(child, ast.Call):
                                func = child.func
                                if isinstance(func, ast.Name):
                                    calls.append((func.id, child.lineno))
                                elif isinstance(func, ast.Attribute):
                                    calls.append((func.attr, child.lineno))
                        break
            if not calls:
                return "No callees found."
            # Deduplicate
            seen = set()
            unique = []
            for name, lineno in calls:
                key = (name, lineno)
                if key not in seen:
                    seen.add(key)
                    unique.append((name, lineno))
            out = f"Callees ({len(unique)}):\n"
            for name, lineno in unique:
                out += f"  {name} — {file_path}:{lineno}\n"
            return out.rstrip()
        except Exception as e:
            return f"Error finding callees: {e}"