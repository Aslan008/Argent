"""LSP Manager — manages language server lifecycle and provides LSP queries.

Uses multilspy (Microsoft Research) under the hood. multilspy is an optional
dependency: when not installed, every method gracefully returns None and
Argent works exactly as before.

The key capability multilspy does NOT provide out of the box is diagnostics
(open issue #12). LSP diagnostics are pushed by the server via
``textDocument/publishDiagnostics`` notifications, not pulled. This module
hooks into multilspy's ``LanguageServerHandler.on_notification_handlers`` to
capture them, storing diagnostics keyed by file URI in a thread-safe dict.

Lifecycle:
    - Language servers are started lazily on first query for a given language.
    - One server per (language, project_root) pair, reused across queries.
    - The asyncio event loop runs in a daemon thread (managed by SyncLanguageServer).
    - Servers stay alive for the rest of the session.

Thread safety:
    - The notification handler runs on the asyncio event loop thread.
    - ``get_diagnostics`` polls from the main thread.
    - A ``threading.Lock`` guards the diagnostics store.
"""

from __future__ import annotations

import os
import pathlib
import threading
import time
from typing import Any

from logger import get_logger

log = get_logger("lsp")

# ── Language detection ──────────────────────────────────────────────────────

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".cs": "csharp",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".h": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".rb": "ruby",
    ".kt": "kotlin",
    ".php": "php",
    ".dart": "dart",
}

# ── Severity names ──────────────────────────────────────────────────────────

_SEVERITY_NAMES = {1: "error", 2: "warning", 3: "info", 4: "hint"}

# ── LSP SymbolKind names ─────────────────────────────────────────────────────

_SYMBOL_KINDS = {
    1: "Class", 2: "Method", 3: "Property", 4: "Field", 5: "Constructor",
    6: "Enum", 7: "Interface", 8: "Function", 9: "Variable", 10: "Constant",
    11: "String", 12: "Number", 13: "Boolean", 14: "Array", 15: "Object",
    16: "Key", 17: "Null", 18: "EnumMember", 19: "Struct", 20: "Event",
    21: "Operator", 22: "TypeParameter",
}


def _uri_to_path(uri: str) -> str:
    """Convert a file:// URI to a filesystem path."""
    if not uri:
        return ""
    if uri.startswith("file:///"):
        # Windows: file:///C:/path → C:\path
        return uri[8:].replace("/", "\\")
    if uri.startswith("file://"):
        return uri[7:]
    return uri


class LSPManager:
    """Manages language server instances and provides LSP-backed queries.

    A module-level singleton ``lsp_manager`` is the intended entry point.
    All public methods degrade to None / empty on any failure, so callers
    can use ``if lsp_manager.is_available():`` as a simple guard.
    """

    def __init__(self):
        self._servers: dict[tuple[str, str], Any] = {}  # (lang, root) → SyncLanguageServer
        self._lock = threading.Lock()
        self._diagnostics: dict[str, list[dict]] = {}   # file URI → diagnostics list
        self._multilspy_checked = False
        self._multilspy_ok = False

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True if multilspy is importable."""
        if not self._multilspy_checked:
            try:
                import multilspy  # noqa: F401
                self._multilspy_ok = True
            except ImportError:
                self._multilspy_ok = False
            self._multilspy_checked = True
        return self._multilspy_ok

    @staticmethod
    def detect_language(file_path: str) -> str | None:
        """Map a file path to a multilspy language code by extension."""
        ext = pathlib.Path(file_path).suffix.lower()
        return _EXT_TO_LANG.get(ext)

    def supported_extensions(self) -> set[str]:
        """File extensions LSP can handle."""
        return set(_EXT_TO_LANG.keys())

    # ── Server lifecycle ─────────────────────────────────────────────────

    def _get_server(self, language: str, project_root: str) -> Any | None:
        """Get or create a SyncLanguageServer for the given language and root.

        Servers are started lazily and cached for the rest of the session.
        Returns None on any failure (multilspy not installed, server crash,
        language not supported, etc.).
        """
        key = (language, project_root)
        if key in self._servers:
            return self._servers[key]

        if not self.is_available():
            return None

        try:
            from multilspy import SyncLanguageServer
            from multilspy.multilspy_config import MultilspyConfig
            from multilspy.multilspy_logger import MultilspyLogger
        except ImportError:
            return None

        try:
            config = MultilspyConfig.from_dict({"code_language": language})
            logger = MultilspyLogger()
            server = SyncLanguageServer.create(config, logger, project_root)

            # Start the server — this creates the event loop thread and
            # launches the language server process.  We replicate the
            # context-manager logic from SyncLanguageServer.start_server()
            # but keep the server alive instead of tearing it down.
            #
            # IMPORTANT: the diagnostics handler must be registered AFTER
            # start_server() completes, because each language server's
            # start_server() registers its own handlers (including a
            # do_nothing for publishDiagnostics) which would overwrite ours.
            import asyncio

            server.loop = asyncio.new_event_loop()
            server.loop_thread = threading.Thread(
                target=server.loop.run_forever, daemon=True
            )
            server.loop_thread.start()
            ctx = server.language_server.start_server()
            server._lsp_ctx = ctx
            asyncio.run_coroutine_threadsafe(
                ctx.__aenter__(), loop=server.loop
            ).result(timeout=30)
            server._server_started = True

            # Now register the diagnostics handler — overwriting the
            # do_nothing placeholder the language server installed.
            async def _on_diagnostics(params):
                if params and "uri" in params:
                    uri = params["uri"]
                    diags = params.get("diagnostics", [])
                    with self._lock:
                        self._diagnostics[uri] = list(diags)

            server.language_server.server.on_notification_handlers[
                "textDocument/publishDiagnostics"
            ] = _on_diagnostics

            self._servers[key] = server
            log.info("LSP server started: %s @ %s", language, project_root)
            return server

        except Exception as e:
            log.warning("Failed to start LSP server for %s: %s", language, e)
            return None

    def shutdown(self):
        """Stop all language servers. Called on Argent exit."""
        for key, server in list(self._servers.items()):
            try:
                import asyncio
                if hasattr(server, "_lsp_ctx") and hasattr(server, "loop"):
                    asyncio.run_coroutine_threadsafe(
                        server._lsp_ctx.__aexit__(None, None, None),
                        loop=server.loop,
                    ).result(timeout=5)
                    server.loop.call_soon_threadsafe(server.loop.stop)
                    if hasattr(server, "loop_thread"):
                        server.loop_thread.join(timeout=3)
            except Exception as e:
                log.warning("Error shutting down LSP server %s: %s", key, e)
        self._servers.clear()
        with self._lock:
            self._diagnostics.clear()

    # ── Path helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_relative(file_path: str, project_root: str) -> str:
        """Convert any path to a path relative to project_root."""
        p = pathlib.Path(file_path)
        if not p.is_absolute():
            p = pathlib.Path(project_root) / p
        try:
            return str(p.resolve().relative_to(pathlib.Path(project_root).resolve()))
        except ValueError:
            # File is outside the project root — use as-is
            return str(p)

    @staticmethod
    def _to_uri(file_path: str, project_root: str) -> str:
        """Convert a file path to a file:// URI."""
        p = pathlib.Path(file_path)
        if not p.is_absolute():
            p = pathlib.Path(project_root) / p
        return p.resolve().as_uri()

    # ── Public API ───────────────────────────────────────────────────────

    def get_diagnostics(
        self, file_path: str, timeout: float = 3.0
    ) -> list[dict] | None:
        """Get LSP diagnostics for a file.

        Opens the file in the language server (triggering analysis), waits
        for ``publishDiagnostics`` to arrive, and returns the diagnostics list.
        Returns None if LSP is not available for this file type. Returns an
        empty list if no issues were found (or the timeout expired before
        diagnostics arrived).

        Each diagnostic dict has: ``severity`` (1-4), ``message`` (str),
        ``source`` (str), ``range`` (dict with start/end line/character),
        and optionally ``code``.
        """
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        uri = self._to_uri(file_path, project_root)

        # Clear previous diagnostics for this URI so we don't return stale data
        with self._lock:
            self._diagnostics.pop(uri, None)

        try:
            with server.open_file(rel_path):
                # The server is now analyzing the file. Poll for diagnostics.
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    with self._lock:
                        if uri in self._diagnostics:
                            return self._diagnostics[uri]
                    time.sleep(0.1)
                # Timeout — return empty (server may not support diagnostics)
                return []
        except Exception as e:
            log.warning("LSP get_diagnostics failed for %s: %s", file_path, e)
            return None

    def find_definition(
        self, file_path: str, line: int, column: int
    ) -> list[dict] | None:
        """Find definition of symbol at line/column. Returns list of
        {file_path, line, column} dicts, or None if LSP unavailable."""
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        try:
            with server.open_file(rel_path):
                results = server.request_definition(rel_path, line, column)
            return [
                {
                    "file_path": str(pathlib.Path(project_root) / loc.get("absolutePath", loc.get("uri", ""))),
                    "line": loc.get("range", {}).get("start", {}).get("line", 0) + 1,
                    "column": loc.get("range", {}).get("start", {}).get("character", 0),
                    "name": "",
                }
                for loc in (results or [])
            ]
        except Exception as e:
            log.warning("LSP find_definition failed for %s: %s", file_path, e)
            return None

    def find_references(
        self, file_path: str, line: int, column: int
    ) -> list[dict] | None:
        """Find references to symbol at line/column. Returns list of
        {file_path, line, column} dicts, or None if LSP unavailable."""
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        try:
            with server.open_file(rel_path):
                results = server.request_references(rel_path, line, column)
            return [
                {
                    "file_path": str(pathlib.Path(project_root) / loc.get("absolutePath", loc.get("uri", ""))),
                    "line": loc.get("range", {}).get("start", {}).get("line", 0) + 1,
                    "column": loc.get("range", {}).get("start", {}).get("character", 0),
                    "name": "",
                }
                for loc in (results or [])
            ]
        except Exception as e:
            log.warning("LSP find_references failed for %s: %s", file_path, e)
            return None

    def get_hover(
        self, file_path: str, line: int, column: int
    ) -> dict | None:
        """Get hover info (type, signature) for symbol at line/column."""
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        try:
            with server.open_file(rel_path):
                result = server.request_hover(rel_path, line, column)
            if result is None:
                return None
            return {
                "content": result.get("contents", {}).get("value", ""),
                "range": result.get("range"),
            }
        except Exception as e:
            log.warning("LSP get_hover failed for %s: %s", file_path, e)
            return None

    def get_document_symbols(self, file_path: str) -> list[dict] | None:
        """Get document symbols (outline) for a file."""
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        try:
            with server.open_file(rel_path):
                symbols, _tree = server.request_document_symbols(rel_path)
            return [
                {
                    "name": s.get("name", ""),
                    "kind": s.get("kind", 0),
                    "line": s.get("location", {}).get("range", {}).get("start", {}).get("line", 0) + 1,
                }
                for s in (symbols or [])
            ]
        except Exception as e:
            log.warning("LSP get_document_symbols failed for %s: %s", file_path, e)
            return None

    # ── Extended LSP operations ──────────────────────────────────────────
    # These use the low-level ``server.send.*`` interface because multilspy
    # does not wrap them in convenience methods (only definition, references,
    # hover, document_symbols, workspace_symbol and completions are wrapped).

    def _send_request(self, server, method: str, params: dict, timeout: float = 30.0):
        """Schedule an async LSP request on the server's event loop and
        return the result synchronously."""
        import asyncio
        send = server.language_server.server.send
        coro = getattr(send, method)(params)
        future = asyncio.run_coroutine_threadsafe(coro, server.loop)
        return future.result(timeout=timeout)

    def find_implementations(
        self, file_path: str, line: int, column: int
    ) -> list[dict] | None:
        """Find implementations of an interface/abstract symbol at line/column.
        Returns list of {file_path, line, column} dicts, or None if LSP
        unavailable or the server doesn't support textDocument/implementation."""
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        uri = self._to_uri(file_path, project_root)
        try:
            with server.open_file(rel_path):
                params = {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": column},
                }
                response = self._send_request(server, "implementation", params)
            if not response:
                return []
            results = []
            if isinstance(response, list):
                for item in response:
                    if "uri" in item and "range" in item:
                        results.append({
                            "file_path": _uri_to_path(item["uri"]),
                            "line": item.get("range", {}).get("start", {}).get("line", 0) + 1,
                            "column": item.get("range", {}).get("start", {}).get("character", 0),
                        })
                    elif "targetUri" in item:
                        results.append({
                            "file_path": _uri_to_path(item["targetUri"]),
                            "line": item.get("targetSelectionRange", {}).get("start", {}).get("line", 0) + 1,
                            "column": item.get("targetSelectionRange", {}).get("start", {}).get("character", 0),
                        })
            elif isinstance(response, dict):
                if "uri" in response and "range" in response:
                    results.append({
                        "file_path": _uri_to_path(response["uri"]),
                        "line": response.get("range", {}).get("start", {}).get("line", 0) + 1,
                        "column": response.get("range", {}).get("start", {}).get("character", 0),
                    })
            return results
        except Exception as e:
            log.warning("LSP find_implementations failed for %s: %s", file_path, e)
            return None

    def get_workspace_symbols(
        self, query: str, timeout: float = 15.0
    ) -> list[dict] | None:
        """Search for symbols across the entire workspace by name.
        Returns list of {name, kind, file_path, line, container} dicts, or
        None if LSP unavailable."""
        project_root = os.getcwd()
        all_results = []
        any_server = False

        for (lang, root), server in self._servers.items():
            if root != project_root:
                continue
            any_server = True
            try:
                response = self._send_request(
                    server, "workspace_symbol", {"query": query}, timeout=timeout)
                if response:
                    for item in response:
                        loc = item.get("location", {})
                        uri = loc.get("uri", "")
                        all_results.append({
                            "name": item.get("name", ""),
                            "kind": item.get("kind", 0),
                            "file_path": _uri_to_path(uri) if uri else "",
                            "line": loc.get("range", {}).get("start", {}).get("line", 0) + 1,
                            "container": item.get("containerName", ""),
                        })
            except Exception as e:
                log.warning("LSP workspace_symbol failed for %s: %s", lang, e)

        if any_server:
            return all_results

        # No server running — try to start one for a common language.
        for ext in [".py", ".ts", ".cs", ".rs", ".go", ".java", ".cpp", ".js"]:
            lang = _EXT_TO_LANG.get(ext)
            if lang:
                server = self._get_server(lang, project_root)
                if server is not None:
                    try:
                        response = self._send_request(
                            server, "workspace_symbol",
                            {"query": query}, timeout=timeout)
                        if response:
                            for item in response:
                                loc = item.get("location", {})
                                uri = loc.get("uri", "")
                                all_results.append({
                                    "name": item.get("name", ""),
                                    "kind": item.get("kind", 0),
                                    "file_path": _uri_to_path(uri) if uri else "",
                                    "line": loc.get("range", {}).get("start", {}).get("line", 0) + 1,
                                    "container": item.get("containerName", ""),
                                })
                    except Exception as e:
                        log.warning("LSP workspace_symbol failed for %s: %s", lang, e)
                    break

        return all_results if all_results else None

    def get_call_hierarchy(
        self, file_path: str, line: int, column: int,
        direction: str = "incoming", timeout: float = 15.0
    ) -> list[dict] | None:
        """Get call hierarchy for a symbol at line/column.

        ``direction`` is "incoming" (who calls this function?) or
        "outgoing" (what does this function call?).

        Returns list of {name, file_path, line, kind, detail} dicts,
        or None if LSP unavailable or the server doesn't support call hierarchy.
        """
        language = self.detect_language(file_path)
        if language is None:
            return None

        project_root = os.getcwd()
        server = self._get_server(language, project_root)
        if server is None:
            return None

        rel_path = self._to_relative(file_path, project_root)
        uri = self._to_uri(file_path, project_root)
        try:
            with server.open_file(rel_path):
                # Step 1: prepare call hierarchy
                params = {
                    "textDocument": {"uri": uri},
                    "position": {"line": line, "character": column},
                }
                items = self._send_request(
                    server, "prepare_call_hierarchy", params, timeout=timeout)
                if not items:
                    return []

                # Step 2: get incoming or outgoing calls for each item
                results = []
                method = "incoming_calls" if direction == "incoming" else "outgoing_calls"
                for item in items:
                    call_params = {"item": item}
                    calls = self._send_request(
                        server, method, call_params, timeout=timeout)
                    if not calls:
                        continue
                    for call in calls:
                        if direction == "incoming":
                            caller = call.get("from", {})
                            results.append({
                                "name": caller.get("name", ""),
                                "file_path": _uri_to_path(caller.get("uri", "")),
                                "line": caller.get("range", {}).get("start", {}).get("line", 0) + 1,
                                "kind": caller.get("kind", 0),
                                "detail": caller.get("detail", ""),
                            })
                        else:
                            callee = call.get("to", {})
                            results.append({
                                "name": callee.get("name", ""),
                                "file_path": _uri_to_path(callee.get("uri", "")),
                                "line": callee.get("range", {}).get("start", {}).get("line", 0) + 1,
                                "kind": callee.get("kind", 0),
                                "detail": callee.get("detail", ""),
                            })
                return results
        except Exception as e:
            log.warning("LSP get_call_hierarchy failed for %s: %s", file_path, e)
            return None

    # ── Formatting ───────────────────────────────────────────────────────

    @staticmethod
    def format_diagnostics(
        file_path: str, diagnostics: list[dict], max_items: int = 30
    ) -> str:
        """Format diagnostics into a human-readable string."""
        if not diagnostics:
            return f"No issues found in {file_path}."

        lines = [f"[LSP Diagnostics for {file_path}]:"]
        shown = 0
        for d in diagnostics:
            if shown >= max_items:
                remaining = len(diagnostics) - shown
                lines.append(f"  ... and {remaining} more")
                break
            severity = _SEVERITY_NAMES.get(
                d.get("severity", 1), "error"
            )
            # LSP lines are 0-indexed; display as 1-indexed
            start = d.get("range", {}).get("start", {})
            line_num = start.get("line", 0) + 1
            msg = d.get("message", "unknown")
            source = d.get("source", "")
            suffix = f" ({source})" if source else ""
            lines.append(f"  Line {line_num}: {severity}: {msg}{suffix}")
            shown += 1
        return "\n".join(lines)

    @staticmethod
    def format_symbols(symbols: list[dict], max_items: int = 50) -> str:
        """Format workspace symbol search results."""
        if not symbols:
            return "No symbols found."
        lines = [f"Found {len(symbols)} symbol(s):"]
        shown = 0
        for s in symbols:
            if shown >= max_items:
                lines.append(f"  ... and {len(symbols) - shown} more")
                break
            kind = _SYMBOL_KINDS.get(s.get("kind", 0), f"kind={s.get('kind', 0)}")
            name = s.get("name", "?")
            container = s.get("container", "")
            fp = s.get("file_path", "")
            line = s.get("line", 0)
            prefix = f"  {kind} {name}"
            if container:
                prefix += f" ({container})"
            lines.append(f"{prefix} — {fp}:{line}")
            shown += 1
        return "\n".join(lines)

    @staticmethod
    def format_call_hierarchy(
        calls: list[dict], direction: str, max_items: int = 50
    ) -> str:
        """Format call hierarchy results."""
        if not calls:
            label = "callers" if direction == "incoming" else "callees"
            return f"No {label} found."
        label = "Callers" if direction == "incoming" else "Callees"
        lines = [f"{label} ({len(calls)}):"]
        shown = 0
        for c in calls:
            if shown >= max_items:
                lines.append(f"  ... and {len(calls) - shown} more")
                break
            kind = _SYMBOL_KINDS.get(c.get("kind", 0), "")
            name = c.get("name", "?")
            fp = c.get("file_path", "")
            line = c.get("line", 0)
            detail = c.get("detail", "")
            entry = f"  {name}"
            if kind:
                entry = f"  {kind} {name}"
            if detail:
                entry += f" — {detail}"
            entry += f"  [{fp}:{line}]"
            lines.append(entry)
            shown += 1
        return "\n".join(lines)


# ── Module-level singleton ──────────────────────────────────────────────────

lsp_manager = LSPManager()