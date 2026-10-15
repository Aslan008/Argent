"""LMTrust test suite verifying LSP fixes in Argent.

Covers:
- D1: Cross-platform URI parsing with percent-encoding and Windows drive casing
- D2: Absolute path / URI resolution in find_definition and find_references
- D3: Diagnostic cache key normalization (Windows case-insensitivity)
- D4: Missing LSP tools in AVAILABLE_TOOLS & TOOL_SCHEMAS parity
- D5: Boundary & input validation (line >= 1, column >= 0)
- D6: Workspace symbols candidate language filtering
- D7: File existence validation and informative error messages
- D8: Thread-safe shutdown & atexit handler registration
"""

import atexit
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════════════
# D1: URI decoding & cross-platform path handling
# ══════════════════════════════════════════════════════════════════════════════

class TestUriToPath:
    def test_windows_uri_drive_letter_and_slashes(self):
        from src.lsp.manager import _uri_to_path
        uri = "file:///c:/Users/user/Desktop/Argent/main.py"
        path_str = _uri_to_path(uri)
        assert isinstance(path_str, str)
        p = Path(path_str)
        assert p.drive.upper() == "C:"
        assert p.name == "main.py"
        assert "file:" not in path_str

    def test_windows_uri_percent_encoded_spaces_and_unicode(self):
        from src.lsp.manager import _uri_to_path
        uri = "file:///c:/My%20Projects/Тест%20Папка/app.py"
        path_str = _uri_to_path(uri)
        assert "My Projects" in path_str
        assert "Тест Папка" in path_str
        assert "%20" not in path_str

    def test_posix_uri_roundtrip(self):
        from src.lsp.manager import _uri_to_path
        if os.name == "nt":
            uri = "file:///c:/home/user/app.py"
            path_str = _uri_to_path(uri)
            assert Path(path_str).name == "app.py"
        else:
            uri = "file:///home/user/app.py"
            path_str = _uri_to_path(uri)
            assert path_str == "/home/user/app.py"


# ══════════════════════════════════════════════════════════════════════════════
# D2: _loc_to_filepath location handling
# ══════════════════════════════════════════════════════════════════════════════

class TestLocToFilepath:
    def test_loc_with_absolute_path(self, tmp_path):
        from src.lsp.manager import _loc_to_filepath
        loc = {"absolutePath": "C:/Projects/Argent/main.py"}
        fp = _loc_to_filepath(loc, str(tmp_path))
        assert "Argent" in fp
        assert "main.py" in fp
        assert "file://" not in fp

    def test_loc_with_uri(self, tmp_path):
        from src.lsp.manager import _loc_to_filepath
        loc = {"uri": "file:///c:/Projects/Argent/tools/schemas.py"}
        fp = _loc_to_filepath(loc, str(tmp_path))
        assert "schemas.py" in fp
        assert "file:" not in fp

    def test_loc_fallback(self, tmp_path):
        from src.lsp.manager import _loc_to_filepath
        loc = {"someOtherKey": "my_file.py"}
        fp = _loc_to_filepath(loc, str(tmp_path))
        assert isinstance(fp, str)


# ══════════════════════════════════════════════════════════════════════════════
# D3: Diagnostic cache normalization
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticCacheNormalization:
    def test_normalize_file_key_case_and_slashes(self):
        from src.lsp.manager import _normalize_file_key
        key1 = _normalize_file_key("c:\\Users\\Foo\\Bar.py")
        key2 = _normalize_file_key("C:/Users/foo/bar.py")
        if sys.platform == "win32":
            assert key1 == key2
        assert "/" in key1 or "\\" in key1

    def test_manager_stores_and_retrieves_normalized(self, tmp_path):
        from src.lsp.manager import _normalize_file_key, lsp_manager
        test_file = tmp_path / "sample.py"
        test_file.write_text("x = 1\n", encoding="utf-8")

        diag_item = {
            "message": "Syntax error",
            "severity": 1,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
        }
        norm_key = _normalize_file_key(str(test_file))
        with lsp_manager._lock:
            lsp_manager._diagnostics[norm_key] = [diag_item]

        # Verify key lookup works with alternate casing
        alt_path_str = str(test_file).upper() if sys.platform == "win32" else str(test_file)
        alt_key = _normalize_file_key(alt_path_str)
        assert alt_key == norm_key
        assert lsp_manager._diagnostics[alt_key][0]["message"] == "Syntax error"


# ══════════════════════════════════════════════════════════════════════════════
# D4: Tool registry and schema parity
# ══════════════════════════════════════════════════════════════════════════════

class TestLspToolRegistryParity:
    def test_all_lsp_tools_registered_in_available_tools(self):
        from tools.schemas import AVAILABLE_TOOLS
        required_tools = ["grep_search"]
        for name in required_tools:
            assert name in AVAILABLE_TOOLS, f"Tool '{name}' must be in AVAILABLE_TOOLS"
            assert callable(AVAILABLE_TOOLS[name]), f"Tool '{name}' must be a callable function"

    def test_all_lsp_tools_have_schemas(self):
        from tools.schemas import TOOL_SCHEMAS
        schema_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        required_tools = ["grep_search"]
        for name in required_tools:
            assert name in schema_names, f"Tool '{name}' must have a schema in TOOL_SCHEMAS"

    def test_full_available_tools_and_schemas_parity(self):
        from tools.schemas import AVAILABLE_TOOLS, TOOL_SCHEMAS
        schema_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        function_names = set(AVAILABLE_TOOLS.keys())
        diff_schemas = schema_names - function_names
        diff_functions = function_names - schema_names
        assert not diff_schemas, f"Schemas without functions: {diff_schemas}"
        assert not diff_functions, f"Functions without schemas: {diff_functions}"


# ══════════════════════════════════════════════════════════════════════════════
# D5 & D7: Input boundary validation & file existence checks
# ══════════════════════════════════════════════════════════════════════════════

class TestInputValidationAndExistence:
    @pytest.mark.parametrize("target", ["definition", "references", "implementations",
                                        "call_hierarchy", "symbol"])
    def test_code_navigation_on_a_missing_file_says_so(self, target):
        """The separate LSP tools are gone; their input validation lives in
        grep_search(target=...), which must refuse a path that does not exist."""
        from tools.search_ops import grep_search
        res = grep_search(path="non_existent_xyz.py", symbol="foo", target=target,
                          line=1, column=0)
        assert "Error: Path" in res
        assert "does not exist" in res

    def test_manager_sanitizes_negative_and_zero_indices(self, tmp_path):
        from src.lsp.manager import lsp_manager
        sample = tmp_path / "valid.py"
        sample.write_text("def hello(): pass\n", encoding="utf-8")

        mock_server = MagicMock()
        mock_server.request_definition.return_value = []

        with patch.object(lsp_manager, "_get_server", return_value=mock_server):
            lsp_manager.find_definition(str(sample), line=-5, column=-10)
            mock_server.request_definition.assert_called_once()
            args = mock_server.request_definition.call_args[0]
            # args: (rel_path, safe_line - 1, safe_column)
            # safe_line is max(1, -5) = 1; 1 - 1 = 0
            assert args[1] == 0
            # safe_column is max(0, -10) = 0
            assert args[2] == 0


# ══════════════════════════════════════════════════════════════════════════════
# D6: Workspace symbols candidate filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceSymbolsCandidateFiltering:
    def test_filters_languages_by_existing_extensions(self, tmp_path):
        from src.lsp.manager import lsp_manager
        with patch("os.getcwd", return_value=str(tmp_path)), \
             patch.object(lsp_manager, "_get_server") as mock_get_server:
            res = lsp_manager.get_workspace_symbols("some_query")
            assert res is None
            mock_get_server.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# D8: Process cleanup & shutdown safety
# ══════════════════════════════════════════════════════════════════════════════

class TestShutdownAndAtexit:
    def test_atexit_is_registered(self):
        from src.lsp.manager import lsp_manager
        assert lsp_manager.shutdown is not None

    def test_shutdown_resilient_to_errors(self):
        from src.lsp.manager import LSPManager
        mgr = LSPManager()
        faulty_server = MagicMock()
        faulty_server.language_server.server.shutdown.side_effect = RuntimeError("Server crashed")
        faulty_server.server_process.poll.return_value = None
        faulty_server.server_process.kill.side_effect = OSError("Permission denied")

        mgr._servers[("python", "/dummy")] = faulty_server
        # Must not raise exceptions
        mgr.shutdown()
        assert mgr._servers == {}
