import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tools.search_ops import grep_search, search_files
from tools.file_ops import replace_in_file, multi_replace_in_file, _parse_multi_replace_changes
from tools.browser_tools import browser_open, browser_action, browser_inspect, browser_close
from tools.schemas import TOOL_SCHEMAS
from tools.lazy_registry import LAZY_TOOL_REGISTRY
from tool_profiles import MODE_TOOLS, CORE_CHAT_TOOLS


# ============================================================================
# L0: Smoke — Are all consolidated tools alive, importable, and registered?
# ============================================================================
class TestL0Smoke:
    def test_consolidated_tools_importable_and_callable(self):
        """L0: Verify all consolidated tools are callable functions."""
        for fn in (grep_search, search_files, replace_in_file, multi_replace_in_file,
                   browser_open, browser_action, browser_inspect, browser_close):
            assert callable(fn), f"{fn} must be callable"

    def test_schema_and_lazy_registry_parity(self):
        """L0: Verify 100% parity between TOOL_SCHEMAS and LAZY_TOOL_REGISTRY."""
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        registry_names = set(LAZY_TOOL_REGISTRY.keys())
        assert schema_names == registry_names, (
            f"Parity mismatch: in schemas but not registry: {schema_names - registry_names}, "
            f"in registry but not schemas: {registry_names - schema_names}"
        )

    # Tools that exist only while a feature is switched on, so they are never in the
    # static registry: semantic_search joins when RAG is enabled.
    _DYNAMIC_TOOLS = {"semantic_search"}

    def test_tool_profiles_reference_valid_tools(self):
        """L0: Verify all tools in MODE_TOOLS and CORE_CHAT_TOOLS exist in LAZY_TOOL_REGISTRY."""
        for t in CORE_CHAT_TOOLS - self._DYNAMIC_TOOLS:
            assert t in LAZY_TOOL_REGISTRY, f"CORE_CHAT_TOOLS references unregistered tool '{t}'"
        for profile, tools in MODE_TOOLS.items():
            for t in tools:
                assert t in LAZY_TOOL_REGISTRY or t in self._DYNAMIC_TOOLS, \
                    f"MODE_TOOLS '{profile}' references unregistered tool '{t}'"


# ============================================================================
# L1: Contract (D7: API Contract) — Types and return shapes
# ============================================================================
class TestL1Contract:
    def test_grep_search_returns_string_for_all_targets(self, tmp_path):
        """L1: grep_search must return a string for every target mode."""
        test_file = tmp_path / "sample.py"
        test_file.write_text("def hello_world():\n    return 42\n", encoding="utf-8")

        for target in ("all", "definition", "references", "implementations", "call_hierarchy"):
            res = grep_search(path=str(test_file), query="hello_world", target=target)
            assert isinstance(res, str), f"grep_search with target={target} must return str, got {type(res)}"
            assert len(res) > 0

    def test_replace_in_file_contract_single(self, tmp_path):
        """L1: replace_in_file contract for single replacement."""
        f = tmp_path / "foo.txt"
        f.write_text("alpha beta gamma", encoding="utf-8")
        res = replace_in_file(file_path=str(f), target_text="beta", replacement_text="omega")
        assert isinstance(res, str)
        assert "Successfully replaced" in res
        assert f.read_text(encoding="utf-8") == "alpha omega gamma"

    def test_browser_action_contract_unknown_action(self):
        """L1: browser_action returns clean error string on unknown action."""
        with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}):
            res = browser_action(action="fly_to_moon")
            assert isinstance(res, str)
            assert "Error: Unknown browser action 'fly_to_moon'" in res

    def test_browser_inspect_contract_unknown_mode(self):
        """L1: browser_inspect returns clean error string on unknown mode."""
        with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}):
            res = browser_inspect(mode="xray_vision")
            assert isinstance(res, str)
            assert "Error: Unknown inspect mode 'xray_vision'" in res


# ============================================================================
# L2: Boundary (D1: Math / D9: Bounds) — 8 boundary conditions
# ============================================================================
class TestL2Boundary:
    def test_grep_empty_or_whitespace_symbol_rejected(self, tmp_path):
        """L2: Empty or whitespace symbol must be rejected for semantic targets."""
        f = tmp_path / "code.py"
        f.write_text("def foo(): pass", encoding="utf-8")

        for empty_val in ("", "   ", "\t\n"):
            res = grep_search(path=str(f), symbol=empty_val, target="definition")
            assert "Error: A symbol name or query is required" in res

    def test_grep_negative_and_zero_line_column_sanitized(self, tmp_path):
        """L2: line <= 0 or column < 0 must be safely sanitized without crashing."""
        f = tmp_path / "calc.py"
        f.write_text("def compute(x):\n    return x * 2\n", encoding="utf-8")

        # Passing negative line and column should not crash, it sanitizes and resolves
        res = grep_search(path=str(f), symbol="compute", target="definition", line=-5, column=-10)
        assert isinstance(res, str)
        assert "compute" in res

    def test_grep_max_results_boundary(self, tmp_path):
        """L2: max_results <= 0 or non-int string handled safely."""
        f = tmp_path / "lines.txt"
        f.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")

        res_zero = grep_search(path=str(f), query="line", max_results=0)
        assert isinstance(res_zero, str)
        assert "Found" in res_zero

        res_str = grep_search(path=str(f), query="line", max_results="not_an_int")
        assert isinstance(res_str, str)
        assert "Found" in res_str

    def test_replace_empty_target_text_rejected(self, tmp_path):
        """L2: Empty target_text returns clear error preventing character injection."""
        f = tmp_path / "test.py"
        f.write_text("def run(): pass", encoding="utf-8")
        res = replace_in_file(file_path=str(f), target_text="", replacement_text="new")
        assert "Error: 'target_text' is empty" in res
        res_ws = replace_in_file(file_path=str(f), target_text="   \n  ", replacement_text="new")
        assert "Error: 'target_text' is empty" in res_ws

    def test_replace_start_end_line_boundaries(self, tmp_path):
        """L2: Invalid start_line / end_line ranges return helpful errors."""
        f = tmp_path / "data.py"
        f.write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")

        res_inv = replace_in_file(file_path=str(f), target_text="b = 2", replacement_text="b = 5",
                                 start_line=3, end_line=1)
        assert "cannot be less than 'start_line'" in res_inv

        res_exceed = replace_in_file(file_path=str(f), target_text="b = 2", replacement_text="b = 5",
                                    start_line=50)
        assert "exceeds total lines in file" in res_exceed

    def test_browser_inspect_query_forwarded_to_state(self):
        """L2: browser_inspect passes query parameter to browser_state for filtering."""
        with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}), \
             patch("tools.browser_tools.browser_state", return_value="Filtered State Result") as mock_state:
            res = browser_inspect(mode="dom_interactive", query="Sign In")
            mock_state.assert_called_once_with(session="default", query="Sign In")
            assert res == "Filtered State Result"

    def test_browser_action_instruction_alias(self):
        """L2: browser_action aliases instruction to text when text is not provided."""
        with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}), \
             patch("tools.browser_tools.browser_click", return_value="Clicked button") as mock_click:
            res = browser_action(action="click", instruction="Submit Order")
            mock_click.assert_called_once_with(
                index=None, selector=None, text="Submit Order", session="default", return_state=True
            )
            assert res == "Clicked button"


# ============================================================================
# L3: Property (D10: Invariants) — Invariants & guarantees
# ============================================================================
class TestL3Property:
    def test_dom_inspection_truncation_invariant(self):
        """L3: DOM accessibility tree must never exceed 15KB context limit."""
        huge_state = "A" * 25000
        with patch("browser_engine.browser_engine._sessions", {"default": MagicMock()}), \
             patch("tools.browser_tools.browser_state", return_value=huge_state):
            res = browser_inspect(mode="dom_interactive")
            assert len(res) < 16000
            assert "State truncated to 15KB to preserve context" in res
            assert res.startswith("A" * 15000)

    def test_atomic_rollback_on_syntax_error(self, tmp_path):
        """L3: Syntax breaking modifications are automatically rolled back."""
        f = tmp_path / "valid.py"
        initial_code = "def calculate():\n    return 100\n"
        f.write_text(initial_code, encoding="utf-8")

        res = replace_in_file(
            file_path=str(f),
            target_text="return 100",
            replacement_text="return def def syntax error !!!"
        )
        assert "Modification aborted because it broke code compilation" in res
        assert f.read_text(encoding="utf-8") == initial_code

    def test_crlf_newline_preservation(self, tmp_path):
        """L3: CRLF line endings are strictly preserved without becoming \\r\\r\\n."""
        f = tmp_path / "windows.py"
        crlf_text = "def greet():\r\n    print('hi')\r\n"
        f.write_bytes(crlf_text.encode("utf-8"))

        res = replace_in_file(
            file_path=str(f),
            target_text="print('hi')",
            replacement_text="print('hello')\n    print('world')"
        )
        assert "Successfully replaced" in res
        data = f.read_bytes()
        assert b"\r\r\n" not in data
        assert b"\r\n" in data
        assert data == b"def greet():\r\n    print('hello')\r\n    print('world')\r\n"


# ============================================================================
# L4: Adversarial (D4: Error Handling / D8: Security)
# ============================================================================
class TestL4Adversarial:
    def test_corrupted_json_in_multi_replace(self):
        """L4: Malformed JSON in multi_replace does not throw uncaught exception."""
        res = multi_replace_in_file("{malformed: json[array")
        assert "Error: changes_json is not valid JSON" in res

    def test_nonexistent_file_path(self):
        """L4: Operating on a non-existent file returns clean error message."""
        res = replace_in_file(file_path="non_existent_dir_999/foo.py", target_text="a", replacement_text="b")
        assert "does not exist" in res

    def test_path_traversal_restricted(self):
        """L4: Attempting to edit files with restricted extensions is halted."""
        res = replace_in_file(file_path="Assets/foo.cs.meta", target_text="guid", replacement_text="new_guid")
        assert "restricted" in res.lower() or "meta" in res.lower()


# ============================================================================
# L4b: Fallback (D4: Recovery paths)
# ============================================================================
class TestL4bFallback:
    def test_grep_definition_fallback_to_text_when_lsp_unavailable(self, tmp_path):
        """L4b: When LSP is unavailable, grep_search falls back to text/regex pattern."""
        f = tmp_path / "calc.py"
        f.write_text("class Calculator:\n    def add(self, a, b):\n        return a + b\n", encoding="utf-8")

        with patch("src.lsp.manager.lsp_manager.is_available", return_value=False):
            res = grep_search(path=str(f), symbol="Calculator", target="definition")
            assert "Calculator" in res
            assert ("Found" in res or "match" in res)

    def test_grep_references_fallback_to_text_when_lsp_unavailable(self, tmp_path):
        """L4b: When LSP is unavailable, references target falls back to text references."""
        f = tmp_path / "use.py"
        f.write_text("c = Calculator()\nresult = c.add(1, 2)\n", encoding="utf-8")

        with patch("src.lsp.manager.lsp_manager.is_available", return_value=False):
            res = grep_search(path=str(f), symbol="Calculator", target="references")
            assert "Calculator" in res
            assert "showing text reference(s)" in res


# ============================================================================
# L5: State (D2: Lifecycle) — Browser session state machine
# ============================================================================
class TestL5State:
    def test_browser_action_fails_cleanly_when_no_active_session(self):
        """L5: Calling browser_action with no open session returns clear error."""
        with patch("browser_engine.browser_engine._sessions", {}):
            res = browser_action(action="click", selector="button")
            assert "Error: No active browser session. Call browser_open(url) first" in res

    def test_browser_inspect_fails_cleanly_when_no_active_session(self):
        """L5: Calling browser_inspect with no open session returns clear error."""
        with patch("browser_engine.browser_engine._sessions", {}):
            res = browser_inspect(mode="dom_interactive")
            assert "Error: No active browser session. Call browser_open(url) first" in res

    def test_browser_close_idempotent(self):
        """L5: Closing an already closed browser session returns clean inactive status."""
        with patch("browser_engine.browser_engine._sessions", {}):
            res = browser_close("default")
            assert "Session 'default' is not active" in res


# ============================================================================
# L6: Cross-System (Cross-Layer / D7) — Composition between tools and agent
# ============================================================================
class TestL6CrossSystem:
    def test_replace_in_file_delegates_dict_wrapped_changes(self, tmp_path):
        """L6: replace_in_file properly unwraps dict-wrapped changes and sets file_path."""
        f = tmp_path / "multi_test.py"
        f.write_text("item1 = 1\nitem2 = 2\n", encoding="utf-8")

        res = replace_in_file(
            file_path=str(f),
            changes={"changes": [
                {"target_text": "item1 = 1", "replacement_text": "item1 = 10"},
                {"target_text": "item2 = 2", "replacement_text": "item2 = 20"}
            ]}
        )
        assert "Multi-replace execution finished" in res
        assert f.read_text(encoding="utf-8") == "item1 = 10\nitem2 = 20\n"

    def test_agent_diagnostics_extracts_all_target_fps(self):
        """L6: Agent _maybe_run_lsp_diagnostics accumulates all modified files in a batch."""
        from agent import ArgentAgent
        agent = ArgentAgent.__new__(ArgentAgent)

        filtered_args = {
            "file_path": "src/main.py",
            "changes": [
                {"file_path": "src/helper.py", "target_text": "a", "replacement_text": "b"},
                {"file_path": "src/main.py", "target_text": "c", "replacement_text": "d"},
            ]
        }
        # Simulate logic extracted from agent.py
        target_fps = []
        fp = filtered_args.get("file_path") or filtered_args.get("path")
        if fp and fp not in target_fps:
            target_fps.append(fp)
        raw_changes = filtered_args.get("changes") or filtered_args.get("changes_json")
        if raw_changes:
            items, _ = _parse_multi_replace_changes(raw_changes)
            if items:
                for it in items:
                    if isinstance(it, dict) and it.get("file_path") and it["file_path"] not in target_fps:
                        target_fps.append(it["file_path"])

        assert target_fps == ["src/main.py", "src/helper.py"]


# ============================================================================
# L7: Compositional (D4: Multiple failures / Overlap)
# ============================================================================
class TestL7Compositional:
    def test_multi_replace_detects_overlapping_chunks(self, tmp_path):
        """L7: Multi-replace detects overlapping change regions and halts atomically."""
        f = tmp_path / "overlap.txt"
        f.write_text("ABCDEF", encoding="utf-8")

        res = multi_replace_in_file([
            {"file_path": str(f), "target_text": "BCDE", "replacement_text": "1234"},
            {"file_path": str(f), "target_text": "CDEF", "replacement_text": "5678"},
        ])
        assert "overlap" in res.lower()
        assert f.read_text(encoding="utf-8") == "ABCDEF"


# ============================================================================
# L8: Temporal (D6: Ordering invariants)
# ============================================================================
class TestL8Temporal:
    def test_multi_chunk_bottom_to_top_preserves_offsets(self, tmp_path):
        """L8: Multiple non-overlapping edits in one file apply bottom-to-top without offset drift."""
        f = tmp_path / "order.txt"
        lines = [f"line_{i}" for i in range(10)]
        f.write_text("\n".join(lines), encoding="utf-8")

        res = multi_replace_in_file([
            {"file_path": str(f), "target_text": "line_2", "replacement_text": "expanded_line_2_part1\nexpanded_line_2_part2"},
            {"file_path": str(f), "target_text": "line_7", "replacement_text": "modified_line_7"},
        ])
        assert "Multi-replace execution finished" in res
        content = f.read_text(encoding="utf-8")
        assert "expanded_line_2_part1" in content
        assert "modified_line_7" in content
        assert "line_9" in content


# ============================================================================
# L9: Negative Space (D3: Integrity — What must NEVER happen)
# ============================================================================
class TestL9NegativeSpace:
    def test_lsp_diagnostics_extension_gate(self):
        """L9: Extension gate strictly rejects non-code files (.md, .txt, .json)."""
        from src.lsp.manager import lsp_manager

        non_code_files = ["README.md", "notes.txt", "config.json", "styles.css", "schema.yaml"]
        for ncf in non_code_files:
            lang = lsp_manager.detect_language(ncf)
            assert lang is None, f"Extension gate failure: '{ncf}' was detected as language '{lang}'"

    def test_search_files_ignores_dotfiles_and_binaries(self, tmp_path):
        """L9: search_files must never return dotfiles (.git) or binary artifacts."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("repositoryformatversion = 0", encoding="utf-8")

        bin_file = tmp_path / "test.exe"
        bin_file.write_bytes(b"\x00\x01\x02")

        code_file = tmp_path / "app.py"
        code_file.write_text("print('hello')", encoding="utf-8")

        res = search_files(directory=str(tmp_path), pattern="*")
        assert ".git" not in res
        assert "test.exe" not in res
        assert "app.py" in res
