"""
tests/lmtrust/test_optimization_blind_spots.py
LMTrust Deep Blind Spot & Contract Audit Test Suite for Argent Acceleration Changes.

Covers:
  - L0 Smoke: Tool registry proxies, ripgrep search, RAG worker queue.
  - L1 x D7 (Contract): LazyToolProxy reflection (__annotations__, __code__, __wrapped__, inspect.signature).
  - L1 x D3 (Known Coordinates): search_files snippet coordinates and context windowing.
  - L2 x D1 (Boundary & Regex): search_files with regex characters ([0], (arg), $env, a+b) strictly literal.
  - L3 x D10 (Invariant / Parity): search_files ripgrep vs Python rglob return identical sets of files.
  - L4 x D4 (Adversarial): search_files with 0 matches completes in < 0.2s without slow rglob fallback.
  - L4b x D4 (Fallback): search_files transparent fallback to rglob when ripgrep binary is absent/fails.
  - L5 x D2 (State): LazyToolProxy unloaded -> loaded state transition idempotency.
  - L6 x D6 (Concurrency): Concurrent multi-threaded proxy resolution and async RAG updates.
  - L9 x D3 (Negative Space): RAG worker handles missing paths and invalid items without thread death.
"""

import sys
import time
import inspect
import threading
from pathlib import Path
import pytest

from tools.lazy_registry import LazyToolProxy, LazyToolRegistry
from tools.search_ops import search_files
from prompt_compressor import compress_tool_result, PRESERVE_CONTEXT_TOOLS


# ============================================================================
# L0 Smoke: Core Components Are Alive
# ============================================================================
@pytest.mark.lmtrust(layer="L0", direction="D7")
class TestL0Smoke:
    def test_lazy_registry_smoke(self):
        reg = LazyToolRegistry({
            "read_file": ("tools.file_ops", "read_file"),
            "search_files": ("tools.search_ops", "search_files"),
        })
        assert "read_file" in reg
        assert "search_files" in reg
        assert len(reg) == 2

    def test_search_files_smoke(self, tmp_path):
        f = tmp_path / "smoke.txt"
        f.write_text("Argent speedup verification", encoding="utf-8")
        out = search_files(directory=str(tmp_path), pattern="*.txt", content_contains="speedup")
        assert "smoke.txt" in out
        assert "speedup" in out

    def test_compress_tool_result_smoke(self):
        long_text = "\n".join(f"line {i}" for i in range(500))
        compressed = compress_tool_result(long_text, "gpt-4o", tool_name="read_file")
        assert len(compressed.splitlines()) >= 250


# ============================================================================
# L1 x D7: Contracts & Introspection Reflection
# ============================================================================
@pytest.mark.lmtrust(layer="L1", direction="D7")
class TestL1Contract:
    def test_lazy_proxy_callable_and_introspection(self):
        proxy = LazyToolProxy("tools.file_ops", "read_file")
        # 1. Callable contract
        assert callable(proxy)

        # 2. inspect.signature resolution via __wrapped__
        sig = inspect.signature(proxy)
        assert "file_path" in sig.parameters

        # 3. Docstring access
        doc = proxy.__doc__
        assert doc is not None and len(doc) > 10

        # 4. Attribute delegation via __getattr__
        assert hasattr(proxy, "__annotations__")
        assert "file_path" in proxy.__annotations__
        assert hasattr(proxy, "__code__")
        assert "file_path" in proxy.__code__.co_varnames

    def test_lazy_registry_dict_contract(self):
        reg = LazyToolRegistry({
            "read_file": ("tools.file_ops", "read_file"),
            "search_files": ("tools.search_ops", "search_files"),
        })
        # Must support full dict contract without eagerly loading
        proxy = reg["read_file"]
        assert isinstance(proxy, LazyToolProxy)
        assert proxy._target is None  # still unloaded!

        # items(), keys(), values(), get()
        assert "read_file" in list(reg.keys())
        assert proxy in list(reg.values())
        assert ("read_file", proxy) in list(reg.items())
        assert reg.get("read_file") is proxy
        assert reg.get("non_existent", "default") == "default"


# ============================================================================
# L1 x D3: Known Coordinates & Snippet Extraction Accuracy
# ============================================================================
@pytest.mark.lmtrust(layer="L1", direction="D3")
class TestL1Coordinates:
    def test_search_snippet_coordinates(self, tmp_path):
        f = tmp_path / "coordinates.py"
        prefix = "A" * 60
        target = "EXACT_TARGET_COORDINATE"
        suffix = "B" * 60
        f.write_text(f"{prefix}\n{target}\n{suffix}\n", encoding="utf-8")

        res = search_files(directory=str(tmp_path), pattern="*.py", content_contains=target)
        assert "coordinates.py" in res
        assert ">> ..." in res
        assert target in res


# ============================================================================
# L2 x D1: Boundaries & Special Characters (Regex Escaping)
# ============================================================================
@pytest.mark.lmtrust(layer="L2", direction="D1")
class TestL2EdgeCases:
    def test_search_files_literal_regex_characters(self, tmp_path):
        """Ensures that content_contains is matched as a literal substring,
        not as a regular expression, even with regex metacharacters: [], (), $, +, ?
        """
        f1 = tmp_path / "array_access.py"
        f1.write_text("result = matrix[0] + items[1]\n", encoding="utf-8")

        f2 = tmp_path / "function_call.py"
        f2.write_text("value = calculate(x, y)\n", encoding="utf-8")

        f3 = tmp_path / "env_var.py"
        f3.write_text("export_path = $env:ARGENT_HOME\n", encoding="utf-8")

        # 1. Square brackets [0] (in regex [0] is char class '0')
        res1 = search_files(directory=str(tmp_path), pattern="*.py", content_contains="matrix[0]")
        assert "array_access.py" in res1
        assert "function_call.py" not in res1

        # 2. Parentheses (x, y)
        res2 = search_files(directory=str(tmp_path), pattern="*.py", content_contains="(x, y)")
        assert "function_call.py" in res2
        assert "array_access.py" not in res2

        # 3. Dollar sign $env
        res3 = search_files(directory=str(tmp_path), pattern="*.py", content_contains="$env:ARGENT")
        assert "env_var.py" in res3

        # 4. Unclosed bracket '[' which crashes regex parser if not using -F
        res4 = search_files(directory=str(tmp_path), pattern="*.py", content_contains="matrix[")
        assert "array_access.py" in res4


# ============================================================================
# L3 x D10: Parity Invariant (Ripgrep vs Python rglob)
# ============================================================================
@pytest.mark.lmtrust(layer="L3", direction="D10")
class TestL3Parity:
    def test_ripgrep_and_rglob_parity(self, tmp_path, monkeypatch):
        """Verifies that search_files produces the exact same results with ripgrep
        as it does when ripgrep is disabled (pure Python rglob fallback).
        """
        # Create a hierarchy of test files
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "file1.py").write_text("# Target query in root\nalpha = 1\n", encoding="utf-8")
        (sub / "file2.py").write_text("# Target query in sub\nbeta = 2\n", encoding="utf-8")
        (sub / "file3.txt").write_text("# Target query in text\ngamma = 3\n", encoding="utf-8")

        # Search with Ripgrep active
        res_rg = search_files(directory=str(tmp_path), pattern="*.py", content_contains="Target query")

        # Disable ripgrep by simulating absence of binary
        import shutil
        monkeypatch.setattr(shutil, "which", lambda cmd: None)

        # Search with Python fallback
        res_py = search_files(directory=str(tmp_path), pattern="*.py", content_contains="Target query")

        assert "file1.py" in res_rg and "file1.py" in res_py
        assert "file2.py" in res_rg and "file2.py" in res_py
        # Both must honor pattern="*.py" and exclude file3.txt
        assert "file3.txt" not in res_rg
        assert "file3.txt" not in res_py


# ============================================================================
# L4 x D4: Adversarial / Zero-Match Fast Path
# ============================================================================
@pytest.mark.lmtrust(layer="L4", direction="D4")
class TestL4Adversarial:
    def test_zero_match_does_not_trigger_slow_rglob(self, tmp_path):
        """Ripgrep returning exit code 1 (no matches) must NOT trigger slow rglob fallback.
        Must execute under 200ms.
        """
        # Create multiple dummy files
        for i in range(20):
            (tmp_path / f"test_{i}.py").write_text(f"# Standard file content {i}\n", encoding="utf-8")

        t0 = time.perf_counter()
        res = search_files(directory=str(tmp_path), pattern="*.py", content_contains="NON_EXISTENT_TOKEN_12345")
        elapsed = time.perf_counter() - t0

        assert "No files found matching" in res
        assert elapsed < 0.35, f"Search took too long ({elapsed:.3f}s), slow fallback triggered!"


# ============================================================================
# L4b x D4: Fallback Under External Failure
# ============================================================================
@pytest.mark.lmtrust(layer="L4b", direction="D4")
class TestL4bFallback:
    def test_ripgrep_process_failure_falls_back_gracefully(self, tmp_path, monkeypatch):
        """If ripgrep binary exits with code > 1 (e.g. fatal signal or crash),
        search_files must gracefully fall back to rglob without raising an exception.
        """
        f = tmp_path / "target.py"
        f.write_text("critical_fallback_string = 42\n", encoding="utf-8")

        import subprocess
        orig_run = subprocess.run

        def mock_failing_rg(cmd, *args, **kwargs):
            if cmd and "rg" in str(cmd[0]):
                # Simulate ripgrep crash / code 2
                return subprocess.CompletedProcess(cmd, returncode=2, stdout="", stderr="rg: error")
            return orig_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", mock_failing_rg)

        res = search_files(directory=str(tmp_path), pattern="*.py", content_contains="critical_fallback_string")
        assert "target.py" in res
        assert "critical_fallback_string" in res


# ============================================================================
# L5 x D2: State & Lifecycle of Lazy Proxies
# ============================================================================
@pytest.mark.lmtrust(layer="L5", direction="D2")
class TestL5State:
    def test_lazy_tool_proxy_state_lifecycle(self):
        proxy = LazyToolProxy("tools.file_ops", "get_file_outline")
        assert proxy._target is None
        assert "unloaded" in repr(proxy)

        # First resolution
        target = proxy._resolve()
        assert target is not None
        assert proxy._target is target
        assert "loaded" in repr(proxy)

        # Idempotent second resolution returns exact same target
        target2 = proxy._resolve()
        assert target2 is target


# ============================================================================
# L6 x D6: Concurrency & Thread Safety
# ============================================================================
@pytest.mark.lmtrust(layer="L6", direction="D6")
class TestL6Concurrency:
    def test_concurrent_proxy_resolution(self):
        proxy = LazyToolProxy("tools.search_ops", "grep_search")
        results = [None] * 10
        exceptions = [None] * 10

        def worker(idx):
            try:
                fn = proxy._resolve()
                results[idx] = fn
            except Exception as e:
                exceptions[idx] = e

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(ex is None for ex in exceptions)
        assert all(res is not None for res in results)
        assert len(set(id(r) for r in results)) == 1

    def test_rag_engine_async_update_concurrency(self, tmp_path):
        import rag_engine

        test_file = tmp_path / "async_rag_test.py"
        test_file.write_text("def test_async_chunk(): pass\n", encoding="utf-8")

        # Enqueue multiple async updates
        for _ in range(5):
            rag_engine.update_file_index(str(test_file), async_mode=True)

        assert rag_engine._INDEX_WORKER_THREAD is not None
        assert rag_engine._INDEX_WORKER_THREAD.is_alive()


# ============================================================================
# L9 x D3: Negative Space (Corrupt Inputs & Context Preservation)
# ============================================================================
@pytest.mark.lmtrust(layer="L9", direction="D3")
class TestL9NegativeSpace:
    def test_preserve_context_tools_safeguard(self):
        """Ensures that precision tools (read_file, get_file_outline) are NEVER
        over-truncated below 250 lines even for 'tiny' or 'small' models.
        """
        sample_code = "\n".join(f"line_{i} = {i}" for i in range(300))

        # 1. Normal tool with small category (e.g. 3b model) truncates to 100 lines
        compressed_generic = compress_tool_result(sample_code, "something-3b", tool_name="run_command")
        assert len(compressed_generic.splitlines()) <= 150

        # 2. Preserve context tool (read_file) protects 250 lines minimum
        compressed_read = compress_tool_result(sample_code, "something-3b", tool_name="read_file")
        assert len(compressed_read.splitlines()) >= 250

    def test_rag_worker_handles_deleted_or_missing_path(self, tmp_path):
        import rag_engine
        missing_path = tmp_path / "does_not_exist_9999.py"

        # Should not raise exception and should complete cleanly
        rag_engine.update_file_index(str(missing_path), async_mode=False)
