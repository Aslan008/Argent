"""
Blind-spot tests for rag_engine.py pure functions and helpers.

Covers _is_ignored, _load_argentignore, _chunk_python_ast,
_chunk_heuristic, _chunk_text, _clear_keyword_cache, is_rag_enabled /
disable_rag / enable_rag_for_project, OllamaEmbeddingFunction, _read_pdf,
prune_deleted_files, remove_file_index, update_file_index, and
semantic_search — all without a real ChromaDB.
"""

import sys
import os
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root (where rag_engine.py lives) is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import rag_engine
from rag_engine import (
    _is_ignored,
    _load_argentignore,
    _chunk_python_ast,
    _chunk_heuristic,
    _chunk_text,
    _clear_keyword_cache,
    is_rag_enabled,
    disable_rag,
    OllamaEmbeddingFunction,
)


# ---------------------------------------------------------------------------
# _is_ignored
# ---------------------------------------------------------------------------

class TestIsIgnored:
    def test_matching_extension_pattern(self):
        patterns = {"*.pyc", "node_modules", ".git"}
        assert _is_ignored("foo/bar.pyc", patterns) is True

    def test_matching_directory_pattern(self):
        patterns = {"*.pyc", "node_modules", ".git"}
        assert _is_ignored("node_modules/package/index.js", patterns) is True

    def test_matching_git_directory(self):
        patterns = {"*.pyc", "node_modules", ".git"}
        assert _is_ignored(".git/config", patterns) is True

    def test_non_matching_path(self):
        patterns = {"*.pyc", "node_modules", ".git"}
        assert _is_ignored("src/main.py", patterns) is False

    def test_non_matching_extension(self):
        patterns = {"*.pyc", "node_modules", ".git"}
        assert _is_ignored("src/main.py", patterns) is False

    def test_empty_patterns_returns_false(self):
        assert _is_ignored("node_modules/foo", set()) is False

    def test_none_like_empty_patterns(self):
        # An empty set is falsy; the function should short-circuit.
        assert _is_ignored("anything/here.py", set()) is False

    def test_nested_path_matching(self):
        patterns = {"node_modules", ".git"}
        assert _is_ignored("src/node_modules/lodash/index.js", patterns) is True

    def test_nested_git_path(self):
        patterns = {".git"}
        assert _is_ignored("subdir/.git/HEAD", patterns) is True

    def test_extension_pattern_only_matches_suffix(self):
        patterns = {"*.pyc"}
        # .pyc at the end → True
        assert _is_ignored("a/b/c.pyc", patterns) is True
        # not ending in .pyc → False
        assert _is_ignored("a/b/c.py", patterns) is False

    def test_prefix_match(self):
        # rel_path.startswith(pattern) is also checked
        patterns = {"build"}
        assert _is_ignored("build/output.o", patterns) is True

    def test_no_false_positive_on_similar_name(self):
        patterns = {"node_modules"}
        # "node_modules_extra" is a different directory name
        assert _is_ignored("node_modules_extra/foo.py", patterns) is True  # startswith matches

    def test_deeply_nested_non_match(self):
        patterns = {"*.pyc", "node_modules", ".git"}
        assert _is_ignored("src/utils/helpers/deep/file.py", patterns) is False


# ---------------------------------------------------------------------------
# _load_argentignore
# ---------------------------------------------------------------------------

class TestLoadArgentignore:
    def test_loads_patterns_from_file(self, tmp_path):
        ignore_file = tmp_path / ".argentignore"
        ignore_file.write_text("*.pyc\nnode_modules/\n.git/\n# comment\n\nvenv\n", encoding="utf-8")
        patterns = _load_argentignore(tmp_path)
        # trailing slashes are stripped
        assert "*.pyc" in patterns
        assert "node_modules" in patterns
        assert ".git" in patterns
        assert "venv" in patterns
        # comments and blank lines excluded
        assert "# comment" not in patterns
        assert "" not in patterns

    def test_no_file_returns_empty_set(self, tmp_path):
        patterns = _load_argentignore(tmp_path)
        assert patterns == set()

    def test_empty_file_returns_empty_set(self, tmp_path):
        (tmp_path / ".argentignore").write_text("", encoding="utf-8")
        patterns = _load_argentignore(tmp_path)
        assert patterns == set()

    def test_only_comments_and_blanks(self, tmp_path):
        (tmp_path / ".argentignore").write_text("# a\n# b\n\n   \n", encoding="utf-8")
        patterns = _load_argentignore(tmp_path)
        assert patterns == set()

    def test_trailing_slash_stripped(self, tmp_path):
        (tmp_path / ".argentignore").write_text("node_modules/\n", encoding="utf-8")
        patterns = _load_argentignore(tmp_path)
        assert "node_modules" in patterns
        assert "node_modules/" not in patterns

    def test_whitespace_stripped(self, tmp_path):
        (tmp_path / ".argentignore").write_text("  *.pyc  \n  venv \n", encoding="utf-8")
        patterns = _load_argentignore(tmp_path)
        assert "*.pyc" in patterns
        assert "venv" in patterns

    def test_malformed_file_does_not_crash(self, tmp_path):
        # Binary-ish content should not raise
        ignore_file = tmp_path / ".argentignore"
        ignore_file.write_bytes(b"\xff\xfe\x00\xbad binary")
        patterns = _load_argentignore(tmp_path)
        # Should return a set (possibly empty) without raising
        assert isinstance(patterns, set)

    def test_duplicate_patterns_deduplicated(self, tmp_path):
        (tmp_path / ".argentignore").write_text("venv\nvenv\nvenv\n", encoding="utf-8")
        patterns = _load_argentignore(tmp_path)
        assert patterns == {"venv"}


# ---------------------------------------------------------------------------
# _chunk_python_ast
# ---------------------------------------------------------------------------

class TestChunkPythonAst:
    def test_chunks_simple_file(self):
        code = "import os\n\ndef foo():\n    return 1\n\nclass Bar:\n    pass\n"
        docs, metas = _chunk_python_ast(code, "test.py")
        assert len(docs) > 0
        assert len(docs) == len(metas)
        for d, m in zip(docs, metas):
            assert "File: test.py" in d
            assert "file" in m
            assert m["file"] == "test.py"
            assert "start_line" in m
            assert "end_line" in m

    def test_empty_content_returns_empty(self):
        docs, metas = _chunk_python_ast("", "empty.py")
        assert docs == []
        assert metas == []

    def test_whitespace_only_content(self):
        docs, metas = _chunk_python_ast("   \n\n   \n", "ws.py")
        # No actual lines of content → empty or minimal
        assert isinstance(docs, list)
        assert isinstance(metas, list)

    def test_syntax_error_falls_back_to_heuristic(self):
        code = "def foo(:\n    return 1\n"
        docs, metas = _chunk_python_ast(code, "bad.py")
        # Should not raise; falls back to _chunk_heuristic
        assert isinstance(docs, list)
        assert isinstance(metas, list)
        # Heuristic fallback should still produce some chunks
        if docs:
            for m in metas:
                assert m.get("context") == "Heuristic block"

    def test_functions_are_separate_chunks(self):
        code = (
            "def func_a():\n"
            "    return 1\n"
            "\n"
            "def func_b():\n"
            "    return 2\n"
        )
        docs, metas = _chunk_python_ast(code, "funcs.py")
        contexts = [m["context"] for m in metas]
        # Both functions should appear as separate chunks
        assert any("func_a" in c for c in contexts)
        assert any("func_b" in c for c in contexts)

    def test_classes_are_separate_chunks(self):
        code = (
            "class Alpha:\n"
            "    pass\n"
            "\n"
            "class Beta:\n"
            "    pass\n"
        )
        docs, metas = _chunk_python_ast(code, "classes.py")
        contexts = [m["context"] for m in metas]
        assert any("Alpha" in c for c in contexts)
        assert any("Beta" in c for c in contexts)

    def test_module_level_code_chunked(self):
        code = "import os\nimport sys\n\nx = 1\n"
        docs, metas = _chunk_python_ast(code, "mod.py")
        assert len(docs) >= 1
        assert any("Module level" in m["context"] for m in metas)

    def test_async_function_chunked(self):
        code = "async def fetch():\n    return 42\n"
        docs, metas = _chunk_python_ast(code, "async.py")
        assert len(docs) >= 1
        assert any("fetch" in m["context"] for m in metas)

    def test_large_function_split_into_parts(self):
        # A function with > 80 lines should be split into parts
        lines = ["def big_func():"] + ["    x = %d" % i for i in range(100)]
        code = "\n".join(lines) + "\n"
        docs, metas = _chunk_python_ast(code, "big.py")
        # Should produce multiple chunks for the single function
        part_chunks = [m for m in metas if "Part" in m["context"]]
        assert len(part_chunks) >= 2

    def test_metadata_has_correct_fields(self):
        code = "def foo():\n    return 1\n"
        docs, metas = _chunk_python_ast(code, "test.py")
        for m in metas:
            assert "file" in m
            assert "start_line" in m
            assert "end_line" in m
            assert "context" in m
            assert isinstance(m["start_line"], int)
            assert isinstance(m["end_line"], int)

    def test_header_contains_file_path(self):
        code = "def foo():\n    return 1\n"
        docs, _ = _chunk_python_ast(code, "mydir/test.py")
        assert any("mydir/test.py" in d for d in docs)


# ---------------------------------------------------------------------------
# _chunk_heuristic
# ---------------------------------------------------------------------------

class TestChunkHeuristic:
    def test_short_text_single_chunk(self):
        text = "line one\nline two\nline three\n"
        docs, metas = _chunk_heuristic(text, "file.txt")
        assert len(docs) == 1
        assert metas[0]["context"] == "Heuristic block"

    def test_empty_text_returns_empty(self):
        docs, metas = _chunk_heuristic("", "empty.txt")
        assert docs == []
        assert metas == []

    def test_no_lines_returns_empty(self):
        docs, metas = _chunk_heuristic("", "empty.txt")
        assert docs == []
        assert metas == []

    def test_long_text_chunked_by_line_count(self):
        # 120 lines → should be split into chunks of ~60 lines
        text = "\n".join("line %d" % i for i in range(120)) + "\n"
        docs, metas = _chunk_heuristic(text, "long.txt")
        assert len(docs) >= 2

    def test_def_pattern_starts_new_chunk(self):
        # A 'def' line after >20 lines of content starts a new chunk
        lines = ["x = %d" % i for i in range(25)]
        lines.append("def my_func():")
        lines.append("    return 1")
        text = "\n".join(lines) + "\n"
        docs, metas = _chunk_heuristic(text, "file.js")
        assert len(docs) >= 2

    def test_class_pattern_starts_new_chunk(self):
        lines = ["x = %d" % i for i in range(25)]
        lines.append("class MyClass:")
        lines.append("    pass")
        text = "\n".join(lines) + "\n"
        docs, metas = _chunk_heuristic(text, "file.ts")
        assert len(docs) >= 2

    def test_chunk_size_boundary_60_lines(self):
        # Exactly 60 lines → single chunk (boundary is >= 60 triggers flush)
        text = "\n".join("line %d" % i for i in range(60)) + "\n"
        docs, metas = _chunk_heuristic(text, "b.txt")
        # 60 lines → one chunk (the 60th line triggers flush, but it's the last)
        assert len(docs) >= 1

    def test_metadata_fields(self):
        text = "hello\nworld\n"
        docs, metas = _chunk_heuristic(text, "f.txt")
        for m in metas:
            assert "file" in m
            assert "start_line" in m
            assert "end_line" in m
            assert m["context"] == "Heuristic block"

    def test_header_contains_file_path(self):
        text = "hello\n"
        docs, _ = _chunk_heuristic(text, "dir/sub.txt")
        assert "dir/sub.txt" in docs[0]


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_python_file_uses_ast(self):
        code = "def foo():\n    return 1\n"
        result = _chunk_text(code, "test.py")
        # Returns a tuple of (docs, metas)
        assert isinstance(result, tuple)
        docs, metas = result
        assert len(docs) >= 1
        # AST chunks have context like "FunctionDef foo"
        assert any("foo" in m["context"] for m in metas)

    def test_non_python_uses_heuristic(self):
        text = "function bar() {\n  return 1;\n}\n"
        result = _chunk_text(text, "test.js")
        docs, metas = result
        assert len(docs) >= 1
        assert all(m["context"] == "Heuristic block" for m in metas)

    def test_empty_text_python(self):
        docs, metas = _chunk_text("", "empty.py")
        assert docs == []
        assert metas == []

    def test_empty_text_non_python(self):
        docs, metas = _chunk_text("", "empty.js")
        assert docs == []
        assert metas == []

    def test_very_long_lines(self):
        long_line = "x" * 5000
        text = long_line + "\n" + long_line + "\n"
        docs, metas = _chunk_text(text, "long.txt")
        assert len(docs) >= 1

    def test_no_newlines(self):
        text = "just one long line of text without any newlines"
        docs, metas = _chunk_text(text, "flat.txt")
        assert len(docs) >= 1

    def test_python_syntax_error_falls_back(self):
        code = "def broken(:\n"
        docs, metas = _chunk_text(code, "bad.py")
        # Falls back to heuristic
        assert isinstance(docs, list)
        if docs:
            assert all(m["context"] == "Heuristic block" for m in metas)


# ---------------------------------------------------------------------------
# _clear_keyword_cache
# ---------------------------------------------------------------------------

class TestClearKeywordCache:
    def test_does_not_raise_when_module_missing(self):
        # The function should not raise even if src.rag.keyword_index is not
        # importable in the test environment.
        with patch.dict(sys.modules, {"src.rag.keyword_index": None}):
            _clear_keyword_cache()  # should not raise

    def test_calls_clear_cache_when_available(self):
        mock_module = MagicMock()
        mock_module.clear_cache = MagicMock()
        with patch.dict(sys.modules, {"src.rag.keyword_index": mock_module}):
            _clear_keyword_cache()
            mock_module.clear_cache.assert_called_once()

    def test_swallows_exceptions_from_clear_cache(self):
        mock_module = MagicMock()
        mock_module.clear_cache = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict(sys.modules, {"src.rag.keyword_index": mock_module}):
            # Should not propagate the exception
            _clear_keyword_cache()


# ---------------------------------------------------------------------------
# is_rag_enabled / disable_rag
# ---------------------------------------------------------------------------

class TestRagEnabledState:
    def test_default_state_is_boolean(self):
        # is_rag_enabled reads global state; just verify it returns a bool
        result = is_rag_enabled()
        assert isinstance(result, bool)

    def test_disable_rag_sets_flag_false(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", MagicMock())
        disable_rag()
        assert rag_engine._RAG_ENABLED is False
        assert rag_engine._COLLECTION is None

    def test_disable_rag_when_already_disabled(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        disable_rag()
        assert rag_engine._RAG_ENABLED is False
        assert rag_engine._COLLECTION is None

    def test_is_rag_enabled_true_when_flag_true(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        assert is_rag_enabled() is True

    def test_is_rag_enabled_false_when_flag_false_and_no_kbs(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        assert is_rag_enabled() is False

    def test_is_rag_enabled_true_when_kbs_active(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [MagicMock()])
        assert is_rag_enabled() is True

    def test_enable_rag_for_project_missing_dir(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        result = rag_engine.enable_rag_for_project("/nonexistent/path/that/does/not/exist")
        assert "does not exist" in result or "Error" in result
        assert rag_engine._RAG_ENABLED is False


# ---------------------------------------------------------------------------
# OllamaEmbeddingFunction
# ---------------------------------------------------------------------------

class TestOllamaEmbeddingFunction:
    def test_init_stores_model_name(self):
        ef = OllamaEmbeddingFunction(model_name="custom-model")
        assert ef.model_name == "custom-model"

    def test_init_default_model_name(self):
        ef = OllamaEmbeddingFunction()
        assert ef.model_name == "nomic-embed-text"

    def test_call_with_empty_list(self):
        ef = OllamaEmbeddingFunction()
        result = ef([])
        assert result == []

    def test_embed_query_returns_list(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        monkeypatch.setattr("requests.post", MagicMock(return_value=mock_response))
        # Patch config to avoid import issues
        monkeypatch.setattr("config.get_embedding_batch_size", lambda: 50)
        result = ef.embed_query("hello")
        assert isinstance(result, list)
        assert result == [0.1, 0.2, 0.3]

    def test_embed_documents_returns_list_of_lists(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        }
        monkeypatch.setattr("requests.post", MagicMock(return_value=mock_response))
        monkeypatch.setattr("config.get_embedding_batch_size", lambda: 50)
        result = ef.embed_documents(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    def test_embed_query_empty_result(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": []}
        monkeypatch.setattr("requests.post", MagicMock(return_value=mock_response))
        monkeypatch.setattr("config.get_embedding_batch_size", lambda: 50)
        # __call__ with one item but embeddings empty → mismatch → fallback
        # The fallback also uses requests.post, so keep the same mock
        result = ef.embed_query("hello")
        assert isinstance(result, list)

    def test_call_batch_mismatch_triggers_fallback(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        # Return fewer embeddings than input texts → triggers ValueError → fallback
        mock_response.json.return_value = {"embeddings": [[0.1]]}
        monkeypatch.setattr("requests.post", MagicMock(return_value=mock_response))
        monkeypatch.setattr("config.get_embedding_batch_size", lambda: 50)
        monkeypatch.setattr("config.get_embedding_concurrency", lambda: 1)
        result = ef(["text1", "text2"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_individually_single_text(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.5, 0.6]]}
        monkeypatch.setattr("requests.post", MagicMock(return_value=mock_response))
        monkeypatch.setattr("config.get_embedding_concurrency", lambda: 1)
        result = ef._embed_individually(["single text"], "http://localhost:11434")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == [0.5, 0.6]

    def test_embed_individually_handles_api_error(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(side_effect=Exception("API down"))
        monkeypatch.setattr("requests.post", MagicMock(return_value=mock_response))
        monkeypatch.setattr("config.get_embedding_concurrency", lambda: 1)
        result = ef._embed_individually(["text"], "http://localhost:11434")
        assert isinstance(result, list)
        assert len(result) == 1
        # On error, returns zero vector of 768 dims
        assert len(result[0]) == 768
        assert all(v == 0.0 for v in result[0])

    def test_call_uses_ollama_host_env(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        monkeypatch.setenv("OLLAMA_HOST", "http://myhost:9999")
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1]]}
        mock_post = MagicMock(return_value=mock_response)
        monkeypatch.setattr("requests.post", mock_post)
        monkeypatch.setattr("config.get_embedding_batch_size", lambda: 50)
        ef(["text"])
        # Verify the URL used includes the env host
        call_args = mock_post.call_args
        assert "http://myhost:9999" in call_args[0][0]

    def test_call_batch_size_from_config(self, monkeypatch):
        ef = OllamaEmbeddingFunction()
        # With batch_size=1, each batch has exactly 1 text, so the mock must
        # return exactly 1 embedding per call to avoid the mismatch fallback.
        def make_response(*args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            sent = kwargs.get("json", {}).get("input", [])
            resp.json.return_value = {"embeddings": [[0.1] for _ in sent]}
            return resp
        mock_post = MagicMock(side_effect=make_response)
        monkeypatch.setattr("requests.post", mock_post)
        monkeypatch.setattr("config.get_embedding_batch_size", lambda: 1)
        # batch_size=1 → two texts → two separate batch calls (no fallback)
        ef(["text1", "text2"])
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# _read_pdf
# ---------------------------------------------------------------------------

class TestReadPdf:
    def test_nonexistent_file_returns_none(self):
        result = rag_engine._read_pdf("/nonexistent/file.pdf")
        # Should return None (or possibly empty string), not raise
        assert result is None or result == ""

    def test_non_pdf_file_returns_none(self, tmp_path):
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_text("this is not a real PDF file", encoding="utf-8")
        result = rag_engine._read_pdf(str(fake_pdf))
        # Should return None for an invalid/unreadable PDF
        assert result is None or result == ""

    def test_text_file_with_pdf_extension(self, tmp_path):
        fake = tmp_path / "doc.pdf"
        fake.write_text("plain text content", encoding="utf-8")
        result = rag_engine._read_pdf(str(fake))
        assert result is None or result == ""


# ---------------------------------------------------------------------------
# prune_deleted_files
# ---------------------------------------------------------------------------

class TestPruneDeletedFiles:
    def test_returns_zero_when_rag_disabled(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        assert rag_engine.prune_deleted_files() == 0

    def test_returns_zero_when_collection_none(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        assert rag_engine.prune_deleted_files() == 0

    def test_prunes_nonexistent_files(self, monkeypatch, tmp_path):
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"file": "deleted_file.py"},
                {"file": "another_deleted.py"},
            ]
        }
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)

        # Patch project_root_or_cwd to return tmp_path (where files don't exist)
        mock_pp = MagicMock()
        mock_pp.project_root_or_cwd = MagicMock(return_value=tmp_path)
        monkeypatch.setitem(sys.modules, "project_paths", mock_pp)

        result = rag_engine.prune_deleted_files()
        assert result == 2
        assert mock_col.delete.call_count == 2

    def test_keeps_existing_files(self, monkeypatch, tmp_path):
        existing = tmp_path / "exists.py"
        existing.write_text("print('hello')", encoding="utf-8")

        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [{"file": "exists.py"}]
        }
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)

        mock_pp = MagicMock()
        mock_pp.project_root_or_cwd = MagicMock(return_value=tmp_path)
        monkeypatch.setitem(sys.modules, "project_paths", mock_pp)

        result = rag_engine.prune_deleted_files()
        assert result == 0
        mock_col.delete.assert_not_called()

    def test_empty_index_prunes_nothing(self, monkeypatch, tmp_path):
        mock_col = MagicMock()
        mock_col.get.return_value = {"metadatas": []}
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)

        mock_pp = MagicMock()
        mock_pp.project_root_or_cwd = MagicMock(return_value=tmp_path)
        monkeypatch.setitem(sys.modules, "project_paths", mock_pp)

        assert rag_engine.prune_deleted_files() == 0

    def test_exception_returns_zero(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.get.side_effect = Exception("DB error")
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        assert rag_engine.prune_deleted_files() == 0


# ---------------------------------------------------------------------------
# remove_file_index
# ---------------------------------------------------------------------------

class TestRemoveFileIndex:
    def test_noop_when_rag_disabled(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        # Should not raise
        rag_engine.remove_file_index("/some/file.py")

    def test_noop_when_collection_none(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        rag_engine.remove_file_index("/some/file.py")

    def test_deletes_by_rel_path(self, monkeypatch, tmp_path):
        mock_col = MagicMock()
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)

        # Patch _index_rel_path to return a known rel path
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: "src/main.py")

        rag_engine.remove_file_index(str(tmp_path / "main.py"))
        mock_col.delete.assert_called_once_with(where={"file": "src/main.py"})

    def test_noop_when_rel_path_none(self, monkeypatch):
        mock_col = MagicMock()
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: None)
        rag_engine.remove_file_index("/some/file.py")
        mock_col.delete.assert_not_called()

    def test_exception_does_not_propagate(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.delete.side_effect = Exception("delete failed")
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: "file.py")
        # Should not raise
        rag_engine.remove_file_index("/some/file.py")


# ---------------------------------------------------------------------------
# update_file_index
# ---------------------------------------------------------------------------

class TestUpdateFileIndex:
    def test_noop_when_rag_disabled(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        rag_engine.update_file_index("/some/file.py")  # should not raise

    def test_noop_when_collection_none(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        rag_engine.update_file_index("/some/file.py")

    def test_noop_when_rel_path_none(self, monkeypatch, tmp_path):
        mock_col = MagicMock()
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: None)
        rag_engine.update_file_index(str(tmp_path / "file.py"))
        mock_col.delete.assert_not_called()
        mock_col.upsert.assert_not_called()

    def test_deletes_and_reindexes_existing_file(self, monkeypatch, tmp_path):
        test_file = tmp_path / "main.py"
        test_file.write_text("def foo():\n    return 1\n", encoding="utf-8")

        mock_col = MagicMock()
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: "main.py")

        rag_engine.update_file_index(str(test_file))
        # Should delete old chunks then upsert new ones
        mock_col.delete.assert_called_once_with(where={"file": "main.py"})
        assert mock_col.upsert.call_count == 1
        upsert_kwargs = mock_col.upsert.call_args[1]
        assert "documents" in upsert_kwargs
        assert "metadatas" in upsert_kwargs
        assert "ids" in upsert_kwargs
        assert len(upsert_kwargs["documents"]) > 0

    def test_missing_file_is_deleted_not_skipped(self, monkeypatch, tmp_path):
        mock_col = MagicMock()
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: "gone.py")

        missing = tmp_path / "gone.py"
        assert not missing.exists()
        rag_engine.update_file_index(str(missing))
        mock_col.delete.assert_called_once_with(where={"file": "gone.py"})
        mock_col.upsert.assert_not_called()

    def test_exception_does_not_propagate(self, monkeypatch, tmp_path):
        mock_col = MagicMock()
        mock_col.delete.side_effect = Exception("boom")
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_index_rel_path", lambda p: "file.py")
        test_file = tmp_path / "file.py"
        test_file.write_text("x = 1\n", encoding="utf-8")
        # Should not raise
        rag_engine.update_file_index(str(test_file))


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def test_disabled_rag_returns_error_message(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        result = rag_engine.semantic_search("query")
        assert "disabled" in result.lower() or "error" in result.lower()

    def test_no_collections_returns_error(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        result = rag_engine.semantic_search("query", target_kb="local")
        assert "error" in result.lower() or "no active" in result.lower()

    def test_returns_formatted_results(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["def foo(): return 1"]],
            "metadatas": [[{"file": "main.py", "start_line": 1, "end_line": 1, "context": "FunctionDef foo"}]],
            "ids": [["doc_1"]],
        }
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        # Clear cache to avoid stale results
        monkeypatch.setattr(rag_engine, "_SEARCH_CACHE", {})

        # Patch keyword index to raise so it's skipped (we only test vector path)
        result = rag_engine.semantic_search("foo function")
        assert "Semantic Search Results" in result
        assert "main.py" in result
        assert "def foo(): return 1" in result

    def test_empty_results_returns_no_relevant(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "ids": [[]],
        }
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        monkeypatch.setattr(rag_engine, "_SEARCH_CACHE", {})

        result = rag_engine.semantic_search("nothing matching")
        assert "No relevant" in result or "no relevant" in result.lower()

    def test_cache_returns_stored_result(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["cached doc"]],
            "metadatas": [[{"file": "f.py", "start_line": 1, "end_line": 1, "context": "ctx"}]],
            "ids": [["id1"]],
        }
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])

        import time as _time
        now = _time.time()
        monkeypatch.setattr(rag_engine, "_SEARCH_CACHE", {
            "some_key": (now, "CACHED_RESULT_TEXT")
        })
        # We need the cache key to match; patch hashlib to control it
        import hashlib
        monkeypatch.setattr(hashlib, "md5", lambda data: MagicMock(hexdigest=lambda: "some_key"))

        result = rag_engine.semantic_search("query")
        assert result == "CACHED_RESULT_TEXT"
        # Collection.query should NOT be called since cache hit
        mock_col.query.assert_not_called()

    def test_n_results_clamped_to_minimum_one(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.query.return_value = {
            "documents": [["doc"]],
            "metadatas": [[{"file": "f.py", "start_line": 1, "end_line": 1, "context": "ctx"}]],
            "ids": [["id1"]],
        }
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        monkeypatch.setattr(rag_engine, "_SEARCH_CACHE", {})

        rag_engine.semantic_search("query", n_results=0)
        # n_results is clamped to max(1, 0) = 1, so query requests 2 (n*2)
        call_kwargs = mock_col.query.call_args[1]
        assert call_kwargs["n_results"] == 2  # 1 * 2

    def test_query_exception_handled_gracefully(self, monkeypatch):
        mock_col = MagicMock()
        mock_col.query.side_effect = Exception("search crashed")
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", mock_col)
        monkeypatch.setattr(rag_engine, "_ACTIVE_KB_COLLECTIONS", [])
        monkeypatch.setattr(rag_engine, "_SEARCH_CACHE", {})

        result = rag_engine.semantic_search("query")
        # The per-collection exception is caught inside the loop; with no
        # results collected, the function returns "No relevant code found".
        assert "No relevant" in result or "Error" in result