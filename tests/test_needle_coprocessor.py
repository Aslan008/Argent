"""
Tests for the Needle 3 Coprocessor module.

Validates:
1. Initialization and graceful fallback when cactus-needle is not installed.
2. BilingualIntentNormalizer: Russian → English action verb mapping.
3. FastDispatchResult: argument extraction and confidence thresholds.
4. Complex-query rejection (reasoning/code/architectural questions → None).
5. Config get/set for needle_enabled / needle_fast_path.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock


# ── 1. Normalizer tests (no Needle dependency needed) ───────────────

class TestBilingualIntentNormalizer:
    """Tests BilingualIntentNormalizer in isolation."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from src.agent.needle_coprocessor import BilingualIntentNormalizer
        self.normalizer = BilingualIntentNormalizer

    def test_read_file_russian(self):
        assert "read file" in self.normalizer.normalize("прочитай main.py")

    def test_open_file_russian(self):
        assert "read file" in self.normalizer.normalize("открой config.py")

    def test_show_file_russian(self):
        assert "read file" in self.normalizer.normalize("покажи содержимое файла test.py")

    def test_git_status_russian(self):
        assert "git status" in self.normalizer.normalize("покажи статус гит")

    def test_git_diff_russian(self):
        result = self.normalizer.normalize("покажи дифф")
        assert "git diff" in result

    def test_search_russian(self):
        assert "search for" in self.normalizer.normalize("найди где находится handle_think")

    def test_run_command_russian(self):
        assert "run command" in self.normalizer.normalize("запусти команду pytest")

    def test_english_passthrough(self):
        """English queries should pass through unchanged."""
        q = "read file config.py"
        assert self.normalizer.normalize(q) == q

    def test_empty_input(self):
        assert self.normalizer.normalize("") == ""
        assert self.normalizer.normalize(None) == ""

    def test_mixed_script_cyrillic_first(self):
        """When text contains Cyrillic, normalization applies."""
        result = self.normalizer.normalize("прочитай file.txt extra")
        assert "read file" in result


# ── 2. FastDispatchResult dataclass tests ────────────────────────────

class TestFastDispatchResult:
    def test_creation(self):
        from src.agent.needle_coprocessor import FastDispatchResult
        r = FastDispatchResult(
            tool_name="read_file",
            arguments={"file_path": "main.py"},
            confidence=0.92,
            elapsed_ms=12.5,
            raw_query="прочитай main.py",
        )
        assert r.tool_name == "read_file"
        assert r.confidence == 0.92
        assert r.elapsed_ms == 12.5


# ── 3. Coprocessor initialization fallback ───────────────────────────

class TestNeedleCoprocessorFallback:
    """When cactus-needle is NOT installed, everything must degrade gracefully."""

    def test_is_available_without_package(self):
        """With needle not installed, is_available returns False."""
        from src.agent.needle_coprocessor import NeedleCoprocessor
        coproc = NeedleCoprocessor()
        # Regardless of actual installation, we can test the fallback path
        if not coproc.is_available:
            assert coproc.initialize() is False
            assert coproc._init_error is not None

    def test_try_fast_dispatch_returns_none_when_unavailable(self):
        from src.agent.needle_coprocessor import NeedleCoprocessor
        coproc = NeedleCoprocessor()
        if not coproc.is_available:
            result = coproc.try_fast_dispatch("прочитай main.py")
            assert result is None

    def test_extract_structured_returns_none_when_unavailable(self):
        from src.agent.needle_coprocessor import NeedleCoprocessor
        coproc = NeedleCoprocessor()
        if not coproc.is_available:
            result = coproc.extract_structured("some text", {"field": str})
            assert result is None


# ── 4. Complex query rejection ───────────────────────────────────────

class TestComplexQueryRejection:
    """Complex prompts (reasoning, code writing, architecture) must return None."""

    @pytest.fixture(autouse=True)
    def _coproc(self):
        from src.agent.needle_coprocessor import NeedleCoprocessor
        self.coproc = NeedleCoprocessor()

    @pytest.mark.parametrize("query", [
        "напиши алгоритм сортировки",
        "объясни как работает async/await",
        "зарефактори этот модуль",
        "создай класс для работы с базой данных",
        "why does this test fail?",
        "write a unit test for this function",
        "explain the architecture",
        "debug the provider connection",
        "проверь текущий проект. Прочитай необходимые файлы",
    ])
    def test_complex_queries_return_none(self, query):
        result = self.coproc.try_fast_dispatch(query)
        assert result is None, f"Expected None for complex query: {query!r}"

    def test_slash_commands_return_none(self):
        assert self.coproc.try_fast_dispatch("/help") is None
        assert self.coproc.try_fast_dispatch("/model") is None

    def test_empty_returns_none(self):
        assert self.coproc.try_fast_dispatch("") is None
        assert self.coproc.try_fast_dispatch("ab") is None  # too short

    def test_bang_prefix_returns_none(self):
        assert self.coproc.try_fast_dispatch("!ls") is None

    @pytest.mark.parametrize("query", [
        "привет",
        "Привет!",
        "здравствуйте",
        "добрый день",
        "hello",
        "hi there",
        "how are you",
        "спасибо за помощь",
    ])
    def test_conversational_greetings_return_none(self, query):
        result = self.coproc.try_fast_dispatch(query)
        assert result is None, f"Expected None for greeting: {query!r}"

    def test_unsafe_command_rejected_even_if_needle_predicts_it(self, monkeypatch):
        """Needle hallucinating random strings as run_command must be blocked."""
        class MockNeedle:
            def reset(self): pass
            def run(self, q, max_steps=1):
                return {
                    "success": True,
                    "confidence": 0.95,
                    "function_calls": [{"name": "run_command", "arguments": {"command": "привет"}}],
                }
        monkeypatch.setattr(self.coproc, "_needle", MockNeedle())
        monkeypatch.setattr(self.coproc, "_initialized", True)
        assert self.coproc.try_fast_dispatch("check something") is None


# ── 5. Config functions ─────────────────────────────────────────────

class TestNeedleConfig:
    """Tests for config.py Needle 3 get/set functions."""

    def test_needle_enabled_default(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "_get", lambda key, default=None: default)
        assert config.get_needle_enabled() is False

    def test_needle_fast_path_default(self, monkeypatch):
        import config
        monkeypatch.setattr(config, "_get", lambda key, default=None: default)
        assert config.get_needle_fast_path() is False

    def test_set_needle_enabled(self, monkeypatch):
        import config
        stored = {}
        monkeypatch.setattr(config, "_set", lambda k, v: stored.update({k: v}))
        config.set_needle_enabled(True)
        assert stored["needle_enabled"] is True

    def test_set_needle_fast_path(self, monkeypatch):
        import config
        stored = {}
        monkeypatch.setattr(config, "_set", lambda k, v: stored.update({k: v}))
        config.set_needle_fast_path(True)
        assert stored["needle_fast_path"] is True


# ── 6. delegate_local micro tier (-1) routing ────────────────────────

class TestDelegateLocalMicroTier:
    """Tests that tier=-1 routes to Needle micro path."""

    def test_micro_tier_fallback_when_needle_unavailable(self, monkeypatch):
        """When cactus-needle is not installed, micro tier falls back to tier 1."""
        from local_inference import delegate_local

        # Mock to prevent actual Ollama/cloud calls
        monkeypatch.setattr(
            "local_inference._is_ollama_online", lambda: False
        )
        monkeypatch.setattr(
            "local_inference.config.is_local_agent_cloud_fallback", lambda: False
        )

        result = delegate_local(task="Extract the file name", tier=-1)
        # Should either succeed via Needle or fall back gracefully
        assert isinstance(result, str)
        assert len(result) > 0
