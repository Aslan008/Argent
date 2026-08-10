"""Tests for the deep_research improvements: dedup, smart_truncate, extended signature."""

from src.research.synthesis import dedup_chunks, _word_set
from deep_research import _smart_truncate, run_deep_research
import inspect


class TestDedupChunks:
    def test_exact_duplicates_removed(self):
        chunks = ["hello world foo", "hello world foo", "different text"]
        result = dedup_chunks(chunks)
        assert len(result) == 2
        assert result[0] == "hello world foo"
        assert result[1] == "different text"

    def test_near_duplicates_removed(self):
        # 8 of 10 unique words overlap → Jaccard = 8/10 = 0.8 ≥ 0.8 → dup
        # (sets deduplicate words, so "the" counts once even if repeated)
        chunks = ["alpha beta gamma delta epsilon zeta eta theta iota",
                  "alpha beta gamma delta epsilon zeta eta theta kappa"]
        result = dedup_chunks(chunks, threshold=0.8)
        assert len(result) == 1

    def test_below_threshold_kept(self):
        # 3 of 5 words overlap = 60% → kept
        chunks = ["the quick brown fox jumps", "the quick brown dog sleeps"]
        result = dedup_chunks(chunks, threshold=0.8)
        assert len(result) == 2

    def test_empty_and_single(self):
        assert dedup_chunks([]) == []
        assert dedup_chunks(["only one"]) == ["only one"]

    def test_preserves_first_occurrence(self):
        chunks = ["first chunk here", "first chunk here"]
        result = dedup_chunks(chunks)
        assert result == ["first chunk here"]

    def test_case_insensitive(self):
        chunks = ["Hello World Foo", "hello world foo"]
        result = dedup_chunks(chunks)
        assert len(result) == 1


class TestWordSet:
    def test_strips_punctuation(self):
        ws = _word_set("Hello, World! Foo-Bar.")
        assert "hello" in ws and "world" in ws and "foo" in ws and "bar" in ws

    def test_lowercase(self):
        ws = _word_set("PYTHON programming")
        assert "python" in ws and "programming" in ws

    def test_empty(self):
        assert _word_set("") == set()
        assert _word_set("...!!!") == set()


class TestSmartTruncate:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert _smart_truncate(text, 100) == text

    def test_truncates_at_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        result = _smart_truncate(text, 35)
        assert len(result) <= 35
        # Should break at a sentence boundary, not mid-word
        assert result.endswith(".") or result.endswith(". ")
        assert not result.endswith("Se")

    def test_fallback_to_hard_cut(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        result = _smart_truncate(text, 50)
        assert len(result) <= 50

    def test_empty(self):
        assert _smart_truncate("", 100) == ""


class TestRunDeepResearchSignature:
    def test_accepts_new_parameters(self):
        sig = inspect.signature(run_deep_research)
        params = list(sig.parameters.keys())
        assert "objective" in params
        assert "focus" in params
        assert "context" in params
        assert "max_rounds" in params
        assert "max_sources" in params

    def test_new_params_have_defaults(self):
        sig = inspect.signature(run_deep_research)
        assert sig.parameters["focus"].default is None
        assert sig.parameters["context"].default is None
        assert sig.parameters["max_rounds"].default == 2
        assert sig.parameters["max_sources"].default == 15

    def test_backward_compatible(self):
        """Calling with just objective should still work (no TypeError)."""
        sig = inspect.signature(run_deep_research)
        required = [p for p, v in sig.parameters.items()
                    if v.default is inspect.Parameter.empty]
        assert required == ["objective"]