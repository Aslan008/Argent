from unittest.mock import patch

from prompt_compressor import compress_system_prompt, compress_tool_result, _strip_section


SAMPLE = (
    "# ROLE\nYou are an agent.\n\n"
    "## 1. OPERATIONAL PROTOCOL\nDo things.\n\n"
    "## 2. PLUGIN DEVELOPMENT\nWrite plugins.\n\n"
    "## 4. PLANNING MODE & ARTIFACTS\nPlan first.\n\n"
    "## 5. UI & TERMINOLOGY STANDARDS\nUse panels.\n\n"
    "## 6. THINK & VERIFY\nCheck results.\n"
)


def _compress(prompt, category):
    with patch("prompt_compressor.get_model_size_category", return_value=category):
        return compress_system_prompt(prompt, "any")


class TestStripSection:
    def test_removes_named_section_only(self):
        out = _strip_section(SAMPLE, "## 2. PLUGIN DEVELOPMENT")
        assert "PLUGIN DEVELOPMENT" not in out
        assert "OPERATIONAL PROTOCOL" in out
        assert "THINK & VERIFY" in out

    def test_missing_section_is_noop(self):
        assert _strip_section(SAMPLE, "## 99. NOPE") == SAMPLE


class TestCompressSystemPrompt:
    def test_cloud_unchanged(self):
        for cat in ("medium", "large", "cloud"):
            assert _compress(SAMPLE, cat) == SAMPLE

    def test_tiny_strips_heavy_sections(self):
        out = _compress(SAMPLE, "tiny")
        assert "PLUGIN DEVELOPMENT" not in out
        assert "PLANNING MODE" not in out
        assert "UI & TERMINOLOGY" not in out
        # Core sections survive.
        assert "OPERATIONAL PROTOCOL" in out
        assert "THINK & VERIFY" in out
        # Short rules suffix appended.
        assert "RULES (SHORT)" in out

    def test_tiny_is_shorter(self):
        assert len(_compress(SAMPLE, "tiny")) < len(SAMPLE) + 400

    def test_small_keeps_sections_adds_reminder(self):
        out = _compress(SAMPLE, "small")
        assert "PLUGIN DEVELOPMENT" in out  # not stripped for small
        assert "ask_user_questions" in out


class TestCompressToolResult:
    def test_short_output_untouched(self):
        with patch("prompt_compressor.get_model_size_category", return_value="tiny"):
            assert compress_tool_result("a\nb\nc", "m") == "a\nb\nc"

    def test_long_output_truncated_for_tiny(self):
        text = "\n".join(str(i) for i in range(200))
        with patch("prompt_compressor.get_model_size_category", return_value="tiny"):
            out = compress_tool_result(text, "m")
        assert "truncated" in out
        assert len(out.splitlines()) < 200
