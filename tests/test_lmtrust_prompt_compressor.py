"""Deep mutation-killing tests for prompt_compressor.

Covers drop_unavailable_tool_lines, _strip_section, compress_system_prompt,
compress_tool_result, and get_adaptive_context_window with edge cases that
kill common mutations (off-by-one, wrong operator, missing branch, etc.).
"""

import pytest

import prompt_compressor
from prompt_compressor import (
    drop_unavailable_tool_lines,
    _strip_section,
    compress_system_prompt,
    compress_tool_result,
    get_adaptive_context_window,
    _TINY_SECTIONS_TO_STRIP,
    _MINI_SYSTEM_SUFFIX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ALL_TOOLS = {
    "read_file", "write_file", "list_directory",
    "create_artifact", "create_plugin", "semantic_search",
}


@pytest.fixture
def tools(monkeypatch):
    """Patch _ALL_TOOL_NAMES to return a deterministic set."""
    monkeypatch.setattr(prompt_compressor, "_ALL_TOOL_NAMES", lambda: ALL_TOOLS)
    return ALL_TOOLS


@pytest.fixture
def cat(monkeypatch):
    """Factory: patch get_model_size_category to return *category*."""
    def _make(category):
        monkeypatch.setattr(
            prompt_compressor, "get_model_size_category",
            lambda *a, **kw: category,
        )
        return category
    return _make


SAMPLE_PROMPT = (
    "# ROLE\nYou are an agent.\n\n"
    "## 1. OPERATIONAL PROTOCOL\nDo things.\n\n"
    "## 2. PLUGIN DEVELOPMENT\nWrite plugins.\n\n"
    "## 4. PLANNING MODE & ARTIFACTS\nPlan first.\n\n"
    "## 5. UI & TERMINOLOGY STANDARDS\nUse panels.\n\n"
    "## 6. THINK & VERIFY\nCheck results.\n"
)


# ---------------------------------------------------------------------------
# drop_unavailable_tool_lines
# ---------------------------------------------------------------------------

class TestDropUnavailableToolLines:

    def test_unavailable_tool_line_dropped(self, tools):
        prompt = "Use read_file to open.\nUse create_plugin to build."
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "read_file" in out
        assert "create_plugin" not in out

    def test_available_tool_line_kept(self, tools):
        prompt = "Use read_file to open.\nUse write_file to save."
        out = drop_unavailable_tool_lines(prompt, {"read_file", "write_file"})
        assert "read_file" in out
        assert "write_file" in out

    def test_heading_with_no_content_dropped(self, tools):
        # Heading followed only by a line referencing an unavailable tool →
        # the heading has no surviving content and must be dropped too.
        prompt = "## FILE TOOLS\nUse create_plugin here.\n## OTHER\nDo stuff.\n"
        out = drop_unavailable_tool_lines(prompt, set())
        # available is empty → passthrough, so let's use a non-empty available
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "FILE TOOLS" not in out
        assert "OTHER" in out
        assert "Do stuff" in out

    def test_heading_with_content_kept(self, tools):
        prompt = "## FILE TOOLS\nUse read_file here.\n## OTHER\nDo stuff.\n"
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "FILE TOOLS" in out
        assert "read_file" in out

    def test_empty_available_set_passthrough(self, tools):
        prompt = "Use read_file.\nUse create_plugin.\n## HEADING\nStuff.\n"
        out = drop_unavailable_tool_lines(prompt, set())
        assert out == prompt

    def test_multiple_sections_mixed(self, tools):
        prompt = (
            "## SECTION A\nUse read_file here.\n"
            "## SECTION B\nUse create_plugin here.\n"
            "## SECTION C\nUse write_file here.\n"
        )
        out = drop_unavailable_tool_lines(prompt, {"read_file", "write_file"})
        assert "SECTION A" in out
        assert "SECTION B" not in out  # heading dropped, content dropped
        assert "SECTION C" in out

    def test_tool_name_as_substring_kept(self, tools):
        """read_file_content is not in _ALL_TOOL_NAMES, so a line mentioning
        it is not treated as a tool reference and is kept even if read_file
        is unavailable."""
        prompt = "Use read_file_content to read deeply."
        out = drop_unavailable_tool_lines(prompt, {"write_file"})
        # read_file_content is not a known tool name, so the line survives.
        assert "read_file_content" in out

    def test_tool_name_as_substring_dropped(self, monkeypatch):
        """When read_file_content IS a known tool name, a line referencing it
        is dropped if read_file_content is not in available — even though
        read_file (a substring) is available."""
        monkeypatch.setattr(
            prompt_compressor, "_ALL_TOOL_NAMES",
            lambda: {"read_file", "read_file_content"},
        )
        prompt = "Use read_file_content to read deeply."
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "read_file_content" not in out

    def test_blank_lines_in_section_preserved_when_content_survives(self, tools):
        prompt = "## SECTION\n\nUse read_file here.\n\nDone.\n"
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "SECTION" in out
        assert "read_file" in out
        assert "Done" in out

    def test_blank_lines_only_under_heading_dropped(self, tools):
        # Heading followed by only blank lines → no content → dropped.
        prompt = "## EMPTY\n\n\n## KEEP\nUse read_file.\n"
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "EMPTY" not in out
        assert "KEEP" in out

    def test_line_with_multiple_tools_all_available_kept(self, tools):
        prompt = "Use read_file and write_file together."
        out = drop_unavailable_tool_lines(prompt, {"read_file", "write_file"})
        assert "read_file" in out
        assert "write_file" in out

    def test_line_with_multiple_tools_one_unavailable_dropped(self, tools):
        prompt = "Use read_file and create_plugin together."
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "read_file" not in out
        assert "create_plugin" not in out

    def test_non_tool_word_not_treated_as_tool(self, tools):
        # 'stuff' matches [a-z_]{4,} but is not in _ALL_TOOL_NAMES, so it
        # doesn't trigger filtering.
        prompt = "Do stuff with read_file."
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "stuff" in out

    def test_consecutive_headings_first_dropped(self, tools):
        # Two headings back-to-back: the first has no content → dropped.
        prompt = "## FIRST\n## SECOND\nUse read_file.\n"
        out = drop_unavailable_tool_lines(prompt, {"read_file"})
        assert "FIRST" not in out
        assert "SECOND" in out

    def test_no_heading_lines_passthrough_logic(self, tools):
        # No headings at all — lines filtered directly into kept.
        prompt = "Use read_file.\nUse create_plugin.\nUse write_file."
        out = drop_unavailable_tool_lines(prompt, {"read_file", "write_file"})
        assert "read_file" in out
        assert "create_plugin" not in out
        assert "write_file" in out


# ---------------------------------------------------------------------------
# _strip_section
# ---------------------------------------------------------------------------

class TestStripSection:

    def test_section_found_and_removed(self):
        out = _strip_section(SAMPLE_PROMPT, "## 2. PLUGIN DEVELOPMENT")
        assert "PLUGIN DEVELOPMENT" not in out
        assert "OPERATIONAL PROTOCOL" in out
        assert "THINK & VERIFY" in out

    def test_section_not_found_passthrough(self):
        assert _strip_section(SAMPLE_PROMPT, "## 99. NOPE") == SAMPLE_PROMPT

    def test_section_at_end_of_prompt(self):
        prompt = "## 1. FIRST\nContent A.\n## 2. LAST\nContent B.\n"
        out = _strip_section(prompt, "## 2. LAST")
        assert "LAST" not in out
        assert "Content B" not in out
        assert "FIRST" in out

    def test_section_in_middle(self):
        prompt = "## 1. FIRST\nA.\n## 2. MID\nB.\n## 3. LAST\nC.\n"
        out = _strip_section(prompt, "## 2. MID")
        assert "MID" not in out
        assert "B." not in out
        assert "FIRST" in out
        assert "LAST" in out
        assert "C." in out

    def test_multiple_sections_stripped_sequentially(self):
        prompt = (
            "## 1. FIRST\nA.\n"
            "## 2. SECOND\nB.\n"
            "## 3. THIRD\nC.\n"
            "## 4. FOURTH\nD.\n"
        )
        out = _strip_section(prompt, "## 2. SECOND")
        out = _strip_section(out, "## 3. THIRD")
        assert "SECOND" not in out
        assert "THIRD" not in out
        assert "FIRST" in out
        assert "FOURTH" in out

    def test_empty_prompt_passthrough(self):
        assert _strip_section("", "## ANY") == ""

    def test_strip_only_heading_line(self):
        # A section that is just a heading with nothing after it (end of prompt).
        prompt = "## 1. KEEP\nContent.\n## 2. GONE\n"
        out = _strip_section(prompt, "## 2. GONE")
        assert "GONE" not in out
        assert "KEEP" in out

    def test_strip_preserves_text_before_section(self):
        prompt = "Preamble line.\n## SECTION\nContent.\n## NEXT\nMore.\n"
        out = _strip_section(prompt, "## SECTION")
        assert "Preamble line." in out
        assert "NEXT" in out
        assert "SECTION" not in out


# ---------------------------------------------------------------------------
# compress_system_prompt
# ---------------------------------------------------------------------------

class TestCompressSystemPrompt:

    def test_tiny_strips_sections_and_appends_suffix(self, cat):
        cat("tiny")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert "RULES (SHORT)" in out
        for section in _TINY_SECTIONS_TO_STRIP:
            assert section not in out

    def test_tiny_strips_each_named_section(self, cat):
        cat("tiny")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert "PLUGIN DEVELOPMENT" not in out
        assert "PLANNING MODE" not in out
        assert "UI & TERMINOLOGY" not in out

    def test_tiny_keeps_non_stripped_sections(self, cat):
        cat("tiny")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert "OPERATIONAL PROTOCOL" in out
        assert "THINK & VERIFY" in out

    def test_tiny_suffix_content_present(self, cat):
        cat("tiny")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert "Use tools via JSON" in out
        assert "ask_user_questions" in out

    def test_small_appends_reminder_only(self, cat):
        cat("small")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        # Sections NOT stripped for small.
        assert "PLUGIN DEVELOPMENT" in out
        assert "PLANNING MODE" in out
        # Reminder appended.
        assert "ask_user_questions" in out
        assert "IMPORTANT" in out

    def test_small_does_not_append_short_rules(self, cat):
        cat("small")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert "RULES (SHORT)" not in out

    def test_medium_passthrough(self, cat):
        cat("medium")
        assert compress_system_prompt(SAMPLE_PROMPT, "m") == SAMPLE_PROMPT

    def test_large_passthrough(self, cat):
        cat("large")
        assert compress_system_prompt(SAMPLE_PROMPT, "m") == SAMPLE_PROMPT

    def test_cloud_passthrough(self, cat):
        cat("cloud")
        assert compress_system_prompt(SAMPLE_PROMPT, "m") == SAMPLE_PROMPT

    def test_category_override_takes_precedence(self, monkeypatch):
        # Even if get_model_size_category says 'medium', the override 'tiny'
        # must win so the prompt is actually compressed.
        monkeypatch.setattr(
            prompt_compressor, "get_model_size_category",
            lambda *a, **kw: "medium",
        )
        out = compress_system_prompt(SAMPLE_PROMPT, "m", category="tiny")
        assert "RULES (SHORT)" in out
        assert "PLUGIN DEVELOPMENT" not in out

    def test_category_override_small(self, monkeypatch):
        monkeypatch.setattr(
            prompt_compressor, "get_model_size_category",
            lambda *a, **kw: "tiny",
        )
        out = compress_system_prompt(SAMPLE_PROMPT, "m", category="small")
        # small does NOT strip sections.
        assert "PLUGIN DEVELOPMENT" in out
        # small does NOT add RULES (SHORT).
        assert "RULES (SHORT)" not in out
        # small DOES add the reminder.
        assert "IMPORTANT" in out

    def test_tiny_sections_actually_removed_not_just_heading(self, cat):
        cat("tiny")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        # The content under stripped sections must also be gone.
        assert "Write plugins" not in out
        assert "Plan first" not in out
        assert "Use panels" not in out

    def test_tiny_result_ends_with_suffix(self, cat):
        cat("tiny")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert out.rstrip().endswith(_MINI_SYSTEM_SUFFIX.rstrip())

    def test_small_result_ends_with_reminder(self, cat):
        cat("small")
        out = compress_system_prompt(SAMPLE_PROMPT, "m")
        assert out.rstrip().endswith(
            "ask_user_questions to clarify before making assumptions."
        )


# ---------------------------------------------------------------------------
# compress_tool_result
# ---------------------------------------------------------------------------

class TestCompressToolResult:

    def test_under_max_lines_passthrough(self, cat):
        cat("tiny")
        text = "\n".join("line" for _ in range(10))
        assert compress_tool_result(text, "m") == text

    def test_over_max_lines_head_tail_separator(self, cat):
        cat("tiny")  # max_lines=40
        text = "\n".join(str(i) for i in range(100))
        out = compress_tool_result(text, "m")
        lines = out.splitlines()
        # head should be first 20 lines (0..19)
        assert lines[0] == "0"
        # tail should be last 20 lines (80..99)
        assert lines[-1] == "99"
        assert "truncated" in out

    def test_exact_max_lines_boundary_passthrough(self, cat):
        cat("tiny")  # max_lines=40
        text = "\n".join(str(i) for i in range(40))
        # Exactly 40 lines → NOT > 40, so no truncation.
        assert compress_tool_result(text, "m") == text

    def test_one_over_max_lines_truncated(self, cat):
        cat("tiny")  # max_lines=40
        text = "\n".join(str(i) for i in range(41))
        out = compress_tool_result(text, "m")
        assert "truncated" in out
        assert "1 lines truncated" in out  # 41 - 40 = 1

    def test_over_max_chars_char_truncation(self, cat):
        cat("tiny")  # max_chars = 40 * 120 = 4800
        text = "x" * 10000
        out = compress_tool_result(text, "m")
        assert "characters truncated" in out
        assert len(out) < 6000
        assert out.startswith("x")
        assert out.endswith("x")

    def test_both_budgets_exceeded(self, cat):
        cat("tiny")  # max_lines=40, max_chars=4800
        # 100 lines, each 200 chars → 20000 chars total. Both budgets exceeded.
        text = "\n".join("y" * 200 for _ in range(100))
        out = compress_tool_result(text, "m")
        # Line budget fires first, then char budget on the result.
        assert "truncated" in out

    def test_non_string_input_coerced(self, cat):
        cat("tiny")
        assert compress_tool_result(12345, "m") == "12345"

    def test_non_string_list_input_coerced(self, cat):
        cat("tiny")
        out = compress_tool_result(["a", "b"], "m")
        assert "a" in out
        assert "b" in out

    def test_separator_contains_line_count(self, cat):
        cat("tiny")  # max_lines=40
        text = "\n".join(str(i) for i in range(200))
        out = compress_tool_result(text, "m")
        assert "160 lines truncated" in out  # 200 - 40 = 160

    def test_separator_contains_char_count(self, cat):
        cat("tiny")  # max_chars=4800
        text = "x" * 10000
        out = compress_tool_result(text, "m")
        # 10000 - 4800 = 5200 chars removed
        assert "5200 characters truncated" in out

    def test_tiny_threshold_40_lines(self, cat):
        cat("tiny")
        text_40 = "\n".join("x" for _ in range(40))
        text_41 = "\n".join("x" for _ in range(41))
        assert compress_tool_result(text_40, "m") == text_40
        assert "truncated" in compress_tool_result(text_41, "m")

    def test_small_threshold_100_lines(self, cat):
        cat("small")
        text_100 = "\n".join("x" for _ in range(100))
        text_101 = "\n".join("x" for _ in range(101))
        assert compress_tool_result(text_100, "m") == text_100
        assert "truncated" in compress_tool_result(text_101, "m")

    def test_medium_threshold_500_lines(self, cat):
        cat("medium")
        text_500 = "\n".join("x" for _ in range(500))
        text_501 = "\n".join("x" for _ in range(501))
        assert compress_tool_result(text_500, "m") == text_500
        assert "truncated" in compress_tool_result(text_501, "m")

    def test_large_threshold_2000_lines(self, cat):
        cat("large")
        text_2000 = "\n".join("x" for _ in range(2000))
        text_2001 = "\n".join("x" for _ in range(2001))
        assert compress_tool_result(text_2000, "m") == text_2000
        assert "truncated" in compress_tool_result(text_2001, "m")

    def test_cloud_threshold_2000_lines(self, cat):
        cat("cloud")
        text_2001 = "\n".join("x" for _ in range(2001))
        assert "truncated" in compress_tool_result(text_2001, "m")

    def test_custom_max_lines_override(self, cat):
        cat("tiny")  # would be 40, but we override to 5
        text = "\n".join(str(i) for i in range(20))
        out = compress_tool_result(text, "m", max_lines=5)
        assert "truncated" in out
        assert "15 lines truncated" in out  # 20 - 5 = 15

    def test_custom_max_chars_override(self, cat):
        cat("tiny")  # would be 4800, but we override to 100
        text = "x" * 500
        out = compress_tool_result(text, "m", max_chars=100)
        assert "characters truncated" in out
        assert "400 characters truncated" in out  # 500 - 100 = 400

    def test_custom_both_overrides(self, cat):
        cat("tiny")
        text = "\n".join("y" * 50 for _ in range(20))
        out = compress_tool_result(text, "m", max_lines=10, max_chars=200)
        assert "truncated" in out

    def test_head_tail_preserve_order(self, cat):
        cat("tiny")  # max_lines=40, half=20
        text = "\n".join(f"LINE_{i}" for i in range(100))
        out = compress_tool_result(text, "m")
        lines = out.splitlines()
        # First surviving content line should be LINE_0
        assert lines[0] == "LINE_0"
        # Last surviving content line should be LINE_99
        assert lines[-1] == "LINE_99"

    def test_empty_string_passthrough(self, cat):
        cat("tiny")
        assert compress_tool_result("", "m") == ""

    def test_single_line_within_budget(self, cat):
        cat("tiny")
        assert compress_tool_result("hello world", "m") == "hello world"

    def test_none_input_coerced(self, cat):
        cat("tiny")
        assert compress_tool_result(None, "m") == "None"


# ---------------------------------------------------------------------------
# get_adaptive_context_window
# ---------------------------------------------------------------------------

class TestGetAdaptiveContextWindow:

    def test_tiny_returns_2048(self, cat):
        cat("tiny")
        assert get_adaptive_context_window("m", base_window=128000) == 2048

    def test_small_returns_4096(self, cat):
        cat("small")
        assert get_adaptive_context_window("m", base_window=128000) == 4096

    def test_medium_returns_8192(self, cat):
        cat("medium")
        assert get_adaptive_context_window("m", base_window=128000) == 8192

    def test_large_returns_base_window(self, cat):
        cat("large")
        assert get_adaptive_context_window("m", base_window=128000) == 128000

    def test_cloud_returns_base_window(self, cat):
        cat("cloud")
        assert get_adaptive_context_window("m", base_window=128000) == 128000

    def test_base_smaller_than_recommendation(self, cat):
        """When base_window < recommendation, min() clamps to base_window."""
        cat("tiny")  # recommendation = 2048
        assert get_adaptive_context_window("m", base_window=1000) == 1000

    def test_base_smaller_than_small_recommendation(self, cat):
        cat("small")  # recommendation = 4096
        assert get_adaptive_context_window("m", base_window=2048) == 2048

    def test_base_smaller_than_medium_recommendation(self, cat):
        cat("medium")  # recommendation = 8192
        assert get_adaptive_context_window("m", base_window=4096) == 4096

    def test_unknown_category_falls_back_to_base(self, monkeypatch):
        monkeypatch.setattr(
            prompt_compressor, "get_model_size_category",
            lambda *a, **kw: "unknown",
        )
        assert get_adaptive_context_window("m", base_window=50000) == 50000

    def test_tiny_base_exactly_2048(self, cat):
        cat("tiny")
        assert get_adaptive_context_window("m", base_window=2048) == 2048

    def test_large_base_exactly_recommendation(self, cat):
        # large recommendation == base_window, so result is base_window.
        cat("large")
        assert get_adaptive_context_window("m", base_window=8192) == 8192
class TestStripSectionExactOutput:
    """Kill ADD_TO_SUB @ pos 3687: ``nxt + 1`` → ``nxt - 1`` in _strip_section.

    The ``+1`` skips the ``\\n`` before the next ``## `` heading.
    With ``-1``, two extra characters leak into the output.
    """

    def test_strip_middle_exact_output(self):
        """Exact output for stripping a middle section.

        prompt = "## 1. FIRST\nA.\n## 2. MID\nB.\n## 3. LAST\nC.\n"
        Stripping "## 2. MID" should give:
            "## 1. FIRST\nA.\n## 3. LAST\nC.\n"
        The mutant (nxt-1) would include an extra ".\n" before "## 3.".
        """
        prompt = "## 1. FIRST\nA.\n## 2. MID\nB.\n## 3. LAST\nC.\n"
        out = _strip_section(prompt, "## 2. MID")
        assert out == "## 1. FIRST\nA.\n## 3. LAST\nC.\n"

    def test_strip_first_section_exact(self):
        """Exact output for stripping the first section."""
        prompt = "## 1. GONE\nBad.\n## 2. KEEP\nGood.\n"
        out = _strip_section(prompt, "## 1. GONE")
        assert out == "## 2. KEEP\nGood.\n"

    def test_strip_no_extra_char_before_next_heading(self):
        """The character before \\n## must NOT appear in output."""
        prompt = "## 1. KEEP\nX.\n## 2. GONE\nY.\n## 3. LAST\nZ.\n"
        out = _strip_section(prompt, "## 2. GONE")
        # The "Y." line ends with ".\n## 3." — the "." before \n must not leak
        assert out == "## 1. KEEP\nX.\n## 3. LAST\nZ.\n"
        assert "Y." not in out  # stripped content must not leak