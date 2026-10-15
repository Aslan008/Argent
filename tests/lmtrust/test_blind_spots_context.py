"""LMTrust Deep Blind-Spot Test Suite for Context Subsystem in Argent.

Tests systematically uncover blind spots and contract violations in:
- src/agent/trimmer.py (distillation, progressive aging, token estimation, rehydration)
- src/agent/context_limit.py (effective budget calculation, floor explosion)
- prompt_compressor.py (infinite loops, heading handling)
"""

import pytest
from unittest.mock import patch
import src.agent.trimmer as trimmer
from src.agent.context_limit import effective_history_budget, is_context_overflow, parse_context_limit
from prompt_compressor import compress_tool_result, _strip_section


class TestTrimmerBlindSpots:
    def test_distillates_accumulate_infinitely_because_of_pinned(self, monkeypatch):
        """Blind Spot L5 State / D3 Data Integrity:
        Old CONTEXT DISTILLATE messages are pinned in Step 1, so they are NEVER
        dropped and NEVER passed to msgs_to_compress, causing infinite accumulation
        and preventing incremental distillation update.
        """
        monkeypatch.setattr(trimmer, "get_provider", lambda: "test")
        monkeypatch.setattr(trimmer, "_summarize_with_llm", lambda *a, **k: "## Done\n- New work done")

        # Session with an existing distillate from previous compression
        messages = [
            {"role": "system", "content": "You are Argent."},
            {"role": "system", "content": "=== CONTEXT DISTILLATE (compression #1) ===\nOld memory\n=== END DISTILLATE ==="},
            {"role": "user", "content": "Task 1: do something"},
            {"role": "assistant", "content": "Did step 1"},
            {"role": "user", "content": "Task 2: continue"},
            {"role": "assistant", "content": "Did step 2"},
            {"role": "user", "content": "Task 3: finish"},
            {"role": "assistant", "content": "Did step 3"},
        ]

        # Trigger soft trim with tight budget
        result = trimmer.soft_trim_with_summarization(
            messages, model_name="test-model",
            max_history_messages=3, max_context_tokens=100
        )

        # Count how many DISTILLATE messages exist in result
        distillate_count = sum(1 for m in result if "CONTEXT DISTILLATE" in m.get("content", ""))
        
        # INVARIANT: There should only EVER be at most ONE active distillate in history!
        # If old distillates are pinned, distillate_count will be 2 (both #1 and #2 exist simultaneously).
        assert distillate_count == 1, f"Expected 1 distillate, but found {distillate_count} accumulated distillates!"

    def test_cyrillic_heuristic_does_not_skew_latin_code(self):
        """Blind Spot L2 Boundary / D1 Math:
        A 10,000 char Python code with a 5-char Russian comment should NOT
        have its entire 10,000 chars divided by 1.8 instead of 4.
        """
        latin_code = "def calculate_statistics(matrix_data):\n    return sum(matrix_data) / len(matrix_data)\n" * 100
        tokens_pure_latin = trimmer._heuristic_tokens(latin_code)

        # Add one tiny Russian word in a comment
        code_with_tiny_comment = latin_code + "\n# Тест"
        tokens_with_comment = trimmer._heuristic_tokens(code_with_tiny_comment)

        # Adding 7 characters should not double the estimated token count!
        ratio = tokens_with_comment / tokens_pure_latin
        assert ratio < 1.1, f"Adding 7 Cyrillic chars to 10k code inflated token count by {(ratio - 1)*100:.1f}%!"

    def test_rehydrate_files_preserves_most_recently_edited_order(self, tmp_path):
        """Blind Spot L8 Temporal / D3 Data Integrity:
        If file A was edited, then file B, then file A again, file A should be
        considered MORE RECENT than file B in rehydration.
        """
        f_a = tmp_path / "a.py"
        f_b = tmp_path / "b.py"
        f_c = tmp_path / "c.py"
        f_d = tmp_path / "d.py"
        f_a.write_text("content a", encoding="utf-8")
        f_b.write_text("content b", encoding="utf-8")
        f_c.write_text("content c", encoding="utf-8")
        f_d.write_text("content d", encoding="utf-8")

        import json
        messages = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"file_path": str(f_a)})}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"file_path": str(f_b)})}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"file_path": str(f_c)})}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"file_path": str(f_a)})}}]}, # edited again!
        ]

        # Max 2 files: should keep f_c and f_a (the last two modified)
        rehydrated = trimmer._rehydrate_files(messages, max_files=2)
        assert "a.py" in rehydrated, "File a.py was edited last and MUST be rehydrated!"
        assert "c.py" in rehydrated, "File c.py was edited before a.py and should be rehydrated!"
        assert "b.py" not in rehydrated, "File b.py is older than c and a, should not be in top 2!"

    def test_no_orphaned_tool_messages_after_trimming(self, monkeypatch):
        """Blind Spot L1 Contract / D7 API Contract:
        All tool messages in trimmed history MUST be preceded by an assistant message
        containing the matching tool_call_id.
        """
        monkeypatch.setattr(trimmer, "get_provider", lambda: "test")
        monkeypatch.setattr(trimmer, "_summarize_with_llm", lambda *a, **k: "Summary")

        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Run command"},
            {"role": "assistant", "content": "running", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "output"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "Now read file"},
            {"role": "assistant", "content": "reading", "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": "file content"},
        ]

        trimmed = trimmer.soft_trim_with_summarization(
            messages, model_name="test-model",
            max_history_messages=2, max_context_tokens=50
        )

        known_tool_call_ids = set()
        for m in trimmed:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls", []):
                    known_tool_call_ids.add(tc.get("id"))
            elif m.get("role") == "tool":
                tid = m.get("tool_call_id")
                assert tid in known_tool_call_ids, f"Orphaned tool message with tool_call_id={tid!r} found in trimmed history!"

    def test_token_cache_thread_safety(self):
        """Blind Spot L8 Concurrency / D6:
        Concurrent token estimations from multiple threads must not corrupt _TOKEN_CACHE.
        """
        import concurrent.futures
        trimmer._TOKEN_CACHE.clear()
        
        def work(i):
            text = f"Some sample text number {i % 50}"
            return trimmer.estimate_tokens(text, "model", "test")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(work, range(500)))
        assert len(results) == 500


class TestContextLimitBlindSpots:
    def test_effective_history_budget_cannot_exceed_window_on_small_models(self):
        """Blind Spot L2 Boundary / D1 Math:
        On small models (window=2048), default floor=2048 causes effective_history_budget
        to return 2048 even when system prompt already consumes 1200 tokens,
        causing guaranteed HTTP 400 context overflow on the provider.
        """
        window = 2048
        sys_tokens = 1200
        # If window is 2048 and system is 1200, remaining budget CANNOT be 2048!
        budget = effective_history_budget(window, sys_tokens)
        assert budget + sys_tokens <= window, (
            f"Budget {budget} + system {sys_tokens} = {budget + sys_tokens}, "
            f"which overflows window {window}!"
        )


class TestPromptCompressorBlindSpots:
    def test_strip_section_does_not_infinite_loop(self):
        """Blind Spot L4 Adversarial:
        Ensure _strip_section terminates even on repeated or malformed headings.
        """
        text = "## 2. PLUGIN DEVELOPMENT\nLine 1\n## 2. PLUGIN DEVELOPMENT\nLine 2\n# Next Section\nHello"
        out = _strip_section(text, "## 2. PLUGIN DEVELOPMENT")
        assert "PLUGIN DEVELOPMENT" not in out
        assert "Next Section" in out
