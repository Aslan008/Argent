"""Tests for Fast Interactive Prompt Detector in command_ops.py."""

import time
import pytest
from unittest.mock import patch

from tools.command_ops import run_command, FAST_INTERACTIVE_SILENCE_LIMIT, INTERACTIVE_PROMPT_PATTERNS


class TestInteractivePromptDetector:
    """Verifies that commands hanging on interactive prompts are terminated rapidly (3s instead of 180s)."""

    def test_interactive_prompt_patterns_match(self):
        prompts = [
            "Do you want to continue? [Y/n] ",
            "Overwrite file? (yes/no)",
            "Enter passphrase:",
            "Select an option: ",
            "Press any key to continue...",
            "Are you sure you want to delete this? ",
        ]
        for p in prompts:
            assert any(rx.search(p.strip()) for rx in INTERACTIVE_PROMPT_PATTERNS), f"Pattern failed for: {p}"

    def test_fast_interactive_timeout_terminates_in_seconds(self, monkeypatch):
        # Auto-approve command for test
        monkeypatch.setattr("tools.command_ops.request_approval", lambda *args, **kwargs: True)

        cmd = 'python -c "print(\'Do you want to proceed? [y/N] \', flush=True); import time; time.sleep(30)"'
        start = time.time()
        res = run_command(cmd)
        elapsed = time.time() - start

        # Must terminate in ~3-5 seconds, NEVER waiting 180s
        assert elapsed < 10.0, f"Took too long: {elapsed}s"
        assert "[FAST INTERACTIVE TIMEOUT" in res
        assert "closed stdin" in res
        assert "start_background_command" in res
