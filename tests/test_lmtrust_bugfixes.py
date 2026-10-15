"""
tests/test_lmtrust_bugfixes.py — LMTrust regression verification test suite.

Verifies fixes for the 10 identified defects:
1. /audit command and tools.security_tools execution
2. /doctor and /skill command dispatch in CLI registry
3. read_file boundary handling (out of bounds start_line)
4. multi_replace_in_file_chunk overlapping detection & CRLF preservation
5. calculate integer overflow protection (_safe_pow logarithmic limits)
6. rag_engine dynamic embedding dimension
"""

import json
import pytest
from pathlib import Path

from src.cli.commands.registry import registry, CommandContext
from tools.file_ops import read_file, multi_replace_in_file_chunk
from tools.misc_tools import calculate
from tools.security_tools import audit_security
from rag_engine import OllamaEmbeddingFunction


class DummyAgent:
    def __init__(self):
        pass

    def get_context_breakdown(self):
        return {"total": 100, "max": 8000}

    def build_system_prompt(self):
        return "system prompt"


def test_security_audit_execution():
    """Verify tools/security_tools.py runs without error and returns report."""
    report = audit_security()
    assert isinstance(report, str)
    assert "Argent Project Security Audit" in report
    assert "Dependencies" in report
    assert "Secrets & Sensitive Files" in report


def test_command_registry_doctor_and_skills():
    """Verify /doctor, /audit, and /skill /skills are properly registered and dispatchable."""
    agent = DummyAgent()

    # /audit
    ctx_audit = CommandContext(agent=agent, user_input="/audit")
    res_audit = registry.dispatch("/audit", ctx_audit)
    assert res_audit.handled is True

    # /doctor
    ctx_doctor = CommandContext(agent=agent, user_input="/doctor")
    res_doctor = registry.dispatch("/doctor", ctx_doctor)
    assert res_doctor.handled is True

    # /skills and /skill
    ctx_skills = CommandContext(agent=agent, user_input="/skills")
    res_skills = registry.dispatch("/skills", ctx_skills)
    assert res_skills.handled is True

    ctx_skill = CommandContext(agent=agent, user_input="/skill")
    res_skill = registry.dispatch("/skill", ctx_skill)
    assert res_skill.handled is True


def test_read_file_out_of_bounds(tmp_path):
    """Verify read_file with start_line > total_lines returns clean message."""
    test_file = tmp_path / "sample.txt"
    test_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    # Request range completely past end of file
    out = read_file(str(test_file), start_line=10, end_line=20)
    assert "exceeds total lines in file" in out or "Error" in out
    assert "[Lines 10-9 of 3]" not in out  # Must never show negative/inverted range


def test_multi_replace_overlapping_chunks(tmp_path):
    """Verify multi_replace_in_file_chunk rejects overlapping chunks."""
    test_file = tmp_path / "test_overlap.py"
    test_file.write_text("def a():\n    return 1\ndef b():\n    return 2\n", encoding="utf-8")

    # Overlapping chunks: chunk 1 (lines 1-3), chunk 2 (lines 2-4)
    overlapping_chunks = [
        {"start_line": 1, "end_line": 3, "target_content": "def a():\n    return 1\n", "replacement_content": "# replaced a"},
        {"start_line": 2, "end_line": 4, "target_content": "    return 1\ndef b():\n", "replacement_content": "# replaced b"},
    ]
    res = multi_replace_in_file_chunk(str(test_file), json.dumps(overlapping_chunks))
    assert "Error: Replacement chunks overlap" in res


def test_multi_replace_preserves_crlf(tmp_path):
    """Verify multi_replace_in_file_chunk preserves CRLF line endings."""
    test_file = tmp_path / "test_crlf.py"
    test_file.write_bytes(b"val_1 = 1\r\nval_2 = 2\r\nval_3 = 3\r\n")

    chunks = [
        {"start_line": 2, "end_line": 2, "target_content": "val_2 = 2\r\n", "replacement_content": "val_2 = 20\r\n"}
    ]
    res = multi_replace_in_file_chunk(str(test_file), json.dumps(chunks))
    assert "Successfully" in res
    content_bytes = test_file.read_bytes()
    assert b"\r\n" in content_bytes
    assert b"val_2 = 20\r\n" in content_bytes


def test_calculate_safe_pow_limits():
    """Verify calculate catches DoS / massive power expressions without crashing."""
    res_large = calculate("10**5000")
    # Must return a clear error or handled response, never unhandled ValueError string conversion limit
    assert "Error" in res_large
    assert "digits" in res_large or "exponent too large" in res_large or "cannot evaluate" in res_large

    res_dos = calculate("9**9**9")
    assert "Error" in res_dos

    # Normal calculations must still work
    res_normal = calculate("2**10")
    assert "1024" in res_normal


def test_rag_engine_dynamic_dim():
    """Verify OllamaEmbeddingFunction supports dynamic dimensionality."""
    embed_fn = OllamaEmbeddingFunction()
    assert embed_fn._dim == 768

    # Empty input returns empty
    assert embed_fn([]) == []

    # If fallback is invoked with an unfilled list, it matches embed_fn._dim
    embed_fn._dim = 384
    fallback_res = embed_fn._embed_individually(["dummy text"], "http://invalid-host-for-testing:9999")
    assert len(fallback_res) == 1
    assert len(fallback_res[0]) == 384
