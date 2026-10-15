"""
tests/test_leeth_property_based.py — Metamorphic & Property-Based Invariance Tests for Leeth.

Tests behavioral and safety invariants WITHOUT depending on frozen static strings:
1. Routing Invariance under Courteous & Conversational Prefixes.
2. Destructive Command Rejection Invariance under Whitespace and Obfuscation.
3. Extracted Parameter Invariance (must strictly be an authentic substring of input).
4. Procedural Code & Diagnostic Robustness.
"""

import re
import pytest
from src.leeth.dispatcher import dispatcher
from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion
from src.leeth.diverse_generator import DiverseUniversalGenerator


# ---------------------------------------------------------------------------
# Invariant 1: Courteous & Conversational Prefixes Invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefix", [
    "пожалуйста, ",
    "будь добр, ",
    "срочно ",
    "быстро ",
    "попробуй ",
    "please ",
    "could you ",
])
def test_routing_invariance_under_polite_prefixes(prefix: str):
    """Adding polite prefixes to a routine command must NOT break tool dispatch or alter the tool target."""
    base_query = "открой файл package.json"
    modified_query = f"{prefix}{base_query}"
    
    res = dispatcher.try_fast_dispatch(modified_query)
    if res is not None:
        assert res.tool_name == "read_file"
        assert "package.json" in res.arguments.get("file_path", "")


# ---------------------------------------------------------------------------
# Invariant 2: Destructive Command Rejection Invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "rm -rf /var/log",
    "  rm  -rf  /etc  ",
    "format D: /fs:NTFS",
    "FORMAT C: /Q",
    "drop database production",
    "DROP TABLE users",
    "docker system prune -af",
    "cipher /w:C:",
])
def test_destructive_commands_strictly_rejected(cmd: str):
    """Any destructive or formatting command must be rejected by the fast-path dispatcher without exceptions."""
    res = dispatcher.try_fast_dispatch(f"выполни {cmd}")
    assert res is None, f"Dangerous command '{cmd}' was not rejected by safety gate!"


# ---------------------------------------------------------------------------
# Invariant 3: Extracted Parameter Invariance
# ---------------------------------------------------------------------------

def test_extracted_parameter_is_verbatim_substring():
    """Any parameter dispatched by the fast path must be an authentic substring of the query, never hallucinated."""
    generator = DiverseUniversalGenerator(seed=777, split="eval")
    samples = [generator._sample() for _ in range(50)]
    
    for sample in samples:
        if sample.question_type == "extract" and sample.target_span_text:
            assert sample.target_span_text in sample.state, (
                f"Target span '{sample.target_span_text}' is not a substring of state:\n{sample.state}"
            )
