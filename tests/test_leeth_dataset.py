"""
tests/test_leeth_dataset.py — Verification tests for Leeth multilingual dataset generator.
"""

from src.leeth.dataset_generator import LeethDatasetGenerator


def test_dataset_generation_coverage(tmp_path):
    # never the default path: that is a tracked-looking file under src/ and the states in it name the machine owner's home
    gen = LeethDatasetGenerator(seed=123)
    samples = gen.build_full_dataset(target_count=500, out_path=tmp_path / "leeth_dataset.jsonl")

    assert len(samples) == 500

    types_found = {s.question_type for s in samples}
    assert "choice" in types_found
    assert "noul" in types_found
    assert "score" in types_found
    assert "extract" in types_found

    # Verify C#, C++, Rust, Python, and Shell presence
    states_text = " ".join(s.state for s in samples)
    assert "C#" in states_text or ".cs" in states_text or "dotnet" in states_text
    assert "C++" in states_text or "clang" in states_text or ".cpp" in states_text
    assert "Rust" in states_text or "cargo" in states_text or "E0382" in states_text
    assert "pytest" in states_text or "Python" in states_text

    # Verify extract spans actually exist in the state strings
    extract_samples = [s for s in samples if s.question_type == "extract" and s.target_span_text]
    assert len(extract_samples) > 20
    for s in extract_samples:
        assert s.target_span_text in s.state, f"Span '{s.target_span_text}' not in state:\n{s.state}"

    # Verify noul targets are strictly 0.0 or 1.0
    noul_samples = [s for s in samples if s.question_type == "noul"]
    for s in noul_samples:
        assert s.target_noul in (0.0, 1.0)

    # Verify score targets are in 0..100
    score_samples = [s for s in samples if s.question_type == "score"]
    for s in score_samples:
        assert 0.0 <= s.target_score <= 100.0
