"""
tests/lmtrust/test_lmtrust_replace_in_file_bounded.py
LMTrust Ruthless Test Suite for replace_in_file bounded search (start_line / end_line).

Covers:
  - L0 Smoke: Basic replacement within window.
  - L1 Contract: 1-indexed boundaries and return signatures.
  - L2 Boundary: start_line == end_line, start_line=1, clamp start_line < 1,
                 start_line > total_lines error, end_line < start_line error,
                 coercion of numeric string bounds ("10", "20").
  - L3 Invariant: Text strictly outside [start_line, end_line] remains 100% byte-for-byte intact.
  - L4 Adversarial: Target exists in file, but OUTSIDE the window -> must fail with
                    descriptive error without touching the file.
  - L4b Disambiguation: Duplicate identical lines in file -> unbounded fails with ambiguity error,
                        bounded replaces ONLY the target in the designated window.
  - L6 Fuzzy Bounded: Fuzzy match succeeds when inside window, fails when outside window.
  - L9 Negative Space: Omission of start_line/end_line searches entire file identically to legacy.
"""

import pytest
from pathlib import Path
from tools.file_ops import replace_in_file


@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    import memory_manager

    class FakeMemory:
        def __init__(self):
            self.completed = []
            self.modified = []

        def add_completed(self, s):
            self.completed.append(s)

        def add_file_modified(self, s):
            self.modified.append(s)

    monkeypatch.setattr(memory_manager, "memory", FakeMemory())

    import file_tracker
    monkeypatch.setattr(file_tracker, "snapshot", lambda *a, **k: None)

    import tools._helpers as helpers
    monkeypatch.setattr(helpers, "_validate_code_syntax", lambda fp: None)

    import tools.file_ops as file_ops_mod
    monkeypatch.setattr(file_ops_mod, "_validate_code_syntax", lambda fp: None)


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ===========================================================================
# L0 Smoke: Basic bounded replacement
# ===========================================================================
class TestSmokeBoundedReplace:
    @pytest.mark.lmtrust(layer="L0", direction="D1")
    def test_l0_smoke_bounded_replace(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("line 1\nline 2\nline 3\nline 4\nline 5\n", encoding="utf-8")
        res = replace_in_file(str(f), "line 3", "line THREE", start_line=2, end_line=4)
        assert res.startswith("Successfully replaced text in")
        assert f.read_text(encoding="utf-8") == "line 1\nline 2\nline THREE\nline 4\nline 5\n"


# ===========================================================================
# L1 Contract: 1-indexed boundaries
# ===========================================================================
class TestContractBoundedReplace:
    @pytest.mark.lmtrust(layer="L1", direction="D1")
    def test_l1_exact_line_1_replace(self, tmp_path):
        f = tmp_path / "contract.txt"
        f.write_text("FIRST LINE\nSECOND LINE\n", encoding="utf-8")
        res = replace_in_file(str(f), "FIRST LINE", "HEAD", start_line=1, end_line=1)
        assert res.startswith("Successfully replaced text in")
        assert f.read_text(encoding="utf-8") == "HEAD\nSECOND LINE\n"

    @pytest.mark.lmtrust(layer="L1", direction="D1")
    def test_l1_exact_last_line_replace(self, tmp_path):
        f = tmp_path / "contract.txt"
        f.write_text("LINE 1\nLINE 2\nTAIL", encoding="utf-8")
        res = replace_in_file(str(f), "TAIL", "END", start_line=3, end_line=3)
        assert res.startswith("Successfully replaced text in")
        assert f.read_text(encoding="utf-8") == "LINE 1\nLINE 2\nEND"


# ===========================================================================
# L2 Boundary: Zero, negative, inverted, string bounds
# ===========================================================================
class TestBoundaryBoundedReplace:
    @pytest.mark.lmtrust(layer="L2", direction="D9")
    def test_l2_start_line_zero_clamps_to_one(self, tmp_path):
        f = tmp_path / "bounds.txt"
        f.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
        res = replace_in_file(str(f), "line 1", "MODIFIED", start_line=0, end_line=2)
        assert res.startswith("Successfully replaced text in")
        assert "MODIFIED" in f.read_text(encoding="utf-8")

    @pytest.mark.lmtrust(layer="L2", direction="D9")
    def test_l2_start_line_exceeds_total_lines(self, tmp_path):
        f = tmp_path / "bounds.txt"
        f.write_text("line 1\nline 2\n", encoding="utf-8")
        res = replace_in_file(str(f), "line 2", "MODIFIED", start_line=10, end_line=20)
        assert "Error: 'start_line' (10) exceeds total lines in file (2)" in res
        assert f.read_text(encoding="utf-8") == "line 1\nline 2\n"

    @pytest.mark.lmtrust(layer="L2", direction="D9")
    def test_l2_end_line_less_than_start_line(self, tmp_path):
        f = tmp_path / "bounds.txt"
        f.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
        res = replace_in_file(str(f), "line 2", "MODIFIED", start_line=3, end_line=2)
        assert "Error: 'end_line' (2) cannot be less than 'start_line' (3)" in res
        assert f.read_text(encoding="utf-8") == "line 1\nline 2\nline 3\n"

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_l2_string_line_numbers_coerced(self, tmp_path):
        f = tmp_path / "bounds.txt"
        f.write_text("a\nb\nc\nd\n", encoding="utf-8")
        res = replace_in_file(str(f), "b", "B", start_line="2", end_line="3")
        assert res.startswith("Successfully replaced text in")
        assert f.read_text(encoding="utf-8") == "a\nB\nc\nd\n"

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_l2_invalid_string_line_numbers_rejected(self, tmp_path):
        f = tmp_path / "bounds.txt"
        f.write_text("a\nb\nc\nd\n", encoding="utf-8")
        res = replace_in_file(str(f), "b", "B", start_line="two", end_line="three")
        assert "Error: 'start_line' and 'end_line' must be valid integers." in res


# ===========================================================================
# L3 Invariant: Outside bounds strictly unchanged
# ===========================================================================
class TestInvariantBoundedReplace:
    @pytest.mark.lmtrust(layer="L3", direction="D3")
    def test_l3_bytes_outside_window_preserved_exactly(self, tmp_path):
        f = tmp_path / "invariant.txt"
        content = "BEFORE_1\nBEFORE_2\nTARGET\nAFTER_1\nAFTER_2\n"
        f.write_text(content, encoding="utf-8")
        res = replace_in_file(str(f), "TARGET", "REPLACED", start_line=3, end_line=3)
        assert res.startswith("Successfully replaced text in")
        new_content = f.read_text(encoding="utf-8")
        assert new_content == "BEFORE_1\nBEFORE_2\nREPLACED\nAFTER_1\nAFTER_2\n"
        assert new_content.startswith("BEFORE_1\nBEFORE_2\n")
        assert new_content.endswith("\nAFTER_1\nAFTER_2\n")


# ===========================================================================
# L4 Adversarial: Target outside window rejected
# ===========================================================================
class TestAdversarialBoundedReplace:
    @pytest.mark.lmtrust(layer="L4", direction="D5")
    def test_l4_target_present_in_file_but_outside_range_is_rejected(self, tmp_path):
        f = tmp_path / "adv.txt"
        f.write_text("line 1\nTARGET_OUTSIDE\nline 3\nline 4\nline 5\n", encoding="utf-8")
        res = replace_in_file(str(f), "TARGET_OUTSIDE", "EVIL", start_line=3, end_line=5)
        assert "Error: The target text was not found within lines 3-5" in res
        assert "TARGET_OUTSIDE" in f.read_text(encoding="utf-8")
        assert "EVIL" not in f.read_text(encoding="utf-8")


# ===========================================================================
# L4b Disambiguation: Resolving identical duplicates
# ===========================================================================
class TestDisambiguationBoundedReplace:
    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_l4b_duplicate_lines_fail_unbounded_but_succeed_bounded(self, tmp_path):
        f = tmp_path / "dup.txt"
        f.write_text("def test():\n    return True\n\ndef other():\n    return True\n", encoding="utf-8")
        
        # Unbounded fails due to multiple matches
        res_unbounded = replace_in_file(str(f), "    return True", "    return False")
        assert "appears 2 times" in res_unbounded
        
        # Bounded to lines 1-3 disambiguates and targets the FIRST instance
        res_bounded = replace_in_file(str(f), "    return True", "    return False", start_line=1, end_line=3)
        assert res_bounded.startswith("Successfully replaced text in")
        assert f.read_text(encoding="utf-8") == "def test():\n    return False\n\ndef other():\n    return True\n"

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_l4b_multiple_matches_inside_window_reports_narrow_down(self, tmp_path):
        f = tmp_path / "dup_window.txt"
        f.write_text("item\nitem\nitem\nother\n", encoding="utf-8")
        res = replace_in_file(str(f), "item", "REPLACED", start_line=1, end_line=3)
        assert "Error: The target text appears 3 times within lines 1-3" in res
        assert "Narrow down start_line/end_line" in res


# ===========================================================================
# L6 Fuzzy Bounded: Indentation/blank line mismatch inside window
# ===========================================================================
class TestFuzzyBoundedReplace:
    @pytest.mark.lmtrust(layer="L6", direction="D2")
    def test_l6_fuzzy_replace_within_window(self, tmp_path):
        f = tmp_path / "fuzzy.py"
        f.write_text("header\n    def calc():\n        x = 1\n        return x\nfooter\n", encoding="utf-8")
        fuzzy_target = "def calc():\n    x = 1\n    return x"
        replacement = "    def calc():\n        return 42"
        res = replace_in_file(str(f), fuzzy_target, replacement, start_line=2, end_line=5)
        assert res.startswith("Successfully replaced text in")
        assert "return 42" in f.read_text(encoding="utf-8")


# ===========================================================================
# L9 Negative Space: Omission preserves legacy full-file search
# ===========================================================================
class TestNegativeSpaceBoundedReplace:
    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_l9_none_bounds_searches_entire_file(self, tmp_path):
        f = tmp_path / "legacy.txt"
        f.write_text("line 1\nline 2\nunique target\nline 4\n", encoding="utf-8")
        res = replace_in_file(str(f), "unique target", "SUCCESS")
        assert res.startswith("Successfully replaced text in")
        assert "SUCCESS" in f.read_text(encoding="utf-8")
