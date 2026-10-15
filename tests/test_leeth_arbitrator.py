"""Tests for the CDCL arbitrator (owner's P = PN solver backend)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.leeth.arbitrator import CDCLArbitrator, LeethCNFEncoder, TOOLS


@pytest.fixture(scope="module")
def arb():
    return CDCLArbitrator()


class TestEncoding:
    def test_valid_dispatch_not_blocked(self, arb):
        """KEY TEST (design): a legitimate dispatch must never be UNSAT."""
        res = arb.arbitrate(
            tool="read_file", need_tool=True,
            param_text="config.py", query="прочитай config.py",
        )
        assert res.error is None
        assert res.sat, f"valid dispatch blocked! core={res.core}"
        assert res.model_tool == "read_file"

    def test_no_tool_chat_sat(self, arb):
        res = arb.arbitrate(tool="NO_TOOL", need_tool=False, param_text=" и", query="продолжай")
        assert res.error is None and res.sat

    def test_unfamiliar_needs_tool_sat(self, arb):
        """UNFAMILIAR is a valid tool-needing choice without param requirements."""
        res = arb.arbitrate(
            tool="UNFAMILIAR", need_tool=True,
            param_text="Haskell-модуль", query="отрефактори этот Haskell-модуль",
        )
        assert res.error is None and res.sat

    def test_ask_user_no_param_needed(self, arb):
        res = arb.arbitrate(tool="ask_user_questions", need_tool=True, param_text="", query="что дальше?")
        assert res.error is None and res.sat


class TestConstraints:
    def test_destructive_run_command_unsat(self, arb):
        res = arb.arbitrate(
            tool="run_command", need_tool=True,
            param_text="rm -rf /", query="rm -rf /",
        )
        assert res.error is None and not res.sat
        assert any("run_command" in c for c in res.core)
        assert any("destructive=True" in c for c in res.core)

    def test_destructive_calculate_unsat(self, arb):
        """A destructive string must not ride along as a calculate param."""
        res = arb.arbitrate(
            tool="calculate", need_tool=True,
            param_text="shutdown /s /t 0", query="shutdown /s /t 0",
        )
        assert res.error is None and not res.sat

    def test_need_tool_inconsistency_unsat(self, arb):
        """tool=read_file with need_tool=False is a joint contradiction."""
        res = arb.arbitrate(
            tool="read_file", need_tool=False,
            param_text="config.py", query="прочитай config.py",
        )
        assert res.error is None and not res.sat
        assert any("need_tool=False" in c for c in res.core)
        assert any("tool=read_file" in c for c in res.core)

    def test_empty_param_unsat(self, arb):
        res = arb.arbitrate(tool="read_file", need_tool=True, param_text="", query="прочитай")
        assert res.error is None and not res.sat
        # The conflict has several minimal explanations (nonempty / in_query /
        # path_like each block read_file alone); any one may anchor the core.
        assert any(
            f"{name}=False" in c for c in res.core
            for name in ("param_nonempty", "param_in_query", "param_path_like")
        )

    def test_param_not_in_query_unsat(self, arb):
        res = arb.arbitrate(
            tool="read_file", need_tool=True,
            param_text="boilerplate/path.txt", query="прочитай config.py",
        )
        assert res.error is None and not res.sat
        assert any("param_in_query=False" in c for c in res.core)

    def test_no_tool_with_need_tool_unsat(self, arb):
        res = arb.arbitrate(tool="NO_TOOL", need_tool=True, param_text="", query="что-то")
        assert res.error is None and not res.sat


class TestCoreQuality:
    def test_core_minimality(self, arb):
        """Removing any single candidate from the core must restore SAT."""
        res = arb.arbitrate(
            tool="run_command", need_tool=True,
            param_text="rm -rf /", query="rm -rf /",
        )
        assert not res.sat and res.core
        # Rebuild the candidate set from the readable core and verify minimality
        enc = LeethCNFEncoder()
        hard = enc.hard_clauses()
        core_vars = {}
        for lit in res.core:
            key, _, val = lit.partition("=")
            if key == "tool":
                core_vars[enc.tool_vars[val]] = True   # "tool=X" always asserts X positively
            else:
                var = {"need_tool": enc.NEED, "param_nonempty": enc.PARAM_NONEMPTY,
                       "param_in_query": enc.PARAM_IN_QUERY, "destructive": enc.DESTRUCTIVE,
                       "safety_allow": enc.SAFE, "execute_immediately": enc.EXEC}[key]
                core_vars[var] = (val == "True")
        for var in core_vars:
            trial = {k: v for k, v in core_vars.items() if k != var}
            units = [[k if v else -k] for k, v in trial.items()]
            model, _ = arb._solve(hard + units, enc.num_vars)
            assert model is not None, f"core not minimal: removing {var} keeps UNSAT"

    def test_sat_has_no_core(self, arb):
        res = arb.arbitrate(tool="list_directory", need_tool=True, param_text="src", query="покажи содержимое папки src")
        assert res.sat and res.core == []


class TestPerformance:
    def test_typical_latency(self, arb):
        """Shadow budget: a single arbitration must stay well under 100 ms."""
        import time
        t0 = time.perf_counter()
        res = arb.arbitrate(tool="read_file", need_tool=True, param_text="requirements.txt", query="прочитай requirements.txt")
        dt = (time.perf_counter() - t0) * 1000
        assert res.sat and dt < 100, f"latency {dt:.1f}ms exceeds shadow budget"

    def test_unsat_core_latency(self, arb):
        import time
        t0 = time.perf_counter()
        res = arb.arbitrate(tool="run_command", need_tool=True, param_text="rm -rf /", query="rm -rf /")
        dt = (time.perf_counter() - t0) * 1000
        assert not res.sat and dt < 300, f"core extraction {dt:.1f}ms too slow"


def test_tools_match_dispatcher():
    """The encoder's tool list must mirror the dispatcher's choices exactly."""
    from src.leeth.dispatcher import LeethDispatcher
    _, questions = LeethDispatcher._build_state_and_questions("тест")
    choices = questions[1].choices
    assert choices == TOOLS, f"Encoder TOOLS out of sync with dispatcher choices: {choices}"


class TestSemanticCoherence:
    """Stage-A constraints the procedural gates do NOT check — the arbitrator's
    unique value: catching param-extraction/tool mismatches."""

    def test_calculate_arithmetic_sat(self, arb):
        res = arb.arbitrate(tool="calculate", need_tool=True, param_text="512 / 8", query="сколько будет 512 / 8")
        assert res.error is None and res.sat and res.model_tool == "calculate"

    def test_calculate_non_arithmetic_unsat(self, arb):
        """Misroute catch: a file path must not ride into calculate."""
        res = arb.arbitrate(tool="calculate", need_tool=True, param_text="config.py", query="прочитай config.py")
        assert res.error is None and not res.sat
        assert any("param_arithmetic=False" in c for c in res.core)

    def test_read_file_non_path_unsat(self, arb):
        """Misroute catch: a command with flags is not a readable path."""
        res = arb.arbitrate(tool="read_file", need_tool=True, param_text="pytest -x", query="запусти pytest -x")
        assert res.error is None and not res.sat
        assert any("param_path_like=False" in c for c in res.core)

    def test_read_file_bare_word_sat(self, arb):
        """Conservative path check: bare words (README, src) stay valid paths."""
        res = arb.arbitrate(tool="read_file", need_tool=True, param_text="README", query="прочитай README")
        assert res.error is None and res.sat

    def test_browser_open_sentence_unsat(self, arb):
        res = arb.arbitrate(tool="browser_open", need_tool=True, param_text="какие у тебя возможности", query="какие у тебя возможности?")
        assert res.error is None and not res.sat
        assert any("param_url_like=False" in c for c in res.core)

    def test_browser_open_domain_sat(self, arb):
        res = arb.arbitrate(tool="browser_open", need_tool=True, param_text="example.com", query="открой example.com")
        assert res.error is None and res.sat

    def test_run_command_free_form_sat(self, arb):
        """run_command stays unconstrained by param-type checks (deny-list covers danger)."""
        res = arb.arbitrate(tool="run_command", need_tool=True, param_text="pytest -x", query="запусти pytest -x")
        assert res.error is None and res.sat


def test_shadow_jsonl_written(tmp_path, monkeypatch):
    """Every arbitration appends one parseable JSONL record with full context."""
    import json as _json
    import src.leeth.arbitrator as arb_mod
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(arb_mod, "_SHADOW_LOG", log_path)
    a = CDCLArbitrator()
    a.arbitrate(tool="read_file", need_tool=True, param_text="config.py",
                query="прочитай config.py", context={"gates": "dispatch", "confidence": 0.91})
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "shadow log not written"
    rec = _json.loads(lines[-1])
    assert rec["sat"] is True and rec["tool"] == "read_file"
    assert rec["gates"] == "dispatch" and rec["confidence"] == 0.91
    assert rec["enforce"] is False and "latency_ms" in rec and rec["error"] is None