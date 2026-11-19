"""tests/test_leeth_review_cli.py — разметка боевого журнала руками человека и единая точка записи исхода.

Зачем этот файл существует: боевой журнал Argent — единственный доступный набор задачи «выбор на боевой трудности»
(корпус фабрики смещён в лёгкую сторону, это измерено). Без меток он бесполезен, а без единой точки записи исхода
он теряет половину сигнала. Оба механизма должны ломаться громко, а не молча.
"""

import json
import types

import pytest

from src.leeth import feedback


@pytest.fixture()
def fb(tmp_path, monkeypatch, _leeth_feedback_off):
    """Isolated feedback store, forced ON (same trick as test_leeth_feedback.py)."""
    monkeypatch.setenv("LEETH_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    monkeypatch.delenv("LEETH_FEEDBACK_DISABLED", raising=False)
    monkeypatch.setenv("LEETH_FEEDBACK_FORCE", "1")
    yield feedback


def _record(query="прочитай config.py", tool="read_file", conf=0.93):
    from src.leeth.primitives import ChoiceQuestion, NoulQuestion

    did = feedback.log_decision(
        query=query,
        state='Incoming user request in Argent session:\n"%s"' % query,
        proposal={
            "need_tool": {"value": True, "probability": 0.97},
            "tool": {"choice": tool, "confidence": conf},
            "param": "config.py",
        },
    )
    feedback.attach_dispatch(did, tool, {"path": "config.py"})
    return did


def _ctx(cmd):
    from src.cli.commands.base import CommandContext

    return CommandContext(agent=None, user_input=cmd)


def test_lists_pending_and_summary(fb, capsys):
    _record()
    from src.cli.commands.leeth_commands import handle_leeth

    handle_leeth(_ctx("/leeth"))
    out = capsys.readouterr().out
    assert "всего 1" in out
    assert "ждёт 1" in out


def test_human_verdict_is_recorded_with_source_human(fb, monkeypatch):
    did = _record()
    import src.cli.commands.leeth_commands as cmd

    answers = iter(["a"])                       # accept, then nothing else is asked
    monkeypatch.setattr(cmd, "_ask", lambda prompt: next(answers, None))

    cmd._review(5)

    rec = feedback.get_record(did)
    assert rec["verdict"] == "accept"
    assert rec["verdict_source"] == "human", "человеческая метка должна отличаться от самооценки агента"


def test_partial_verdict_keeps_correction(fb, monkeypatch):
    did = _record(tool="read_file")
    import src.cli.commands.leeth_commands as cmd

    answers = iter(["p", "grep_search", "pattern=leeth", "не тот инструмент"])
    monkeypatch.setattr(cmd, "_ask", lambda prompt: next(answers, None))

    cmd._review(5)

    rec = feedback.get_record(did)
    assert rec["verdict"] == "partial"
    assert rec["correction"] == {"tool": "grep_search", "param": "pattern=leeth"}
    assert rec["reason"] == "не тот инструмент"


def test_skip_leaves_record_untouched(fb, monkeypatch):
    did = _record()
    import src.cli.commands.leeth_commands as cmd

    monkeypatch.setattr(cmd, "_ask", lambda prompt: "s")
    cmd._review(5)

    assert feedback.get_record(did).get("verdict") is None


def test_quit_stops_the_loop(fb, monkeypatch):
    ids = [_record(query="запрос %d" % i) for i in range(3)]
    import src.cli.commands.leeth_commands as cmd

    monkeypatch.setattr(cmd, "_ask", lambda prompt: "q")
    cmd._review(10)

    assert all(feedback.get_record(i).get("verdict") is None for i in ids)


def test_no_terminal_is_not_a_crash(fb, monkeypatch):
    """In a TUI run there is no stdin for input(): the command must end quietly, not raise."""
    _record()
    import src.cli.commands.leeth_commands as cmd

    def boom(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    cmd._review(5)                                # no exception


def test_outcome_lands_on_the_same_record_as_the_proposal(fb):
    """Единая точка исполнения: вызов инструмента и его исход — на одной записи, без второго места."""
    from argent.agent import ArgentAgent

    did = _record()

    stub = types.SimpleNamespace(_execute_fast_tool=lambda name, args: "file contents")
    result = ArgentAgent._execute_dispatched_tool(stub, "read_file", {"path": "config.py"}, did)

    assert result == "file contents"
    rec = feedback.get_record(did)
    assert rec["proposal"]["tool"]["choice"] == "read_file"
    assert rec["outcome"]["type"] == "executed_ok"
    assert rec["outcome"]["tool"] == "read_file"


def test_a_raising_tool_is_recorded_as_an_outcome(fb):
    from argent.agent import ArgentAgent

    did = _record()

    def explode(name, args):
        raise RuntimeError("disk on fire")

    stub = types.SimpleNamespace(_execute_fast_tool=explode)
    result = ArgentAgent._execute_dispatched_tool(stub, "read_file", {"path": "x"}, did)

    assert result.startswith("Error executing read_file")
    assert feedback.get_record(did)["outcome"]["type"] == "executed_error"


def test_arbitrator_conflict_is_recorded_and_never_blocks(fb, monkeypatch):
    """Спор солвера — это приоритет в очереди, а не отказ: вызов обязан состояться."""
    from src.leeth import arbitrator as arb_mod
    from src.leeth import dispatcher as disp_mod
    from src.leeth.arbitrator import ArbitrationResult

    class _Stub:
        def arbitrate(self, **kwargs):
            return ArbitrationResult(sat=False, core=["param_arithmetic=False", "tool=calculate"],
                                     model_tool=None, latency_ms=1.5, conflicts=1, decisions=2, error=None)

    monkeypatch.setattr(arb_mod, "get_arbitrator", lambda: _Stub())
    res = disp_mod.dispatcher.try_fast_dispatch("посчитай 128 * 64")
    assert res is not None, "солвер не имеет права блокировать вызов"
    rec = feedback.get_record(res.decision_id)
    assert rec["arbitrator"]["sat"] is False
    assert "param_arithmetic=False" in rec["arbitrator"]["core"]


def test_solver_conflicts_are_reviewed_first(fb, monkeypatch):
    _record(query="старый запрос без спора")
    _record(query="новый запрос без спора")
    conflict = _record(query="запрос, где солвер возражает")
    feedback.attach_arbitrator(conflict, False, ["tool=calculate"], 1.0)

    import src.cli.commands.leeth_commands as cmd

    seen = []
    monkeypatch.setattr(cmd, "_label_one", lambda rec: seen.append(rec.get("id")) or None)
    cmd._review(1)

    assert seen == [conflict], "в очередь первым должен попасть спор с солвером, даже если он старше"


def test_summary_counts_solver_conflicts(fb, capsys):
    did = _record()
    feedback.attach_arbitrator(did, False, ["tool=read_file"], 0.4)

    from src.cli.commands.leeth_commands import handle_leeth

    handle_leeth(_ctx("/leeth"))
    assert "Солвер возражает" in capsys.readouterr().out


def test_no_decision_id_means_no_outcome_attempt(fb):
    """Инструмент, выбранный самой LLM (не Leeth), не имеет записи решения — и не должен её выдумывать."""
    from argent.agent import ArgentAgent

    stub = types.SimpleNamespace(_execute_fast_tool=lambda name, args: "ok")
    assert ArgentAgent._execute_dispatched_tool(stub, "read_file", {}, None) == "ok"
