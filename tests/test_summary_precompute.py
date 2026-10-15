"""The summary a compaction will need is written ahead of time, in the background, and used only when it is
provably about the same messages."""

import threading
import time

import pytest

from src.agent import governor, summary_precompute as sp, trimmer


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.delenv("ARGENT_PRECOMPUTE", raising=False)
    sp.reset()
    yield
    sp.reset()


def _msgs(n, start=0):
    out = []
    for i in range(start, start + n):
        role = "user" if i % 2 == 0 else "assistant"
        out.append({"role": role, "content": f"сообщение {i}: " + "слово " * 120})
    return out


def _wait(timeout=5.0):
    t = sp._running
    if t is not None:
        t.join(timeout)
        assert not t.is_alive()


def test_a_job_runs_at_background_priority_and_leaves_a_ready_summary():
    seen = []
    fresh = _msgs(12)

    def work():
        seen.append(governor.current_priority())
        return "Готовая сводка: " + "x" * 80

    assert sp.start(fresh, "", work)
    _wait()
    assert seen == [governor.P_BACKGROUND]
    assert sp.status()["ready_messages"] == 12


def test_claim_returns_the_summary_and_only_the_newer_messages():
    fresh = _msgs(12)
    sp.start(fresh, "", lambda: "S" * 100)
    _wait()
    grown = fresh + _msgs(3, start=12)
    got = sp.claim(grown, "")
    assert got is not None
    summary, newer = got
    assert summary == "S" * 100 and [m["content"] for m in newer] == [m["content"] for m in grown[12:]]
    assert sp.claim(grown, "") is None, "a ready summary is used once"


def test_an_edited_history_does_not_match():
    fresh = _msgs(12)
    sp.start(fresh, "", lambda: "S" * 100)
    _wait()
    edited = _msgs(12)
    edited[3] = {"role": "user", "content": "совсем другое сообщение"}
    assert sp.claim(edited, "") is None


def test_a_different_earlier_summary_does_not_match():
    fresh = _msgs(12)
    sp.start(fresh, "прежняя сводка", lambda: "S" * 100)
    _wait()
    assert sp.claim(fresh, "другая сводка") is None
    assert sp.claim(fresh, "прежняя сводка") is not None


def test_aged_tool_results_still_match():
    base = _msgs(10)
    tool = {"role": "tool", "tool_call_id": "c1", "content": "полный вывод " * 400}
    fresh = base + [tool]
    sp.start(fresh, "", lambda: "S" * 100)
    _wait()
    aged = base + [{"role": "tool", "tool_call_id": "c1", "content": "[вывод сокращён]"}]
    assert sp.claim(aged, "") is not None


def test_nothing_is_written_for_a_short_history():
    assert sp.start(_msgs(3), "", lambda: "S" * 100) is False
    assert sp.status()["running"] is False


def test_a_reset_drops_a_job_that_is_still_running():
    release = threading.Event()

    def work():
        release.wait(5)
        return "S" * 100

    assert sp.start(_msgs(12), "", work)
    sp.reset()
    release.set()
    _wait()
    assert sp.status()["ready_messages"] == 0


def test_only_one_job_at_a_time():
    release = threading.Event()
    assert sp.start(_msgs(12), "", lambda: release.wait(5) and "S" * 100)
    assert sp.start(_msgs(14), "", lambda: "T" * 100) is False
    release.set()
    _wait()


def test_a_failed_job_is_not_repeated_at_once():
    assert sp.start(_msgs(12), "", lambda: None)
    _wait()
    assert sp.start(_msgs(12), "", lambda: "S" * 100) is False
    assert sp.status()["ready_messages"] == 0


def test_a_close_enough_summary_is_not_written_again():
    fresh = _msgs(20)
    sp.start(fresh, "", lambda: "S" * 100)
    _wait()
    assert sp.start(_msgs(22), "", lambda: "T" * 100) is False


# ── through the real compaction ─────────────────────────────────────────────────

BUDGET = 3000


def _history(n=70):
    return [{"role": "system", "content": "System prompt"}] + _msgs(n)


@pytest.fixture
def asked(monkeypatch):
    calls = []

    def fake(pre_distilled_text, active_goal, prior_context, model_name, timeout=120.0, ui=None):
        calls.append({"text": pre_distilled_text, "prior": prior_context, "priority": governor.current_priority()})
        return f"## Intent\nСводка номер {len(calls)} " + "и ещё слова " * 12

    monkeypatch.setattr(trimmer, "_summarize_with_llm", fake)
    return calls


def _distillate(msgs):
    return next(m["content"] for m in msgs if "CONTEXT DISTILLATE" in m.get("content", ""))


def test_a_compaction_uses_the_summary_written_ahead_without_asking_again(asked):
    history = _history()
    assert trimmer.precompute_summary(history, "mock", BUDGET) is True
    _wait()
    assert len(asked) == 1 and asked[0]["priority"] == governor.P_BACKGROUND

    out = trimmer.soft_trim_with_summarization(history, "mock", 20, BUDGET)
    assert len(asked) == 1, "nothing changed since: no second request"
    assert "Сводка номер 1" in _distillate(out)


def test_a_compaction_updates_the_ready_summary_with_what_came_after(asked):
    history = _history()
    trimmer.precompute_summary(history, "mock", BUDGET)
    _wait()
    grown = history + _msgs(8, start=70)
    out = trimmer.soft_trim_with_summarization(grown, "mock", 20, BUDGET)
    assert len(asked) == 2
    second = asked[1]
    assert "Сводка номер 1" in second["prior"], "the ready summary is what the model updates"
    assert second["priority"] == governor.P_INTERACTIVE, "the update is the person's own wait"
    assert len(second["text"]) < len(asked[0]["text"]), "and it sees only the newer messages"
    assert "Сводка номер 2" in _distillate(out)


def test_without_a_ready_summary_the_compaction_is_what_it_always_was(asked):
    history = _history()
    trimmer.soft_trim_with_summarization(history, "mock", 20, BUDGET)
    assert len(asked) == 1 and not asked[0]["prior"]


def test_a_failed_update_falls_back_to_the_ready_summary(monkeypatch, asked):
    history = _history()
    trimmer.precompute_summary(history, "mock", BUDGET)
    _wait()
    monkeypatch.setattr(trimmer, "_summarize_with_llm", lambda *a, **k: None)
    out = trimmer.soft_trim_with_summarization(history + _msgs(8, start=70), "mock", 20, BUDGET)
    assert "Сводка номер 1" in _distillate(out)


def test_disabled_by_the_setting(monkeypatch, asked):
    monkeypatch.setattr("config.is_summary_precompute_enabled", lambda: False)
    assert trimmer.precompute_summary(_history(), "mock", BUDGET) is False
    assert asked == []


def test_a_streamed_background_answer_yields_to_the_person(monkeypatch, tmp_path):
    """The background request goes through the lane as a worker's would: a better request pushes it out."""
    from src.agent import lanes, pressure
    from src.agent.local_backend import Backend

    monkeypatch.delenv("ARGENT_GOVERNOR", raising=False)
    monkeypatch.setenv("ARGENT_LANES_DIR", str(tmp_path / "lanes"))
    lanes.forget_lanes()
    local = Backend(True, "ollama", "localhost", 11434)
    monkeypatch.setattr(governor, "classify", lambda provider, model="": local)
    calm = type("Calm", (), {"wait_until_calm": staticmethod(lambda **kw: True)})()
    monkeypatch.setattr(pressure, "monitor", lambda: calm)

    class P:
        name = "ollama"
        streams = 0
        syncs = 0

        def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None, format_schema=None):
            P.streams += 1
            yield {"content": "ответ "}
            yield {"content": "целиком"}

        def sync_chat(self, model, messages, temperature=0.3, json_format=False, timeout=None):
            P.syncs += 1
            return "sync"

    p = governor.govern(P())
    with governor.priority(governor.P_BACKGROUND):
        assert p.sync_chat("m", []) == "ответ целиком"
        assert p.sync_chat("m", [], json_format=True) == "sync", "JSON answers keep the provider's own path"
    assert p.sync_chat("m", []) == "sync", "the person's own request is not rerouted"
    assert (P.streams, P.syncs) == (1, 2)
    lanes.forget_lanes()
