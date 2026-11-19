"""Compaction reports what it saved and refuses to loop when it cannot save anything."""

import json
from unittest.mock import patch

import pytest

from src.agent import trimmer


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    trimmer.reset_compaction_history()
    monkeypatch.setattr("argent.project_paths.project_root_or_cwd", lambda *a, **k: tmp_path)
    yield
    trimmer.reset_compaction_history()


def _record(saving, at, n=1):
    for _ in range(n):
        pre = 10000
        trimmer.record_compaction("auto", 1, pre, int(pre * (1 - saving)), "mock", now=at)


def test_saving_is_computed_and_described():
    entry = trimmer.record_compaction("auto", 3, 128450, 12000, "m")
    assert entry["pre_tokens"] == 128450 and entry["post_tokens"] == 12000
    assert entry["saving"] == pytest.approx(0.907, abs=0.001)
    text = trimmer.describe_compaction(entry)
    assert "128.4k" in text and "12k" in text and "91%" in text


def test_a_compaction_that_saved_enough_is_not_a_thrash():
    _record(0.8, at=1000.0, n=3)
    assert trimmer.is_thrashing(now=1100.0) is False


def test_three_close_compactions_with_no_gain_are_a_thrash():
    _record(0.05, at=1000.0, n=3)
    assert trimmer.is_thrashing(now=1100.0) is True


def test_old_compactions_are_not_a_thrash():
    _record(0.05, at=1000.0, n=3)
    assert trimmer.is_thrashing(now=1000.0 + trimmer.THRASH_WINDOW_SECONDS + 1) is False


def test_two_runs_are_not_enough():
    _record(0.0, at=1000.0, n=2)
    assert trimmer.is_thrashing(now=1001.0) is False


def test_the_boundary_log_carries_token_counts(tmp_path):
    before = [{"role": "system", "content": "s"}, {"role": "user", "content": "x " * 400},
              {"role": "assistant", "content": "y " * 400}]
    after = [before[0], {"role": "system", "content": "short"}]
    trimmer._write_compact_boundary(before, after, 5, model_name="mock", kind="auto")
    line = (tmp_path / ".argent" / "compact_log.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1]
    entry = json.loads(line)
    assert entry["type"] == "compact_boundary" and entry["kind"] == "auto"
    assert entry["pre_tokens"] > entry["post_tokens"] >= 0
    assert 0 < entry["saving"] <= 1


def _long_history(n=26):
    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(n):
        messages.append({"role": "user", "content": f"Сделай шаг номер {i} в большом проекте"})
        messages.append({"role": "assistant", "content": f"Шаг {i} готов, результат записан"})
    return messages


def test_while_thrashing_the_summariser_is_not_called():
    _record(0.0, at=__import__("time").time(), n=3)
    assert trimmer.is_thrashing()
    with patch.object(trimmer, "_create_distillate", side_effect=AssertionError("model must not be called")), \
         patch("src.agent.trimmer._summarize_with_llm", side_effect=AssertionError("model must not be called")):
        out = trimmer.soft_trim_with_summarization(_long_history(), "mock", max_history_messages=10,
                                                   max_context_tokens=30000)
    assert any("CONTEXT DISTILLATE" in m.get("content", "") for m in out)
    assert len(out) < 53


def test_a_normal_compaction_records_its_metrics_and_announces_them(capsys):
    said = []
    with patch("src.agent.trimmer._summarize_with_llm", return_value="## Done\n- old turns"), \
         patch("argent.ui.print_system", side_effect=lambda t, *a, **k: said.append(t)):
        out = trimmer.soft_trim_with_summarization(_long_history(), "mock", max_history_messages=10,
                                                   max_context_tokens=30000)
    assert len(out) < 53
    rec = trimmer.last_compaction()
    assert rec is not None and rec["kind"] == "auto" and rec["pre_tokens"] > rec["post_tokens"]
    assert any("Контекст сжат" in s for s in said)


def test_the_turn_counter_counts_only_new_compactions():
    """Метка берётся на старте хода; старое в этот ход не попадает.

    Длина списка для этого не годится: он обрезается до двадцати записей, и
    долгий ход с двумя десятками сжатий иначе показывал бы ноль.
    """
    mark = trimmer.compaction_seq()
    assert trimmer.count_since(mark) == 0

    _record(0.5, at=1000.0, n=2)
    assert trimmer.count_since(mark) == 2

    fresh = trimmer.compaction_seq()
    assert trimmer.count_since(fresh) == 0, "новый ход считается с новой метки"


def test_the_counter_survives_more_compactions_than_the_list_keeps():
    from src.agent import compaction_log as clog

    keep = clog._COMPACTIONS_KEEP
    mark = trimmer.compaction_seq()
    _record(0.5, at=1000.0, n=keep + 5)

    assert len(clog._COMPACTIONS) == keep, "список обрезан"
    assert trimmer.count_since(mark) == keep + 5, "счётчик не обрезан"


def test_the_agent_reports_the_compactions_of_its_own_turn():
    """Свойство агента, с которого читают оба фронтенда."""
    from argent.agent import ArgentAgent

    agent = ArgentAgent.__new__(ArgentAgent)          # без __init__: нужно одно поле
    agent._turn_compaction_seq = trimmer.compaction_seq()
    assert agent.turn_compactions == 0

    _record(0.5, at=1000.0, n=2)
    assert agent.turn_compactions == 2


def test_the_turn_note_is_silent_without_compactions():
    assert trimmer.turn_note(0) == ""
    assert trimmer.turn_note(0, {"pre_tokens": 1, "post_tokens": 1}) == ""


def test_the_turn_note_names_the_count_and_the_last_saving():
    entry = trimmer.record_compaction("auto", 7, 128450, 12000, "m")

    assert trimmer.turn_note(1, entry) == "1 сжатие · 128.4k→12k"
    assert trimmer.turn_note(2, entry) == "2 сжатия · 128.4k→12k"
    assert trimmer.turn_note(5, entry) == "5 сжатий · 128.4k→12k"
    assert trimmer.turn_note(11, entry).startswith("11 сжатий")
    assert trimmer.turn_note(21, entry).startswith("21 сжатие")
    # Без записи — только число, но строка всё равно правдива.
    assert trimmer.turn_note(2) == "2 сжатия"
