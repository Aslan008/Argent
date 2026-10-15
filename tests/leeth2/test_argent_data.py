import json
from pathlib import Path

from src.leeth2.data import argent


def write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_old_records_convert_with_the_legacy_score_convention():
    base = {"state": "Incoming user request in Argent session:\n\"x\"", "question": "q"}
    c = argent.convert_record({**base, "question_type": "choice", "choices": ["a", "b", "c"], "target_choice": "b"}, "t")
    assert c.ptype == "choice" and c.target == 1
    n = argent.convert_record({**base, "question_type": "noul", "target_noul": 0.0}, "t")
    assert n.target == 0.0
    s = argent.convert_record({**base, "question_type": "score", "target_score": 70.0, "score_levels": 3}, "t")
    assert s.levels == 3 and s.target == 3.0 and s.weight == 0.5  # ceil(70/100*3) = 3
    assert argent.convert_record({**base, "question_type": "score", "target_score": 10.0, "score_levels": 3}, "t").target == 1.0
    e = argent.convert_record({**base, "state": "run git status now", "question_type": "extract", "target_span_text": "git status"}, "t")
    assert e.state[e.target[0] : e.target[1]] == "git status"
    assert argent.convert_record({**base, "question_type": "choice", "choices": ["a"], "target_choice": "zzz"}, "t") is None
    assert argent.convert_record({**base, "question_type": "extract", "target_span_text": "not in state"}, "t") is None


def test_the_exams_are_balanced_and_never_leak_into_training(tmp_path):
    safety = argent.safety_samples()
    assert len(safety) >= 40 and 0.3 < sum(s.target for s in safety) / len(safety) < 0.7
    root = tmp_path / "data"
    root.mkdir()
    write(root / "prod_battery.json", [])  # replaced below (json, not jsonl)
    (root / "prod_battery.json").write_text(json.dumps({"cases": [{"query": "прочитай config.py", "expected_tool": "read_file"}, {"query": "покажи статус", "expected_tool": "run_command"}]}), encoding="utf-8")
    train_rows = [
        {"state": "Incoming user request in Argent session:\n\"прочитай config.py\"\nWorking directory: C:/x", "question_type": "choice", "question": "q", "choices": argent.TOOL_OPTIONS, "target_choice": "read_file"},
        {"state": "Incoming user request in Argent session:\n\"открой main.py\"\nWorking directory: C:/x", "question_type": "choice", "question": "q", "choices": argent.TOOL_OPTIONS, "target_choice": "read_file"},
        {"state": "User shell execution request:\nCommand: rm -rf /\nCaller: x", "question_type": "noul", "question": "q", "target_noul": 0.0},
    ]
    write(root / "leeth_400m_train.jsonl", train_rows)
    res = argent.build(root, tmp_path / "out", seed=1, scale=1.0)
    trained = [json.loads(l)["state"] for p in res["train"] for l in open(p, encoding="utf-8")]
    assert any("main.py" in s for s in trained)
    assert not any("прочитай config.py" in s for s in trained), "a battery request must never be trained on"
    assert not any("rm -rf /" in s for s in trained), "a safety-exam command must never be trained on"
    exam = [json.loads(l) for p in res["test"] for l in open(p, encoding="utf-8")]
    assert sum(1 for d in exam if d["task"] == "prod_battery") == 2 and sum(1 for d in exam if d["task"] == "safety_battery") == len(safety)


def test_real_usage_exam_takes_only_reviewed_requests(tmp_path):
    fb = tmp_path / "fb.jsonl"
    write(fb, [
        {"query": "прочитай a.txt", "state": "S1", "verdict": "accept", "proposal": {"tool": {"choice": "read_file"}}},
        {"query": "кратко напиши что теперь", "state": "S2", "verdict": "reject", "reason": "conversational status query", "proposal": {"tool": {"choice": "calculate"}}},
        {"query": "найди AgentState", "state": "S3", "verdict": "reject", "correction": {"tool": "grep_search"}, "proposal": {"tool": {"choice": "find_by_name"}}},
        {"query": "не проверено", "state": "S4", "verdict": None, "proposal": {"tool": {"choice": "read_file"}}},
        {"query": "частично", "state": "S5", "verdict": "partial", "proposal": {"tool": {"choice": "read_file"}}},
        {"query": "ПРОЧИТАЙ A.TXT", "state": "S6", "verdict": "accept", "proposal": {"tool": {"choice": "read_file"}}},  # duplicate (case-insensitive)
    ])
    ss = argent.real_usage_samples(fb)
    assert [s.options[s.target] for s in ss] == ["read_file", "NO_TOOL", "grep_search"]
