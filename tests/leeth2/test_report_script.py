import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("leeth2_report", Path(__file__).resolve().parents[2] / "scripts" / "leeth2_report.py")
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def rep():
    return {
        "task:imdb": {"choice": {"n": 100, "acc": 0.95, "ece": 0.02, "brier": 0.08, "nll": 0.2}},
        "task:arc_easy": {"choice": {"n": 50, "acc": 0.8}},
        "task:arc_challenge": {"choice": {"n": 150, "acc": 0.6}},
        "task:squad2": {"extract": {"n": 10, "has_ans": 8, "f1": 0.75, "em": 0.6}},
        "ptype:choice": {"choice": {"n": 300, "acc": 0.9, "ece": 0.031, "brier": 0.12, "nll": 0.3}},
        "ptype:noul": {"noul": {"n": 100, "acc": 0.93, "ece": 0.02, "brier": 0.05, "nll": 0.2}},
    }


def test_pooling_is_weighted_by_example_count():
    v, n = R.pool(rep(), ["arc_easy", "arc_challenge"], "choice", "acc")
    assert n == 200 and abs(v - (0.8 * 50 + 0.6 * 150) / 200) < 1e-9
    v, n = R.pool(rep(), ["squad2"], "extract", "f1")
    assert n == 8 and v == 0.75  # only answerable questions count for span F1
    assert R.pool(rep(), ["missing"], "choice", "acc") == (None, 0)


def test_markdown_lists_found_tasks_marks_status_and_skips_missing():
    md = R.render(rep(), {"steps": 1000, "seconds": 7200, "temperatures": {"temp_choice": 1.1}})
    assert "IMDB sentiment" in md and "**95.0%**" in md and "96.5%" in md and "in-domain" in md
    assert "ARC Easy+Challenge" in md and "**65.0%**" in md
    assert "CLINC150" not in md, "tasks absent from the run are not invented"
    assert "Calibration" in md and "0.031" in md and "0.028" in md
