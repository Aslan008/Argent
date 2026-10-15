"""scripts/leeth2_kaggle.py: the arena kernel is written with the settings that were asked for (sample size, task subset, reused Laya answers)."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".argent" / "kaggle" / "leeth2" / "kernel_arena" / "arena.py"

pytestmark = pytest.mark.skipif(not TEMPLATE.exists(), reason="the Kaggle kernel template lives in the local .argent folder")


@pytest.fixture
def kg(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("leeth2_kaggle_under_test", ROOT / "scripts" / "leeth2_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "kernel_arena").mkdir()
    (tmp_path / "kernel_arena" / "arena.py").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(mod, "WORK", tmp_path)
    return mod


def setting(d: Path, name: str) -> str:
    return re.search(rf"^{name} = (.*?)\s*(#.*)?$", (d / "arena_x.py").read_text(encoding="utf-8"), re.M).group(1)


def test_a_plain_arena_keeps_the_template_sizes(kg):
    d = kg.stage_arena_kernel({"base": "leeth2-final-base"})
    assert setting(d, "ONLY_TASKS") == "()" and setting(d, "LAYA_REUSE") == '""'
    assert int(setting(d, "N_COMMON")) == 120 and int(setting(d, "N_OURS")) == 300


def test_a_precision_run_gets_its_tasks_and_sample_size(kg):
    d = kg.stage_arena_kernel({"base": "leeth2-final-base"}, only=("sst5", "clinc150"), samples=500)
    assert setting(d, "ONLY_TASKS") == "('sst5', 'clinc150')"
    assert int(setting(d, "N_COMMON")) == 500 and int(setting(d, "N_OURS")) == 500


def test_reused_laya_answers_come_from_that_kernel(kg):
    d = kg.stage_arena_kernel({"base": "leeth2-final-base"}, reuse_laya="leeth2-arena")
    assert setting(d, "LAYA_REUSE") == '"leeth2-arena"'
    assert "hunzlee/leeth2-arena" in json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))["kernel_sources"]


def test_a_candidate_can_name_the_subfolder_of_its_weights(kg):
    d = kg.stage_arena_kernel({"soup": "leeth2-interp-g2:soup0.5", "base": "leeth2-final-base"})
    assert re.search(r'^CANDIDATES = (.*)$', (d / "arena_x.py").read_text(encoding="utf-8"), re.M).group(1) == \
        json.dumps({"soup": ["leeth2-interp-g2", "soup0.5"], "base": ["leeth2-final-base", "final"]})
    assert json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))["kernel_sources"] == ["hunzlee/leeth2-interp-g2", "hunzlee/leeth2-final-base"]
