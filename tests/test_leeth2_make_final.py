"""scripts/leeth2_make_final.py: the three kinds of final kernel are written with the right settings and sources."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".argent" / "kaggle" / "leeth2" / "kernel_final" / "final_v1.py"

pytestmark = pytest.mark.skipif(not TEMPLATE.exists(), reason="the Kaggle kernel template lives in the local .argent folder")


@pytest.fixture
def mk(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("leeth2_make_final_under_test", ROOT / "scripts" / "leeth2_make_final.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "kernel_final").mkdir()
    (tmp_path / "kernel_final" / "final_v1.py").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(mod, "WORK", tmp_path)
    return mod


def meta(d: Path) -> dict:
    return json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))


def setting(d: Path, name: str, code: str) -> str:
    return re.search(rf"^{name} = (.*?)\s*(#.*)?$", (d / code).read_text(encoding="utf-8"), re.M).group(1)


def test_a_self_contained_kernel_builds_its_own_corpus(mk):
    d = mk.make("base", "leeth2-merge-v2", 2.6, False)
    m = meta(d)
    assert setting(d, "MODE", "final_base.py") == '"all"' and m["enable_gpu"] is True
    assert m["dataset_sources"] == ["hunzlee/leeth2-code", "hunzlee/leeth2-targeted-v4"]
    assert m["kernel_sources"] == ["hunzlee/leeth2-corpus-v1", "hunzlee/leeth2-merge-v2"]


def test_the_data_kernel_runs_on_the_cpu_and_needs_no_start_model(mk):
    d = mk.make("data", "leeth2-merge-v2", 0, False, data_only=True)
    m = meta(d)
    assert setting(d, "MODE", "final_data.py") == '"data"' and m["enable_gpu"] is False
    assert m["kernel_sources"] == ["hunzlee/leeth2-corpus-v1"] and "hunzlee/leeth2-targeted-v4" in m["dataset_sources"]
    assert setting(d, "DATA_KERNEL", "final_data.py") == '"leeth2-final-data"'


def test_a_training_kernel_takes_the_corpus_of_the_data_kernel(mk):
    d = mk.make("grown", "leeth2-grow-v1", 5.0, False, data_kernel="leeth2-final-data")
    m = meta(d)
    assert setting(d, "MODE", "final_grown.py") == '"train"' and setting(d, "TRAIN_S", "final_grown.py") == "int(5.0 * 3600)"
    assert m["enable_gpu"] is True and m["dataset_sources"] == ["hunzlee/leeth2-code"]
    assert m["kernel_sources"] == ["hunzlee/leeth2-grow-v1", "hunzlee/leeth2-final-data"]


def test_growing_inside_the_kernel_is_a_flag(mk):
    d = mk.make("g2", "leeth2-merge-v2", 4.0, True)
    assert setting(d, "GROW", "final_g2.py") == "True"


def test_the_template_keeps_the_markers_the_script_rewrites(mk):
    text = TEMPLATE.read_text(encoding="utf-8")
    for name in ("INIT_KERNEL", "INIT_SUB", "GROW", "MODE", "DATA_KERNEL", "TRAIN_S"):
        assert len(re.findall(rf"^{name} = ", text, re.M)) == 1, name


def test_the_play_scale_of_the_newer_games_is_set_per_kernel(mk):
    d = mk.make("games2", "leeth2-final-base", 1.5, False, play_more_scale=1.2)
    assert setting(d, "PLAY_MORE_SCALE", "final_games2.py") == "1.2"
    assert "scale=PLAY_MORE_SCALE" in (d / "final_games2.py").read_text(encoding="utf-8")
    assert setting(mk.make("plain", "leeth2-final-base", 1.5, False), "PLAY_MORE_SCALE", "final_plain.py") == "0.45"


def test_the_replay_share_of_the_general_corpus_is_set_per_kernel(mk):
    assert setting(mk.make("rp", "leeth2-final-base", 1.5, False, replay_rows=330000), "REPLAY_ROWS", "final_rp.py") == "330000"
    assert setting(mk.make("rq", "leeth2-final-base", 1.5, False), "REPLAY_ROWS", "final_rq.py") == "190000"
