"""scripts/leeth2_install.py: a candidate that does not work is never switched in, and the model it replaces is always kept."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "leeth2_install.py"


@pytest.fixture
def inst(monkeypatch):
    spec = importlib.util.spec_from_file_location("leeth2_install_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_run(root: Path, name: str, marker: str) -> Path:
    run = root / name
    (run / "final").mkdir(parents=True)
    for f in ("model.safetensors", "leeth2_config.json", "tokenizer.json"):
        (run / "final" / f).write_text(marker, encoding="utf-8")
    (run / "report.json").write_text(f'{{"who": "{marker}"}}', encoding="utf-8")
    return run


def run_install(inst, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["leeth2_install.py", *map(str, argv)])
    inst.main()


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_good_candidate_replaces_and_keeps_the_old_one(inst, monkeypatch, tmp_path):
    dest = make_run(tmp_path / "models", "leeth2", "old")
    new = make_run(tmp_path, "run", "new")
    monkeypatch.setattr(inst, "smoke", lambda final: (True, "fine"))
    run_install(inst, monkeypatch, new, "--dest", dest, "--backup-as", "leeth2_v31")
    assert read(dest / "final" / "model.safetensors") == "new" and read(dest / "report.json") == '{"who": "new"}'
    assert read(tmp_path / "models" / "leeth2_v31" / "final" / "model.safetensors") == "old"
    assert not (tmp_path / "models" / ".leeth2_staging").exists()


def test_first_install_has_nothing_to_back_up(inst, monkeypatch, tmp_path):
    new = make_run(tmp_path, "run", "new")
    dest = tmp_path / "models" / "leeth2"
    monkeypatch.setattr(inst, "smoke", lambda final: (True, "fine"))
    run_install(inst, monkeypatch, new, "--dest", dest)
    assert read(dest / "final" / "model.safetensors") == "new" and [p.name for p in (tmp_path / "models").iterdir()] == ["leeth2"]


def test_a_candidate_that_fails_the_check_changes_nothing(inst, monkeypatch, tmp_path):
    dest = make_run(tmp_path / "models", "leeth2", "old")
    new = make_run(tmp_path, "run", "new")
    monkeypatch.setattr(inst, "smoke", lambda final: (False, "NaN in the output"))
    with pytest.raises(SystemExit) as e:
        run_install(inst, monkeypatch, new, "--dest", dest)
    assert "nothing was installed" in str(e.value)
    assert read(dest / "final" / "model.safetensors") == "old"
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["leeth2"]          # no staging folder, no backup


def test_a_copy_that_fails_after_the_swap_is_rolled_back(inst, monkeypatch, tmp_path):
    dest = make_run(tmp_path / "models", "leeth2", "old")
    new = make_run(tmp_path, "run", "new")
    answers = iter([(True, "staged copy fine"), (False, "the installed copy cannot load")])
    monkeypatch.setattr(inst, "smoke", lambda final: next(answers))
    with pytest.raises(SystemExit) as e:
        run_install(inst, monkeypatch, new, "--dest", dest, "--backup-as", "keep")
    assert "restored" in str(e.value)
    assert read(dest / "final" / "model.safetensors") == "old" and read(dest / "report.json") == '{"who": "old"}'
    assert sorted(p.name for p in (tmp_path / "models").iterdir()) == ["leeth2"]


def test_an_existing_backup_name_is_not_overwritten(inst, monkeypatch, tmp_path):
    dest = make_run(tmp_path / "models", "leeth2", "old")
    make_run(tmp_path / "models", "leeth2_v31", "precious")
    new = make_run(tmp_path, "run", "new")
    monkeypatch.setattr(inst, "smoke", lambda final: (True, "fine"))
    with pytest.raises(SystemExit):
        run_install(inst, monkeypatch, new, "--dest", dest, "--backup-as", "leeth2_v31")
    assert read(dest / "final" / "model.safetensors") == "old"
    assert read(tmp_path / "models" / "leeth2_v31" / "final" / "model.safetensors") == "precious"


def test_a_folder_without_the_weights_is_refused(inst, monkeypatch, tmp_path):
    bad = make_run(tmp_path, "run", "x")
    (bad / "final" / "model.safetensors").unlink()
    with pytest.raises(SystemExit) as e:
        run_install(inst, monkeypatch, bad, "--dest", tmp_path / "models" / "leeth2")
    assert "missing model.safetensors" in str(e.value)


def test_the_check_rejects_a_file_that_is_not_a_model(inst, tmp_path):
    ok, seen = inst.smoke(make_run(tmp_path, "run", "not a model") / "final")
    assert not ok and seen
