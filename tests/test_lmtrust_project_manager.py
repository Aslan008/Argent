"""Blind-spot tests for project_manager.ProjectManager.

Each test uses a fresh tmp_path and monkeypatches
``project_manager.resolve_project_file`` so that the manager reads/writes
a throwaway ``.argent_project.json`` inside the tmp directory.
"""

import json
import pytest

import project_manager


@pytest.fixture
def pm(tmp_path, monkeypatch):
    """Return a ProjectManager whose project file lives in tmp_path."""
    project_file = tmp_path / ".argent_project.json"
    monkeypatch.setattr(
        project_manager, "resolve_project_file", lambda: project_file
    )
    return project_manager.ProjectManager()


# ─── 1. create sets up data dict with correct defaults ───────────────────

def test_create_sets_up_data_dict_with_defaults(pm):
    pm.create("Build a calculator")
    assert pm.data is not None
    expected_keys = {
        "mode", "objective", "created_at", "status", "research_data",
        "architecture", "file_specs", "spec", "tasks", "files_created",
        "files_to_edit", "files_to_create", "work_strategy",
        "work_auto_mode", "tdd_mode",
    }
    assert expected_keys.issubset(pm.data.keys())
    assert pm.data["objective"] == "Build a calculator"
    assert pm.data["mode"] == "project"
    assert pm.data["status"] == "specifying"
    assert pm.data["research_data"] == ""
    assert pm.data["architecture"] == ""
    assert pm.data["file_specs"] == {}
    assert pm.data["spec"] == ""
    assert pm.data["tasks"] == []
    assert pm.data["files_created"] == []
    assert pm.data["files_to_edit"] == []
    assert pm.data["files_to_create"] == []
    assert pm.data["work_strategy"] == ""
    assert pm.data["work_auto_mode"] is False
    assert pm.data["tdd_mode"] is False


def test_create_accepts_custom_params(pm):
    pm.create("Obj", status="planning", mode="work", auto_mode=True, tdd_mode=True)
    assert pm.data["status"] == "planning"
    assert pm.data["mode"] == "work"
    assert pm.data["work_auto_mode"] is True
    assert pm.data["tdd_mode"] is True


# ─── 2. active returns False when no project file, True after create ─────

def test_active_false_when_no_project_file(pm):
    assert pm.active is False


def test_active_true_after_create(pm):
    pm.create("Objective")
    assert pm.active is True


# ─── 3. destroy removes file and sets data to None ───────────────────────

def test_destroy_removes_file_and_sets_data_none(pm):
    pm.create("Objective")
    assert pm._project_file.exists()
    pm.destroy()
    assert not pm._project_file.exists()
    assert pm.data is None
    assert pm.active is False


def test_destroy_when_no_file_is_noop(pm):
    # Should not raise even if file does not exist.
    pm.destroy()
    assert pm.data is None


# ─── 4. save_research_data stores content ───────────────────────────────

def test_save_research_data_stores_content(pm):
    pm.create("Objective")
    pm.save_research_data("synthesis report text")
    assert pm.data["research_data"] == "synthesis report text"
    # Persisted to disk
    raw = json.loads(pm._project_file.read_text(encoding="utf-8"))
    assert raw["research_data"] == "synthesis report text"


# ─── 5. set_architecture / get_architecture roundtrip ───────────────────

def test_set_get_architecture_roundtrip(pm):
    pm.create("Objective")
    pm.set_architecture("arch map text", files=["a.py", "b.py"])
    assert pm.get_architecture() == "arch map text"
    assert pm.data["architecture_files"] == ["a.py", "b.py"]


def test_set_architecture_without_files(pm):
    pm.create("Objective")
    pm.set_architecture("arch map text")
    assert pm.get_architecture() == "arch map text"
    # architecture_files should not be created when files omitted
    assert "architecture_files" not in pm.data


# ─── 6. has_architecture False initially, True after setting ─────────────

def test_has_architecture_false_initially(pm):
    pm.create("Objective")
    assert pm.has_architecture() is False


def test_has_architecture_true_after_setting(pm):
    pm.create("Objective")
    pm.set_architecture("arch map text")
    assert pm.has_architecture() is True


def test_has_architecture_false_for_whitespace_only(pm):
    pm.create("Objective")
    pm.set_architecture("   \n  ")
    assert pm.has_architecture() is False


# ─── 7. set_file_spec / get_file_spec roundtrip ──────────────────────────

def test_set_get_file_spec_roundtrip(pm):
    pm.create("Objective")
    pm.set_file_spec("src/calculator.py", "detailed spec")
    assert pm.get_file_spec("src/calculator.py") == "detailed spec"


def test_get_file_spec_missing_returns_empty(pm):
    pm.create("Objective")
    assert pm.get_file_spec("nonexistent.py") == ""


def test_set_file_spec_creates_file_specs_dict_if_missing(pm):
    pm.create("Objective")
    del pm.data["file_specs"]
    pm.set_file_spec("a.py", "spec")
    assert pm.data["file_specs"] == {"a.py": "spec"}


# ─── 8. get_pending_spec_files returns files without specs ───────────────

def test_get_pending_spec_files_returns_files_without_specs(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py", "b.py", "c.py"])
    pm.set_file_spec("a.py", "spec for a")
    pending = pm.get_pending_spec_files()
    assert set(pending) == {"b.py", "c.py"}


def test_get_pending_spec_files_empty_when_no_architecture_files(pm):
    pm.create("Objective")
    assert pm.get_pending_spec_files() == []


def test_get_pending_spec_files_empty_when_all_specified(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py", "b.py"])
    pm.set_file_spec("a.py", "spec a")
    pm.set_file_spec("b.py", "spec b")
    assert pm.get_pending_spec_files() == []


# ─── 9. has_pending_specs True when pending, False when all done ─────────

def test_has_pending_specs_true_when_pending(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py", "b.py"])
    pm.set_file_spec("a.py", "spec a")
    assert pm.has_pending_specs() is True


def test_has_pending_specs_false_when_all_done(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py", "b.py"])
    pm.set_file_spec("a.py", "spec a")
    pm.set_file_spec("b.py", "spec b")
    assert pm.has_pending_specs() is False


def test_has_pending_specs_false_when_no_files(pm):
    pm.create("Objective")
    assert pm.has_pending_specs() is False


# ─── 10. all_specs_done True when no pending and specs exist ──────────────

def test_all_specs_done_true_when_no_pending_and_specs_exist(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py", "b.py"])
    pm.set_file_spec("a.py", "spec a")
    pm.set_file_spec("b.py", "spec b")
    assert pm.all_specs_done() is True


def test_all_specs_done_false_when_pending_exists(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py", "b.py"])
    pm.set_file_spec("a.py", "spec a")
    assert pm.all_specs_done() is False


def test_all_specs_done_false_when_no_specs_at_all(pm):
    pm.create("Objective")
    pm.set_architecture("arch", files=["a.py"])
    # No specs written at all → pending == ["a.py"], so not done
    assert pm.all_specs_done() is False


# ─── 11. add_task returns incrementing IDs ──────────────────────────────

def test_add_task_returns_incrementing_ids(pm):
    pm.create("Objective")
    id1 = pm.add_task("task one")
    id2 = pm.add_task("task two")
    id3 = pm.add_task("task three")
    assert id1 == 1
    assert id2 == 2
    assert id3 == 3


def test_add_task_stores_description_and_target_files(pm):
    pm.create("Objective")
    pm.add_task("do something", target_files=["x.py", "y.py"])
    task = pm.data["tasks"][0]
    assert task["description"] == "do something"
    assert task["target_files"] == ["x.py", "y.py"]
    assert task["status"] == "pending"
    assert task["result_summary"] == ""
    assert task["files_affected"] == []


def test_add_task_default_target_files_empty(pm):
    pm.create("Objective")
    pm.add_task("do something")
    assert pm.data["tasks"][0]["target_files"] == []


# ─── 12. complete_task marks status and stores summary ──────────────────

def test_complete_task_marks_status_and_stores_summary(pm):
    pm.create("Objective")
    tid = pm.add_task("task one")
    pm.complete_task(tid, "did the thing")
    task = pm.data["tasks"][0]
    assert task["status"] == "completed"
    assert task["result_summary"] == "did the thing"


def test_complete_task_without_files_leaves_files_affected_empty(pm):
    pm.create("Objective")
    tid = pm.add_task("task one")
    pm.complete_task(tid, "summary")
    assert pm.data["tasks"][0]["files_affected"] == []


# ─── 13. complete_task adds files to files_created ───────────────────────

def test_complete_task_adds_files_to_files_created(pm):
    pm.create("Objective")
    tid = pm.add_task("task one")
    pm.complete_task(tid, "summary", files=["new_a.py", "new_b.py"])
    assert "new_a.py" in pm.data["files_created"]
    assert "new_b.py" in pm.data["files_created"]


def test_complete_task_does_not_duplicate_files_created(pm):
    pm.create("Objective")
    pm.data["files_created"] = ["existing.py"]
    tid = pm.add_task("task one")
    pm.complete_task(tid, "summary", files=["existing.py", "new.py"])
    assert pm.data["files_created"].count("existing.py") == 1
    assert "new.py" in pm.data["files_created"]


# ─── 14. get_next_pending returns first pending task ─────────────────────

def test_get_next_pending_returns_first_pending(pm):
    pm.create("Objective")
    pm.add_task("task one")
    pm.add_task("task two")
    pm.add_task("task three")
    nxt = pm.get_next_pending()
    assert nxt is not None
    assert nxt["id"] == 1
    assert nxt["description"] == "task one"


def test_get_next_pending_skips_completed(pm):
    pm.create("Objective")
    pm.add_task("task one")
    pm.add_task("task two")
    pm.complete_task(1, "done one")
    nxt = pm.get_next_pending()
    assert nxt is not None
    assert nxt["id"] == 2


def test_get_next_pending_returns_none_when_all_done(pm):
    pm.create("Objective")
    pm.add_task("task one")
    pm.complete_task(1, "done")
    assert pm.get_next_pending() is None


def test_get_next_pending_returns_none_when_no_tasks(pm):
    pm.create("Objective")
    assert pm.get_next_pending() is None


# ─── 15. has_pending True when pending exists ───────────────────────────

def test_has_pending_true_when_pending_exists(pm):
    pm.create("Objective")
    pm.add_task("task one")
    assert pm.has_pending() is True


def test_has_pending_false_when_no_tasks(pm):
    pm.create("Objective")
    assert pm.has_pending() is False


def test_has_pending_false_when_all_done(pm):
    pm.create("Objective")
    pm.add_task("task one")
    pm.complete_task(1, "done")
    assert pm.has_pending() is False


# ─── 16. is_complete True when all tasks done ────────────────────────────

def test_is_complete_true_when_all_tasks_done(pm):
    pm.create("Objective")
    pm.add_task("task one")
    pm.add_task("task two")
    pm.complete_task(1, "done one")
    pm.complete_task(2, "done two")
    assert pm.is_complete() is True


def test_is_complete_false_when_pending_exists(pm):
    pm.create("Objective")
    pm.add_task("task one")
    pm.add_task("task two")
    pm.complete_task(1, "done one")
    assert pm.is_complete() is False


def test_is_complete_false_when_no_tasks(pm):
    pm.create("Objective")
    assert pm.is_complete() is False


# ─── 17. set_status updates status ──────────────────────────────────────

def test_set_status_updates_status(pm):
    pm.create("Objective")
    assert pm.data["status"] == "specifying"
    pm.set_status("implementing")
    assert pm.data["status"] == "implementing"
    # Persisted to disk
    raw = json.loads(pm._project_file.read_text(encoding="utf-8"))
    assert raw["status"] == "implementing"


# ─── 18. _load returns None for missing file ─────────────────────────────

def test_load_returns_none_for_missing_file(tmp_path, monkeypatch):
    project_file = tmp_path / ".argent_project.json"
    monkeypatch.setattr(
        project_manager, "resolve_project_file", lambda: project_file
    )
    mgr = project_manager.ProjectManager()
    assert mgr.data is None


# ─── 19. _load returns None for corrupt JSON ──────────────────────────────

def test_load_returns_none_for_corrupt_json(tmp_path, monkeypatch):
    project_file = tmp_path / ".argent_project.json"
    project_file.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(
        project_manager, "resolve_project_file", lambda: project_file
    )
    mgr = project_manager.ProjectManager()
    assert mgr.data is None


# ─── Bonus: persistence across instances ────────────────────────────────

def test_state_persists_across_instances(tmp_path, monkeypatch):
    project_file = tmp_path / ".argent_project.json"
    monkeypatch.setattr(
        project_manager, "resolve_project_file", lambda: project_file
    )
    mgr1 = project_manager.ProjectManager()
    mgr1.create("Persistent objective")
    mgr1.add_task("persistent task")

    mgr2 = project_manager.ProjectManager()
    assert mgr2.active is True
    assert mgr2.data["objective"] == "Persistent objective"
    assert len(mgr2.data["tasks"]) == 1
    assert mgr2.data["tasks"][0]["description"] == "persistent task"