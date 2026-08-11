"""Blind-spot tests for MemoryManager.

These exercise the truncation limits, dedup logic, caps, empty-input
guards, context-note assembly, clear, and timestamp behaviour that the
durability suite does not cover.
"""

import json

import pytest

import memory_manager
from memory_manager import MemoryManager


@pytest.fixture
def memory(tmp_path, monkeypatch):
    target = tmp_path / ".argent" / "memory.json"
    monkeypatch.setattr(memory_manager, "resolve_memory_file", lambda: target)
    return target


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestObjectiveTruncation:
    def test_objective_is_truncated_to_500(self, memory):
        manager = MemoryManager()
        manager.set_objective("x" * 600)
        assert len(manager.data["objective"]) == 500


class TestCurrentTaskTruncation:
    def test_current_task_is_truncated_to_300(self, memory):
        manager = MemoryManager()
        manager.set_current_task("x" * 400)
        assert len(manager.data["current_task"]) == 300


class TestFactTruncation:
    def test_fact_is_truncated_to_300(self, memory):
        manager = MemoryManager()
        manager.add_fact("x" * 400)
        assert len(manager.data["key_facts"][0]) == 300


class TestErrorTruncation:
    def test_error_is_truncated_to_200(self, memory):
        manager = MemoryManager()
        manager.add_error("x" * 300)
        assert len(manager.data["errors_encountered"][0]) == 200


class TestCompletedTruncation:
    def test_completed_is_truncated_to_200(self, memory):
        manager = MemoryManager()
        manager.add_completed("x" * 300)
        assert len(manager.data["completed"][0]) == 200


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

class TestCompletedDedup:
    def test_duplicate_within_last_three_is_rejected(self, memory):
        manager = MemoryManager()
        manager.add_completed("action1")
        manager.add_completed("action1")  # in last 3 -> rejected
        assert len(manager.data["completed"]) == 1

        manager.add_completed("action2")
        manager.add_completed("action3")
        # last 3 are now [action1, action2, action3] -> action1 still rejected
        manager.add_completed("action1")
        assert len(manager.data["completed"]) == 3

        manager.add_completed("action4")
        # last 3 are now [action2, action3, action4] -> action1 accepted
        manager.add_completed("action1")
        assert len(manager.data["completed"]) == 5
        assert manager.data["completed"][-1] == "action1"


class TestErrorsDedup:
    def test_duplicate_error_within_last_five_is_rejected(self, memory):
        manager = MemoryManager()
        manager.add_error("err1")
        manager.add_error("err1")  # in last 5 -> rejected
        assert len(manager.data["errors_encountered"]) == 1


class TestFileModifiedDedup:
    def test_duplicate_file_is_not_added(self, memory):
        manager = MemoryManager()
        manager.add_file_modified("a.py")
        manager.add_file_modified("a.py")
        assert len(manager.data["files_modified"]) == 1


class TestFactDedup:
    def test_duplicate_fact_is_not_added(self, memory):
        manager = MemoryManager()
        manager.add_fact("fact1")
        manager.add_fact("fact1")
        assert len(manager.data["key_facts"]) == 1


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

class TestCompletedCap:
    def test_completed_capped_at_20(self, memory):
        manager = MemoryManager()
        for i in range(25):
            manager.add_completed(f"action_{i}")
        assert len(manager.data["completed"]) == 20
        # the last 20 should be kept
        assert manager.data["completed"][0] == "action_5"
        assert manager.data["completed"][-1] == "action_24"


class TestFilesModifiedCap:
    def test_files_modified_capped_at_30(self, memory):
        manager = MemoryManager()
        for i in range(35):
            manager.add_file_modified(f"file_{i}.py")
        assert len(manager.data["files_modified"]) == 30
        assert manager.data["files_modified"][0] == "file_5.py"
        assert manager.data["files_modified"][-1] == "file_34.py"


class TestFactsCap:
    def test_facts_capped_at_15(self, memory):
        manager = MemoryManager()
        for i in range(20):
            manager.add_fact(f"fact_{i}")
        assert len(manager.data["key_facts"]) == 15
        assert manager.data["key_facts"][0] == "fact_5"
        assert manager.data["key_facts"][-1] == "fact_19"


class TestErrorsCap:
    def test_errors_capped_at_10(self, memory):
        manager = MemoryManager()
        for i in range(15):
            manager.add_error(f"error_{i}")
        assert len(manager.data["errors_encountered"]) == 10
        assert manager.data["errors_encountered"][0] == "error_5"
        assert manager.data["errors_encountered"][-1] == "error_14"


# ---------------------------------------------------------------------------
# Empty / None guards
# ---------------------------------------------------------------------------

class TestAddCompletedEmpty:
    def test_empty_string_is_ignored(self, memory):
        manager = MemoryManager()
        manager.add_completed("")
        assert len(manager.data["completed"]) == 0


class TestAddCompletedNone:
    def test_none_is_ignored(self, memory):
        manager = MemoryManager()
        manager.add_completed(None)
        assert len(manager.data["completed"]) == 0


# ---------------------------------------------------------------------------
# build_context_note
# ---------------------------------------------------------------------------

class TestBuildContextNoteEmpty:
    def test_fresh_manager_returns_empty_string(self, memory):
        manager = MemoryManager()
        assert manager.build_context_note() == ""


class TestBuildContextNotePartial:
    def test_only_objective_shows_just_objective(self, memory):
        manager = MemoryManager()
        manager.set_objective("ship it")
        note = manager.build_context_note()
        assert "OBJECTIVE:" in note
        assert "CURRENT TASK:" not in note
        assert "COMPLETED ACTIONS:" not in note


class TestBuildContextNoteFull:
    def test_all_sections_present(self, memory):
        manager = MemoryManager()
        manager.set_objective("build the thing")
        manager.set_current_task("writing tests")
        manager.add_completed("did step 1")
        manager.add_file_modified("src/main.py")
        manager.add_fact("discovered X")
        manager.add_error("approach Y failed")
        note = manager.build_context_note()
        assert "OBJECTIVE:" in note
        assert "CURRENT TASK:" in note
        assert "COMPLETED ACTIONS:" in note
        assert "FILES MODIFIED:" in note
        assert "KEY FACTS:" in note
        assert "KNOWN ERRORS" in note


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_resets_all_fields_and_persists(self, memory):
        manager = MemoryManager()
        manager.set_objective("temp objective")
        manager.set_current_task("temp task")
        manager.add_completed("temp action")
        manager.add_file_modified("temp.py")
        manager.add_fact("temp fact")
        manager.add_error("temp error")

        manager.clear()

        assert manager.data["objective"] == ""
        assert manager.data["current_task"] == ""
        assert manager.data["completed"] == []
        assert manager.data["files_modified"] == []
        assert manager.data["key_facts"] == []
        assert manager.data["errors_encountered"] == []

        # verify the file on disk reflects the cleared state
        on_disk = json.loads(memory.read_text(encoding="utf-8"))
        assert on_disk["objective"] == ""
        assert on_disk["completed"] == []
        assert on_disk["files_modified"] == []
        assert on_disk["key_facts"] == []
        assert on_disk["errors_encountered"] == []


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

class TestSaveUpdatesTimestamp:
    def test_updated_at_is_set_on_save(self, memory):
        manager = MemoryManager()
        manager.set_objective("x")
        on_disk = json.loads(memory.read_text(encoding="utf-8"))
        assert on_disk["updated_at"]
        assert on_disk["updated_at"] != ""