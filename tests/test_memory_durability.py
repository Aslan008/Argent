"""The memory file is what survives a context reset — it must survive a crash.

memory.json holds the objective, the completed steps and the files touched:
everything a small model still knows after its history is compacted away. It
was written with a plain write_text, which truncates the target before it
writes, so a process killed mid-write left a half file. _load then swallowed
the parse error and returned blank defaults, and the next _save wrote those
defaults over the damage.

The result looked exactly like the model forgetting, with nothing to
distinguish it from one. atomic_write_text has been in the project for this
since config.py and session.py adopted it.
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


class TestDamagedMemory:
    def test_a_truncated_file_is_kept(self, memory):
        memory.parent.mkdir(parents=True)
        memory.write_text('{"objective": "ship the release", "completed": ["a"', encoding="utf-8")

        manager = MemoryManager()
        assert manager.data["objective"] == ""          # blank, but not silently

        backups = list(memory.parent.glob("memory.json.broken-*"))
        assert len(backups) == 1
        assert "ship the release" in backups[0].read_text(encoding="utf-8")

    def test_the_next_save_cannot_overwrite_it(self, memory):
        memory.parent.mkdir(parents=True)
        memory.write_text('{"objective": "ship the release"', encoding="utf-8")
        manager = MemoryManager()
        manager.set_objective("something else")

        backup = next(memory.parent.glob("memory.json.broken-*"))
        assert "ship the release" in backup.read_text(encoding="utf-8")
        assert json.loads(memory.read_text(encoding="utf-8"))["objective"] == "something else"

    def test_it_is_recorded(self, memory, caplog):
        memory.parent.mkdir(parents=True)
        memory.write_text("{oops", encoding="utf-8")
        with caplog.at_level("WARNING", logger="memory"):
            MemoryManager()
        assert any("unreadable" in r.message for r in caplog.records)

    def test_json_of_the_wrong_shape_is_damage_too(self, memory):
        memory.parent.mkdir(parents=True)
        memory.write_text("[]", encoding="utf-8")
        manager = MemoryManager()
        assert manager.data["completed"] == []          # a list would crash add_completed
        manager.add_completed("still works")


class TestTheWriteIsAtomic:
    def test_save_goes_through_atomic_write(self, memory, monkeypatch):
        """A partial memory.json is indistinguishable from a forgetful model."""
        seen = []
        real = memory_manager.atomic_write_text
        monkeypatch.setattr(memory_manager, "atomic_write_text",
                            lambda p, t, **k: seen.append(p) or real(p, t, **k))
        MemoryManager().set_objective("x")
        assert seen == [memory]

    def test_no_temp_file_is_left_behind(self, memory):
        manager = MemoryManager()
        manager.set_objective("x")
        manager.add_completed("y")
        assert [p.name for p in memory.parent.iterdir()] == ["memory.json"]

    def test_the_content_round_trips(self, memory):
        manager = MemoryManager()
        manager.set_objective("собрать релиз")
        manager.add_file_modified("src/main.py")
        reloaded = json.loads(memory.read_text(encoding="utf-8"))
        assert reloaded["objective"] == "собрать релиз"
        assert "src/main.py" in reloaded["files_modified"]
