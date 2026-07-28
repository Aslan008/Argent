"""RAG garbage collection: a file that no longer exists must stop being
searchable, or the model gets snippets of code that isn't there any more."""

import pytest

import rag_engine


class FakeCollection:
    """Minimal ChromaDB stand-in that records deletes and holds metadata."""

    def __init__(self, files=()):
        self.deleted = []
        self.upserts = 0
        self._files = list(files)

    def delete(self, where=None):
        self.deleted.append(where)
        target = (where or {}).get("file")
        self._files = [f for f in self._files if f != target]

    def get(self, include=None):
        return {"metadatas": [{"file": f} for f in self._files]}

    def upsert(self, **kw):
        self.upserts += 1


@pytest.fixture
def rag(monkeypatch, tmp_path):
    """RAG enabled against a fake collection, rooted at a tmp project."""
    home = tmp_path / "home"
    home.mkdir()
    root = home / "proj"
    (root / ".argent").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))
    monkeypatch.chdir(root)

    col = FakeCollection()
    monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
    monkeypatch.setattr(rag_engine, "_COLLECTION", col)
    return root, col


class TestUpdateFileIndex:
    def test_missing_file_drops_its_chunks(self, rag):
        root, col = rag
        ghost = root / "src" / "gone.py"

        rag_engine.update_file_index(str(ghost))

        # It used to return silently, leaving the chunks searchable forever.
        assert {"file": str(ghost.relative_to(root))} in col.deleted
        assert col.upserts == 0

    def test_existing_file_is_reindexed(self, rag):
        root, col = rag
        f = root / "mod.py"
        f.write_text("def a():\n    return 1\n", encoding="utf-8")

        rag_engine.update_file_index(str(f))
        assert {"file": "mod.py"} in col.deleted     # old chunks cleared
        assert col.upserts == 1                       # new chunks added

    def test_unresolvable_root_is_a_no_op(self, monkeypatch, tmp_path):
        col = FakeCollection()
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", True)
        monkeypatch.setattr(rag_engine, "_COLLECTION", col)
        monkeypatch.setattr("project_paths.find_project_root", lambda start=None: None)
        rag_engine.update_file_index(str(tmp_path / "x.py"))
        assert col.deleted == []

    def test_disabled_rag_is_a_no_op(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        rag_engine.update_file_index(str(tmp_path / "x.py"))    # must not raise


class TestRemoveFileIndex:
    def test_removes_by_relative_path(self, rag):
        root, col = rag
        f = root / "pkg" / "old.py"
        f.parent.mkdir()
        f.write_text("x = 1\n", encoding="utf-8")

        rag_engine.remove_file_index(str(f))
        assert col.deleted == [{"file": str(f.relative_to(root))}]

    def test_delete_file_tool_drops_the_index(self, rag, monkeypatch):
        root, col = rag
        import tools.file_ops as file_ops
        monkeypatch.setattr(file_ops, "snapshot", lambda *a, **k: True)
        monkeypatch.setattr(file_ops, "memory", type("M", (), {"add_completed": staticmethod(lambda *a: None)})())
        monkeypatch.setattr("approval.request_approval", lambda *a, **k: True)

        f = root / "doomed.py"
        f.write_text("x = 1\n", encoding="utf-8")
        result = file_ops.delete_file(str(f))

        assert result.startswith("Successfully deleted")
        assert {"file": "doomed.py"} in col.deleted


class TestPruneDeletedFiles:
    def test_prunes_only_vanished_files(self, rag):
        root, col = rag
        alive = root / "alive.py"
        alive.write_text("x = 1\n", encoding="utf-8")
        col._files = ["alive.py", "ghost.py", "sub/renamed_away.py"]

        pruned = rag_engine.prune_deleted_files()

        assert pruned == 2
        assert col.deleted == [{"file": "ghost.py"}, {"file": "sub/renamed_away.py"}] or \
               sorted(d["file"] for d in col.deleted) == ["ghost.py", "sub/renamed_away.py"]

    def test_nothing_to_prune(self, rag):
        root, col = rag
        (root / "a.py").write_text("x = 1\n", encoding="utf-8")
        col._files = ["a.py"]
        assert rag_engine.prune_deleted_files() == 0
        assert col.deleted == []

    def test_disabled_rag_returns_zero(self, monkeypatch):
        monkeypatch.setattr(rag_engine, "_RAG_ENABLED", False)
        monkeypatch.setattr(rag_engine, "_COLLECTION", None)
        assert rag_engine.prune_deleted_files() == 0
