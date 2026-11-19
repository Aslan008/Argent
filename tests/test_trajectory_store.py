"""The trajectory store: append-only, bounded, and readable.

A turn used to update its trajectory by loading the whole JSONL and rewriting it — at
797 records (7.9 MB) that was tens of megabytes of I/O per event, and it grows with the
file. What is pinned here is the property that replaced it: a turn writes one record and
then only small patch lines, and reading applies them.
"""

import json

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    path = tmp_path / "trajectories.jsonl"
    monkeypatch.setenv("LEETH_TRAJECTORY_PATH", str(path))
    monkeypatch.setenv("LEETH_TRAJECTORY_FORCE", "1")
    from src.leeth import trajectory

    trajectory._local.stack = []
    trajectory._known_ids.clear()
    yield trajectory, path
    trajectory._local.stack = []
    trajectory._known_ids.clear()


def _lines(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestAppendOnly:
    def test_a_turn_writes_one_record_then_only_patches(self, store):
        trajectory, path = store
        traj_id = trajectory.start_trajectory("почини тест")
        first_line = _lines(path)[0]
        size_after_start = path.stat().st_size

        for i in range(30):
            trajectory.record_event("tool_exec", tool="read_file", ok=True, detail=f"шаг {i}")

        lines = _lines(path)
        assert len(lines) == 31, "одна запись плюс по строке на событие"
        assert lines[0] == first_line, "запись хода не переписывается"
        assert all("patch" in json.loads(line) for line in lines[1:])
        # 30 small lines, not 30 rewrites of everything that came before.
        assert path.stat().st_size - size_after_start < 30 * 700

    def test_closing_a_turn_is_a_patch_too(self, store):
        trajectory, path = store
        trajectory.start_trajectory("вопрос")
        before = len(_lines(path))

        trajectory.end_trajectory("completed")

        lines = _lines(path)
        assert len(lines) == before + 1
        assert json.loads(lines[-1])["set"]["status"] == "completed"

    def test_reading_merges_the_patches(self, store):
        trajectory, path = store
        traj_id = trajectory.start_trajectory("собери отчёт")
        trajectory.record_event("tool_exec", tool="grep_search", ok=True)
        trajectory.attach_decision("dec_1")
        trajectory.record_event("command_exit", command="pytest -q", code=1)
        trajectory.end_trajectory("completed")

        rec = trajectory.get_trajectory(traj_id)

        assert [e["type"] for e in rec["events"]] == ["tool_exec", "command_exit"]
        assert rec["decision_ids"] == ["dec_1"]
        assert rec["status"] == "completed"
        assert rec["end_ts"] is not None


class TestTolerance:
    def test_a_corrupt_line_costs_one_record_not_the_file(self, store):
        trajectory, path = store
        trajectory.start_trajectory("первый")
        trajectory.end_trajectory("completed")
        with open(path, "a", encoding="utf-8") as f:
            f.write("{половина строки\n")
        trajectory.start_trajectory("второй")
        trajectory.end_trajectory("completed")

        records = trajectory._load_all()

        assert len(records) == 2, "две целые записи пережили мусорную строку"

    def test_a_patch_without_its_record_is_ignored(self, store):
        trajectory, path = store
        trajectory.start_trajectory("целый")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"patch": "tr_нет", "set": {"status": "x"}}) + "\n")

        records = trajectory._load_all()

        assert len(records) == 1
        assert records[0]["status"] == "open"

    def test_an_unknown_trajectory_is_refused_rather_than_written(self, store):
        trajectory, path = store
        lines_before = len(_lines(path)) if path.exists() else 0

        trajectory._local.stack = ["tr_чужой"]        # as if restored from a dead process
        wrote = trajectory.record_event("tool_exec", tool="x")
        trajectory._local.stack = []

        assert wrote is False, "событие для чужой траектории не пишется"
        assert (len(_lines(path)) if path.exists() else 0) == lines_before


class TestBounds:
    @staticmethod
    def _write_records(path, queries):
        """A store built by hand: compaction runs on start_trajectory, so a test that
        grows it through the API would compact on every step and measure the wrong thing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, query in enumerate(queries):
                f.write(json.dumps({
                    "id": f"tr_{i}", "ts": 1000 + i, "query": query,
                    "status": "completed", "events": [], "decision_ids": [],
                }) + "\n")

    def test_compaction_keeps_the_newest_and_archives_the_rest(self, store, monkeypatch):
        trajectory, path = store
        self._write_records(path, [f"ход {i}" for i in range(5)])
        monkeypatch.setattr(trajectory, "COMPACT_AT_BYTES", 1)
        monkeypatch.setattr(trajectory, "KEEP_RECORDS", 2)

        trajectory._maybe_compact()

        kept = trajectory._load_all()
        assert len(kept) == 2
        assert [r["query"] for r in kept] == ["ход 3", "ход 4"], "остаются самые новые"

        archived = sorted((path.parent / "archive").glob("*.jsonl"))
        assert len(archived) == 1
        assert len(_lines(archived[0])) == 5, "архив содержит всё, что было"

    def test_the_number_of_archives_is_capped(self, store, monkeypatch):
        trajectory, path = store
        self._write_records(path, ["ход"])
        monkeypatch.setattr(trajectory, "COMPACT_AT_BYTES", 1)
        monkeypatch.setattr(trajectory, "KEEP_RECORDS", 1)
        monkeypatch.setattr(trajectory, "ARCHIVE_KEEP", 2)
        archive = path.parent / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        for i in range(4):
            (archive / f"{path.stem}-2020010{i}-000000{path.suffix}").write_text("{}\n", encoding="utf-8")

        trajectory._maybe_compact()

        assert len(list(archive.glob("*.jsonl"))) <= 2


class TestMigration:
    def test_the_old_in_tree_store_is_moved_once(self, tmp_path, monkeypatch):
        from src.leeth import trajectory

        monkeypatch.delenv("LEETH_TRAJECTORY_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        legacy = tmp_path / trajectory.DEFAULT_STORE_PATH
        legacy.parent.mkdir(parents=True)
        legacy.write_text('{"id": "tr_old", "ts": 1, "status": "open"}\n', encoding="utf-8")

        first = trajectory._store_path()
        second = trajectory._store_path()

        assert first == second
        assert first.name == trajectory.STORE_NAME
        assert first.parent.name == "leeth"
        assert first.exists() and first.read_text(encoding="utf-8").startswith('{"id": "tr_old"')
        assert not legacy.exists(), "старый файл переехал, а не скопировался"

    def test_an_explicit_path_is_never_second_guessed(self, tmp_path, monkeypatch):
        from src.leeth import trajectory

        explicit = tmp_path / "custom" / "mine.jsonl"
        monkeypatch.setenv("LEETH_TRAJECTORY_PATH", str(explicit))

        assert trajectory._store_path() == explicit


class TestSecrets:
    def test_a_key_in_an_event_never_reaches_the_file(self, store):
        trajectory, path = store
        key = "sk-proj-" + "a1b2c3d4" * 6
        trajectory.start_trajectory("деплой")

        trajectory.record_event("command_exit", command=f"deploy --token {key}", code=0)

        raw = path.read_text(encoding="utf-8")
        assert key not in raw, "ключ не должен оседать в долгом логе"
