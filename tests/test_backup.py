"""/backup: what goes in (an allowlist), that keys and heavy folders stay out, and that restoring can
neither write outside the allowed places nor lose what it replaces."""

import json
import zipfile

import pytest

from src.project import backup


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An Argent folder with rooms and a pile of things that must never be backed up."""
    argent = tmp_path / "argent" / ".argent"
    monkeypatch.setenv("ARGENT_ROOMS_DIR", str(argent / "rooms"))
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda *a, **k: project)
    home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))

    (argent / "rooms" / ".history" / "r1").mkdir(parents=True)
    (argent / "rooms" / "r1.json").write_text('{"room": "r1"}', encoding="utf-8")
    (argent / "rooms" / ".history" / "r1" / "20260101-000000-000.json").write_text("{}", encoding="utf-8")
    (argent / "rooms_stats.json").write_text("{}", encoding="utf-8")
    (argent / "automations.json").write_text("[]", encoding="utf-8")
    (argent / "webhooks.json").write_text("{}", encoding="utf-8")
    # things that must stay out
    (argent / "models").mkdir()
    (argent / "models" / "weights.bin").write_bytes(b"x" * 1000)
    (argent / "chroma_db").mkdir()
    (argent / "chroma_db" / "db.sqlite").write_bytes(b"y")
    (argent / "web_cache.db").write_bytes(b"z")
    (argent / ".env").write_text("SECRET=1", encoding="utf-8")
    (project / ".argent" / "memory").mkdir(parents=True)
    (project / ".argent" / "memory" / "note.md").write_text("---\nname: note\n---\nhi", encoding="utf-8")
    (project / ".argent" / "memory" / ".snapshots" / "s1").mkdir(parents=True)
    (project / ".argent" / "memory" / ".snapshots" / "s1" / "old.md").write_text("old", encoding="utf-8")
    (home / ".argent" / "memory").mkdir(parents=True)
    (home / ".argent" / "memory" / "g.md").write_text("global", encoding="utf-8")
    return tmp_path


def names(path):
    with zipfile.ZipFile(path) as zf:
        return set(zf.namelist())


def test_the_archive_holds_the_allowlist_and_nothing_heavy_or_secret(world):
    path, count, _ = backup.create()
    got = names(path)
    assert {"argent/rooms/r1.json", "argent/rooms/.history/r1/20260101-000000-000.json",
            "argent/rooms_stats.json", "argent/automations.json", "argent/webhooks.json",
            "project_memory/note.md", "global_memory/g.md", "manifest.json"} <= got
    assert not any(n.startswith(("argent/models", "argent/chroma_db")) for n in got)
    assert not any(n.endswith((".env", "web_cache.db")) for n in got)
    assert not any(".snapshots" in n for n in got)
    assert count == len(got) - 1


def test_a_backup_is_never_put_inside_the_next_one(world):
    backup.create()
    second, _, _ = backup.create()
    assert not any("backups/" in n for n in names(second))


def test_restore_puts_the_files_back_and_saves_what_it_replaces(world):
    path, _, _ = backup.create()
    argent = world / "argent" / ".argent"
    (argent / "rooms" / "r1.json").write_text('{"room": "CHANGED"}', encoding="utf-8")
    (argent / "automations.json").unlink()
    count, safety = backup.restore(path, confirm=lambda text: True)
    assert count > 0
    assert json.loads((argent / "rooms" / "r1.json").read_text(encoding="utf-8"))["room"] == "r1"
    assert (argent / "automations.json").is_file()
    assert safety is not None and safety.is_file()
    with zipfile.ZipFile(safety) as zf:                       # the edit we just overwrote is in the safety copy
        assert "CHANGED" in zf.read("argent/rooms/r1.json").decode("utf-8")


def test_saying_no_changes_nothing(world):
    path, _, _ = backup.create()
    argent = world / "argent" / ".argent"
    (argent / "rooms" / "r1.json").write_text("EDIT", encoding="utf-8")
    assert backup.restore(path, confirm=lambda text: False) == (0, None)
    assert (argent / "rooms" / "r1.json").read_text(encoding="utf-8") == "EDIT"


def _evil(tmp_path, member, label="argent"):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(member, "pwned")
        zf.writestr("manifest.json", json.dumps({"format": 1, "created": "x", "roots": {label: [member.split("/", 1)[1]]}}))
    return archive


@pytest.mark.parametrize("member", [
    "argent/../../evil.txt", "argent/rooms/../../evil.txt", "argent/C:/evil.txt",
    "argent/models/weights.bin", "argent/.env", "argent/run_pytest.bat",
])
def test_a_path_outside_the_allowlist_is_refused_before_anything_is_written(world, member):
    archive = _evil(world, member)
    with pytest.raises(backup.BackupError):
        backup.restore(archive, confirm=lambda text: True)
    assert not (world / "evil.txt").exists()
    assert (world / "argent" / ".argent" / "models" / "weights.bin").read_bytes() == b"x" * 1000


def test_an_unknown_part_and_a_foreign_zip_are_refused(world, tmp_path):
    with pytest.raises(backup.BackupError):
        backup.restore(_evil(tmp_path, "mystery/x.txt", label="mystery"), confirm=lambda t: True)
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(backup.BackupError):
        backup.inspect(other)
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    with pytest.raises(backup.BackupError):
        backup.inspect(junk)


def test_the_command_is_registered_and_lists(world, capsys):
    from src.cli.commands import registry
    from src.cli.commands.base import CommandContext
    res = registry.dispatch("/backup", CommandContext(agent=None))
    assert res.handled
    assert len(backup.list_backups()) == 1
    assert registry.dispatch("/backup list", CommandContext(agent=None)).handled
