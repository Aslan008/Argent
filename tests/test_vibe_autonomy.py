"""/vibe must be able to work unattended: it deletes what it can put back.

The mode's job is not to refuse deletes — an agent that cannot clear its own temp
or rebuild a directory cannot work while the person sleeps. The job is to make a
delete recoverable, and to refuse exactly what cannot be recovered: anything
outside the project, Argent's own state, a target it cannot name, and a delete
too large to copy.
"""

from pathlib import Path

import pytest

import argent.approval as approval
import argent.trash as trash


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project: every path helper follows it, so nothing is real."""
    root = tmp_path / "proj"
    (root / ".argent" / "artifacts").mkdir(parents=True)
    (root / ".argent" / "checkpoints").mkdir(parents=True)
    (root / "src").mkdir()
    monkeypatch.setattr("argent.project_paths.project_root_or_cwd", lambda *a, **k: str(root))
    # Only this project's own scratch counts here: pytest's tmp_path lives under
    # the OS temp dir, so the real scratch roots would make every "outside" file
    # look like housekeeping.
    monkeypatch.setattr(approval, "_scratch_roots", lambda: [
        root / ".argent" / "artifacts",
        root / ".argent" / "tmp",
        root / ".argent" / "scratch",
    ])
    yield root
    trash.use_root(None)


@pytest.fixture
def vibe():
    approval.set_policy(approval.POLICY_AUTO)
    yield
    approval.set_policy(approval.POLICY_ASK)


def run(command: str) -> bool:
    return approval.request_approval(
        f"{approval.RUN_COMMAND_ACTION}{command}",
        destructive=approval.is_destructive_command(command),
        grant_key=None)


# --- reading the target out of a variable --------------------------------------

def test_a_delete_target_written_as_a_variable_is_resolved():
    targets = approval._delete_targets('$tmp = ".argent/artifacts/probe.txt"; Remove-Item -Recurse $tmp')
    assert targets == [".argent/artifacts/probe.txt"]


def test_a_variable_is_resolved_for_the_whole_path_not_only_the_name():
    targets = approval._delete_targets('$d = ".argent/artifacts"; Remove-Item "$d/nested/x.bin"')
    assert targets == [".argent/artifacts/nested/x.bin"]


def test_a_computed_value_stays_unreadable():
    # `$p = $env:TEMP + "\\x"` is not a literal, so the target stays unknown and
    # the delete keeps being refused — the old behaviour, on purpose.
    targets = approval._delete_targets('$p = $env:TEMP; Remove-Item $p')
    assert targets == ["$p"]


# --- housekeeping that needs no copy ------------------------------------------

def test_cleaning_scratch_written_through_a_variable_runs(project, vibe):
    probe = project / ".argent" / "artifacts" / "probe.txt"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text("x", encoding="utf-8")

    assert run('$tmp = ".argent/artifacts/probe.txt"; Remove-Item -Recurse $tmp') is True
    assert trash.list_snapshots() == [], "scratch не нужно копировать"


def test_derived_data_is_removed_without_a_copy(project, vibe):
    (project / "build").mkdir()
    (project / "build" / "artifact.bin").write_bytes(b"x" * 100)

    assert run("Remove-Item -Recurse -Force build") is True
    assert trash.list_snapshots() == [], "build можно собрать снова"


def test_a_delete_whose_target_is_already_gone_runs(project, vibe):
    assert run("Remove-Item -Recurse -Force .argent/artifacts/never-existed") is True


def test_the_temp_variable_still_means_the_temp_directory(project, vibe, monkeypatch):
    import tempfile
    # `$env:TEMP` is housekeeping wherever it is written as a variable, and the
    # check that recognises it must not be lost when the path is expanded.
    monkeypatch.setattr(approval, "_scratch_roots", lambda: [Path(tempfile.gettempdir())])
    assert run("Remove-Item $env:TEMP\\argent-probe.log") is True


# --- the person's files: allowed, but only because they come back --------------

def test_a_project_file_is_deleted_only_after_it_is_copied(project, vibe):
    target = project / "src" / "main.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    assert run("Remove-Item src/main.py") is True
    snapshots = trash.list_snapshots()
    assert len(snapshots) == 1, "удаление без копии не разрешается"
    assert snapshots[0].files == 1

    target.unlink()                               # the command does this
    restored = trash.restore(snapshots[0].id)
    assert restored == [target]
    assert target.read_text(encoding="utf-8") == "print('hello')\n"


def test_the_transcript_says_how_to_undo(project, vibe):
    from argent.ui import console
    import io

    (project / "src" / "probe.py").write_text("x", encoding="utf-8")
    buf = io.StringIO()
    original = console.file
    console.file = buf
    try:
        run("Remove-Item src/probe.py")
    finally:
        console.file = original

    text = buf.getvalue()
    assert "/trash restore" in text, text
    assert "отменяемо" in text, text


def test_a_wildcard_is_expanded_before_the_copy(project, vibe):
    (project / "src" / "a.tmp").write_text("a", encoding="utf-8")
    (project / "src" / "b.tmp").write_text("b", encoding="utf-8")
    (project / "src" / "keep.py").write_text("k", encoding="utf-8")

    assert run("Remove-Item src\\*.tmp") is True
    snapshot = trash.list_snapshots()[0]
    assert sorted(Path(e.original).name for e in snapshot.entries) == ["a.tmp", "b.tmp"]


# --- what is still refused -----------------------------------------------------

def test_a_file_outside_the_project_is_refused(tmp_path, project, vibe):
    outside = tmp_path / "outside.txt"
    outside.write_text("важное", encoding="utf-8")

    assert run(f'Remove-Item "{outside}"') is False
    assert trash.list_snapshots() == [], "чужое не трогаем даже копией"


def test_a_variable_pointing_outside_the_project_is_refused(tmp_path, project, vibe):
    outside = tmp_path / "outside.txt"
    outside.write_text("важное", encoding="utf-8")

    assert run(f'$f = "{outside}"; Remove-Item $f') is False


def test_argents_own_state_is_never_deleted(project, vibe):
    (project / ".argent" / "checkpoints" / "c.json").write_text("{}", encoding="utf-8")

    assert run("Remove-Item .argent/checkpoints") is False
    assert run("Remove-Item -Recurse -Force .argent") is False


def test_a_git_operation_that_destroys_work_is_refused(project, vibe):
    # No path to copy, and `reset --hard` throws the work away: nothing to undo with.
    assert run("git reset --hard HEAD~1") is False
    assert run("git clean -fd") is False


def test_catastrophic_and_power_are_still_refused(project, vibe):
    assert run("rm -rf /") is False
    assert run("shutdown /s /t 0") is False
    assert run("Invoke-Expression $payload") is False


def test_a_delete_too_large_to_copy_is_refused(project, vibe, monkeypatch):
    big = project / "src" / "big.bin"
    big.write_bytes(b"x" * 4096)
    monkeypatch.setattr(trash, "MAX_BYTES", 10)

    assert run("Remove-Item src/big.bin") is False
    assert trash.list_snapshots() == []


# --- the snapshot store itself -------------------------------------------------

def test_restoring_never_overwrites_what_is_there_now(project, vibe):
    target = project / "src" / "notes.md"
    target.write_text("исходное", encoding="utf-8")
    assert run("Remove-Item src/notes.md") is True
    snapshot = trash.list_snapshots()[0]

    target.unlink()
    target.write_text("новое, написанное после", encoding="utf-8")

    restored = trash.restore(snapshot.id)
    assert restored and restored[0] != target, "занятый путь не перезаписывается"
    assert target.read_text(encoding="utf-8") == "новое, написанное после"
    assert Path(restored[0]).read_text(encoding="utf-8") == "исходное"


def test_the_trash_is_pruned_to_a_budget(project, vibe, monkeypatch):
    monkeypatch.setattr(trash, "KEEP_SNAPSHOTS", 2)
    for index in range(4):
        (project / "src" / f"f{index}.txt").write_text("x" * 10, encoding="utf-8")
        assert run(f"Remove-Item src/f{index}.txt") is True

    assert len(trash.list_snapshots()) == 2


# --- the /trash command --------------------------------------------------------

def test_the_trash_command_lists_and_restores(project, vibe):
    from src.cli.commands.agent_commands import handle_trash
    from src.cli.commands.base import CommandContext

    target = project / "src" / "report.md"
    target.write_text("данные", encoding="utf-8")
    assert run("Remove-Item src/report.md") is True
    snapshot = trash.list_snapshots()[0]

    result = handle_trash(CommandContext(agent=None, user_input="/trash"))
    assert result.should_continue is True

    target.unlink()
    handle_trash(CommandContext(agent=None, user_input=f"/trash restore {snapshot.id}"))
    assert target.read_text(encoding="utf-8") == "данные"
