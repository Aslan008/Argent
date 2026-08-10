"""You must be told which file you are agreeing to destroy.

delete_file asked «удалить файл 'notes.md'» — the string the model passed, not
the path that would be unlinked. Those differ: _resolve_path falls back to the
configured Obsidian vault when a relative path does not exist locally, so a
file absent from the project but present in the vault resolves there, and the
prompt named neither the directory nor the fact that it had moved.

The vault fallback predates the removal of every Obsidian tool and is still
live for anyone who set a vault back when /obsidian existed.
"""

from pathlib import Path

import pytest

import tools.file_ops as file_ops
import tools._helpers as helpers


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A vault holding notes.md, and a working directory that does not."""
    vault_dir = tmp_path / "vault"
    (vault_dir / "Notes").mkdir(parents=True)
    note = vault_dir / "Notes" / "notes.md"
    note.write_text("years of writing", encoding="utf-8")

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(helpers, "get_obsidian_vault", lambda: str(vault_dir))
    return note


class TestTheApprovalNamesTheResolvedPath:
    def test_deleting_through_the_vault_fallback_says_so(self, vault, monkeypatch):
        asked = []

        def _record(action, **kwargs):
            asked.append(action)
            return False                       # refuse, so nothing is destroyed

        monkeypatch.setattr("approval.request_approval", _record)
        result = file_ops.delete_file("Notes/notes.md")

        assert asked, "удаление прошло без запроса подтверждения"
        assert str(vault) in asked[0], (
            f"пользователю показали {asked[0]!r}, а удалён был бы {vault}")
        assert "aborted" in result.lower()
        assert vault.exists()

    def test_an_ordinary_local_delete_names_an_absolute_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(helpers, "get_obsidian_vault", lambda: None)
        target = tmp_path / "scratch.txt"
        target.write_text("x", encoding="utf-8")

        asked = []
        monkeypatch.setattr("approval.request_approval",
                            lambda action, **k: asked.append(action) or False)
        file_ops.delete_file("scratch.txt")

        assert asked and Path(asked[0].split("'")[1]).is_absolute()


class TestTheRedirectIsRecorded:
    def test_resolving_into_the_vault_is_logged(self, vault, caplog):
        with caplog.at_level("INFO", logger="tools"):
            resolved = helpers._resolve_path("Notes/notes.md")
        assert resolved == vault.resolve()
        assert any("Obsidian vault" in r.message for r in caplog.records), \
            "молчаливое перенаправление пути снова не оставляет следа"

    def test_a_local_file_is_not_logged(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(helpers, "get_obsidian_vault", lambda: None)
        (tmp_path / "here.txt").write_text("x", encoding="utf-8")
        with caplog.at_level("INFO", logger="tools"):
            helpers._resolve_path("here.txt")
        assert not [r for r in caplog.records if "Obsidian" in r.message]
