"""Файлы промпта: две области, шаблон и запуск во внешнем редакторе.

Встроенного редактора намеренно нет: поле ввода — это то же место, куда идёт
каждый символ хода, и ошибка там дороже удобства. Файл открывается в том, чем
человек уже пользуется; если редактора нет — системной программой.
"""

from pathlib import Path

import pytest

from argent import prompt_files


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARGENT_HOME", str(tmp_path / "home"))
    return tmp_path


def test_the_two_scopes_are_separate(workspace):
    assert prompt_files.path_for("project") == workspace / "AGENTS.md"
    assert prompt_files.path_for("global") == workspace / "home" / "AGENTS.md"
    assert prompt_files.path_for("project") != prompt_files.path_for("global")


def test_a_missing_file_says_so_and_does_not_pretend(workspace):
    text = prompt_files.describe("project")
    assert "нет" in text


def test_the_template_is_created_once(workspace):
    assert "Создан" in prompt_files.create("project")
    path = prompt_files.path_for("project")
    assert path.exists()
    assert "## Команды" in path.read_text(encoding="utf-8")

    path.write_text("мои правила", encoding="utf-8")
    again = prompt_files.create("project")
    assert "уже есть" in again, "шаблон не должен затирать написанное"
    assert path.read_text(encoding="utf-8") == "мои правила"


def test_opening_a_missing_file_offers_to_create_it(workspace):
    message = prompt_files.open_in("project")
    assert "/prompt new" in message


def test_opening_starts_the_chosen_editor(workspace, monkeypatch):
    started = []
    monkeypatch.setattr(prompt_files, "_launch", lambda editor, path: started.append((editor, path)) or None)
    prompt_files.create("project")

    message = prompt_files.open_in("project", editor="subl")
    assert started and started[0][0] == "subl"
    assert "Sublime" in message


def test_a_missing_editor_falls_back_to_the_system_program(workspace, monkeypatch):
    calls = []

    def fake(editor, path):
        calls.append(editor)
        return "'subl' не найден в PATH" if editor == "subl" else None

    monkeypatch.setattr(prompt_files, "_launch", fake)
    prompt_files.create("project")

    message = prompt_files.open_in("project", editor="subl")
    assert calls == ["subl", "system"]
    assert "систем" in message.lower()


def test_the_status_line_counts_the_lines(workspace):
    prompt_files.create("project")
    assert "есть" in prompt_files.describe("project")
