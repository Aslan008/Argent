"""A task pasted as several lines, and the message for a room that is not in this working folder."""

from src.rooms import cli


class Term:
    def __init__(self):
        self.out, self.err = [], []

    def io(self):
        return cli.Io(out=self.out.append, err=self.err.append, confirm=lambda q: True,
                      choose=lambda q, o: None)


def test_first_word_ends_at_any_whitespace():
    assert cli._first_word("run clode_code Создай\nпесочницу") == ("run", "clode_code Создай\nпесочницу")
    assert cli._first_word("clode_code\nСоздай песочницу") == ("clode_code", "Создай песочницу")
    assert cli._first_word("  one  ") == ("one", "")
    assert cli._first_word("") == ("", "")


def test_a_multiline_task_keeps_the_room_name_and_all_its_lines(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_start_run", lambda io, lib, name, task, params, *a: seen.update(name=name, task=task))
    task = "Создай песочницу. Она должна включать:\nКисть (песок, вода)\nФизика песка"
    for text in (f"run analyze {task}", f"run analyze\n{task}"):
        seen.clear()
        cli.handle(text, Term().io())
        assert seen["name"] == "analyze"
        assert seen["task"].count("\n") == 2 and "Физика песка" in seen["task"]


def test_the_missing_room_message_names_the_room():
    assert cli._no_room("ghost ") == "Нет комнаты «ghost». /rooms list — список."
