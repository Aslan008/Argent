"""A new chat is actually new.

Measured from the input history: /clear was pressed 115 times — more than the
other 47 commands combined — and asked why, the answer was "чтобы в новом чате
точно не осталось контекста предыдущих разговоров".

That ritual was justified. Startup offered to restore the saved CONVERSATION,
while .argent/memory.json came back regardless: the previous run's objective,
its completed steps and its recorded failures, all feeding the objective anchor
in the prompt. Declining the restore did not give a clean slate, and nothing
said so.
"""

import pytest

import main as main_module
from main import carried_over_memory, offer_previous_context


class _Agent:
    def __init__(self):
        self.messages = [{"role": "system", "content": "s"}]
        self.model_name = "m"
        self.session_id = None


@pytest.fixture
def memory(monkeypatch, tmp_path):
    """A real MemoryManager pointed at a scratch file."""
    import memory_manager

    monkeypatch.setattr(memory_manager, "resolve_memory_file",
                        lambda: tmp_path / "memory.json")
    mem = memory_manager.MemoryManager()
    monkeypatch.setattr(memory_manager, "memory", mem)
    return mem


@pytest.fixture
def ui(monkeypatch):
    printed = []
    monkeypatch.setattr(main_module, "print_system", lambda t, *a, **k: printed.append(str(t)))
    monkeypatch.setattr(main_module, "print_error", lambda t, *a, **k: printed.append(str(t)))
    return printed


def _answer(monkeypatch, value):
    class _Ans:
        def ask(self):
            return value

    class _Q:
        @staticmethod
        def select(*a, **k):
            return _Ans()

    monkeypatch.setattr(main_module, "questionary", _Q)


class TestWhatCarriesOver:
    def test_an_empty_memory_carries_nothing(self, memory):
        assert carried_over_memory() == []

    def test_the_objective_is_named(self, memory):
        memory.set_objective("починить меню инвентаря")
        assert any("починить меню" in p for p in carried_over_memory())

    def test_counts_are_named(self, memory):
        memory.add_completed("шаг один")
        memory.add_completed("шаг два")
        assert any("2" in p and "сделано" in p for p in carried_over_memory())


class TestTheChoice:
    def test_nothing_is_asked_when_there_is_nothing_to_carry(self, memory, ui, monkeypatch):
        monkeypatch.setattr(main_module, "get_last_session", lambda: None)
        monkeypatch.setattr(main_module, "questionary",
                            type("Q", (), {"select": staticmethod(
                                lambda *a, **k: pytest.fail("нечего спрашивать"))}))
        offer_previous_context(_Agent())

    def test_working_memory_alone_is_enough_to_ask(self, memory, ui, monkeypatch):
        """The old code only asked when a saved SESSION existed, so memory-only
        carry-over passed in silence."""
        monkeypatch.setattr(main_module, "get_last_session", lambda: None)
        memory.set_objective("прошлая цель")
        _answer(monkeypatch, "Начать с чистого листа (забыть всё выше)")
        offer_previous_context(_Agent())
        assert memory.data["objective"] == ""

    def test_a_clean_start_really_clears_the_memory(self, memory, ui, monkeypatch):
        """The whole point: "начать с чистого листа" is now literally true."""
        monkeypatch.setattr(main_module, "get_last_session", lambda: None)
        memory.set_objective("старая цель")
        memory.add_completed("старый шаг")
        _answer(monkeypatch, "Начать с чистого листа (забыть всё выше)")
        offer_previous_context(_Agent())
        assert memory.data["objective"] == "" and memory.data["completed"] == []

    def test_keeping_the_memory_keeps_it(self, memory, ui, monkeypatch):
        monkeypatch.setattr(main_module, "get_last_session", lambda: None)
        memory.set_objective("нужная цель")
        _answer(monkeypatch, "Новый чат, но сохранить рабочую память проекта")
        offer_previous_context(_Agent())
        assert memory.data["objective"] == "нужная цель"

    def test_restoring_a_session_loads_the_conversation(self, memory, ui, monkeypatch):
        restored = []
        monkeypatch.setattr(main_module, "get_last_session",
                            lambda: {"id": "s1", "saved_at": "2026-08-09T10:00:00",
                                     "preview": "почини меню", "cwd": ""})
        monkeypatch.setattr(main_module, "restore_session",
                            lambda a, m: restored.append(m["id"]))
        _answer(monkeypatch, "Продолжить прошлую сессию")
        offer_previous_context(_Agent())
        assert restored == ["s1"]

    def test_what_would_carry_over_is_shown_before_the_choice(self, memory, ui, monkeypatch):
        """This is the part that used to arrive uninvited."""
        monkeypatch.setattr(main_module, "get_last_session", lambda: None)
        memory.set_objective("прошлая цель")
        _answer(monkeypatch, "Начать с чистого листа (забыть всё выше)")
        offer_previous_context(_Agent())
        assert any("Рабочая память проекта" in line for line in ui)

    def test_escaping_the_prompt_changes_nothing(self, memory, ui, monkeypatch):
        """Ctrl+C must not silently pick an option that discards work."""
        monkeypatch.setattr(main_module, "get_last_session", lambda: None)
        memory.set_objective("нужная цель")
        _answer(monkeypatch, None)
        offer_previous_context(_Agent())
        assert memory.data["objective"] == "нужная цель"
