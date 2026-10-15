"""What a person sees of a room's work: lists as lists (not Python's repr), what each step produced on the
board, a long question put in the conversation and only its last line asked, and no emoji in the interface."""

import pathlib
import re

import pytest

from src.rooms import template
from src.rooms.ask import ask_human, split_question
from src.rooms.board import RoomBoard
from src.rooms.models import Node


# ── lists are lists ───────────────────────────────────────────────────────────

def test_a_list_is_one_line_per_item_not_a_python_repr():
    text = template.render("Замечания: {{findings}}", {"findings": ["Первое: утечка", "Второе: гонка"]})
    assert text == "Замечания: - Первое: утечка\n- Второе: гонка"
    assert "[" not in text and "'" not in text


def test_a_dict_is_one_line_per_key_and_scalars_are_unchanged():
    assert template.render("{{d}}", {"d": {"a": 1, "b": [2, 3]}}) == "a: 1\nb: 2, 3"
    assert template.render("{{n}} {{ok}} {{s}}", {"n": 3, "ok": True, "s": "x"}) == "3 True x"
    assert template.as_text(["multi\nline", "one"]) == "- multi\n  line\n- one"


def test_an_empty_list_renders_nothing():
    assert template.render("[{{x}}]", {"x": []}) == "[]"


# ── the board says what a step produced ──────────────────────────────────────

class _Store:
    def __init__(self):
        self.items = None

    def get(self, scope):
        return None

    def set(self, items, scope):
        self.items = [(t.content, t.status) for t in items]

    def clear(self, scope):
        self.items = None


class _Lib:
    def get(self, name):
        return None


def _board():
    said, store = [], _Store()
    return RoomBoard(_Lib(), printer=said.append, todos=store), said, store


def test_a_critics_findings_are_all_on_the_screen_one_per_row():
    board, said, _ = _board()
    board.on_event({"type": "node_start", "room": "critic", "node": "review", "kind": "agent"})
    findings = [f"Замечание {i}: " + "слово " * 20 for i in range(5)]
    board.on_event({"type": "node_end", "room": "critic", "node": "review", "kind": "agent", "seconds": 12.0,
                    "produced": {"findings": findings, "critical_count": 2}})
    text = "\n".join(said)
    for i in range(5):
        assert f"Замечание {i}" in text
    assert text.count("•") == 5, "a bullet per finding"
    assert "critical_count" in text and "2" in text


def test_the_checklist_is_the_steps_that_began_and_comes_back_after_the_run():
    board, said, store = _board()
    board.on_event({"type": "node_start", "room": "r", "node": "a", "kind": "tool"})
    board.on_event({"type": "node_end", "room": "r", "node": "a", "kind": "tool", "seconds": 0.1})
    board.on_event({"type": "node_start", "room": "r", "node": "b", "kind": "agent"})
    assert store.items == [("a", "completed"), ("b", "in_progress")]
    board.close()
    assert store.items is None


def test_a_value_with_square_brackets_does_not_break_the_markup():
    board, said, _ = _board()
    board.on_event({"type": "node_start", "room": "r", "node": "a", "kind": "tool"})
    board.on_event({"type": "node_end", "room": "r", "node": "a", "kind": "tool", "seconds": 2,
                    "produced": {"out": "see [bold]this[/bold] and [x]"}})
    from rich.text import Text
    rendered = Text.from_markup(chr(10).join(said)).plain       # raises on markup the board let through
    assert "see [bold]this[/bold] and [x]" in rendered


def test_the_engine_reports_what_a_node_wrote():
    from src.rooms.engine import Engine
    from src.rooms.models import Room, State

    room = Room.model_validate({
        "room": "r", "entry": "n", "budget": {"max_iterations": 3, "max_tokens": 1000},
        "nodes": [{"id": "n", "type": "agent", "writes": ["findings"],
                   "budget": {"max_iterations": 1, "max_tokens": 500}}],
        "edges": [{"from": "n", "to": "EXIT:done"}]})
    events = []

    def run_node(node, state):
        state.data["findings"] = ["a", "b"]
        state.data["other"] = "not claimed by the node"

    Engine().run(room, State(data={}), run_node, on_event=events.append)
    ends = [e for e in events if e["type"] == "node_end"]
    assert ends[0]["produced"] == {"findings": ["a", "b"]}


# ── a long question ──────────────────────────────────────────────────────────

def test_the_last_line_is_the_question_and_the_rest_is_context():
    context, question = split_question("Критик видит проблемы:\n- одна\n- вторая\n\nПродолжить?\n\n")
    assert question == "Продолжить?"
    assert context == "Критик видит проблемы:\n- одна\n- вторая"
    assert split_question("Вы согласны?") == ("", "Вы согласны?")
    assert split_question("") == ("", "")


def test_the_context_goes_to_the_conversation_and_the_modal_gets_one_line():
    node = Node(id="accept", type="human_pause", prompt="Плохо:\n- а\n- б\n\nПродолжить?", into="accept")
    from src.rooms.models import State
    state, said, asked = State(data={}), [], []
    ask_human(node, state, said.append, lambda q: asked.append(q) or "да", lambda q, o: None)
    assert asked == ["Продолжить?"]
    assert "- а" in said[0] and "- б" in said[0] and "accept" in said[0]
    assert state.data["accept.output"] == "да"


def test_a_question_with_options_is_a_selection():
    node = Node(id="pick", type="human_pause", prompt="Что делаем?", options=["Да", "Нет"])
    from src.rooms.models import State
    state, picked = State(data={}), []
    ask_human(node, state, lambda t: None, lambda q: None, lambda q, o: picked.append((q, o)) or "Нет")
    assert picked == [("[pick] Что делаем?", ["Да", "Нет"])]
    assert state.data["pick.output"] == "Нет"


# ── no emoji in the interface ────────────────────────────────────────────────

# Telegram is left out on purpose: its buttons and reactions are emoji by the Bot API's design.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF⭐✅❌❎❓-❕❗⚠⚡✨"
                    "⏳⏰⚙❤ℹ⌛⏱]")
_ROOT = pathlib.Path(__file__).resolve().parents[1]
_FOLDERS = ["src", "tools", "plugins"]
_FILES = ["agent.py", "main.py", "command_handler.py", "approval.py", "ui.py"]


def _interface_sources():
    for folder in _FOLDERS:
        for path in (_ROOT / folder).rglob("*"):
            if path.suffix in {".py", ".js", ".html", ".css"} and "telegram" not in path.parts \
                    and "__pycache__" not in path.parts:
                yield path
    for name in _FILES:
        if (_ROOT / name).is_file():
            yield _ROOT / name


def test_the_interface_has_no_emoji():
    offenders = {}
    for path in _interface_sources():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = sorted(set(_EMOJI.findall(text)))
        # the data tables of the language/emoji handling itself may name them in comments
        if found:
            offenders[str(path.relative_to(_ROOT))] = "".join(found)
    assert not offenders, f"emoji in the interface: {offenders}"
