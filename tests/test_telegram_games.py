"""Quizzes: the model writes the question, the bridge runs the round.

The round must be cheap (no extra requests), honest (answers hidden until the reveal) and
bounded (one round per chat, the clock always closes it). These tests hold all three.
"""

import time

import pytest

from src.telegram import games
from src.telegram.bot import TelegramBridge

GOOD = '{"q":"Столица Австралии?","o":["Сидней","Канберра"],"a":1,"t":5}'


# ─── parsing ─────────────────────────────────────────────────────────────────

def test_a_valid_spec_becomes_a_question():
    spec = games.parse(GOOD)

    assert spec == {"q": "Столица Австралии?", "o": ["Сидней", "Канберра"], "a": 1, "t": 5,
                    "n": None, "open": False}


def test_the_tag_is_cut_out_of_the_answer():
    text, spec = games.extract(f"Держи вопрос.\n[[quiz:{GOOD}]]")

    assert text == "Держи вопрос."
    assert spec and spec["a"] == 1


def test_the_tag_is_removed_even_when_the_spec_is_broken():
    """A broken spec must not reach the chat as raw JSON, and must not half-draw a game."""
    text, spec = games.extract('Ответ [[quiz:{"q":"?","o":["a"],"a":9}]] конец')

    assert spec is None
    assert "quiz" not in text


def test_an_index_outside_the_options_is_refused():
    assert games.parse('{"q":"?","o":["a","b"],"a":5}') is None


def test_too_few_options_are_refused():
    assert games.parse('{"q":"?","o":["only"],"a":0}') is None


def test_the_clock_is_bounded_and_has_a_default():
    assert games.parse('{"q":"?","o":["a","b"],"a":0}')["t"] == games.DEFAULT_SECONDS
    assert games.parse('{"q":"?","o":["a","b"],"a":0,"t":10000}')["t"] == games.MAX_SECONDS
    assert games.parse('{"q":"?","o":["a","b"],"a":0,"t":1}')["t"] == 5


def test_long_options_are_trimmed_and_their_number_capped():
    spec = games.parse('{"q":"?","o":["%s"],"a":0}' % ("x" * 500))
    assert spec is None, "one option is not a quiz"


def test_options_are_capped_at_six():
    spec = games.parse('{"q":"?","o":["a","b","c","d","e","f","g"],"a":0}')
    assert len(spec["o"]) == 6


def test_the_expected_players_count_is_optional():
    assert games.parse(GOOD)["n"] is None
    assert games.parse('{"q":"?","o":["a","b"],"a":0,"n":3}')["n"] == 3
    assert games.parse('{"q":"?","o":["a","b"],"a":0,"n":0}')["n"] is None


def test_zero_seconds_means_the_question_waits():
    """`t: 0` is the async game: somebody may open the chat a day later and still play."""
    spec = games.parse('{"q":"?","o":["a","b"],"a":1,"t":0}')

    assert spec["open"] is True
    assert spec["t"] == 0


def test_a_timed_round_is_not_open_ended():
    assert games.parse(GOOD)["open"] is False


# ─── the question that waits ─────────────────────────────────────────────────

def test_an_open_question_does_not_block_a_new_round():
    manager = games.Games()
    waiting = manager.start(-100, {"q": "?", "o": ["a", "b"], "a": 0, "t": 0, "open": True})
    timed = manager.start(-100, {"q": "?", "o": ["a", "b"], "a": 0, "t": 30, "open": False})

    assert waiting is not None and timed is not None
    assert manager.active(-100) is timed, "only the timed round holds the chat"
    assert manager.find(-100, waiting.id) is waiting


def test_an_open_question_allows_one_answer_per_person():
    manager = games.Games()
    round_ = manager.start(-100, {"q": "?", "o": ["a", "b"], "a": 0, "t": 0, "open": True})

    assert manager.answer(-100, round_.id, 7, "Иван", 0) == "correct"
    assert manager.answer(-100, round_.id, 7, "Иван", 1) == "already"


def test_only_a_few_open_questions_are_kept():
    manager = games.Games()
    for index in range(games.Games.MAX_OPEN + 3):
        manager.start(-100, {"q": str(index), "o": ["a", "b"], "a": 0, "t": 0, "open": True})

    assert len(manager._open["-100"]) == games.Games.MAX_OPEN


# ─── a round ─────────────────────────────────────────────────────────────────

def make_round(seconds=5, expected=None):
    manager = games.Games()
    return manager, manager.start(-100, {"q": "2+2?", "o": ["3", "4"], "a": 1,
                                         "t": seconds, "n": expected})


def test_one_round_per_chat():
    manager = games.Games()
    first = manager.start(-100, {"q": "2+2?", "o": ["3", "4"], "a": 1, "t": 5})
    second = manager.start(-100, {"q": "3+3?", "o": ["5", "6"], "a": 1, "t": 5})

    assert first is not None
    assert second is None, "a second round would draw a second set of buttons"


def test_one_answer_per_person():
    manager, round_ = make_round()

    assert manager.answer(-100, round_.id, 7, "Иван", 1) == "ok"
    assert manager.answer(-100, round_.id, 7, "Иван", 0) == "already"
    assert round_.answers == {"7": 1}, "the first answer stands"


def test_closing_scores_only_the_correct_option():
    manager, round_ = make_round()
    manager.answer(-100, round_.id, 7, "Иван", 1)      # right
    manager.answer(-100, round_.id, 8, "Мария", 0)     # wrong

    manager.close(-100)

    assert manager.scoreboard(-100) == [("Иван", 1)]


def test_closing_twice_scores_once():
    manager, round_ = make_round()
    manager.answer(-100, round_.id, 7, "Иван", 1)

    assert manager.close(-100) is not None
    assert manager.close(-100) is None
    assert manager.scoreboard(-100) == [("Иван", 1)]


def test_the_expected_number_closes_the_round_early():
    manager, round_ = make_round(expected=2)

    assert manager.answer(-100, round_.id, 7, "Иван", 1) == "ok"
    assert manager.answer(-100, round_.id, 8, "Мария", 1) == "complete"


def test_a_late_answer_is_refused():
    manager, round_ = make_round(seconds=5)
    round_.started_at -= 10          # the clock ran out

    assert manager.answer(-100, round_.id, 7, "Иван", 1) == "timeout"


def test_cancel_closes_without_scoring():
    manager, round_ = make_round()
    manager.answer(-100, round_.id, 7, "Иван", 1)

    assert manager.cancel(-100) is not None
    assert manager.scoreboard(-100) == []


def test_an_unknown_round_id_is_refused():
    manager, round_ = make_round()
    assert manager.answer(-100, "zzzzzz", 7, "Иван", 1) == "closed"


def test_the_scoreboard_keeps_the_best_first():
    manager, round_ = make_round()
    manager.answer(-100, round_.id, 7, "Иван", 1)
    manager.close(-100)

    second = manager.start(-100, {"q": "3+3?", "o": ["5", "6"], "a": 1, "t": 5})
    manager.answer(-100, second.id, 8, "Мария", 1)
    manager.close(-100)

    assert manager.scoreboard(-100)[0] == ("Иван", 1)
    assert len(manager.scoreboard(-100)) == 2


# ─── rendering ───────────────────────────────────────────────────────────────

def test_the_question_carries_the_buttons():
    _, round_ = make_round()
    keyboard = games.keyboard(round_)

    labels = [button["text"] for row in keyboard["inline_keyboard"] for button in row]
    assert labels == ["3", "4"]
    payloads = [button["callback_data"] for row in keyboard["inline_keyboard"] for button in row]
    assert payloads == [f"qz:{round_.id}:0", f"qz:{round_.id}:1"]


def test_the_result_names_the_answer_and_the_score():
    manager, round_ = make_round()
    manager.answer(-100, round_.id, 7, "Иван", 1)
    manager.answer(-100, round_.id, 8, "Мария", 0)
    manager.close(-100)

    text = games.result_text(round_, manager.scoreboard(-100))

    assert "Правильный ответ" in text and "4" in text
    assert "Угадали: Иван" in text
    assert "Не угадали: Мария" in text


def test_an_empty_scoreboard_says_so():
    assert "Пока пусто" in games.score_text([])


# ─── the button path through the bridge ──────────────────────────────────────

class _Client:
    def __init__(self):
        self.toasts = []
        self.edits = []

    def send_message(self, chat_id, text, **kw):
        return {"message_id": 10}

    def answer_callback_query(self, query_id, text=None, alert=False):
        self.toasts.append(text)

    def edit_message_text(self, chat_id, message_id, text, **kw):
        self.edits.append((message_id, text))
        return {"message_id": message_id}

    def send_chat_action(self, *a, **kw):
        return True

    def set_reaction(self, *a, **kw):
        return True


def _bridge(monkeypatch, groups=("-100",)):
    import src.telegram.bot as bot_module
    monkeypatch.setattr(bot_module, "get_telegram_chat_id", lambda: "42")
    monkeypatch.setattr(bot_module, "is_telegram_group_allowed", lambda cid: str(cid) in set(groups))
    monkeypatch.setattr(bot_module, "get_telegram_group_agent", lambda cid: False)
    monkeypatch.setattr(bot_module, "get_telegram_mode", lambda: "isolated")
    bridge = TelegramBridge()
    bridge.client = _Client()
    bridge._bot_info = {"id": 7, "username": "argent_bot"}
    return bridge


def _press(bridge, chat_id, data, user_id):
    bridge._handle_callback_query({
        "id": "cb", "from": {"id": user_id, "first_name": "Мария"}, "data": data,
        "message": {"message_id": 10, "chat": {"id": int(chat_id), "type": "supergroup"}},
    })


def test_any_member_may_press_a_quiz_button(monkeypatch):
    bridge = _bridge(monkeypatch)
    round_ = bridge._games.start(-100, {"q": "2+2?", "o": ["3", "4"], "a": 1, "t": 60})
    round_.message_id = 10

    _press(bridge, -100, f"qz:{round_.id}:1", 999)

    assert bridge.client.toasts[-1] == "Принято ✅"
    assert round_.answers == {"999": 1}


def test_a_member_still_cannot_press_an_approval(monkeypatch):
    bridge = _bridge(monkeypatch)
    _press(bridge, -100, "appr:1:once", 999)

    assert bridge.client.toasts[-1] and "владелец" in bridge.client.toasts[-1]


def test_an_open_question_grades_a_late_player_privately(monkeypatch):
    """Nothing is published for an open question: the verdict goes to the presser alone."""
    bridge = _bridge(monkeypatch)
    round_ = bridge._games.start(-100, {"q": "2+2?", "o": ["3", "4"], "a": 1,
                                        "t": 0, "open": True})
    round_.message_id = 10
    round_.started_at -= 86_400          # somebody opens the chat the next day

    _press(bridge, -100, f"qz:{round_.id}:1", 999)

    assert "Верно" in bridge.client.toasts[-1]
    assert bridge.client.edits == [], "no public reveal: others may still be thinking"


def test_the_round_is_revealed_once_the_expected_players_answer(monkeypatch):
    bridge = _bridge(monkeypatch)
    round_ = bridge._games.start(-100, {"q": "2+2?", "o": ["3", "4"], "a": 1, "t": 60, "n": 1})
    round_.message_id = 10

    _press(bridge, -100, f"qz:{round_.id}:1", 999)

    assert bridge.client.edits, "the answer is shown as soon as the last expected player presses"
    assert "Правильный ответ" in bridge.client.edits[-1][1]
