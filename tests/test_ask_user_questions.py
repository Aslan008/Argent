"""Asking the user a question they can actually answer.

A cloud model sent `options` as objects instead of the declared strings.
questionary renders any dict without a "name" key as the literal text "None",
so the user was shown a menu of four Nones, could not answer, and nothing
anywhere recorded why. The model was wrong; the tool turning a recoverable
deviation into an unanswerable prompt was ours.
"""

import pytest

from tools import misc_tools
from tools.misc_tools import _normalize_options, _option_text, ask_user_questions


class TestOptionText:
    def test_a_plain_string(self):
        assert _option_text("Полное покрытие") == "Полное покрытие"

    @pytest.mark.parametrize("shape", [
        {"name": "Полное покрытие"},
        {"label": "Полное покрытие"},
        {"title": "Полное покрытие"},
        {"text": "Полное покрытие"},
        {"option": "Полное покрытие"},
        {"value": "Полное покрытие"},
    ])
    def test_the_shapes_a_model_actually_reaches_for(self, shape):
        assert _option_text(shape) == "Полное покрытие"

    def test_a_label_keeps_its_explanation(self):
        """The model wrote the description for a reason; dropping it loses the
        very information that makes the choice decidable."""
        assert _option_text({"label": "Только критические пути",
                             "description": "быстрее, но дыры останутся"}) == (
            "Только критические пути — быстрее, но дыры останутся")

    def test_numbers_are_usable(self):
        assert _option_text(3) == "3"

    def test_nothing_displayable(self):
        assert _option_text({"weight": 5}) is None
        assert _option_text("   ") is None
        assert _option_text(None) is None


class TestNormalize:
    def test_objects_become_readable_choices(self):
        raw = [{"label": "A"}, {"label": "B"}]
        assert _normalize_options(raw) == (["A", "B"], 0)

    def test_unusable_entries_are_counted_not_shown(self):
        """Showing them is what produced the menu of Nones."""
        options, dropped = _normalize_options(["A", {"weight": 1}, None])
        assert options == ["A"] and dropped == 2

    def test_a_json_string_is_accepted(self):
        assert _normalize_options('["A", "B"]') == (["A", "B"], 0)

    def test_newline_separated_text_is_accepted(self):
        assert _normalize_options("A\nB\n") == (["A", "B"], 0)

    def test_duplicates_collapse(self):
        assert _normalize_options(["A", "A", {"label": "A"}]) == (["A"], 0)

    def test_garbage_yields_nothing_rather_than_raising(self):
        assert _normalize_options(42) == ([], 0)


class TestAsking:
    @pytest.fixture
    def picks_first(self, monkeypatch):
        """A user who always takes the first offered choice."""
        seen = {}

        class _Ask:
            def __init__(self, choices):
                seen["choices"] = choices

            def ask(self):
                return seen["choices"][0]

        class _Q:
            @staticmethod
            def select(_msg, choices):
                return _Ask(choices)

            @staticmethod
            def checkbox(_msg, choices):
                return _Ask(choices)

        monkeypatch.setattr(misc_tools, "questionary", _Q)
        monkeypatch.setattr(misc_tools.memory, "add_fact", lambda *a, **k: None)
        return seen

    def test_object_options_produce_a_real_menu(self, picks_first):
        """This is the reported bug: four choices, all rendering as None."""
        out = ask_user_questions([{
            "type": "single_choice",
            "question": "Какой объём покрытия тестами?",
            "options": [{"label": "Только критические пути"},
                        {"label": "Полное покрытие"}],
        }])
        assert "None" not in picks_first["choices"]
        assert picks_first["choices"][0] == "Только критические пути"
        assert "Только критические пути" in out

    def test_the_model_is_told_it_sent_the_wrong_shape(self, picks_first):
        """It can only stop doing this if something says so within the turn."""
        out = ask_user_questions([{
            "type": "single_choice", "question": "Q",
            "options": ["A", {"weight": 1}],
        }])
        assert "must be an array of STRINGS" in out

    def test_a_clean_call_says_nothing_extra(self, picks_first):
        out = ask_user_questions([{
            "type": "single_choice", "question": "Q", "options": ["A", "B"]}])
        assert "must be an array of STRINGS" not in out

    def test_a_custom_answer_is_always_offered(self, picks_first):
        ask_user_questions([{"type": "single_choice", "question": "Q",
                             "options": ["A"]}])
        assert any("Свой вариант" in c for c in picks_first["choices"])


class TestDegradedInput:
    @pytest.fixture
    def types_text(self, monkeypatch):
        monkeypatch.setattr(misc_tools.memory, "add_fact", lambda *a, **k: None)
        import prompt_toolkit
        monkeypatch.setattr(prompt_toolkit, "prompt", lambda *a, **k: "мой ответ")
        return None

    def test_a_choice_with_no_usable_options_becomes_free_text(self, types_text):
        """An empty menu is unanswerable; keeping the question is better than
        making the user escape out of it."""
        out = ask_user_questions([{"type": "single_choice", "question": "Сколько?",
                                   "options": [{"weight": 1}]}])
        assert "мой ответ" in out

    def test_a_single_question_sent_unwrapped(self, types_text):
        out = ask_user_questions({"type": "text", "question": "Как назвать?"})
        assert "мой ответ" in out

    def test_the_whole_payload_as_a_json_string(self, types_text):
        out = ask_user_questions('[{"type": "text", "question": "Как назвать?"}]')
        assert "мой ответ" in out

    def test_unparseable_input_explains_the_shape(self):
        out = ask_user_questions("не json")
        assert out.startswith("Error:") and "single_choice" in out

    def test_an_empty_list(self):
        assert ask_user_questions([]).startswith("Error:")
