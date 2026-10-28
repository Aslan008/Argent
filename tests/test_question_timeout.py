"""An unanswered question must not be reported as a chosen answer.

Two faults came out of one incident: a long answer was cut off by a five-minute limit
(and nothing in the result said "timeout"), and the empty result was then written into
long-term memory as "User preference on <question>: No answer" — an invented preference
that would quietly shape later sessions.

So: a question gets a much longer limit than an approval, and an empty answer is reported
as a fact about the prompt, never as a preference.
"""

import pytest

from argent.tools.misc_tools import _summarize_answers


class TestTheSummary:
    def test_an_empty_answer_is_not_a_preference(self, monkeypatch):
        facts = []
        monkeypatch.setattr("argent.tools.misc_tools.memory.add_fact",
                            lambda text: facts.append(text))

        out = _summarize_answers([{"question": "С чего начать?"}], {"С чего начать?": "No answer"})

        # An empty Enter IS an answer (the prompt was shown and accepted) — but it is
        # not a choice, and it must never become a remembered preference.
        assert out.splitlines()[0].startswith("ANSWERED")
        assert facts == [], f"в память попало вымышленное предпочтение: {facts}"
        assert "do not treat as a choice" in out

    def test_a_dismissed_prompt_is_not_an_answer(self, monkeypatch):
        facts = []
        monkeypatch.setattr("argent.tools.misc_tools.memory.add_fact",
                            lambda text: facts.append(text))

        out = _summarize_answers([{"question": "С чего начать?"}], {"С чего начать?": "Skipped"})

        assert out.splitlines()[0].startswith("SKIPPED")
        assert facts == []

    def test_a_real_answer_is_still_remembered(self, monkeypatch):
        facts = []
        monkeypatch.setattr("argent.tools.misc_tools.memory.add_fact",
                            lambda text: facts.append(text))

        out = _summarize_answers([{"question": "Тема?"}], {"Тема?": "git"})

        assert facts == ["User preference on 'Тема?': git"]
        assert "ANSWERED: all 1" in out
        assert "Тема?" in out and "git" in out

    def test_a_partial_answer_says_which_is_missing(self, monkeypatch):
        monkeypatch.setattr("argent.tools.misc_tools.memory.add_fact", lambda text: None)

        out = _summarize_answers(
            [{"question": "A"}, {"question": "B"}],
            {"A": "да", "B": "Skipped"},
        )

        assert "PARTLY ANSWERED: 1 of 2" in out
        assert "- B: SKIPPED" in out

    def test_an_empty_answer_and_a_dismissal_read_differently(self, monkeypatch):
        monkeypatch.setattr("argent.tools.misc_tools.memory.add_fact", lambda text: None)

        empty = _summarize_answers([{"question": "A"}], {"A": "No answer"})
        skipped = _summarize_answers([{"question": "A"}], {"A": "Skipped"})

        assert "empty" in empty and "SKIPPED" not in empty
        assert "SKIPPED" in skipped and "empty" not in skipped

    def test_malformed_options_are_told_to_the_model(self, monkeypatch):
        monkeypatch.setattr("argent.tools.misc_tools.memory.add_fact", lambda text: None)

        out = _summarize_answers([{"question": "A"}], {"A": "да"}, malformed=["A"])

        assert "array of STRINGS" in out


class TestTheTimeout:
    def test_a_question_waits_much_longer_than_an_approval(self, monkeypatch):
        from src.bridge import session

        monkeypatch.delenv("ARGENT_QUESTION_TIMEOUT", raising=False)
        monkeypatch.delenv("ARGENT_ANSWER_TIMEOUT", raising=False)

        assert session.question_timeout() > session.answer_timeout() * 5, (
            "вопрос — это человек думает, а не действие, которое блокирует ход"
        )

    def test_zero_means_wait_forever(self, monkeypatch):
        from src.bridge import session

        monkeypatch.setenv("ARGENT_QUESTION_TIMEOUT", "0")
        assert session.question_timeout() == 0.0

    def test_the_override_is_honoured(self, monkeypatch):
        from src.bridge import session

        monkeypatch.setenv("ARGENT_QUESTION_TIMEOUT", "42")
        assert session.question_timeout() == 42.0

    def test_nonsense_falls_back_to_the_default(self, monkeypatch):
        from src.bridge import session

        monkeypatch.setenv("ARGENT_QUESTION_TIMEOUT", "soon")
        assert session.question_timeout() == session.DEFAULT_QUESTION_TIMEOUT
