"""Lone surrogates: invisible, inert, and fatal on the way out.

A real failure: "'utf-8' codec can't encode characters in position
53000-53001: surrogates not allowed". Two ADJACENT surrogates is a decomposed
emoji, and the damage was not cosmetic — once one is in the conversation
history, every following request fails to encode, so the chat stays bricked
until /clear, for a character the user never sees.
"""

import json

import pytest

from text_safety import REPLACEMENT, has_lone_surrogates, repair_structure, repair_surrogates

HIGH = chr(0xD83D)
LOW = chr(0xDE00)
EMOJI = "😀"


def _encodable(value) -> bool:
    try:
        json.dumps(value, ensure_ascii=False).encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


class TestDetection:
    def test_ordinary_text_is_clean(self):
        assert has_lone_surrogates("привет 😀 мир") is False

    def test_a_surrogate_is_spotted(self):
        assert has_lone_surrogates("x" + HIGH) is True

    def test_non_strings(self):
        assert has_lone_surrogates(None) is False
        assert has_lone_surrogates(42) is False


class TestRepair:
    def test_a_split_pair_becomes_the_character_again(self):
        """This is the reported case — position 53000-53001, one emoji."""
        assert repair_surrogates("a" + HIGH + LOW + "b") == "a😀b"

    def test_an_orphan_is_replaced_not_dropped(self):
        """Dropping it would silently eat one character out of the middle of
        somebody's file; the replacement character says something was here."""
        assert repair_surrogates("a" + HIGH + "b") == "a" + REPLACEMENT + "b"

    def test_a_lone_low_surrogate(self):
        assert repair_surrogates(LOW) == REPLACEMENT

    def test_a_high_surrogate_at_the_very_end(self):
        assert repair_surrogates("text" + HIGH) == "text" + REPLACEMENT

    def test_clean_text_is_returned_unchanged(self):
        text = "обычный текст с эмодзи 😀 и всем прочим"
        assert repair_surrogates(text) is text

    def test_the_result_always_encodes(self):
        for bad in (HIGH, LOW, HIGH + LOW, HIGH + HIGH, LOW + HIGH, "x" + HIGH + "y" + LOW):
            repair_surrogates(bad).encode("utf-8")     # must not raise

    def test_two_separate_orphans_are_not_glued_together(self):
        """HIGH+HIGH is not a pair; treating it as one would invent a character."""
        assert repair_surrogates(HIGH + HIGH) == REPLACEMENT * 2


class TestStructures:
    def test_nested_messages(self):
        messages = [{"role": "user", "content": "a" + HIGH + LOW},
                    {"role": "tool", "content": ["ok", "b" + HIGH]}]
        fixed = repair_structure(messages)
        assert fixed[0]["content"] == "a😀"
        assert fixed[1]["content"][1] == "b" + REPLACEMENT
        assert _encodable(fixed)

    def test_a_clean_structure_is_not_rebuilt(self):
        """Rebuilding every dict on every turn to fix nothing would be a
        per-request cost paid forever."""
        messages = [{"role": "user", "content": "чисто"}]
        assert repair_structure(messages) is messages

    def test_non_text_values_survive(self):
        value = {"n": 5, "flag": True, "none": None}
        assert repair_structure(value) is value


class TestTheProducer:
    def test_an_escaped_pair_still_decodes_to_the_emoji(self):
        from src.agent.parser import decode_json_escapes
        assert decode_json_escapes(r"😀") == EMOJI

    def test_a_half_written_escape_no_longer_poisons_anything(self):
        """A \\ud83d whose partner never arrived — truncated output, a split
        stream chunk, a malformed low escape."""
        from src.agent.parser import decode_json_escapes
        out = decode_json_escapes(r"text \ud83d more")
        out.encode("utf-8")
        assert REPLACEMENT in out

    def test_ordinary_escapes_are_untouched(self):
        from src.agent.parser import decode_json_escapes
        assert decode_json_escapes(r"line\nnext\ttab") == "line\nnext\ttab"


class TestTheRequestPath:
    def test_an_already_poisoned_history_heals_itself(self):
        """Fixing the producer cannot rescue a conversation that is ALREADY
        carrying one — and that conversation fails on every single turn."""
        from src.agent.trimmer import clean_messages_for_llm

        poisoned = [{"role": "user", "content": "x" * 50 + HIGH + LOW + "y"},
                    {"role": "assistant", "content": "ok" + HIGH}]
        assert _encodable(poisoned) is False
        cleaned = clean_messages_for_llm(poisoned, False)
        assert _encodable(cleaned) is True
        assert "😀" in cleaned[0]["content"]

    def test_the_stored_history_is_not_mutated(self):
        from src.agent.trimmer import clean_messages_for_llm

        original = [{"role": "user", "content": "x" + HIGH}]
        clean_messages_for_llm(original, False)
        assert original[0]["content"] == "x" + HIGH


class TestTheStoragePath:
    def test_a_session_with_a_surrogate_still_saves(self, tmp_path, monkeypatch):
        """The one moment the history most needs to be written is the moment it
        would have been lost."""
        import session as session_module

        d = tmp_path / "sessions"
        d.mkdir()
        monkeypatch.setattr(session_module, "SESSIONS_DIR", d)
        monkeypatch.setattr(session_module, "INDEX_PATH", d / "index.json")
        monkeypatch.chdir(tmp_path)

        sid = session_module.save_session([
            {"role": "system", "content": "s"},
            {"role": "user", "content": "посчитай " + HIGH + LOW},
        ], {"model": "m"})
        assert sid is not None
        assert "😀" in session_module.load_session(sid)["messages"][1]["content"]
