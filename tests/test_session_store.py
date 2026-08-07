"""Saving and restoring a conversation.

Three defects shaped this file, all found by reading the user's real store of
fifty sessions:

* autosave filed a NEW snapshot every five turns, so one long chat crowded the
  other forty-nine out;
* four of the fifty held a single message — a system prompt and nothing else;
* where the conversation happened was never recorded, so restoring one from a
  different project looked exactly like restoring one from this project.
"""

import gzip
import json

import pytest

import session as session_module
from session import (
    delete_session, find_session, get_last_session, list_sessions,
    load_session, save_session,
)


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(session_module, "SESSIONS_DIR", d)
    monkeypatch.setattr(session_module, "INDEX_PATH", d / "index.json")
    monkeypatch.chdir(tmp_path)
    return d


def _chat(user="привет", n=1):
    msgs = [{"role": "system", "content": "you are argent"}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"{user} {i}" if n > 1 else user})
        msgs.append({"role": "assistant", "content": "ok"})
    return msgs


class TestNothingToSave:
    def test_a_system_prompt_alone_is_not_a_session(self):
        """Four of the user's fifty stored sessions were exactly this — a slot
        each, with nothing to restore."""
        assert save_session([{"role": "system", "content": "you are argent"}]) is None
        assert list_sessions() == []

    def test_argents_own_user_messages_do_not_count(self):
        """Continuation nudges are user-role but nobody typed them."""
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "You stopped after thinking, without..."}]
        assert save_session(msgs) is None

    def test_a_real_message_is_saved(self):
        assert save_session(_chat()) is not None


class TestContinuity:
    def test_saving_again_updates_the_same_session(self):
        """Autosave used to file a fresh copy every five turns."""
        sid = save_session(_chat(n=1), {"model": "m"})
        again = save_session(_chat(n=3), {"model": "m"}, session_id=sid)
        assert again == sid
        assert len(list_sessions()) == 1
        assert load_session(sid)["message_count"] == 7

    def test_without_an_id_a_new_session_is_created(self):
        a = save_session(_chat(), {"model": "m"})
        b = save_session(_chat(user="другое"), {"model": "m"})
        assert a != b and len(list_sessions()) == 2

    def test_an_interrupted_overwrite_leaves_the_old_copy_intact(self, store):
        """Updating in place is a risk the old append-only scheme did not have:
        a crash mid-write would destroy the only copy."""
        sid = save_session(_chat(user="важное"), {"model": "m"})
        real_gzipfile = session_module.gzip.GzipFile
        session_module.gzip.GzipFile = lambda **kw: (_ for _ in ()).throw(OSError("disk full"))
        try:
            with pytest.raises(OSError):
                save_session(_chat(user="новое"), {"model": "m"}, session_id=sid)
        finally:
            session_module.gzip.GzipFile = real_gzipfile
        assert load_session(sid)["preview"] == "важное"
        assert not list(store.glob("*.tmp"))          # no debris left behind


class TestWhereItHappened:
    def test_the_directory_is_recorded(self, tmp_path):
        sid = save_session(_chat())
        assert load_session(sid)["cwd"] == str(tmp_path)

    def test_an_explicit_cwd_wins(self):
        sid = save_session(_chat(), {"cwd": "C:/Projects/Game"})
        assert load_session(sid)["cwd"] == "C:/Projects/Game"

    def test_it_reaches_the_listing_without_opening_the_file(self):
        save_session(_chat(), {"cwd": "C:/Projects/Game"})
        assert list_sessions()[0]["cwd"] == "C:/Projects/Game"


class TestIndex:
    def test_listing_does_not_decompress_sessions(self, monkeypatch):
        """The old listing unpacked every full message array — megabytes — to
        read six fields, on every save, because cleanup called it."""
        save_session(_chat(n=5), {"model": "m"})
        monkeypatch.setattr(session_module.gzip, "open",
                            lambda *a, **k: pytest.fail("listing must not decompress"))
        assert len(list_sessions()) == 1

    def test_a_file_that_appeared_from_outside_is_picked_up(self, store):
        """The index is a cache, not the truth: a restored backup must show up."""
        save_session(_chat(), {"model": "m"})
        payload = {"id": "20200101_000000_x", "saved_at": "2020-01-01T00:00:00",
                   "model": "x", "provider": "ollama", "message_count": 2,
                   "preview": "из бэкапа", "cwd": "", "label": "",
                   "messages": _chat()}
        with gzip.open(store / "20200101_000000_x.json.gz", "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        assert any(s["preview"] == "из бэкапа" for s in list_sessions())

    def test_a_file_deleted_by_hand_disappears_from_the_listing(self, store):
        sid = save_session(_chat(), {"model": "m"})
        (store / f"{sid}.json.gz").unlink()
        assert list_sessions() == []

    def test_a_corrupt_index_is_rebuilt_not_fatal(self, store):
        save_session(_chat(), {"model": "m"})
        session_module.INDEX_PATH.write_text("{not json", encoding="utf-8")
        assert len(list_sessions()) == 1

    def test_a_corrupt_session_is_listed_rather_than_hidden(self, store):
        (store / "20200101_000000_broken.json.gz").write_bytes(b"not gzip")
        assert list_sessions()[0]["preview"] == "[corrupted session]"


class TestFinding:
    def test_by_position(self):
        save_session(_chat(user="старое"), {"model": "m"})
        save_session(_chat(user="новое"), {"model": "m"})
        assert find_session("1")["preview"] == "новое"      # newest first

    def test_by_label(self):
        save_session(_chat(), {"model": "m"}, label="шейдеры")
        assert find_session("шейдеры")["label"] == "шейдеры"

    def test_by_a_fragment_of_what_was_said(self):
        """A position shifts every time anything is saved, so the number you
        read a minute ago can point somewhere else by the time you type it."""
        save_session(_chat(user="как работает освещение в Unity"), {"model": "m"})
        assert "освещение" in find_session("освещен")["preview"]

    def test_a_miss_is_none_not_a_wrong_session(self):
        save_session(_chat(), {"model": "m"})
        assert find_session("не существует") is None
        assert find_session("99") is None
        assert find_session("") is None

    def test_query_filters_the_listing(self):
        save_session(_chat(user="про шейдеры"), {"model": "m"})
        save_session(_chat(user="про физику"), {"model": "m"})
        assert len(list_sessions("шейдер")) == 1
        assert len(list_sessions()) == 2


class TestHousekeeping:
    def test_the_store_is_capped(self, monkeypatch):
        monkeypatch.setattr(session_module, "MAX_SESSIONS", 3)
        for i in range(6):
            save_session(_chat(user=f"чат {i}"), {"model": "m"})
        assert len(list_sessions()) == 3

    def test_delete_removes_the_index_entry_too(self):
        sid = save_session(_chat(), {"model": "m"})
        assert delete_session(sid) is True
        assert list_sessions() == [] and delete_session(sid) is False

    def test_last_session_is_the_newest(self):
        save_session(_chat(user="первое"), {"model": "m"})
        save_session(_chat(user="второе"), {"model": "m"})
        assert get_last_session()["preview"] == "второе"

    def test_unserializable_values_do_not_break_the_save(self):
        msgs = _chat()
        msgs.append({"role": "assistant", "content": "x", "obj": object()})
        sid = save_session(msgs, {"model": "m"})
        assert isinstance(load_session(sid)["messages"][-1]["obj"], str)

    def test_a_model_name_with_path_characters_is_still_a_valid_filename(self):
        sid = save_session(_chat(), {"model": "minimax-m3:cloud"})
        assert ":" not in sid and load_session(sid) is not None

    def test_two_saves_in_the_same_second_do_not_overwrite_each_other(self, monkeypatch):
        """The id carries a one-second timestamp, so two conversations saved
        inside the same second used to collide — and the second silently
        destroyed the first."""
        fixed = session_module.datetime(2026, 8, 8, 12, 0, 0)

        class _Clock(session_module.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed

        monkeypatch.setattr(session_module, "datetime", _Clock)
        a = save_session(_chat(user="первое"), {"model": "m"})
        b = save_session(_chat(user="второе"), {"model": "m"})
        assert a != b
        assert {load_session(a)["preview"], load_session(b)["preview"]} == {"первое", "второе"}
