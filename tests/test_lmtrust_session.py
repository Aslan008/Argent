"""Blind-spot tests for session.py.

Covers the helpers a happy-path save/load would never exercise alone:
filename sanitisation edge cases, serialisation of non-JSON values,
first-user-message filtering of injected prompts, surrogate survival,
positional vs substring session lookup, and newest-first ordering.
"""

import time
from pathlib import Path

import pytest

import session as session_mod


# ---------------------------------------------------------------------------
# Fixtures — isolate every test in a throwaway tmp_path so we never touch the
# user's real ~/.argent/sessions directory.
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_sessions(tmp_path, monkeypatch):
    """Point SESSIONS_DIR and INDEX_PATH at a tmp_path subdir."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_path = sessions_dir / "index.json"

    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_mod, "INDEX_PATH", index_path)
    return sessions_dir


def _user(content, **extra):
    msg = {"role": "user", "content": content}
    msg.update(extra)
    return msg


def _assistant(content):
    return {"role": "assistant", "content": content}


# ---------------------------------------------------------------------------
# 1 — _sanitize_for_filename
# ---------------------------------------------------------------------------
class TestSanitizeForFilename:
    def test_replaces_dots_slashes_colons_spaces(self):
        assert session_mod._sanitize_for_filename("a.b/c\\d:e f") == "a-b-c-d-e-f"

    def test_replaces_all_problematic_chars(self):
        # Every char in the forbidden set becomes a dash
        name = 'file*?"<>| name'
        out = session_mod._sanitize_for_filename(name)
        for c in '.\\/:*?"<>| ':
            assert c not in out

    def test_empty_string_returns_unknown(self):
        assert session_mod._sanitize_for_filename("") == "unknown"

    def test_only_dashes_returns_unknown(self):
        # After stripping leading/trailing dashes, nothing is left
        assert session_mod._sanitize_for_filename("...") == "unknown"
        assert session_mod._sanitize_for_filename("   ") == "unknown"

    def test_strips_leading_trailing_dashes(self):
        assert session_mod._sanitize_for_filename(".name.") == "name"

    def test_non_string_input_stringified(self):
        # str(name) is taken, so an int is fine
        assert session_mod._sanitize_for_filename(123) == "123"


# ---------------------------------------------------------------------------
# 2 & 3 — _make_serializable
# ---------------------------------------------------------------------------
class TestMakeSerializable:
    def test_none_returns_empty_list(self):
        assert session_mod._make_serializable(None) == []

    def test_empty_list_returns_empty_list(self):
        assert session_mod._make_serializable([]) == []

    def test_non_serializable_value_converted_to_str(self):
        # A set is not JSON-serializable — placed directly as a field value so
        # json.dumps(value) fails and the whole value is stringified.
        msgs = [{"role": "user", "content": "hi", "tags": {1, 2, 3}}]
        out = session_mod._make_serializable(msgs)
        assert len(out) == 1
        # The set value should have been stringified
        val = out[0]["tags"]
        assert isinstance(val, str)
        assert "1" in val and "2" in val and "3" in val

    def test_serializable_values_preserved(self):
        msgs = [{"role": "user", "content": "hello", "count": 5}]
        out = session_mod._make_serializable(msgs)
        assert out[0]["content"] == "hello"
        assert out[0]["count"] == 5

    def test_does_not_mutate_input(self):
        msgs = [{"role": "user", "content": "hi"}]
        session_mod._make_serializable(msgs)
        assert msgs == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# 4 — _first_user_message
# ---------------------------------------------------------------------------
class TestFirstUserMessage:
    def test_finds_first_real_user_message(self):
        msgs = [
            _assistant("hi there"),
            _user("What is 2+2?"),
            _user("Another question"),
        ]
        assert session_mod._first_user_message(msgs) == "What is 2+2?"

    def test_skips_you_are_prefixed(self):
        msgs = [
            {"role": "user", "content": "You are a helpful assistant."},
            _user("Please act as a coder."),
            _user("Real question here"),
        ]
        assert session_mod._first_user_message(msgs) == "Real question here"

    def test_skips_all_injected_prefixes(self):
        msgs = [
            _user("You stopped after the last edit."),
            _user("Your file write was cut off mid-way."),
            _user("Your tool call was cut off."),
            _user("Please act as a reviewer."),
            _user("Finally a real one"),
        ]
        assert session_mod._first_user_message(msgs) == "Finally a real one"

    def test_returns_none_if_no_user_messages(self):
        msgs = [_assistant("only assistant")]
        assert session_mod._first_user_message(msgs) is None

    def test_returns_none_if_only_injected(self):
        msgs = [_user("You are a helpful assistant.")]
        assert session_mod._first_user_message(msgs) is None

    def test_skips_non_string_content(self):
        msgs = [
            {"role": "user", "content": ["list", "content"]},  # not a str
            _user("real message"),
        ]
        assert session_mod._first_user_message(msgs) == "real message"

    def test_skips_empty_content(self):
        msgs = [
            {"role": "user", "content": "   "},
            _user("real"),
        ]
        assert session_mod._first_user_message(msgs) == "real"

    def test_case_insensitive_prefix_check(self):
        msgs = [
            _user("YOU ARE an assistant"),
            _user("real question"),
        ]
        assert session_mod._first_user_message(msgs) == "real question"


# ---------------------------------------------------------------------------
# 5 — save_session returns None when no user message
# ---------------------------------------------------------------------------
class TestSaveNoUserMessage:
    def test_returns_none_when_no_user_message(self, isolated_sessions):
        msgs = [_assistant("only assistant")]
        result = session_mod.save_session(msgs, metadata={"model": "test"})
        assert result is None

    def test_returns_none_when_only_injected(self, isolated_sessions):
        msgs = [_user("You are a helpful assistant.")]
        result = session_mod.save_session(msgs, metadata={"model": "test"})
        assert result is None

    def test_no_file_created_when_no_user_message(self, isolated_sessions):
        msgs = [_assistant("only assistant")]
        session_mod.save_session(msgs, metadata={"model": "test"})
        gz_files = list(isolated_sessions.glob("*.json.gz"))
        assert gz_files == []


# ---------------------------------------------------------------------------
# 6 — save_session / load_session roundtrip
# ---------------------------------------------------------------------------
class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_messages(self, isolated_sessions):
        msgs = [
            _user("Hello, how are you?"),
            _assistant("I'm fine, thanks!"),
            _user("Can you help me with Python?"),
        ]
        sid = session_mod.save_session(
            msgs, metadata={"model": "test-model", "provider": "test-provider"},
            label="My Session",
        )
        assert sid is not None

        loaded = session_mod.load_session(sid)
        assert loaded is not None
        assert loaded["messages"] == msgs
        assert loaded["model"] == "test-model"
        assert loaded["provider"] == "test-provider"
        assert loaded["label"] == "My Session"
        assert loaded["id"] == sid

    def test_roundtrip_preserves_message_count(self, isolated_sessions):
        msgs = [_user("hi"), _assistant("hello"), _user("bye")]
        sid = session_mod.save_session(msgs, metadata={"model": "m"})
        loaded = session_mod.load_session(sid)
        assert loaded["message_count"] == 3

    def test_load_nonexistent_returns_none(self, isolated_sessions):
        assert session_mod.load_session("does_not_exist") is None


# ---------------------------------------------------------------------------
# 7 — surrogates in content don't crash
# ---------------------------------------------------------------------------
class TestSurrogates:
    def test_save_with_surrogates_does_not_crash(self, isolated_sessions):
        # A lone surrogate that cannot be encoded as UTF-8
        surrogate_content = "Hello \ud800 world \udc00 end"
        msgs = [_user(surrogate_content)]
        sid = session_mod.save_session(msgs, metadata={"model": "test"})
        assert sid is not None

        loaded = session_mod.load_session(sid)
        assert loaded is not None
        # The surrogate should have been repaired (replaced or recombined)
        assert "Hello" in loaded["messages"][0]["content"]


# ---------------------------------------------------------------------------
# 8 — delete_session
# ---------------------------------------------------------------------------
class TestDeleteSession:
    def test_delete_existing_returns_true(self, isolated_sessions):
        sid = session_mod.save_session([_user("delete me")], metadata={"model": "m"})
        assert sid is not None
        assert session_mod.delete_session(sid) is True
        # File is gone
        assert not session_mod._session_path(sid).exists()

    def test_delete_nonexistent_returns_false(self, isolated_sessions):
        assert session_mod.delete_session("never_existed") is False

    def test_delete_removes_from_index(self, isolated_sessions):
        sid = session_mod.save_session([_user("index me")], metadata={"model": "m"})
        index = session_mod._read_index()
        assert sid in index
        session_mod.delete_session(sid)
        index = session_mod._read_index()
        assert sid not in index


# ---------------------------------------------------------------------------
# 9 — _session_id_from_path
# ---------------------------------------------------------------------------
class TestSessionIdFromPath:
    def test_strips_suffix(self):
        p = Path("/some/dir/20240101_120000_model.json.gz")
        assert session_mod._session_id_from_path(p) == "20240101_120000_model"

    def test_no_suffix_returns_name(self):
        p = Path("/some/dir/plain_name")
        assert session_mod._session_id_from_path(p) == "plain_name"

    def test_preserves_subdirectories_in_name(self):
        p = Path("nested/deep/20240101_model.json.gz")
        assert session_mod._session_id_from_path(p) == "20240101_model"


# ---------------------------------------------------------------------------
# 10 & 11 — find_session
# ---------------------------------------------------------------------------
class TestFindSession:
    def _make_sessions(self, isolated_sessions, count=3):
        """Create a few sessions and return their metadata sorted newest-first."""
        sids = []
        for i in range(count):
            sid = session_mod.save_session(
                [_user(f"message number {i}")],
                metadata={"model": f"model-{i}"},
                label=f"session-{i}",
            )
            sids.append(sid)
            time.sleep(0.01)  # ensure distinct saved_at timestamps
        return session_mod.list_sessions()

    def test_digit_returns_positional(self, isolated_sessions):
        sessions = self._make_sessions(isolated_sessions, count=3)
        # "1" should return the newest (first in the sorted list)
        result = session_mod.find_session("1", sessions=sessions)
        assert result is not None
        assert result == sessions[0]

    def test_digit_out_of_range_returns_none(self, isolated_sessions):
        sessions = self._make_sessions(isolated_sessions, count=2)
        assert session_mod.find_session("5", sessions=sessions) is None

    def test_substring_matches(self, isolated_sessions):
        sessions = self._make_sessions(isolated_sessions, count=3)
        # "session-1" is a unique label substring
        result = session_mod.find_session("session-1", sessions=sessions)
        assert result is not None
        assert "session-1" == result.get("label")

    def test_empty_reference_returns_none(self, isolated_sessions):
        self._make_sessions(isolated_sessions, count=1)
        assert session_mod.find_session("", sessions=[]) is None
        assert session_mod.find_session(None, sessions=[]) is None

    def test_no_match_returns_none(self, isolated_sessions):
        sessions = self._make_sessions(isolated_sessions, count=2)
        assert session_mod.find_session("zzzznomatch", sessions=sessions) is None


# ---------------------------------------------------------------------------
# 12 — list_sessions sorted newest first
# ---------------------------------------------------------------------------
class TestListSessionsOrdering:
    def test_sorted_newest_first(self, isolated_sessions):
        sids = []
        for i in range(4):
            sid = session_mod.save_session(
                [_user(f"msg {i}")], metadata={"model": "m"}, label=f"sess-{i}",
            )
            sids.append(sid)
            time.sleep(0.02)  # distinct timestamps

        sessions = session_mod.list_sessions()
        assert len(sessions) == 4
        # saved_at should be in descending order
        timestamps = [s.get("saved_at", "") for s in sessions]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_empty_when_no_sessions(self, isolated_sessions):
        assert session_mod.list_sessions() == []

    def test_query_filters_results(self, isolated_sessions):
        session_mod.save_session([_user("alpha")], metadata={"model": "m"}, label="first")
        time.sleep(0.02)
        session_mod.save_session([_user("beta")], metadata={"model": "m"}, label="second")

        results = session_mod.list_sessions("first")
        assert len(results) == 1
        assert results[0]["label"] == "first"