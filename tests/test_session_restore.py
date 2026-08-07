"""Putting a session back — and saying what does NOT match.

Restoring the messages is the easy half. The system prompt is rebuilt for the
CURRENT directory and model, while the conversation still talks about the old
project's file paths and the working memory in .argent/memory.json belongs to
wherever you are standing now. Nothing downstream can detect that split, so
the restore has to say it out loud.
"""

import os

import pytest

import main as main_module
import session as session_module
from main import restore_session
from session import save_session


class _Agent:
    def __init__(self, model="local-7b"):
        self.messages = [{"role": "system", "content": "s"}]
        self.model_name = model
        self.session_id = None


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(session_module, "SESSIONS_DIR", d)
    monkeypatch.setattr(session_module, "INDEX_PATH", d / "index.json")
    monkeypatch.chdir(tmp_path)
    return d


@pytest.fixture
def said(monkeypatch):
    lines = []
    monkeypatch.setattr(main_module, "print_system", lambda t, *a, **k: lines.append(str(t)))
    monkeypatch.setattr(main_module, "print_error", lambda t, *a, **k: lines.append(str(t)))
    return lines


@pytest.fixture
def never_asks(monkeypatch):
    """Default: no prompt is expected. Tests that want one override it."""
    class _Q:
        def confirm(self, *a, **k):
            pytest.fail("nothing here should have asked the user")
    monkeypatch.setattr(main_module, "questionary", _Q())


def _saved(**meta):
    msgs = [{"role": "system", "content": "old prompt"},
            {"role": "user", "content": "почини Player.cs"},
            {"role": "assistant", "content": "готово"}]
    return save_session(msgs, meta)


class TestRestoring:
    def test_messages_come_back(self, said, never_asks):
        sid = _saved(model="local-7b")
        agent = _Agent()
        assert restore_session(agent, {"id": sid}) is True
        assert len(agent.messages) == 3

    def test_the_conversation_keeps_its_identity(self, said, never_asks):
        """So the next autosave updates this session instead of forking it."""
        sid = _saved(model="local-7b")
        agent = _Agent()
        restore_session(agent, {"id": sid})
        assert agent.session_id == sid

    def test_a_missing_session_is_reported_not_crashed(self, said, never_asks):
        assert restore_session(_Agent(), {"id": "nope"}) is False
        assert any("Не удалось" in l for l in said)


class TestModelMismatch:
    def test_a_different_model_is_called_out(self, said, never_asks):
        """The history is full of tool calls a weaker model may not reproduce."""
        sid = _saved(model="minimax-m3:cloud")
        restore_session(_Agent(model="qwen3.5:9b"), {"id": sid})
        warning = " ".join(said)
        assert "minimax-m3:cloud" in warning and "qwen3.5:9b" in warning

    def test_the_same_model_is_not_mentioned(self, said, never_asks):
        sid = _saved(model="local-7b")
        restore_session(_Agent(model="local-7b"), {"id": sid})
        assert not any("Сессия велась" in l for l in said)


class TestDirectory:
    def test_the_same_directory_asks_nothing(self, said, never_asks, tmp_path):
        sid = _saved(model="local-7b", cwd=str(tmp_path))
        restore_session(_Agent(), {"id": sid})
        assert not any("Перейти" in l for l in said)

    def test_a_different_directory_shows_both(self, said, monkeypatch, tmp_path):
        other = tmp_path / "OtherProject"
        other.mkdir()
        sid = _saved(model="local-7b", cwd=str(other))

        class _Q:
            def confirm(self, *a, **k):
                return type("A", (), {"ask": lambda s: False})()

        monkeypatch.setattr(main_module, "questionary", _Q())
        restore_session(_Agent(), {"id": sid})
        text = " ".join(said)
        assert "OtherProject" in text and str(tmp_path) in text

    def test_declining_leaves_the_directory_alone(self, said, monkeypatch, tmp_path):
        """A silent cd would send later file writes somewhere the user never
        asked for — the one class of action Argent always confirms."""
        other = tmp_path / "OtherProject"
        other.mkdir()
        sid = _saved(model="local-7b", cwd=str(other))
        before = os.getcwd()

        class _Q:
            def confirm(self, *a, **k):
                return type("A", (), {"ask": lambda s: False})()

        monkeypatch.setattr(main_module, "questionary", _Q())
        restore_session(_Agent(), {"id": sid})
        assert os.getcwd() == before
        assert any("другого" in l for l in said)      # the mismatch is restated

    def test_accepting_changes_the_directory(self, said, monkeypatch, tmp_path):
        other = tmp_path / "OtherProject"
        other.mkdir()
        sid = _saved(model="local-7b", cwd=str(other))

        class _Q:
            def confirm(self, *a, **k):
                return type("A", (), {"ask": lambda s: True})()

        monkeypatch.setattr(main_module, "questionary", _Q())
        restore_session(_Agent(), {"id": sid})
        assert os.path.samefile(os.getcwd(), other)

    def test_a_directory_that_no_longer_exists_is_not_offered(self, said, never_asks,
                                                              tmp_path):
        sid = _saved(model="local-7b", cwd=str(tmp_path / "deleted-project"))
        restore_session(_Agent(), {"id": sid})
        assert any("больше нет" in l for l in said)

    def test_a_session_from_before_cwd_was_recorded(self, said, never_asks, store):
        """Old sessions have no cwd at all; they must restore, not explode."""
        import gzip
        import json
        payload = {"id": "legacy", "saved_at": "2026-01-01T00:00:00", "model": "local-7b",
                   "provider": "ollama", "message_count": 2, "preview": "старое",
                   "messages": [{"role": "system", "content": "s"},
                                {"role": "user", "content": "привет"}]}
        with gzip.open(store / "legacy.json.gz", "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        assert restore_session(_Agent(), {"id": "legacy"}) is True
