"""Remembering what a scheduled run already reported.

Without this, a monitoring automation finds the same twenty items every run and
calls all twenty news — which is the same as having no automation at all.
"""

from datetime import datetime, timedelta

import pytest

from src.automation import memory
from tools.misc_tools import filter_new_items


@pytest.fixture(autouse=True)
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("project_paths.project_root_or_cwd", lambda: tmp_path)
    (tmp_path / ".argent").mkdir()
    memory.reset_scope()
    yield tmp_path
    memory.reset_scope()


class TestFiltering:
    def test_first_run_reports_everything(self):
        assert memory.filter_new(["a", "b"], scope="jobs") == ["a", "b"]

    def test_second_run_reports_only_the_new_one(self):
        memory.filter_new(["a", "b"], scope="jobs")
        assert memory.filter_new(["a", "b", "c"], scope="jobs") == ["c"]

    def test_scopes_do_not_bleed(self):
        memory.filter_new(["a"], scope="jobs")
        assert memory.filter_new(["a"], scope="prices") == ["a"]

    def test_duplicates_within_one_call_collapse(self):
        assert memory.filter_new(["a", "a", "b"], scope="s") == ["a", "b"]

    def test_whitespace_variants_are_the_same_item(self):
        """The same URL scraped twice often differs only by a stray newline;
        two spellings of one item would defeat the whole point."""
        memory.filter_new(["http://x/1"], scope="s")
        assert memory.filter_new([" http://x/1 \n"], scope="s") == []

    def test_long_items_are_hashed_not_stored_whole(self):
        blob = "x" * 5000
        memory.filter_new([blob], scope="s")
        stored = list(memory._load()["s"])
        assert len(stored) == 1 and stored[0].startswith("sha1:")

    def test_empty_input(self):
        assert memory.filter_new([], scope="s") == []


class TestTransaction:
    def test_a_run_that_finishes_commits(self):
        memory.set_scope("jobs")
        assert memory.filter_new(["a"]) == ["a"]
        assert memory.commit() == 1
        memory.reset_scope()
        assert memory.filter_new(["a"], scope="jobs") == []

    def test_a_run_that_fails_forgets_what_it_never_reported(self):
        """Marking on read would mean a crash silently swallows items that were
        never shown to anyone — losing exactly what the automation exists to
        catch. A repeat is annoying; a miss is invisible."""
        memory.set_scope("jobs")
        memory.filter_new(["a", "b"])
        assert memory.rollback() == 2
        memory.reset_scope()
        assert memory.filter_new(["a", "b"], scope="jobs") == ["a", "b"]

    def test_staged_items_do_not_repeat_within_one_run(self):
        memory.set_scope("jobs")
        memory.filter_new(["a"])
        assert memory.filter_new(["a", "b"]) == ["b"]

    def test_interactive_use_commits_immediately(self):
        """Nobody will commit for a call made outside a run, and the human is
        already looking at the answer."""
        assert memory.filter_new(["a"], scope="manual") == ["a"]
        assert memory.filter_new(["a"], scope="manual") == []


class TestPruning:
    def test_old_keys_expire(self):
        old = datetime.now() - timedelta(days=90)
        memory.filter_new(["ancient"], scope="s", now=old)
        assert memory.filter_new(["ancient"], scope="s", ttl_days=30) == ["ancient"]

    def test_recent_keys_survive(self):
        memory.filter_new(["fresh"], scope="s")
        assert memory.filter_new(["fresh"], scope="s", ttl_days=30) == []

    def test_scope_is_capped(self, monkeypatch):
        monkeypatch.setattr(memory, "MAX_KEYS_PER_SCOPE", 10)
        memory.filter_new([f"i{n}" for n in range(50)], scope="s")
        assert len(memory._load()["s"]) == 10


class TestAdmin:
    def test_forget_clears_one_scope(self):
        memory.filter_new(["a"], scope="jobs")
        memory.filter_new(["a"], scope="prices")
        assert memory.forget("jobs") == 1
        assert memory.filter_new(["a"], scope="jobs") == ["a"]
        assert memory.filter_new(["a"], scope="prices") == []

    def test_forget_unknown_scope(self):
        assert memory.forget("nope") == 0

    def test_stats(self):
        memory.filter_new(["a", "b"], scope="jobs")
        assert memory.stats() == {"jobs": 2}
        assert memory.stats("jobs") == {"jobs": 2}

    def test_corrupt_file_is_survivable(self, project):
        memory.store_path().write_text("{not json", encoding="utf-8")
        assert memory.filter_new(["a"], scope="s") == ["a"]


class TestTool:
    def test_reports_the_new_ones(self):
        memory.set_scope("jobs")
        out = filter_new_items(["http://a", "http://b"])
        assert "2 of 2 items are new" in out and "http://a" in out

    def test_tells_the_model_to_stay_quiet_when_nothing_changed(self):
        """A weak model handed an empty list will happily re-summarise the old
        one, so the instruction has to be in the tool result."""
        memory.filter_new(["http://a"], scope="jobs")
        memory.set_scope("jobs")
        out = filter_new_items(["http://a"])
        assert "0 of 1" in out and "already" in out

    def test_accepts_a_json_string(self):
        memory.set_scope("s")
        out = filter_new_items('["a", "b"]')
        assert "2 of 2" in out

    def test_accepts_newline_separated_text(self):
        memory.set_scope("s")
        out = filter_new_items("a\nb\nc")
        assert "3 of 3" in out

    def test_rejects_nonsense(self):
        assert filter_new_items(42).startswith("Error:")

    def test_empty_list(self):
        assert "nothing is new" in filter_new_items([])
