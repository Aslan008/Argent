"""What the LAST request took off the balance, and that it survives a restart.

The session totals answer "what has this run cost so far". After a restart they
start at zero, and the question people actually ask — "what did the last task
cost me" — was unanswerable exactly then. Two things had to be true:

1. the number is recorded per request (exact cost when the provider reports it,
   the balance drop otherwise, labelled approximate);
2. it is written into the session file and put back on restore, without the
   restored balance being mistaken for a spend of this run.
"""

import pytest

from argent.session import _spent_summary, load_session, save_session
from argent.usage_tracker import SessionUsage


class TestLastSpend:
    def test_exact_cost_wins_over_the_balance_drop(self):
        u = SessionUsage()
        u.set_balance(5.00, "USD", provider="openrouter")
        u.add({"prompt": 100, "completion": 50, "cost": 0.0021})
        assert u.last_spent == pytest.approx(0.0021)
        assert u.last_spent_approx is False
        assert u.format_last_spend() == "last $0.0021"

    def test_balance_drop_is_approximate(self):
        u = SessionUsage()
        u.set_balance(5.00, "USD", provider="deepseek")
        u.add({"prompt": 100, "completion": 50})       # DeepSeek reports no cost
        u.set_balance(4.97, "USD", provider="deepseek")
        assert u.last_spent == pytest.approx(0.03)
        assert u.last_spent_approx is True
        # Sub-cent precision, like every other cost in the UI: a balance drop
        # of a few cents is exactly the case where "$0.03" hides the detail.
        assert u.format_last_spend() == "last~ $0.0300"

    def test_a_rise_in_balance_is_not_a_spend(self):
        u = SessionUsage()
        u.set_balance(5.00, "USD", provider="deepseek")
        u.set_balance(9.00, "USD", provider="deepseek")   # topped up
        assert u.last_spent is None
        assert u.format_last_spend() == ""

    def test_rounding_noise_is_not_a_spend(self):
        u = SessionUsage()
        u.set_balance(5.0000, "USD", provider="deepseek")
        u.set_balance(4.9999, "USD", provider="deepseek")
        assert u.last_spent is None

    def test_nothing_known_says_nothing(self):
        # A local model: no cost, no balance. A "$0.00" here would read as
        # "this turn was free", which is a claim, not an absence.
        u = SessionUsage()
        u.add({"prompt": 100, "completion": 50})
        assert u.format_last_spend() == ""

    def test_non_usd_currency_is_labelled(self):
        u = SessionUsage()
        u.set_balance(100.0, "CNY", provider="deepseek")
        u.set_balance(99.5, "CNY", provider="deepseek")
        assert u.format_last_spend() == "last~ CNY 0.50"


class TestSnapshotRoundTrip:
    def test_snapshot_and_restore_keep_the_last_spend(self):
        u = SessionUsage()
        u.set_balance(5.00, "USD", provider="deepseek")
        u.add({"prompt": 1200, "completion": 340, "cached": 900})
        u.set_balance(4.97, "USD", provider="deepseek")

        fresh = SessionUsage()
        assert fresh.restore(u.snapshot()) is True
        assert fresh.last_spent == pytest.approx(0.03)
        assert fresh.last_spent_approx is True
        assert fresh.last_prompt == 1200 and fresh.last_completion == 340
        assert fresh.last_cached == 900
        assert fresh.balance == pytest.approx(4.97)
        assert fresh.format_last_spend() == "last~ $0.0300"

    def test_restoring_does_not_charge_the_previous_run(self):
        """The restored balance is a reading, not a spend: the next reading must
        measure only what THIS run took."""
        u = SessionUsage()
        u.set_balance(5.00, "USD", provider="deepseek")
        u.set_balance(4.97, "USD", provider="deepseek")

        fresh = SessionUsage()
        fresh.restore(u.snapshot())
        fresh.set_balance(4.97, "USD", provider="deepseek")   # same reading
        assert fresh.last_spent == pytest.approx(0.03)        # unchanged
        fresh.set_balance(4.95, "USD", provider="deepseek")   # this run spent
        assert fresh.last_spent == pytest.approx(0.02)

    def test_restore_of_nothing_is_false(self):
        assert SessionUsage().restore(None) is False
        assert SessionUsage().restore({}) is False

    def test_restore_tolerates_garbage(self):
        u = SessionUsage()
        assert u.restore({"prompt": "abc", "cost": None, "last_spent": "x"}) is True
        assert u.prompt == 0 and u.cost == 0.0 and u.last_spent == 0.0


class TestSessionFile:
    def test_usage_is_saved_and_loaded(self, tmp_path, monkeypatch):
        import argent.session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(session_mod, "INDEX_PATH", tmp_path / "index.json")

        u = SessionUsage()
        u.add({"prompt": 500, "completion": 100, "cost": 0.0042})
        sid = save_session([{"role": "user", "content": "hi"}],
                           {"model": "m"}, usage=u.snapshot())
        assert sid
        data = load_session(sid)
        assert data["usage"]["last_spent"] == pytest.approx(0.0042)
        assert data["spent"] == "$0.0042"

    def test_a_session_without_usage_still_saves(self, tmp_path, monkeypatch):
        import argent.session as session_mod

        monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(session_mod, "INDEX_PATH", tmp_path / "index.json")
        sid = save_session([{"role": "user", "content": "hi"}], {"model": "m"})
        assert sid and "usage" not in load_session(sid)


class TestSpentSummary:
    def test_exact_cost(self):
        assert _spent_summary({"cost": 0.0042}) == "$0.0042"
        assert _spent_summary({"cost": 1.5}) == "$1.50"

    def test_approximate_drop(self):
        assert _spent_summary({"last_spent": 0.03, "last_spent_approx": True}) == "~$0.0300"

    def test_nothing_known(self):
        assert _spent_summary({}) == ""
        assert _spent_summary({"cost": 0, "last_spent": None}) == ""
