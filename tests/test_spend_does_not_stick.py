"""A balance delta must not stay on screen after the turn it described.

DeepSeek reports no per-request cost, so `last_spent` comes from a drop in the
account balance — and that balance only moves when a whole cent is billed. A cheap
turn leaves both the balance and `last_spent` untouched, so the same cent was
shown again on every following turn ("last~ $0.0100" five times in a row).
"""

import pytest

from argent.usage_tracker import SessionUsage


def after_a_cent_drop():
    u = SessionUsage()
    u.set_balance(3.85, "USD", provider="deepseek")
    u.set_balance(3.84, "USD", provider="deepseek")
    assert u.format_last_spend() == "last~ $0.0100"
    return u


class TestTheValueDoesNotStick:
    def test_a_cheap_turn_clears_the_previous_delta(self):
        u = after_a_cent_drop()
        u.add({"prompt": 800, "completion": 150})          # DeepSeek: cost == 0
        assert u.last_spent is None
        assert u.format_last_spend() == ""

    def test_five_cheap_turns_do_not_claim_five_cents(self):
        u = after_a_cent_drop()
        for _ in range(5):
            u.add({"prompt": 800, "completion": 150})
            u.set_balance(3.84, "USD", provider="deepseek")  # balance unmoved
            assert u.format_last_spend() == ""

    def test_the_next_real_drop_shows_again(self):
        u = after_a_cent_drop()
        u.add({"prompt": 800, "completion": 150})
        assert u.format_last_spend() == ""
        u.set_balance(3.83, "USD", provider="deepseek")
        assert u.format_last_spend() == "last~ $0.0100"

    def test_the_session_total_keeps_everything(self):
        u = after_a_cent_drop()
        u.add({"prompt": 800, "completion": 150})
        u.set_balance(3.83, "USD", provider="deepseek")
        assert u.session_spent == pytest.approx(0.02)


class TestAReportedCostIsNotThrownAway:
    def test_a_measured_cost_survives_a_costless_request(self):
        u = SessionUsage()
        u.add({"prompt": 100, "completion": 20, "cost": 0.0021})
        assert u.format_last_spend() == "last $0.0021"
        u.add({"prompt": 100, "completion": 20})           # no cost reported
        assert u.format_last_spend() == "last $0.0021", \
            "a measured figure is not overwritten by a different provider's silence"

    def test_nothing_known_stays_nothing(self):
        u = SessionUsage()
        u.add({"prompt": 100, "completion": 20})
        assert u.format_last_spend() == ""
