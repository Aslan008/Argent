from datetime import datetime, timezone

import pytest

from argent.model_pricing import estimate_cost, is_peak
from src.bridge.repl_bridge import _spend_breakdown
from argent.usage_tracker import SessionUsage


OFF_PEAK = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)  # Tuesday
PEAK = datetime(2026, 10, 27, 7, 0, tzinfo=timezone.utc)      # Tuesday, 06-10 UTC
WEEKEND = datetime(2026, 10, 25, 7, 0, tzinfo=timezone.utc)   # Sunday


class TestDeepSeekPricing:
    def test_flash_peak_is_double_off_peak(self):
        kwargs = dict(provider="deepseek", model="deepseek-flash",
                      prompt=1_000_000, completion=1_000_000)
        off = estimate_cost(**kwargs, at=OFF_PEAK)
        peak = estimate_cost(**kwargs, at=PEAK)
        assert off == pytest.approx(0.75)
        assert peak == pytest.approx(1.50)

    def test_cached_prefix_uses_the_cache_hit_rate(self):
        cost = estimate_cost(
            "deepseek", "deepseek-flash", prompt=1_000_000, completion=0,
            cached=1_000_000, at=OFF_PEAK,
        )
        assert cost == pytest.approx(0.003)

    def test_pro_uses_its_own_rates(self):
        cost = estimate_cost(
            "deepseek", "deepseek-v4-pro", prompt=1_000_000,
            completion=1_000_000, at=OFF_PEAK,
        )
        assert cost == pytest.approx(2.64)

    def test_weekend_is_off_peak(self):
        assert is_peak(WEEKEND) is False
        assert is_peak(PEAK) is True
        assert is_peak(OFF_PEAK) is False

    def test_unknown_provider_or_model_has_no_estimate(self):
        assert estimate_cost("ollama", "llama3", 1000, 500, at=OFF_PEAK) is None
        assert estimate_cost("deepseek", "unknown-model", 1000, 500, at=OFF_PEAK) is None


class TestSpendHover:
    def test_hover_contains_estimate_and_a_clear_warning(self):
        u = SessionUsage()
        u.add({"prompt": 1_000_000, "completion": 1_000_000})
        u.set_balance(3.80, "USD", provider="deepseek")
        text = _spend_breakdown(u, "deepseek", "deepseek-flash")
        assert "est $" in text
        assert "не биллинг" in text

    def test_hover_explains_that_spent_is_account_wide(self):
        u = SessionUsage()
        u.add({"prompt": 100, "completion": 50})
        u.set_balance(3.85, "USD", provider="deepseek")
        u.set_balance(3.80, "USD", provider="deepseek")
        text = _spend_breakdown(u, "deepseek", "deepseek-flash")
        assert "Списано (аккаунт)" in text
        assert "другой клиент" in text.lower()
