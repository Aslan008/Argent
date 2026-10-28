"""Тарифы не вводятся руками: их тянет публичный список и кладёт в кеш.

Проверяется три вещи: поиск модели в кеше, расчёт по нему и то, что неизвестная
модель не роняет ход и не ходит в сеть на каждом запросе.
"""

import json

import pytest

from argent import model_pricing as mp


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A price table on disk, with the module's in-memory copy reset."""
    home = tmp_path / "argent"
    home.mkdir()
    (home / "model_prices.json").write_text(json.dumps({
        "version": "test",
        "models": {
            "openai/gpt-4o-mini": {"in": 1.5e-07, "out": 6e-07, "cache": 7.5e-08},
            "anthropic/claude-3-5-sonnet": {"in": 3e-06, "out": 1.5e-05, "cache": 3e-07},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("ARGENT_HOME", str(home))
    monkeypatch.setattr(mp, "_PRICES", None)
    return home


def test_a_model_is_found_by_its_short_name(cache):
    # Argent is given `gpt-4o-mini`; the public list calls it `openai/gpt-4o-mini`.
    assert mp.lookup_price("gpt-4o-mini")["in"] == pytest.approx(1.5e-07)
    assert mp.lookup_price("claude-3-5-sonnet")["out"] == pytest.approx(1.5e-05)
    assert mp.lookup_price("no-such-model") is None


def test_the_cost_of_an_unknown_family_comes_from_the_table(cache):
    cost = mp.estimate_cost("openai", "gpt-4o-mini", 100_000, 2_000)
    assert cost == pytest.approx(0.0162, rel=1e-6)

    # Cached input is billed at the cache rate, not the miss rate.
    cached_cost = mp.estimate_cost("openai", "gpt-4o-mini", 100_000, 0, cached=100_000)
    assert cached_cost == pytest.approx(0.0075, rel=1e-6)


def test_an_unknown_model_is_not_a_crash(cache, monkeypatch):
    started = []
    monkeypatch.setattr(mp, "ensure_prices_async", lambda: started.append(True))

    assert mp.estimate_cost("nobody", "unknown-model", 1000, 100) is None
    assert started, "a miss should schedule one background refresh, not a blocking fetch"


def test_a_fresh_table_is_not_downloaded_again(cache, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("свежий кеш не должен скачиваться повторно")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    assert mp.refresh_prices(force=False) is False


def test_deepseek_prices_stay_built_in(cache):
    """DeepSeek needs the peak-hour multiplier, so its list stays in the module."""
    cost = mp.estimate_cost("deepseek", "deepseek-chat", 1_000_000, 0, at=None)
    assert cost is not None and cost > 0
