"""Approximate per-token prices for Argent's own requests.

This is deliberately labelled an estimate: it uses the public list prices and
the token counts Argent received, but not the provider's invoice. Cache misses,
reasoning tokens, retries and Chinese public holidays can move the real bill.

DeepSeek prices are from the Models & Pricing page checked 2026-10-17:
off-peak (peak is doubled) USD per 1M tokens.

    flash:       cache hit 0.003, cache miss 0.15, output 0.60
    v4-pro:      cache hit 0.022, cache miss 0.66, output 1.98

Peak hours are 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday, excluding
Chinese public holidays. The weekdays are applied; holidays are not, so the
result is an estimate by design.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from argent.logger import get_logger

log = get_logger(__name__)

#: Providers that run on this machine: their tokens are never billed, so no table
#: entry (or a same-named hosted model) may turn into a cost estimate.
LOCAL_PROVIDERS = frozenset({"ollama", "koboldcpp", "local", "llamacpp", "lmstudio"})

PRICE_VERSION = "2026-10-17"

#: Public price list for thousands of models, kept by the LiteLLM project. It is a
#: plain JSON of per-token costs, so nothing is typed in by hand and a model released
#: tomorrow is priced the day the list is refreshed. Downloaded at most weekly, in a
#: background thread, into a local file — a turn never waits for the network.
PRICE_URL = ("https://raw.githubusercontent.com/BerriAI/litellm/main/"
             "model_prices_and_context_window.json")
CACHE_TTL_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class TokenPrice:
    cache_hit: float
    cache_miss: float
    output: float


_DEEPSEEK_PRICES: dict[str, TokenPrice] = {
    "deepseek-flash": TokenPrice(0.003, 0.15, 0.60),
    "deepseek-v4-pro": TokenPrice(0.022, 0.66, 1.98),
}


def _model_family(provider: str, model: str) -> Optional[str]:
    if (provider or "").strip().lower() != "deepseek":
        return None
    name = (model or "").strip().lower()
    if "v4-pro" in name or "v4_pro" in name or name.endswith("-pro"):
        return "deepseek-v4-pro"
    if "flash" in name or "chat" in name or "reasoner" in name:
        return "deepseek-flash"
    return None


def is_peak(at: Optional[datetime] = None) -> bool:
    """DeepSeek peak hours in UTC, weekdays only (holidays are not modelled)."""
    now = at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    return 1 <= now.hour < 4 or 6 <= now.hour < 10


def _cache_path() -> Path:
    base = Path(os.environ.get("ARGENT_HOME") or (Path.home() / ".argent"))
    return base / "model_prices.json"


_PRICES: Optional[Dict[str, dict]] = None
_PRICES_LOCK = threading.Lock()
_REFRESH_STARTED = False


def _load_cache() -> Dict[str, dict]:
    """The downloaded price table, read once per process."""
    global _PRICES
    with _PRICES_LOCK:
        if _PRICES is not None:
            return _PRICES
        try:
            path = _cache_path()
            raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            _PRICES = raw.get("models", {}) if isinstance(raw, dict) else {}
        except Exception as e:
            log.debug("price cache unreadable: %s", e)
            _PRICES = {}
        return _PRICES


def prices_age() -> Optional[float]:
    """Seconds since the table was downloaded, or None when there is none."""
    try:
        path = _cache_path()
        return time.time() - path.stat().st_mtime if path.exists() else None
    except Exception:
        return None


def lookup_price(model: str) -> Optional[dict]:
    """Per-token prices for `model` from the local table, or None.

    The public list names models as `provider/model`; Argent is given just the
    model, so an exact match is tried first and a suffix match after that.
    """
    table = _load_cache()
    name = (model or "").strip().lower()
    if not table or not name:
        return None
    if name in table:
        return table[name]
    for key, value in table.items():
        low = key.lower()
        # Only a `provider/model` tail counts. A substring test used to live here and
        # it priced whatever happened to contain the name: an unknown model got a
        # neighbour's price, and `llama3` matched every local entry in the table.
        if low.endswith("/" + name):
            return value
    return None


def refresh_prices(force: bool = False, timeout: int = 10) -> bool:
    """Download the public price list. True when the local table was replaced.

    Nothing is typed in by hand: the list is a JSON of per-token costs published
    for thousands of models, refreshed at most weekly rather than per turn.
    """
    global _PRICES
    age = prices_age()
    if not force and age is not None and age < CACHE_TTL_SECONDS:
        return False
    try:
        import urllib.request

        with urllib.request.urlopen(PRICE_URL, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        log.debug("price list not downloaded: %s", e)
        return False
    models: Dict[str, dict] = {}
    for name, info in raw.items() if isinstance(raw, dict) else []:
        if not isinstance(info, dict):
            continue
        try:
            inp = float(info.get("input_cost_per_token") or 0.0)
            out = float(info.get("output_cost_per_token") or 0.0)
        except (TypeError, ValueError):
            continue
        if inp <= 0 and out <= 0:
            continue                  # a free or local model says nothing about price
        cache = info.get("cache_read_input_token_cost")
        try:
            cache_hit = float(cache) if cache else inp
        except (TypeError, ValueError):
            cache_hit = inp
        models[name] = {"in": inp, "out": out, "cache": cache_hit}
    if not models:
        return False
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"version": PRICE_VERSION, "models": models}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        log.debug("price cache not written: %s", e)
        return False
    with _PRICES_LOCK:
        _PRICES = models
    return True


def ensure_prices_async() -> None:
    """Refresh a stale table in the background; a request never waits for the network."""
    global _REFRESH_STARTED
    age = prices_age()
    if age is not None and age < CACHE_TTL_SECONDS:
        return
    with _PRICES_LOCK:
        if _REFRESH_STARTED:
            return
        _REFRESH_STARTED = True
    threading.Thread(target=lambda: refresh_prices(force=True), name="price-refresh", daemon=True).start()


def estimate_cost(
    provider: str,
    model: str,
    prompt: int,
    completion: int,
    cached: int = 0,
    at: Optional[datetime] = None,
) -> Optional[float]:
    """Estimated USD cost of Argent's own tokens, or None for an unknown model.

    ``prompt`` includes the cached prefix; ``cached`` is the part billed at the
    cache-hit rate. DeepSeek's list is built in (it also needs the peak-hour
    multiplier); everything else is looked up in the downloaded public table, so no
    price is ever entered by hand. Nothing here is an invoice.
    """
    prompt = max(0, int(prompt or 0))
    completion = max(0, int(completion or 0))
    cached = min(prompt, max(0, int(cached or 0)))
    miss = prompt - cached

    # A model running on this machine costs nothing per token, whatever the public
    # table says about someone's hosted build with the same name.
    if str(provider or "").strip().lower() in LOCAL_PROVIDERS:
        return None

    family = _model_family(provider, model)
    if family is not None:
        price = _DEEPSEEK_PRICES[family]
        multiplier = 2.0 if is_peak(at) else 1.0
        per_million = cached * price.cache_hit + miss * price.cache_miss + completion * price.output
        return per_million / 1_000_000.0 * multiplier

    entry = lookup_price(model)
    if entry is None:
        # No table yet (first run or an unknown model): fetch it for the NEXT turn.
        ensure_prices_async()
        return None
    per_million = (cached * entry["cache"] + miss * entry["in"] + completion * entry["out"]) * 1_000_000.0
    return per_million / 1_000_000.0
