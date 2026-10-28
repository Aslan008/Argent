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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

PRICE_VERSION = "2026-10-17"


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
    cache-hit rate. Nothing here is an invoice.
    """
    family = _model_family(provider, model)
    if family is None:
        return None
    price = _DEEPSEEK_PRICES[family]
    prompt = max(0, int(prompt or 0))
    completion = max(0, int(completion or 0))
    cached = min(prompt, max(0, int(cached or 0)))
    miss = prompt - cached
    multiplier = 2.0 if is_peak(at) else 1.0
    per_million = cached * price.cache_hit + miss * price.cache_miss + completion * price.output
    return per_million / 1_000_000.0 * multiplier
