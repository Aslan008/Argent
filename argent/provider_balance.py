"""Provider prepaid balance, fetched off the UI thread.

The TUI asks for state many times per second. A synchronous ``GET /user/balance``
inside that snapshot would freeze the screen for the duration of an HTTP round
trip, so this module keeps the last value and refreshes it in a daemon thread.
Only providers that actually expose a balance endpoint participate; the rest
simply omit the chip.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Balance:
    amount: float
    currency: str = "USD"


_TTL = 60.0
_FAIL_TTL = 20.0
_lock = threading.Lock()
_cache: dict[str, tuple[float, Optional[Balance]]] = {}
_inflight: set[str] = set()


def _provider_name(provider: Optional[str]) -> str:
    if provider:
        return provider
    try:
        from argent.config import get_provider

        return get_provider()
    except Exception:
        return ""


def get_balance_now(provider: Optional[str] = None) -> Optional[Balance]:
    """Synchronous fetch; used by the background refresher and tests."""
    name = _provider_name(provider)
    if not name:
        return None
    try:
        from argent.providers import create_provider

        prov = create_provider(name)
        fn = getattr(prov, "get_balance", None)
        if not callable(fn):
            return None
        result = fn()
        if not result:
            return None
        amount, currency = result
        return Balance(float(amount), str(currency or "USD").upper())
    except Exception:
        return None


def _refresh(name: str, force: bool) -> None:
    try:
        balance = get_balance_now(name)
        with _lock:
            _cache[name] = (time.monotonic(), balance)
    finally:
        with _lock:
            _inflight.discard(name)


def prime_balance(provider: Optional[str] = None, force: bool = False) -> None:
    """Start a refresh when the cached value is stale; never blocks the caller."""
    name = _provider_name(provider)
    if not name:
        return
    now = time.monotonic()
    with _lock:
        if name in _inflight:
            return
        cached = _cache.get(name)
        if cached and not force:
            ttl = _TTL if cached[1] is not None else _FAIL_TTL
            if now - cached[0] < ttl:
                return
        _inflight.add(name)
    threading.Thread(
        target=_refresh,
        args=(name, force),
        daemon=True,
        name=f"balance-{name}",
    ).start()


def current_balance(provider: Optional[str] = None) -> Optional[Balance]:
    """Last known balance; never performs network I/O."""
    name = _provider_name(provider)
    if not name:
        return None
    with _lock:
        cached = _cache.get(name)
        return cached[1] if cached else None


def clear_balance_cache() -> None:
    """Tests and provider switches: forget every cached reading."""
    with _lock:
        _cache.clear()
        _inflight.clear()
