"""Telling "the network is down" from "the provider is down", and waiting out either.

Until now a dropped connection ended the turn: the streaming path had no retry at all, `with_retry`
only knew HTTP status codes written in the message, and a cloud client waited ten minutes on a dead
socket before noticing. Everything that decides *whether* a failure is worth waiting for lives here,
so the agent loop, the providers and the tools agree:

* ``is_transport_error``   — the exception (or anything it was raised from) is a broken/absent connection;
* ``reachability``         — "up" | "provider_down" | "offline", measured with plain TCP connects
                             (a local server such as Ollama needs no internet and is probed alone);
* ``RecoveryWait``         — waits for the connection to come back (polling cheaply) or for a busy provider
                             to calm down (exponential back-off, honouring Retry-After), narrating it;
* ``retry_transient``      — the small in-place retry for idempotent tool requests (web fetch, search).

Nothing here talks to a model; it is pure standard library and every clock/sleep/probe is injectable.
"""

from __future__ import annotations

import ipaddress
import os
import random
import socket
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Tuple
from urllib.parse import urlparse

# ── is this a broken connection? ─────────────────────────────────────────────

#: class-name fragments (lower case) of the exceptions HTTP stacks raise for a connection that is not there
_TRANSPORT_NAMES = (
    "connect", "connection", "timeout", "timedout", "readerror", "writeerror", "remoteprotocol",
    "protocolerror", "chunkedencoding", "gaierror", "brokenpipe", "incompleteread", "networkerror",
    "transporterror",
)
#: message fragments (lower case) that say the same in words
_TRANSPORT_HINTS = (
    "connection error", "connection reset", "connection aborted", "connection refused", "connection closed",
    "remote end closed", "server disconnected", "peer closed", "incomplete chunked", "name resolution",
    "getaddrinfo", "temporary failure in name", "network is unreachable", "no route to host",
    "nodename nor servname", "timed out", "no such host", "eof occurred", "broken pipe", "stream idle",
    "connection lost", "network error",
)


def _chain(exc: Optional[BaseException]):
    seen, stack = [], [exc]
    while stack and len(seen) < 8:
        e = stack.pop()
        if e is None or any(e is s for s in seen):
            continue
        seen.append(e)
        yield e
        stack.extend([getattr(e, "original_error", None), getattr(e, "__cause__", None), getattr(e, "__context__", None)])


def is_transport_error(exc: Optional[BaseException]) -> bool:
    """True when `exc`, or an exception it was raised from, is a connection that failed, dropped or timed out."""
    for e in _chain(exc):
        if isinstance(e, (ConnectionError, TimeoutError, socket.gaierror, socket.timeout, ssl.SSLError, EOFError)):
            return True
        if any(n in type(e).__name__.lower() for n in _TRANSPORT_NAMES):
            return True
        text = str(e).lower()
        if any(h in text for h in _TRANSPORT_HINTS):
            return True
    return False


# ── is the other end reachable? ──────────────────────────────────────────────

#: well-known anchors reachable by IP (no DNS needed); one answering means "the internet is up"
INTERNET_ANCHORS: Tuple[Tuple[str, int], ...] = (("1.1.1.1", 443), ("8.8.8.8", 443), ("9.9.9.9", 443))


def tcp_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _is_local(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host in ("localhost", "") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local
    except ValueError:
        return False


def endpoint_of(provider) -> Optional[Tuple[str, int, bool]]:
    """(host, port, is_local) the provider talks to, or None when it cannot be told."""
    if provider is None:
        return None
    name = str(getattr(provider, "name", "") or "").lower()
    url = getattr(provider, "_base_url", None)
    if not url and name == "ollama":
        url = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if not url and name == "claude":
        url = "https://api.anthropic.com"
    if not url:
        return None
    url = str(url)
    if "://" not in url:
        url = "http://" + url
    try:
        parts = urlparse(url)
        host = parts.hostname or ""
        if not host:
            return None
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None
    return host, port, _is_local(host)


def internet_up(timeout: float = 3.0, probe: Callable[[str, int, float], bool] = tcp_open) -> bool:
    """Any anchor answering within `timeout` (they are tried in parallel, so the wait is one timeout, not three)."""
    hits = []

    def _one(host: str, port: int) -> None:
        if probe(host, port, timeout):
            hits.append(1)

    threads = [threading.Thread(target=_one, args=a, daemon=True) for a in INTERNET_ANCHORS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 0.5)
    return bool(hits)


def reachability(provider=None, timeout: float = 3.0, probe: Callable[[str, int, float], bool] = tcp_open) -> str:
    """"up": the provider's endpoint answers; "provider_down": the internet is there but the endpoint is not;
    "offline": nothing answers. A local endpoint (Ollama, a LAN server) is judged by itself alone."""
    ep = endpoint_of(provider)
    if ep is None:                       # nothing to ask but the internet itself
        return "up" if internet_up(timeout, probe) else "offline"
    host, port, local = ep
    if probe(host, port, timeout):
        return "up"
    if local:
        return "provider_down"
    return "provider_down" if internet_up(timeout, probe) else "offline"


# ── how long to be patient ───────────────────────────────────────────────────

DEFAULT_PATIENCE_S = 900.0


def patience_seconds() -> float:
    """How long a turn waits for a lost connection or a busy provider before giving up (0 = do not wait)."""
    raw = os.environ.get("ARGENT_NETWORK_PATIENCE")
    if raw is None:
        try:
            from config import get_network_patience
            return get_network_patience()
        except Exception:
            return DEFAULT_PATIENCE_S
    try:
        return max(0.0, min(86400.0, float(raw)))
    except ValueError:
        return DEFAULT_PATIENCE_S


#: pause before the next attempt when the provider answered but is busy (seconds, by attempt number)
BACKOFF = (2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
MAX_BACKOFF = 120.0
#: while the connection itself is gone, look again this often (a TCP connect is nearly free)
POLL_S = 3.0
#: a streak of failures that are the provider's own is not retried for ever, whatever the patience says
MAX_PROVIDER_ATTEMPTS = 12


def backoff_delay(attempt: int, retry_after: Optional[float] = None, rng: Optional[random.Random] = None) -> float:
    rng = rng or random
    base = BACKOFF[min(max(0, attempt), len(BACKOFF) - 1)]
    wanted = min(MAX_BACKOFF, max(base, retry_after or 0.0))
    return wanted * (0.85 + 0.3 * rng.random())


def clock_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


@dataclass
class Streak:
    """The failures of one request in a row; success (a whole stream received) clears it."""
    started: Optional[float] = None
    attempts: int = 0

    def fail(self, now: float) -> int:
        if self.started is None:
            self.started = now
        self.attempts += 1
        return self.attempts - 1

    def reset(self) -> None:
        self.started, self.attempts = None, 0

    def waited(self, now: float) -> float:
        return 0.0 if self.started is None else max(0.0, now - self.started)


@dataclass
class RecoveryWait:
    """Wait for a failed request to be worth sending again, narrating it. Iterate `run()`: it yields the lines to
    show the person; afterwards `recovered` says whether to retry (True) or give up (False) and `reason` why."""

    provider: object = None
    streak: Streak = field(default_factory=Streak)
    patience: float = DEFAULT_PATIENCE_S
    kind: str = "busy"                  # "transport" | "busy" | "rate_limited"
    retry_after: Optional[float] = None
    detail: str = ""
    stop: Callable[[], bool] = lambda: False
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    probe: Callable[[str, int, float], bool] = tcp_open
    progress_every: float = 30.0
    recovered: bool = False
    reason: str = ""
    state: str = "up"

    def _nap(self, seconds: float) -> bool:
        """Sleep in slices so a stop request is honoured within a fraction of a second; True when stopped."""
        end = self.clock() + seconds
        while True:
            if self.stop():
                return True
            left = end - self.clock()
            if left <= 0:
                return False
            self.sleep(min(0.25, left))

    def run(self) -> Iterator[str]:
        now = self.clock()
        attempt = self.streak.fail(now)
        deadline = (self.streak.started or now) + self.patience
        provider_side = self.kind != "transport"
        if self.patience <= 0 or (provider_side and attempt >= MAX_PROVIDER_ATTEMPTS):
            self.recovered, self.reason = False, "no-patience" if self.patience <= 0 else "too-many-attempts"
            return

        self.state = reachability(self.provider, probe=self.probe) if self.kind == "transport" else "up"
        if self.state == "up":
            delay = backoff_delay(attempt, self.retry_after)
            if self.clock() + delay > deadline:
                self.recovered, self.reason = False, "patience-exhausted"
                return
            if attempt > 0 or delay > 3.5:
                what = "перегружен или лимит запросов" if self.kind != "transport" else "соединение оборвалось"
                tail = f" ({self.detail})" if self.detail else ""
                yield f"[Argent: провайдер: {what}{tail}. Повтор через {delay:.0f} с, попытка {attempt + 1}. Esc — прервать.]"
            if self._nap(delay):
                self.recovered, self.reason = False, "stopped"
                return
            self.recovered, self.reason = True, "retry"
            return

        # The connection itself is gone: wait for it, looking cheaply and often.
        lost = "интернет-соединение потеряно" if self.state == "offline" else "провайдер недоступен"
        until = clock_text(max(0.0, deadline - self.clock()))
        yield (f"[Argent: {lost}. Жду восстановления (до {until}); диалог сохранён, ход продолжится сам. "
               f"Esc — прервать.]")
        next_note = self.clock() + self.progress_every
        while True:
            if self._nap(POLL_S):
                self.recovered, self.reason = False, "stopped"
                return
            if self.clock() >= deadline:
                self.recovered, self.reason = False, "patience-exhausted"
                return
            state = reachability(self.provider, probe=self.probe)
            if state == "up":
                self.recovered, self.reason = True, "back"
                yield f"[Argent: связь восстановлена через {clock_text(self.streak.waited(self.clock()))}, повторяю запрос.]"
                return
            if state != self.state:
                self.state = state
            if self.clock() >= next_note:
                next_note = self.clock() + self.progress_every
                yield (f"[Argent: всё ещё нет связи ({clock_text(self.streak.waited(self.clock()))}); "
                       f"жду, осталось до {clock_text(max(0.0, deadline - self.clock()))}.]")


# ── small in-place retry for idempotent tool requests ───────────────────────

def retry_transient(fn: Callable[[], object], attempts: int = 3, base: float = 0.7,
                    sleep: Callable[[float], None] = time.sleep, rng: Optional[random.Random] = None):
    """Call `fn`; when it fails with a broken connection, try again a couple of times with a short growing pause.
    Only for requests that are safe to repeat (GET). Anything that is not a transport failure propagates at once."""
    rng = rng or random
    for attempt in range(max(1, attempts)):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts - 1 or not is_transport_error(e):
                raise
            sleep(base * (2 ** attempt) * (0.8 + 0.4 * rng.random()))
