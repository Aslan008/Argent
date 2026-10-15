"""A context summary written ahead of time, in the background (roadmap M1b).

Compaction asks a model for a summary at the moment the history no longer fits. On a local model that is a
long wait in the middle of the person's turn. When the history has reached `START` of its budget, the summary
of what would be dropped is written NOW, between turns, at the lowest lane priority (the PC's own load and the
person's next message both come first). When the compaction does come, the ready summary is the PREVIOUS
SUMMARY the model only has to bring up to date with the few messages that arrived since: a short request, or no
request at all when nothing changed.

A ready summary is used only when it is provably about the same messages:

* the messages it covered are a prefix of what is being compacted (role and text; a tool result counts by its
  call id only, because aging shortens those in place and the summariser sees them cut anyway);
* the summary it was built on top of (the earlier distillate's) is the same one.

Anything else — an edited history, a new conversation, a failed call — and the compaction simply does what it
always did. The feature can only make a compaction faster, never different in kind.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from logger import get_logger

log = get_logger("summary_precompute")

#: Fraction of the history budget at which a summary starts to be written.
START = 0.85
#: Less than this many messages / characters to summarise is not worth a request ahead of time.
MIN_MESSAGES = 6
MIN_CHARS = 6000
#: After a failure the background request is not repeated for this long.
RETRY_AFTER = 180.0
#: A ready summary that already covers this share of what would be dropped is not written again.
REFRESH_SHARE = 0.35

_LOCK = threading.Lock()
_ready: Optional["Ready"] = None
_running: Optional[threading.Thread] = None
_generation = 0
_failed_at = -1e9


@dataclass(frozen=True)
class Ready:
    summary: str
    digests: Tuple[str, ...]
    prior: str
    made_at: float


def enabled() -> bool:
    if os.environ.get("ARGENT_PRECOMPUTE") == "0":
        return False
    try:
        import config
        return bool(config.is_summary_precompute_enabled())
    except Exception:
        return True


def _h(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def digest_of(message: Dict) -> str:
    """What identifies a message across the small edits a history gets in place."""
    from src.agent import compaction
    role = str(message.get("role", ""))
    if role == "tool":
        return f"tool:{message.get('tool_call_id', '')}"
    return f"{role}:{_h(compaction.message_text(message))}"


def fingerprint(messages: Sequence[Dict]) -> Tuple[str, ...]:
    return tuple(digest_of(m) for m in messages)


def _size(messages: Sequence[Dict]) -> int:
    from src.agent import compaction
    return sum(len(compaction.message_text(m)) for m in messages)


def worth_writing(fresh: Sequence[Dict]) -> bool:
    return len(fresh) >= MIN_MESSAGES and _size(fresh) >= MIN_CHARS


def _is_prefix(prefix: Sequence[str], whole: Sequence[str]) -> bool:
    return len(prefix) <= len(whole) and tuple(whole[:len(prefix)]) == tuple(prefix)


def start(fresh: Sequence[Dict], prior_summary: str, work: Callable[[], Optional[str]]) -> bool:
    """Write the summary of `fresh` in the background. `work` makes the request (and nothing else); it is
    called on a thread of its own, at the background lane priority. Returns True when a job was started."""
    global _running, _generation
    if not worth_writing(fresh):
        return False
    digests = fingerprint(fresh)
    prior = _h(prior_summary or "")
    with _LOCK:
        if _running is not None and _running.is_alive():
            return False
        if time.monotonic() - _failed_at < RETRY_AFTER:
            return False
        if (_ready is not None and _ready.prior == prior and _is_prefix(_ready.digests, digests)
                and len(digests) - len(_ready.digests) <= max(2, REFRESH_SHARE * len(digests))):
            return False                                     # what is ready is close enough: the rest is a short update
        generation = _generation

        def job() -> None:
            global _ready, _failed_at
            text = None
            try:
                from src.agent import governor
                with governor.priority(governor.P_BACKGROUND):
                    text = work()
            except Exception:
                log.exception("background summary failed")
            with _LOCK:
                if generation != _generation:
                    return                                   # claimed, reset or replaced meanwhile
                if text:
                    _ready = Ready(text, digests, prior, time.time())
                    log.info("background summary ready (%d messages, %d chars)", len(digests), len(text))
                else:
                    _failed_at = time.monotonic()

        _running = threading.Thread(target=job, daemon=True, name="summary-precompute")
        _running.start()
        return True


def claim(fresh: Sequence[Dict], prior_summary: str) -> Optional[Tuple[str, List[Dict]]]:
    """The ready summary and the messages it does not cover yet, when it is valid for `fresh`; else None.
    Taking it clears it: a summary is used once, and a job still running is told its result is not wanted."""
    global _ready, _generation
    with _LOCK:
        ready = _ready
        if ready is None:
            if _running is not None and _running.is_alive():
                _generation += 1
            return None
        digests = fingerprint(fresh)
        if ready.prior != _h(prior_summary or "") or not _is_prefix(ready.digests, digests):
            return None
        _ready = None
        _generation += 1
        return ready.summary, list(fresh[len(ready.digests):])


def reset() -> None:
    """A new conversation (or a test): nothing written so far belongs to it."""
    global _ready, _generation, _failed_at
    with _LOCK:
        _ready = None
        _failed_at = -1e9
        _generation += 1


def status() -> Dict[str, object]:
    with _LOCK:
        running = _running is not None and _running.is_alive()
        ready = _ready
    return {"running": running, "ready_messages": len(ready.digests) if ready else 0,
            "age": (time.time() - ready.made_at) if ready else None}
