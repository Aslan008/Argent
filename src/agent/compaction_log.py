"""What compaction did, and whether it is doing any good.

Kept apart from `trimmer` on purpose: anything that starts a new conversation (a restored session, /clear) must
be able to say "forget the old one's history" without importing the whole trimming machinery.

The history belongs to ONE conversation. It is process-wide state, so whoever replaces the conversation resets it
(`agent.clear_history`, restoring a session) and the tests start every test with it empty (tests/conftest.py).
"""

import threading
import time
from typing import Any, Dict, List, Optional

#: Consecutive compactions that each saved too little, within the window, make a "thrash".
THRASH_RUNS = 3
THRASH_WINDOW_SECONDS = 900.0
THRASH_MIN_SAVING = 0.25
_COMPACTIONS: List[Dict[str, Any]] = []
_COMPACTIONS_LOCK = threading.Lock()
_COMPACTIONS_KEEP = 20


def record_compaction(kind: str, compression_id: int, pre_tokens: int, post_tokens: int,
                      model_name: str = "", now: Optional[float] = None) -> Dict[str, Any]:
    """Remember one compaction: what it started from, what it left, and when."""
    saving = (1.0 - post_tokens / pre_tokens) if pre_tokens > 0 else 0.0
    entry = {"kind": kind, "id": compression_id, "pre_tokens": int(pre_tokens), "post_tokens": int(post_tokens),
             "saving": round(saving, 3), "model": model_name, "at": time.time() if now is None else now}
    with _COMPACTIONS_LOCK:
        _COMPACTIONS.append(entry)
        del _COMPACTIONS[:-_COMPACTIONS_KEEP]
    return entry


def last_compaction() -> Optional[Dict[str, Any]]:
    with _COMPACTIONS_LOCK:
        return dict(_COMPACTIONS[-1]) if _COMPACTIONS else None


def reset_compaction_history() -> None:
    with _COMPACTIONS_LOCK:
        _COMPACTIONS.clear()
    # a summary written ahead of time belongs to the conversation that is being replaced as well
    from src.agent import summary_precompute
    summary_precompute.reset()


def is_thrashing(now: Optional[float] = None) -> bool:
    """True when the last THRASH_RUNS compactions came close together and each saved little.

    The usual cause: every request brings tens of thousands of tokens that are NOT in the history yet
    (a huge file pasted again and again, the active window itself over the budget). More compaction
    cannot help, and each one costs a model call: the person should be told, and the summariser left alone."""
    now = time.time() if now is None else now
    with _COMPACTIONS_LOCK:
        recent = _COMPACTIONS[-THRASH_RUNS:]
    if len(recent) < THRASH_RUNS:
        return False
    if now - recent[0]["at"] > THRASH_WINDOW_SECONDS:
        return False
    return all(r["saving"] < THRASH_MIN_SAVING for r in recent)


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k".replace(".0k", "k") if n >= 1000 else str(n)


def describe_compaction(entry: Dict[str, Any]) -> str:
    return (f"Контекст сжат: {_fmt_tokens(entry['pre_tokens'])} → {_fmt_tokens(entry['post_tokens'])} токенов "
            f"({entry['saving'] * 100:.0f}% экономии)")


THRASH_ADVICE = ("Контекст сжимается слишком часто и почти ничего не выигрывает: в каждом запросе столько "
                 "данных, что они не умещаются даже после сжатия. Не вставляйте файлы целиком; пусть агент "
                 "читает нужные диапазоны строк. Модель для сжатия больше не вызывается, пока это не пройдёт.")
