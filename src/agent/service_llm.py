"""A service-LLM call (context compression) that survives a busy, limited or dead free model.

Compression asks a model, usually a free OpenRouter one, to write a summary. Free models
fail in ways that call for different answers:

* BUSY      HTTP 5xx / 408, a timeout, "provider returned error", a body with no choices.
            The model is fine and the server is loaded: ask the SAME model again after a pause.
* RATE      HTTP 429 that clears soon (per-minute, "temporarily rate-limited upstream"):
            wait for it and ask the same model again.
* EXHAUSTED HTTP 429 "free-models-per-day", HTTP 402: asking again is pointless until the
            limit resets. Move to ANOTHER free model, the next one on the list or the one
            the user picks.
* UNFIT     404, an invalid model id, a context window too small: the same, move on.
* AUTH      HTTP 401/403: another model on the same key will not help. Stop.

Whatever happens the caller gets text or `None`, never an exception, in bounded time:

* every attempt runs on its own thread with a deadline and a late answer from an
  abandoned one is ignored, so a model that dies mid-answer cannot hang the turn or write
  into a later attempt;
* a whole call has a time budget, and a stop request (Esc / Ctrl+C) ends it at once;
* an answer is checked before it is believed (not empty, not an error message, reasoning
  tags removed), and a failed check counts as a failure of that model;
* a model that failed is remembered for a while (until its quota resets, or a couple of
  minutes when it was only busy) so the next compression does not walk into it again.

The caller keeps the history untouched until it holds a result, so `None` simply means
"build the summary without a model".
"""

from __future__ import annotations

import contextvars
import enum
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from logger import get_logger

log = get_logger("service_llm")


class Kind(enum.Enum):
    BUSY = "busy"
    RATE_LIMITED = "rate_limited"
    EXHAUSTED = "exhausted"
    UNFIT = "unfit"
    AUTH = "auth"
    BAD_OUTPUT = "bad_output"
    OTHER = "other"


_WHY = {
    Kind.BUSY: "перегружена или не отвечает",
    Kind.RATE_LIMITED: "временно ограничена по частоте запросов",
    Kind.EXHAUSTED: "исчерпала лимит",
    Kind.UNFIT: "недоступна для этого запроса",
    Kind.AUTH: "отклонила ключ доступа",
    Kind.BAD_OUTPUT: "вернула пустой или негодный ответ",
    Kind.OTHER: "вернула ошибку",
}

# How long a failed model is skipped (seconds).
_COOLDOWN = {Kind.BUSY: 120, Kind.RATE_LIMITED: 90, Kind.BAD_OUTPUT: 600, Kind.UNFIT: 86400, Kind.OTHER: 300}

BACKOFF = (3.0, 8.0, 20.0, 30.0, 30.0, 30.0)
MAX_DELAY = 45.0
MAX_SWITCHES = 5
MIN_SUMMARY_CHARS = 60
_FREE_CACHE_TTL = 600.0
# Timing floors, as constants so a test can shrink them.
_MIN_CALL = 4.0        # the shortest timeout an attempt is given
_SLACK = 8.0           # extra time an attempt thread gets beyond twice its own timeout
_MIN_LEFT = 6.0        # less budget than this left: no point starting another attempt
_MIN_SWITCH_LEFT = 20.0  # less than this left: no point moving to another model


class Cancelled(Exception):
    """The user asked the turn to stop while a compression was in progress."""


# ── stop requests ────────────────────────────────────────────────────────────

_cancel_stamp = 0.0


def request_cancel() -> None:
    """The agent was told to stop. Compressions that started before now give up."""
    global _cancel_stamp
    _cancel_stamp = time.monotonic()


def _cancelled_since(started: float) -> bool:
    # A stop pressed BEFORE this call started (an earlier turn) must not cancel it.
    return _cancel_stamp >= started


def _pause(seconds: float, started: float) -> bool:
    """Sleep up to `seconds`; True when a stop request cut the wait short."""
    end = time.monotonic() + seconds
    while True:
        if _cancelled_since(started):
            return True
        left = end - time.monotonic()
        if left <= 0:
            return False
        time.sleep(min(0.2, left))


# ── what went wrong ──────────────────────────────────────────────────────────

@dataclass
class ErrorInfo:
    kind: Kind
    status: Optional[int] = None
    detail: str = ""
    retry_after: Optional[float] = None
    reset_at: Optional[float] = None  # epoch seconds, when the quota is known to come back

    def why(self) -> str:
        return _WHY[self.kind]


_EXHAUSTED_HINTS = (
    "per-day", "per day", "daily", "free-models-per-day", "quota", "insufficient credits",
    "out of credits", "credit balance", "add credits", "billing", "requires more credits",
)
_UNFIT_HINTS = (
    "no endpoints found", "not a valid model", "model not found", "does not exist",
    "no longer available", "deprecated", "unknown model", "invalid model", "is not supported",
)
_CONTEXT_HINTS = ("context length", "context window", "maximum context", "too many tokens", "too long", "token limit")
_BUSY_HINTS = (
    "overload", "unavailable", "provider returned error", "busy", "capacity", "temporar", "timed out",
    "timeout", "connection", "reset by peer", "bad gateway", "gateway", "no choices", "empty response",
    "try again", "high demand", "upstream", "no available",
)
_AUTH_HINTS = ("invalid api key", "incorrect api key", "no auth", "unauthorized", "invalid key")


def _chain(exc: BaseException):
    seen = []
    stack = [exc]
    while stack and len(seen) < 6:
        e = stack.pop(0)
        if e is None or any(e is s for s in seen):
            continue
        seen.append(e)
        yield e
        stack.extend([getattr(e, "original_error", None), getattr(e, "__cause__", None), getattr(e, "__context__", None)])


def _epoch(value) -> Optional[float]:
    """OpenRouter's X-RateLimit-Reset is epoch milliseconds; Retry-After is seconds."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v / 1000.0 if v > 1e11 else v


def classify(exc: BaseException, now: Optional[float] = None) -> ErrorInfo:
    """Decide what kind of failure `exc` is, from its status code, body, headers and text."""
    now = time.time() if now is None else now
    status: Optional[int] = None
    texts: List[str] = []
    headers: dict = {}
    transport = False
    for e in _chain(exc):
        texts.append(str(e))
        s = getattr(e, "status_code", None)
        if isinstance(s, int) and status is None:
            status = s
        body = getattr(e, "body", None)
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                code = err.get("code")
                if isinstance(code, int) and status is None:
                    status = code
                if err.get("message"):
                    texts.append(str(err["message"]))
                meta = err.get("metadata")
                if isinstance(meta, dict):
                    if meta.get("raw"):
                        texts.append(str(meta["raw"]))
                    h = meta.get("headers")
                    if isinstance(h, dict):
                        headers.update({str(k).lower(): v for k, v in h.items()})
        rh = getattr(getattr(e, "response", None), "headers", None)
        if rh is not None:
            try:
                headers.update({str(k).lower(): v for k, v in rh.items()})
            except Exception:
                pass
    from src.agent.netstate import is_transport_error
    transport = is_transport_error(exc)
    text = " ".join(texts).lower()
    if status is None:
        m = re.search(r"http (\d{3})", text)
        if m:
            status = int(m.group(1))

    retry_after = None
    ra = headers.get("retry-after")
    if ra is not None:
        try:
            retry_after = max(0.0, float(ra))
        except (TypeError, ValueError):
            retry_after = None
    reset_at = _epoch(headers.get("x-ratelimit-reset"))
    if reset_at is not None and reset_at <= now:
        reset_at = None
    if retry_after is None and reset_at is not None and reset_at - now <= MAX_DELAY:
        retry_after = reset_at - now

    detail = " ".join(str(exc).split())[:200]

    def info(kind: Kind) -> ErrorInfo:
        return ErrorInfo(kind, status, detail, retry_after, reset_at)

    if status in (401, 403) or any(h in text for h in _AUTH_HINTS):
        return info(Kind.AUTH)
    if status == 402:
        return info(Kind.EXHAUSTED)
    if status == 429:
        if any(h in text for h in _EXHAUSTED_HINTS) or (reset_at is not None and reset_at - now > 600):
            return info(Kind.EXHAUSTED)
        return info(Kind.RATE_LIMITED)
    if status == 404 or any(h in text for h in _UNFIT_HINTS):
        return info(Kind.UNFIT)
    if status in (400, 413, 422) and any(h in text for h in _CONTEXT_HINTS):
        return info(Kind.UNFIT)
    if (status is not None and (status >= 500 or status == 408)) or transport:
        return info(Kind.BUSY)
    if any(h in text for h in _BUSY_HINTS):
        return info(Kind.BUSY)
    return info(Kind.OTHER)


# ── models that failed recently ──────────────────────────────────────────────

_LOCK = threading.Lock()
_COOL: dict = {}  # (provider, model) -> (until epoch seconds, reason text)


def mark_unavailable(provider: str, model: str, seconds: float, reason: str = "", now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    with _LOCK:
        _COOL[(provider, model)] = (now + max(0.0, seconds), reason)


def cooldown_left(provider: str, model: str, now: Optional[float] = None) -> float:
    now = time.time() if now is None else now
    with _LOCK:
        entry = _COOL.get((provider, model))
        if not entry:
            return 0.0
        if entry[0] <= now:
            _COOL.pop((provider, model), None)
            return 0.0
        return entry[0] - now


def clear_cooldowns() -> None:
    with _LOCK:
        _COOL.clear()
        _FREE_CACHE.clear()


def _pname(provider) -> str:
    return str(getattr(provider, "name", "") or "")


def _seconds_to_utc_midnight(now: float) -> float:
    return 86400.0 - (now % 86400.0)


def _cooldown_for(info: ErrorInfo, now: float) -> float:
    if info.kind is Kind.EXHAUSTED:
        if info.reset_at is not None:
            return max(60.0, info.reset_at - now)
        # OpenRouter's free-tier counters reset at 00:00 UTC.
        return min(86400.0, max(300.0, _seconds_to_utc_midnight(now)))
    return float(_COOLDOWN.get(info.kind, 300))


def _until(seconds: float, now: float) -> str:
    return time.strftime("%H:%M", time.localtime(now + seconds))


# ── the free-model list ──────────────────────────────────────────────────────

_FREE_CACHE: dict = {}  # provider name -> (fetched at, [models])


def live_free_models(provider) -> List[str]:
    """Free chat models the provider offers right now, best first. Cached for a few minutes."""
    now = time.time()
    with _LOCK:
        hit = _FREE_CACHE.get(_pname(provider))
        if hit and now - hit[0] < _FREE_CACHE_TTL:
            return list(hit[1])
    models: List[str] = []
    try:
        if hasattr(provider, "list_free_models"):
            models = list(provider.list_free_models())
        else:
            models = [m for m in provider.list_models() if str(m).endswith(":free")]
    except Exception as e:
        log.warning("Could not fetch the free model list: %s", e)
    if models:
        with _LOCK:
            _FREE_CACHE[_pname(provider)] = (now, models)
    return models


# ── checking an answer ───────────────────────────────────────────────────────

_REFUSAL_STARTS = ("error", '{"error"', "i'm sorry", "i am sorry", "i cannot", "i can't", "извините", "не могу")


def clean_summary(text) -> Optional[str]:
    """The summary text, or None when the answer is not one."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    # A model that died inside its reasoning leaves an unterminated tag: nothing usable follows it.
    t = re.sub(r"<think>.*\Z", "", t, flags=re.S | re.I).strip()
    if not t or t == "[Compression Failed]" or len(t) < MIN_SUMMARY_CHARS:
        return None
    if len(t) < 400 and t.lower().startswith(_REFUSAL_STARTS):
        return None
    return t


# ── the person at the keyboard ───────────────────────────────────────────────

def _noop(*_a, **_k):
    return None


@dataclass
class UI:
    """How a compression talks to the user. The defaults say nothing and ask nothing."""
    notify: Callable[[str], None] = _noop
    ask: Optional[Callable[[str, List[str]], Optional[str]]] = None
    pause: Callable[[], None] = _noop
    resume: Callable[[], None] = _noop


def default_ui(status=None) -> UI:
    """The terminal's own way: a system line per event, a select list for a question. `status` is
    the spinner shown while compressing; it is stopped while a question is on screen."""
    from src.agent.interaction import can_ask_user

    def notify(msg: str) -> None:
        log.warning(msg)
        try:
            from rich.markup import escape
            from ui import print_system
            print_system("[yellow]" + escape(msg) + "[/yellow]")
        except Exception:
            pass

    def ask(title: str, options: List[str]) -> Optional[str]:
        from ui import _select_from_list
        return _select_from_list(title, options, options[0])

    def pause() -> None:
        try:
            if status is not None:
                status.stop()
        except Exception:
            pass

    def resume() -> None:
        try:
            if status is not None:
                status.start()
        except Exception:
            pass

    return UI(notify=notify, ask=ask if can_ask_user() else None, pause=pause, resume=resume)


# ── one attempt ──────────────────────────────────────────────────────────────

def _attempt(provider, model: str, prompt: str, call_timeout: float, started: float):
    """One request on its own thread. Returns (text, exception); a thread that overruns is
    abandoned and whatever it eventually produces is thrown away with its result box."""
    box: dict = {}

    def work() -> None:
        try:
            box["text"] = provider.sync_chat(
                model=model, messages=[{"role": "user", "content": prompt}], timeout=call_timeout,
            )
        except BaseException as e:  # noqa: BLE001 - a worker must report, never die silently
            box["exc"] = e

    # the attempt runs on a thread of its own, but is still the caller's request: it keeps the caller's lane
    # priority (a background summary must queue as one)
    ctx = contextvars.copy_context()
    th = threading.Thread(target=lambda: ctx.run(work), daemon=True, name="service-llm")
    th.start()
    # sync_chat may retry once itself, so allow for two calls and the pause between them.
    limit = 2 * call_timeout + _SLACK
    deadline = time.monotonic() + limit
    while th.is_alive():
        th.join(0.05)
        if _cancelled_since(started):
            return None, Cancelled()
        if time.monotonic() > deadline:
            log.warning("Context summarization timed out after %.0fs (model %s); the attempt is abandoned.", limit, model)
            return None, TimeoutError(f"{model}: no answer within {limit:.0f}s")
    return box.get("text"), box.get("exc")


@dataclass
class Result:
    text: Optional[str]
    model: Optional[str] = None
    provider: Optional[str] = None
    switched: bool = False
    reason: str = ""
    tried: List[str] = field(default_factory=list)


def _try_model(provider, model, prompt, *, started, deadline, per_call, retries, ui, tell):
    """Ask one model, repeating while it is only busy. Returns (text, ErrorInfo or None)."""
    attempt = 0
    bad_output = 0
    info: Optional[ErrorInfo] = None
    while True:
        attempt += 1
        remaining = deadline - time.monotonic()
        if remaining < _MIN_LEFT:
            return None, info or ErrorInfo(Kind.BUSY, detail="закончилось время, отведённое на сжатие")
        call_timeout = max(_MIN_CALL, min(per_call, (remaining - _SLACK) / 2.0))
        text, exc = _attempt(provider, model, prompt, call_timeout, started)
        if isinstance(exc, Cancelled):
            raise exc
        if exc is None:
            cleaned = clean_summary(text)
            if cleaned:
                return cleaned, None
            info = ErrorInfo(Kind.BAD_OUTPUT, detail="ответ пуст или это не краткое изложение")
            bad_output += 1
            if bad_output < 2:
                tell(f"{model} {info.why()}, пробую ещё раз")
                continue
            return None, info
        info = classify(exc)
        log.warning("service LLM %s attempt %d: %s -> %s", model, attempt, info.detail, info.kind.value)
        if info.kind in (Kind.BUSY, Kind.RATE_LIMITED) and attempt < retries:
            delay = min(MAX_DELAY, max(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)], info.retry_after or 0.0))
            delay += random.uniform(0.0, 1.5)
            if delay > deadline - time.monotonic() - _MIN_LEFT:
                return None, info
            tell(f"{model} {info.why()}{_status(info)}. Повтор {attempt + 1}/{retries} через {delay:.0f} с")
            if _pause(delay, started):
                raise Cancelled()
            continue
        return None, info


def _status(info: ErrorInfo) -> str:
    return f" (HTTP {info.status})" if info.status else ""


def _ordered_candidates(provider, primary: str, tried: List[str], extra: Optional[Callable[[], List[str]]]):
    """(fresh, cooling): other free models worth trying, split by whether they failed recently."""
    from config import get_compress_fallback_models

    order: List[str] = []
    source = extra() if extra is not None else list(get_compress_fallback_models()) + live_free_models(provider)
    for m in source:
        if m and m not in order and m != primary and m not in tried:
            order.append(m)
    fresh = [m for m in order if cooldown_left(_pname(provider), m) <= 0]
    cooling = [m for m in order if m not in fresh]
    return fresh, cooling


_RETRY = "↻ Повторить ту же модель"
_GIVE_UP = "Без модели (детерминированная выжимка)"


def _choose_next(provider, model, info, tried, policy, ui, extra, now):
    """The next model to try, `model` itself to try it again, or None to stop."""
    fresh, cooling = _ordered_candidates(provider, model, tried, extra)
    if policy == "ask" and ui.ask is not None:
        labels = {}
        for m in fresh:
            labels[m] = m
        for m in cooling:
            labels[f"{m}  (недавно не ответила, до {_until(cooldown_left(_pname(provider), m, now), now)})"] = m
        options = [_RETRY] + list(labels) + [_GIVE_UP]
        title = f"Сжатие контекста: модель {model} {info.why()}{_status(info)}. Что сделать?"
        ui.pause()
        try:
            answer = ui.ask(title, options)
        except Exception as e:
            log.warning("Asking for a model failed: %s", e)
            answer = None
        finally:
            ui.resume()
        if answer is None or answer == _GIVE_UP:
            return None
        if answer == _RETRY:
            return model
        return labels.get(answer, answer)
    return fresh[0] if fresh else None


def run(prompt: str, *, per_call_timeout: float = 120.0, total_budget: Optional[float] = None,
        ui: Optional[UI] = None, retries: Optional[int] = None, policy: Optional[str] = None,
        service=None, candidates: Optional[Callable[[], List[str]]] = None) -> Result:
    """Get a summary for `prompt` from the service model, falling back as described above.

    Never raises (a stop request aside, which is reported as `Result(None, reason="stopped")`)."""
    ui = ui or UI()
    started = time.monotonic()
    try:
        return _run(prompt, per_call_timeout, total_budget, ui, retries, policy, service, candidates, started)
    except Cancelled:
        return Result(None, reason="stopped")
    except Exception as e:  # noqa: BLE001 - compression must never be what breaks a turn
        log.exception("service LLM call failed unexpectedly")
        try:
            ui.notify(f"Сжатие контекста: внутренняя ошибка ({type(e).__name__}), делаю выжимку без модели")
        except Exception:
            pass
        return Result(None, reason=f"{type(e).__name__}: {e}")


def _run(prompt, per_call_timeout, total_budget, ui, retries, policy, service, candidates, started) -> Result:
    from config import get_compress_on_failure, get_compress_retries

    retries = max(1, int(retries if retries is not None else get_compress_retries()))
    policy = policy or get_compress_on_failure()
    if service is None:
        from providers import create_service_provider
        provider, primary = create_service_provider()
    else:
        provider, primary = service
    problem = provider.validate_config()
    if problem:
        ui.notify(f"Сжатие контекста: {problem}. Делаю выжимку без модели")
        return Result(None, provider=_pname(provider), reason=str(problem))

    deadline = started + (total_budget if total_budget else max(per_call_timeout * 2, 240.0))
    # Only a free OpenRouter model has a free sibling to move to.
    can_switch = _pname(provider) == "openrouter" and str(primary).endswith(":free")
    model, tried, switches = primary, [], 0
    last_info: Optional[ErrorInfo] = None

    def tell(msg: str) -> None:
        ui.notify(f"Сжатие контекста: {msg}")

    while True:
        # A model that failed a little while ago is skipped when there is another to take.
        if can_switch and cooldown_left(_pname(provider), model) > 0 and model not in tried:
            fresh, _ = _ordered_candidates(provider, model, tried, candidates)
            if fresh:
                tell(f"{model} недавно не отвечала, беру {fresh[0]}")
                tried.append(model)
                model = fresh[0]
                switches += 1
                continue
        tried.append(model)
        text, info = _try_model(
            provider, model, prompt, started=started, deadline=deadline,
            per_call=per_call_timeout, retries=retries, ui=ui, tell=tell,
        )
        if text:
            if model != primary:
                tell(f"готово, сжатие выполнила {model}")
            return Result(text, model=model, provider=_pname(provider), switched=model != primary, tried=list(tried))
        last_info = info
        now = time.time()
        if info is not None and info.kind is not Kind.AUTH:
            mark_unavailable(_pname(provider), model, _cooldown_for(info, now), info.kind.value, now)
        if info is None or info.kind is Kind.AUTH or not can_switch:
            break
        if switches >= MAX_SWITCHES or deadline - time.monotonic() < _MIN_SWITCH_LEFT:
            break
        nxt = _choose_next(provider, model, info, [m for m in tried if m != model], policy, ui, candidates, now)
        if nxt is None:
            if policy != "ask" or ui.ask is None:
                tell(f"{model} {info.why()}{_status(info)}, других свободных бесплатных моделей не осталось")
            break
        if nxt == model:
            tried.remove(model)
            with _LOCK:
                _COOL.pop((_pname(provider), model), None)
            tell(f"повторяю {model}")
        else:
            tell(f"{model} {info.why()}{_status(info)} — перехожу на {nxt}")
            switches += 1
        model = nxt

    why = last_info.why() if last_info else "нет ответа"
    tell(f"модель {why}; делаю детерминированную выжимку без LLM")
    return Result(None, provider=_pname(provider), reason=why, tried=list(tried))
