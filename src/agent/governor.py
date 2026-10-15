"""The resource governor: puts the lanes around every call to a model that runs on this machine.

    provider = create_provider("ollama")         # providers.py wraps it with `govern`
    provider.stream_chat(...)                    # takes a ticket in the server's lane first

What changes for a request depends on who is asking (`current_priority`):

* the person's own turn (P0) and the advisor/critic (P1) take the lane and stream as before — they are the front
  of the queue;
* a worker (P2) or background job (P3) first waits until the PC is not busy with something else
  (`pressure.py`), then waits its turn in the lane. Its answer is collected before it is passed on: if a better
  request arrives meanwhile the worker's generation is dropped and made again later, with nothing half-written
  reaching the sub-agent. After MAX_PREEMPTS drops the worker is pinned and allowed to finish.

Cloud models are not touched: no lane, no waiting.

Switches (config): `local_governor` (on), `local_parallelism` (1), `lower_model_server_priority` (off),
`limit_server_threads` (off). `ARGENT_GOVERNOR=0` turns the whole thing off for one process.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import threading
import time
from typing import Callable, Iterator, Optional

from logger import get_logger
from src.agent import lanes
from src.agent.lanes import P_ADVISOR, P_BACKGROUND, P_INTERACTIVE, P_WORKER  # noqa: F401 (re-exported)
from src.agent.local_backend import Backend, TUNER, classify, thread_budget

log = get_logger("governor")

#: How often a worker's generation may be dropped for a better request before it is pinned.
MAX_PREEMPTS = 3

_priority: "contextvars.ContextVar[Optional[int]]" = contextvars.ContextVar("argent_lane_priority", default=None)
_held = threading.local()
_notify: Optional[Callable[[str], None]] = None
_lowered_at: dict = {}


def set_notifier(fn: Optional[Callable[[str], None]]) -> None:
    """Where "waiting for the model" notes go (the TUI registers its notice emitter)."""
    global _notify
    _notify = fn


_said: dict = {}


def _say(text: str) -> None:
    """Log it; tell the person too, but not the same thing again within a few seconds."""
    log.info("%s", text)
    now = time.monotonic()
    if now - _said.get(text, -1e9) < 5.0:
        return
    _said[text] = now
    if len(_said) > 64:
        _said.clear()
    if _notify is not None:
        try:
            _notify(text)
        except Exception:
            pass


def current_priority() -> int:
    value = _priority.get()
    if value is not None:
        return value
    env = os.environ.get("ARGENT_LANE_PRIORITY", "")
    return int(env) if env.isdigit() else P_INTERACTIVE


@contextlib.contextmanager
def priority(level: int) -> Iterator[None]:
    """Requests made inside the block queue at `level` (this thread / task only)."""
    token = _priority.set(int(level))
    try:
        yield
    finally:
        _priority.reset(token)


@contextlib.contextmanager
def capped(level: int) -> Iterator[None]:
    """Requests inside the block queue no better than `level` (a worker's own critic stays a worker)."""
    with priority(max(current_priority(), int(level))):
        yield


def lower_process_priority(pid: int) -> bool:
    """Run a worker process below the person's own programs: its tools (builds, tests) must not freeze the PC
    either. The model server is handled separately (`ServerTuner`). Children inherit the class."""
    try:
        import psutil
        psutil.Process(pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        return True
    except Exception as e:
        log.debug("cannot lower priority of process %s: %s", pid, e)
        return False


def enabled() -> bool:
    if os.environ.get("ARGENT_GOVERNOR") == "0":
        return False
    try:
        import config
        return bool(config.is_local_governor_enabled())
    except Exception:
        return True


def _capacity() -> int:
    try:
        import config
        return max(1, int(config.get_local_parallelism()))
    except Exception:
        return 1


def lane_for(backend: Backend) -> lanes.Lane:
    return lanes.get_lane(backend.lane_id, _capacity())


class _NoLease:
    """Stands in for a lease when the lane cannot be used (an unwritable home folder, a full disk): a request
    must never fail because the queue in front of it is broken."""

    pinned = False

    def pin(self) -> None: ...
    def release(self) -> None: ...
    def preempted(self) -> bool: return False
    def holds(self) -> bool: return True
    def __enter__(self): return self
    def __exit__(self, *exc) -> None: ...


def _acquire(lane: lanes.Lane, prio: int, label: str):
    try:
        return lane.acquire(prio, label, on_wait=_on_wait_note(label))
    except OSError as e:
        log.warning("lane %s cannot be used (%s): the request goes without it", lane.lane_id, e)
        return _NoLease()


def server_options(provider_name: str, model: str = "") -> dict:
    """Extra request options that keep a local server from taking every core (Ollama: num_thread)."""
    try:
        import config
        if not config.is_limit_server_threads_enabled():
            return {}
    except Exception:
        return {}
    backend = classify(provider_name, model)
    if backend.local and backend.family == "ollama":
        return {"num_thread": thread_budget()}
    return {}


def _after_acquire(backend: Backend) -> None:
    """Once in a while, while we use the server: make sure it runs below the person's own programs."""
    try:
        import config
        if not config.is_lower_server_priority_enabled():
            return
    except Exception:
        return
    now = time.monotonic()
    if now - _lowered_at.get(backend.lane_id, -1e9) < 30.0:
        return
    _lowered_at[backend.lane_id] = now
    TUNER.lower(backend)


def _already_holding(lane_id: str) -> bool:
    return getattr(_held, "ids", {}).get(lane_id, 0) > 0


@contextlib.contextmanager
def _hold(lane_id: str) -> Iterator[None]:
    ids = getattr(_held, "ids", None)
    if ids is None:
        ids = _held.ids = {}
    ids[lane_id] = ids.get(lane_id, 0) + 1
    try:
        yield
    finally:
        ids[lane_id] -= 1


def _wait_for_calm(prio: int) -> None:
    """Only background priorities defer to the rest of the PC; the person's own turn never waits here."""
    if prio < P_WORKER:
        return
    from src.agent.pressure import monitor
    monitor().wait_until_calm(on_wait=lambda why: _say(f"Фоновая работа ждёт: {why}."))


def _on_wait_note(label: str) -> Callable[[int], None]:
    return lambda ahead: _say(f"Локальная модель занята, запрос ждёт своей очереди ({label}, впереди: {ahead}).")


def settings() -> dict:
    import config
    return {
        "enabled": enabled(),
        "parallelism": _capacity(),
        "lower_priority": bool(config.is_lower_server_priority_enabled()),
        "limit_threads": bool(config.is_limit_server_threads_enabled()),
        "thread_budget": thread_budget(),
    }


def status_lines() -> list:
    """What `/lanes` and `/doctor` show: the switches, who is in which lane, and how busy the PC is."""
    cfg = settings()
    on = lambda v: "вкл" if v else "выкл"          # noqa: E731
    lines = [
        f"Очередь локальных моделей: {on(cfg['enabled'])}; запросов к серверу одновременно: {cfg['parallelism']}",
        f"Пониженный приоритет процесса сервера: {on(cfg['lower_priority'])}; "
        f"лимит потоков Ollama: {on(cfg['limit_threads'])} (до {cfg['thread_budget']})",
    ]
    root = lanes.lanes_root()
    shown = False
    try:
        folders = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        folders = []
    for folder in folders:
        snap = lanes.Lane(folder.name, _capacity()).snapshot()
        if not snap:
            continue
        shown = True
        lines.append(f"  сервер {folder.name}:")
        for item in snap:
            state = "работает" if item["holding"] else "ждёт"
            pin = ", закреплён" if item["pinned"] else ""
            lines.append(f"    {state}: {item['label'] or '—'} [{item['name']}{pin}, pid {item['pid']}]")
    if not shown:
        lines.append("  очереди пусты")
    try:
        from src.agent.pressure import monitor
        mon = monitor()
        mon.sample()
        r = mon.last
        if r is not None:
            vram = f", видеопамять свободно {r.vram_free_mb:.0f} МБ" if r.vram_free_mb is not None else ""
            verdict = f"ПЕРЕГРУЖЕН: {mon.reason}" if mon.overloaded else "спокоен"
            lines.append(f"Компьютер: {verdict} (чужая нагрузка на процессор {r.cpu_external:.0f}%, "
                         f"свободно памяти {r.ram_free_gb:.1f} ГБ{vram})")
    except Exception as e:
        log.debug("pressure reading failed: %s", e)
    return lines


def brief() -> str:
    """One short line for a progress panel: who waits in the local model's queue and how the PC is doing.
    Empty when there is nothing worth saying (a cloud model, no queue, a calm PC)."""
    parts = []
    try:
        waiting = holding = 0
        for folder in sorted(p for p in lanes.lanes_root().iterdir() if p.is_dir()):
            for item in lanes.Lane(folder.name, _capacity()).snapshot():
                if item["holding"]:
                    holding += 1
                else:
                    waiting += 1
        if holding or waiting:
            parts.append(f"модель: {holding} работает" + (f", {waiting} ждёт" if waiting else ""))
    except Exception as e:
        log.debug("lane summary failed: %s", e)
    try:
        from src.agent.pressure import monitor
        mon = monitor()
        if mon.last is not None and mon.overloaded:
            parts.append(f"ПК занят: {mon.reason}")
    except Exception as e:
        log.debug("pressure summary failed: %s", e)
    return " · ".join(parts)


@contextlib.contextmanager
def slot(backend: Backend, label: str, prio: Optional[int] = None) -> Iterator[None]:
    """One request by hand, for code that talks to a model without a provider object (`delegate_local`)."""
    if not backend.local or not enabled() or _already_holding(backend.lane_id):
        yield
        return
    level = current_priority() if prio is None else prio
    _wait_for_calm(level)
    lease = _acquire(lane_for(backend), level, label)
    try:
        _after_acquire(backend)
        with _hold(backend.lane_id):
            yield
    finally:
        lease.release()


#: A GGUF file loaded into this very process: no port, but just as much RAM. One lane for all of them.
GGUF = Backend(local=True, family="gguf", host="in-process", port=0)


def _wants_json(args: tuple, kwargs: dict) -> bool:
    """`sync_chat(model, messages, temperature=0.3, json_format=False, timeout=None)`: only the stream-less
    path of a provider knows how to ask for JSON."""
    return bool(kwargs.get("json_format", args[1] if len(args) > 1 else False))


def govern(provider):
    """Wrap `stream_chat` and `sync_chat` of a provider instance. Idempotent; cloud calls pass straight through."""
    if getattr(provider, "_governed", False):
        return provider
    inner_stream = provider.stream_chat
    inner_sync = provider.sync_chat

    def stream_chat(model, messages, *args, **kwargs):
        backend = classify(provider.name, model)
        if not backend.local or not enabled() or _already_holding(backend.lane_id):
            yield from inner_stream(model, messages, *args, **kwargs)
            return
        prio = current_priority()
        lane = lane_for(backend)
        label = f"{provider.name}:{model}"
        if prio >= P_WORKER:
            chunks = _collect_as_worker(lane, backend, prio, label,
                                        lambda: inner_stream(model, messages, *args, **kwargs))
            yield from chunks
            return
        lease = _acquire(lane, prio, label)
        try:
            _after_acquire(backend)
            with _hold(backend.lane_id):
                yield from inner_stream(model, messages, *args, **kwargs)
        finally:
            lease.release()

    def sync_chat(model, messages, *args, **kwargs):
        backend = classify(provider.name, model)
        if not backend.local or not enabled() or _already_holding(backend.lane_id):
            return inner_sync(model, messages, *args, **kwargs)
        prio = current_priority()
        lane = lane_for(backend)
        label = f"{provider.name}:{model}"
        if prio >= P_WORKER and not _wants_json(args, kwargs):
            # A background answer that cannot be interrupted would hold the server against the person's next
            # message for as long as the model takes to write it. Streamed, it can be dropped for a better
            # request and made again, like a worker's.
            temperature = kwargs.get("temperature", args[0] if args else 0.3)
            # the context the person's own turns use: a server that gets another one reloads the model
            try:
                import config
                window = int(config.get_context_window()) or None
            except Exception:
                window = None
            chunks = _collect_as_worker(
                lane, backend, prio, label,
                lambda: inner_stream(model, messages, context_window=window, temperature=temperature))
            return "".join(str(c.get("content") or "") for c in chunks if isinstance(c, dict))
        _wait_for_calm(prio)
        with _acquire(lane, prio, label):
            _after_acquire(backend)
            with _hold(backend.lane_id):
                return inner_sync(model, messages, *args, **kwargs)

    provider.stream_chat = stream_chat
    provider.sync_chat = sync_chat
    provider._governed = True
    return provider


def _collect_as_worker(lane: lanes.Lane, backend: Backend, prio: int, label: str, call) -> list:
    """Run one streamed request as a worker: wait for calm and for the lane, collect the answer whole, and make it
    again if a better request pushed it out of the lane meanwhile."""
    drops = 0
    while True:
        _wait_for_calm(prio)
        lease = _acquire(lane, prio, label)
        try:
            if drops >= MAX_PREEMPTS:
                lease.pin()
            _after_acquire(backend)
            chunks: list = []
            generator = call()
            last = time.monotonic()
            bumped = False
            with _hold(backend.lane_id):
                for chunk in generator:
                    chunks.append(chunk)
                    now = time.monotonic()
                    if now - last >= 0.1:
                        last = now
                        if lease.preempted():
                            bumped = True
                            break
            if bumped:
                with contextlib.suppress(Exception):
                    generator.close()
                drops += 1
                log.info("worker request %s dropped for a better one (%d/%d)", label, drops, MAX_PREEMPTS)
                continue
            return chunks
        finally:
            lease.release()
