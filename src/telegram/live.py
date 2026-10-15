"""The live status message: what Argent is doing right now, edited in place.

While a task runs, one message in the chat shows the plan, the last few steps and a
timer, with a Stop button.  It is edited, never re-sent, so the chat does not fill
with progress noise and the phone does not buzz for every tool call; the answer
arrives as its own message when the work is done.  When the task ends the panel
collapses to a one-line verdict with the full step list folded into an expandable
quote.

Edits are rate-limited (Telegram throttles bots that hammer ``editMessageText``)
and coalesced: many events inside one interval cost one edit.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from logger import get_logger
from src.telegram.render import (
    diff_counts,
    esc,
    fmt_duration,
    one_line,
    steps_word,
    tool_label,
)

log = get_logger("telegram_bot")

STOP_CALLBACK = "stop_turn"
MIN_EDIT_INTERVAL = 1.7
TICK_INTERVAL = 4.0
VISIBLE_STEPS = 6
KEPT_STEPS = 60
MAX_PLAN_ROWS = 8
_GIVE_UP_AFTER = 4          # consecutive failed edits before the panel stops trying


def stop_keyboard() -> Dict[str, Any]:
    return {"inline_keyboard": [[{"text": "⏹ Прервать", "callback_data": STOP_CALLBACK}]]}


class _Step:
    __slots__ = ("name", "emoji", "verb", "detail", "state", "started", "duration", "extra")

    def __init__(self, name: str, emoji: str, verb: str, detail: str, started: float):
        self.name, self.emoji, self.verb, self.detail = name, emoji, verb, detail
        self.state = "run"           # run | ok | err
        self.started = started
        self.duration: Optional[float] = None
        self.extra = ""

    def html(self, final: bool = False) -> str:
        mark = {"run": "▸", "ok": "✓", "err": "✗"}[self.state]
        text = f"{mark} {self.emoji} {esc(self.verb)}"
        if self.detail:
            text += f" <code>{esc(self.detail)}</code>"
        if self.extra:
            text += f" {esc(self.extra)}"
        if self.state != "run" and self.duration is not None and self.duration >= 1.0:
            text += f" <i>{fmt_duration(self.duration)}</i>"
        return text


class LiveStatus:
    """One editable progress message for one task."""

    def __init__(self, client: Any, chat_id: str | int, title: str = "", *, stop_button: bool = True,
                 clock: Callable[[], float] = time.monotonic, min_interval: float = MIN_EDIT_INTERVAL,
                 tick: float = TICK_INTERVAL, label: str = "Работаю"):
        self.client = client
        self.chat_id = chat_id
        self.title = one_line(title, 90)
        self.stop_button = stop_button
        self.label = label
        self._clock = clock
        self._min_interval = min_interval
        self._tick = tick
        self._lock = threading.RLock()
        self._steps: List[_Step] = []
        self._dropped = 0
        self._plan: List[Dict[str, str]] = []
        self._phase = ""
        self._notes: List[str] = []
        self._t0 = clock()
        self._last_edit = 0.0
        self._last_text = ""
        self._dirty = False
        self._failures = 0
        self._start_attempts = 0
        self._finished = False
        self.message_id: Optional[int] = None
        self._ticker: Optional[threading.Thread] = None
        self._ticker_stop = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────────────

    @property
    def started(self) -> bool:
        return self.message_id is not None

    @property
    def step_count(self) -> int:
        return len(self._steps) + self._dropped

    def elapsed(self) -> float:
        return self._clock() - self._t0

    def start(self) -> bool:
        """Send the panel. False when Telegram would not take it (the work goes on regardless)."""
        with self._lock:
            if self.started or self._finished or self._start_attempts >= 2:
                return self.started
            self._start_attempts += 1
            text = self._render()
            try:
                res = self.client.send_message(
                    self.chat_id, text,
                    reply_markup=stop_keyboard() if self.stop_button else None,
                    parse_mode="HTML", disable_notification=True,
                )
            except TypeError:        # a client without the newer keyword arguments
                res = self.client.send_message(self.chat_id, text, reply_markup=stop_keyboard() if self.stop_button else None, parse_mode="HTML")
            if isinstance(res, dict) and res.get("message_id") is not None:
                self.message_id = res["message_id"]
                self._last_text = text
                self._last_edit = self._clock()
                self._start_ticker()
                self._follow_checklist()
            return self.started

    def _follow_checklist(self) -> None:
        """The checklist is also changed from inside a long tool (an Ultracode workflow keeps its steps there):
        mirror it here as it changes, not only when the model calls todo_write."""
        try:
            from src.agent.todos import TODOS
            TODOS.subscribe(self._on_checklist)
        except Exception as e:
            log.debug("cannot follow the checklist: %s", e)

    def _unfollow_checklist(self) -> None:
        try:
            from src.agent.todos import TODOS
            TODOS.unsubscribe(self._on_checklist)
        except Exception:
            pass

    def _on_checklist(self, scope: str, items: Any) -> None:
        if scope != "main" or self._finished:
            return
        self.set_plan([t.as_dict() if hasattr(t, "as_dict") else t for t in items])

    def _start_ticker(self) -> None:
        if self._ticker is not None or self._tick <= 0:
            return

        def _run() -> None:
            while not self._ticker_stop.wait(self._tick):
                try:
                    self._flush(force=True)
                except Exception as e:                       # never let the clock thread die loudly
                    log.debug("live status tick failed: %s", e)

        self._ticker = threading.Thread(target=_run, daemon=True, name="ArgentTelegramLiveStatus")
        self._ticker.start()

    def _stop_ticker(self) -> None:
        self._ticker_stop.set()
        t, self._ticker = self._ticker, None
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=1.0)

    # ── what happens ───────────────────────────────────────────────────────

    def begin(self, name: str, args: Optional[Dict[str, Any]] = None) -> None:
        emoji, verb, detail = tool_label(name, args)
        with self._lock:
            self._phase = ""
            if name == "todo_write":
                # The plan has its own block; a step line for it would only repeat it.
                self.set_plan((args or {}).get("todos"))
            else:
                self._steps.append(_Step(name, emoji, verb, detail, self._clock()))
                self._trim()
            self._dirty = True
        self._ensure_and_flush()

    def end(self, name: str, result: Any = "", duration: Optional[float] = None) -> None:
        with self._lock:
            if name == "todo_write":
                self._dirty = True
                step = None
            else:
                step = (next((s for s in reversed(self._steps) if s.state == "run" and s.name == name), None)
                        or next((s for s in reversed(self._steps) if s.state == "run"), None))
            if step is not None:
                step.state = "err" if _looks_failed(result) else "ok"
                step.duration = duration if duration is not None else self._clock() - step.started
            self._dirty = True
        self._flush()

    def diff(self, file: str, diff_text: str) -> None:
        added, removed = diff_counts(diff_text)
        with self._lock:
            step = next((s for s in reversed(self._steps) if s.state in ("run", "ok") and s.emoji in ("✏️", "📝")), None)
            if step is not None:
                step.extra = f"+{added} −{removed}"
            self._dirty = True
        self._flush()

    def set_plan(self, todos: Any) -> None:
        rows: List[Dict[str, str]] = []
        if isinstance(todos, list):
            for t in todos:
                if isinstance(t, dict):
                    text = str(t.get("content") or t.get("text") or t.get("title") or "").strip()
                    status = str(t.get("status") or "pending")
                elif isinstance(t, str):
                    text, status = t.strip(), "pending"
                else:
                    continue
                if text:
                    rows.append({"text": one_line(text, 70), "status": status})
        with self._lock:
            self._plan = rows
            self._dirty = True

    def phase(self, text: str) -> None:
        with self._lock:
            if self._phase == text:
                return
            self._phase = text
            self._dirty = True
        self._flush()

    def note(self, text: str) -> None:
        text = one_line(text, 120)
        if not text:
            return
        with self._lock:
            if self._notes and self._notes[-1] == text:
                return
            self._notes = (self._notes + [text])[-2:]
            self._dirty = True
        self._ensure_and_flush()

    def _trim(self) -> None:
        while len(self._steps) > KEPT_STEPS:
            self._steps.pop(0)
            self._dropped += 1

    # ── drawing ────────────────────────────────────────────────────────────

    def _render(self, verdict: Optional[str] = None) -> str:
        secs = fmt_duration(self.elapsed())
        n = self.step_count
        if verdict is None:
            head = f"⏳ <b>{esc(self.label)}</b> · {secs}"
            if n:
                head += f" · {n} {steps_word(n)}"
        else:
            head = f"{verdict} · {secs}" + (f" · {n} {steps_word(n)}" if n else "")
        lines = [head]
        if self.title:
            lines.append(f"<i>«{esc(self.title)}»</i>")

        if self._plan:
            lines.append("")
            lines.extend(self._plan_lines())

        if verdict is None:
            shown = self._steps[-VISIBLE_STEPS:]
            hidden = n - len(shown)
            if shown:
                lines.append("")
                if hidden > 0:
                    lines.append(f"<i>… ещё {hidden} {steps_word(hidden)} выше</i>")
                lines.extend(s.html() for s in shown)
            if self._phase:
                lines.append("")
                lines.append(esc(self._phase))
            for note in self._notes:
                lines.append(f"⚠️ <i>{esc(note)}</i>")
        else:
            if self._steps:
                body = "\n".join(s.html(final=True) for s in self._steps)
                if self._dropped:
                    body = f"… ещё {self._dropped} {steps_word(self._dropped)} раньше\n" + body
                lines.append(f"<blockquote expandable>{body}</blockquote>")
        return "\n".join(lines)

    def _plan_lines(self) -> List[str]:
        icons = {"completed": "✅", "in_progress": "▶️", "pending": "☐"}
        rows = self._plan
        lines = [f"{icons.get(r['status'], '☐')} {esc(r['text'])}" for r in rows[:MAX_PLAN_ROWS]]
        if len(rows) > MAX_PLAN_ROWS:
            done = sum(1 for r in rows[MAX_PLAN_ROWS:] if r["status"] == "completed")
            lines.append(f"<i>… ещё {len(rows) - MAX_PLAN_ROWS} (готово {done})</i>")
        return lines

    # ── editing ────────────────────────────────────────────────────────────

    def _ensure_and_flush(self) -> None:
        if not self.started and not self._finished:
            self.start()
        else:
            self._flush()

    def _flush(self, force: bool = False) -> None:
        with self._lock:
            if not self.started or self._finished or self._failures >= _GIVE_UP_AFTER:
                return
            now = self._clock()
            if not force and now - self._last_edit < self._min_interval:
                return
            if not force and not self._dirty:
                return
            text = self._render()
            if text == self._last_text:
                self._dirty = False
                return
            res = self.client.edit_message_text(
                chat_id=self.chat_id, message_id=self.message_id, text=text,
                reply_markup=stop_keyboard() if self.stop_button else None,
            )
            self._last_edit = now
            if res is None:
                self._failures += 1
            else:
                self._failures = 0
                self._last_text = text
                self._dirty = False

    def finish(self, state: str = "done", detail: str = "") -> None:
        """Collapse the panel: ``done`` / ``stopped`` / ``failed``."""
        verdict = {
            "done": "✅ <b>Готово</b>",
            "stopped": "⏹ <b>Остановлено</b>",
            "failed": "❌ <b>Сбой</b>",
        }.get(state, "✅ <b>Готово</b>")
        with self._lock:
            if self._finished:
                return
            self._finished = True
            for s in self._steps:
                if s.state == "run":
                    s.state = "err" if state != "done" else "ok"
                    s.duration = self._clock() - s.started
            mid = self.message_id
            text = self._render(verdict=verdict if not detail else f"{verdict} ({esc(one_line(detail, 60))})")
        self._unfollow_checklist()
        self._stop_ticker()
        if mid is not None:
            self.client.edit_message_text(
                chat_id=self.chat_id, message_id=mid, text=text, reply_markup={"inline_keyboard": []},
            )

    def request_stop_ui(self) -> None:
        """Stop was pressed: take the button away (it was heard) and say so until the task unwinds."""
        with self._lock:
            self.stop_button = False
            self._phase = "⏹ Останавливаю…"
            self._dirty = True
        self._flush(force=True)

    def drop(self) -> None:
        """Remove the panel altogether (nothing worth keeping)."""
        with self._lock:
            self._finished = True
            mid = self.message_id
        self._unfollow_checklist()
        self._stop_ticker()
        if mid is not None:
            delete = getattr(self.client, "delete_message", None)
            if callable(delete):
                delete(self.chat_id, mid)


def _looks_failed(result: Any) -> bool:
    head = str(result or "")[:160].lstrip().lower()
    return head.startswith(("error", "ошибка", "[error", "[ошибка", "traceback", "failed", "exception", "✗", "❌"))
