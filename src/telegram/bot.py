"""Telegram bridge for Argent: your agent, from the phone.

What you get in the chat:

* **A live status message** while a task runs -- the plan, the last steps, a timer and a
  Stop button, edited in place -- and the answer as its own message, with a quiet
  footer (time, tool calls, tokens, context).  Long answers arrive as a file.
* **Approvals as cards**: the exact action in a code block, Allow / Always / Deny
  buttons, and the card is rewritten with who decided and what.  In companion mode the
  same question is asked at the PC and on the phone at once; whichever answers first wins.
* **A control panel** (``/menu``): status, Stop, Vibe, mode, new dialog, queue -- plus a
  command menu in the Telegram UI and a persistent keyboard for the common ones.
* **A queue**: a message sent while Argent is busy waits its turn instead of bouncing.
* **Files both ways**: photos and documents you send land in the project's upload
  folder (images also go to the model); ``/file <path>`` brings a project file back.
* **Both frontends mirrored**: a task started at the PC -- in the classic terminal or in
  the TUI -- shows up on the phone with progress, approvals and its result.

Two operation modes:
    'shared' (companion): one agent and context with the terminal.
    'isolated' (standalone): a separate, quiet session.

Only the configured chat is obeyed; everyone else is dropped without an answer.
"""

import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (  # noqa: F401  (set_* / get_telegram_enabled are re-exported: older code and tests import them from here)
    get_telegram_chat_id,
    get_telegram_enabled,
    get_telegram_mode,
    get_telegram_token,
    set_telegram_chat_id,
    set_telegram_enabled,
    set_telegram_mode,
    set_telegram_token,
)
from logger import get_logger
from src.bridge.events import to_event
from src.bridge.session import AgentSession
from src.telegram.client import MAX_MSG_LEN, TELEGRAM_API_BASE, TelegramClient  # noqa: F401  (re-exported)
from src.telegram.live import STOP_CALLBACK, LiveStatus, stop_keyboard  # noqa: F401  (re-exported)
from src.telegram.render import (  # noqa: F401  (format_markdown_for_telegram, split_html: re-exported)
    MAX_HTML,
    bar,
    esc,
    fmt_duration,
    format_html,
    format_markdown_for_telegram,
    one_line,
    split_html,
    turn_footer,
)

log = get_logger("telegram_bot")

TYPING_INTERVAL = 4.0           # seconds between sendChatAction calls
MAX_QUEUE = 5                   # messages that may wait for a busy agent
STALE_AFTER = 900               # a message older than this was sent while Argent was away: not obeyed
LONG_ANSWER = 3 * (MAX_HTML - 200)   # beyond this an answer goes as a file, with a preview
MAX_ATTACH = 20 * 1024 * 1024

MODE_LABEL = {"shared": "🔗 Компаньон", "isolated": "🔒 Изолированный"}
MODE_DESC = {
    "shared": "единая сессия с ПК: сообщения видны в терминале, подтверждения приходят на оба устройства",
    "isolated": "отдельная тихая сессия, терминал не затрагивается",
}
DECISION_LABELS = {
    "once": "✅ Разрешено (однократно)",
    "always": "🔒 Разрешено (всегда в этой сессии)",
    "deny": "❌ Отклонено",
}

#: The list Telegram shows when the user types "/".
BOT_COMMANDS: List[Tuple[str, str]] = [
    ("menu", "Панель управления"),
    ("status", "Состояние: модель, контекст, режим"),
    ("stop", "Прервать текущую задачу"),
    ("new", "Начать новый диалог"),
    ("vibe", "Автоодобрение безопасных действий"),
    ("mode", "Режим: компаньон или изолированный"),
    ("queue", "Очередь сообщений"),
    ("file", "Получить файл из проекта"),
    ("help", "Справка"),
]

#: The persistent keyboard under the input field -> the command each button stands for.
REPLY_BUTTONS = {
    "📊 Статус": "/status",
    "⏹ Стоп": "/stop",
    "🆕 Новый диалог": "/new_confirm",
    "☰ Меню": "/menu",
}

_SENSITIVE_NAMES = re.compile(
    r"(^\.env|\.pem$|\.key$|\.pfx$|\.p12$|^id_(rsa|dsa|ecdsa|ed25519)|credentials|secret|\.argent_coder_config|^\.netrc$|^\.npmrc$|^\.pypirc$)",
    re.IGNORECASE,
)
_SENSITIVE_DIRS = {".git", ".ssh", ".gnupg", ".aws", ".kube"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
_CAPTURE_TOOLS = {"capture_desktop_screen", "capture_desktop_window", "get_clipboard_image"}


def _console_print(markup: str) -> None:
    """Echo to the PC terminal when the bridge runs beside it; never an error."""
    try:
        from ui import console
        console.print(markup)
    except Exception:
        pass


def reply_keyboard() -> Dict[str, Any]:
    return {
        "keyboard": [[{"text": "📊 Статус"}, {"text": "⏹ Стоп"}], [{"text": "🆕 Новый диалог"}, {"text": "☰ Меню"}]],
        "resize_keyboard": True,
        "is_persistent": True,
        "input_field_placeholder": "Задача для Argent…",
    }


def publish_commands(client: Any) -> bool:
    """Put the command list into Telegram's '/' menu."""
    try:
        return bool(client.set_my_commands([{"command": c, "description": d} for c, d in BOT_COMMANDS]))
    except Exception as e:
        log.debug("could not publish bot commands: %s", e)
        return False


def welcome_html(mode: Optional[str] = None) -> str:
    mode = mode or get_telegram_mode()
    return (
        "🤖 <b>Argent на связи</b>\n\n"
        "Напишите задачу обычным сообщением — я выполню её на вашем компьютере и буду показывать ход работы. "
        "Можно прислать скриншот или файл: он попадёт в контекст.\n\n"
        f"<b>Режим:</b> {MODE_LABEL.get(mode, mode)} — {esc(MODE_DESC.get(mode, ''))}\n\n"
        "Кнопки под полем ввода — для частых действий, а ☰ Меню открывает полную панель."
    )


def send_welcome(client: Any, chat_id: str | int) -> None:
    """The first message after pairing: what this is, the command menu, the quick keyboard."""
    publish_commands(client)
    client.send_message(chat_id, welcome_html(), reply_markup=reply_keyboard(), parse_mode="HTML")


def approval_card(action: str, destructive: bool, source: str = "") -> str:
    head = "🚨 <b>Деструктивное действие</b>" if destructive else "🔐 <b>Нужно подтверждение</b>"
    if source:
        head += f" · {esc(source)}"
    shown = action if len(action) <= 1200 else action[:1200] + "…"
    return f"{head}\n<pre>{esc(shown)}</pre>"


def approval_keyboard(prefix: str, aid: Any, grant_key: Optional[str], destructive: bool) -> Dict[str, Any]:
    row = [{"text": "✅ Разрешить", "callback_data": f"{prefix}:{aid}:once"}]
    if grant_key and not destructive:
        row.append({"text": f"🔒 Всегда «{one_line(grant_key, 22)}»", "callback_data": f"{prefix}:{aid}:always"})
    return {"inline_keyboard": [row, [{"text": "❌ Отклонить", "callback_data": f"{prefix}:{aid}:deny"}]]}


def _cut_markdown(text: str, limit: int) -> str:
    """The start of a Markdown text, cut at a paragraph and with any open code fence closed."""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", limit // 3, limit)
    if cut == -1:
        cut = text.rfind("\n", limit // 3, limit)
    head = text[: cut if cut != -1 else limit].rstrip()
    if head.count("```") % 2:
        head += "\n```"
    return head


class _TerminalTurn:
    """A task running at the PC, as the phone sees it."""

    def __init__(self, user_input: str, live: LiveStatus, source: str):
        self.user_input = user_input
        self.live = live
        self.source = source
        self.content: List[str] = []
        self.errors: List[str] = []
        self.snapshot: Optional[Dict[str, Any]] = None
        self.stopped = False


class TelegramBridge:
    """Bridge coordinating Telegram polling, authorization, approvals and Argent sessions."""

    def __init__(self, session: Optional[AgentSession] = None):
        self._isolated_session = session or AgentSession()
        self._terminal_agent = None
        self._terminal_session: Optional[AgentSession] = None
        self._terminal_frontend: Any = None     # the TUI bridge, which can answer its own approvals

        self._pending_terminal_approvals: Dict[int, Dict[str, Any]] = {}
        self._approval_counter = 1000
        self._cards: Dict[Tuple[str, str], Dict[str, Any]] = {}     # open approval/select cards by (kind, id)
        self._card_html: Dict[Tuple[str, int], str] = {}            # (chat, message) -> the HTML it was sent with
        self._queue: List[Dict[str, Any]] = []
        self._term: Optional[_TerminalTurn] = None
        self._active_lives: List[LiveStatus] = []
        self._stale_skipped: List[str] = []
        self._turn_started_at: Optional[float] = None
        # Network work on behalf of the PC frontends runs here, in order, never on the agent's thread: an
        # unreachable Telegram must not be able to stall a task that is running at the PC.
        self._pump_q: "queue.Queue[Tuple[Callable[..., Any], tuple]]" = queue.Queue()
        self._pump_thread: Optional[threading.Thread] = None
        self._pump_lock = threading.Lock()

        self.client: Optional[TelegramClient] = None
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._typing_stop_event = threading.Event()
        self._typing_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._bot_info: Optional[Dict[str, Any]] = None
        self._active_terminal_msg_id: Optional[int] = None

    # ─── wiring ───────────────────────────────────────────────────────────────

    def attach_terminal_agent(self, agent: Any) -> None:
        """Attach the active terminal agent so shared mode can reuse its history and context."""
        self._terminal_agent = agent
        self._terminal_session = AgentSession(agent=agent)

    def attach_terminal_frontend(self, frontend: Any) -> None:
        """The TUI bridge: lets the phone answer an approval or question the PC is showing."""
        self._terminal_frontend = frontend

    @property
    def session(self) -> AgentSession:
        """Return the active AgentSession based on the configured mode ('shared' vs 'isolated')."""
        mode = get_telegram_mode()
        if mode == "shared" and self._terminal_session is not None:
            return self._terminal_session
        return self._isolated_session

    @session.setter
    def session(self, new_session: AgentSession) -> None:
        mode = get_telegram_mode()
        if mode == "shared" and self._terminal_session is not None:
            self._terminal_session = new_session
        else:
            self._isolated_session = new_session

    @property
    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def mirrors_terminal(self) -> bool:
        """True when PC turns should be shown on the phone."""
        return self.is_running and get_telegram_mode() == "shared"

    def start(self, token: Optional[str] = None, chat_id: Optional[str] = None) -> tuple[bool, str]:
        """Start the background Telegram bridge polling thread."""
        with self._lock:
            if self.is_running:
                return True, "Telegram bridge already running."

            tok = token or get_telegram_token()
            if not tok:
                return False, "Telegram token is not configured. Run /telegram setup first."

            self.client = TelegramClient(tok)
            me = self.client.get_me()
            if not me:
                return False, "Failed to authenticate with Telegram API. Please check your bot token."

            self._bot_info = me
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._polling_loop, daemon=True, name="ArgentTelegramBridge")
            self._worker_thread.start()

            bot_username = me.get("username", "bot")
            log.info("Telegram bridge started for @%s (mode: %s)", bot_username, get_telegram_mode())
            return True, f"Telegram bridge started successfully (@{bot_username})."

    def stop(self) -> None:
        """Stop the background polling thread and clean up."""
        with self._lock:
            self._stop_event.set()
            self._stop_typing()
            if self.session and self.session.busy:
                self.session.cancel()
            for aid, entry in list(self._pending_terminal_approvals.items()):
                if not entry.get("resolved"):
                    entry["decision"] = "deny"
                    entry["resolved"] = True
                    entry["event"].set()
            self._queue.clear()
            for live in list(self._active_lives):
                try:
                    live.finish("stopped")
                except Exception:
                    pass
            self._active_lives.clear()
            self._term = None
            if self.client and hasattr(self.client, "session"):
                try:
                    self.client.session.close()
                except Exception:
                    pass

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._worker_thread = None
        log.info("Telegram bridge stopped.")

    # ─── typing indicator ─────────────────────────────────────────────────────

    def _start_typing(self, chat_id: str | int) -> None:
        """Start repeating typing action in the background until stopped."""
        self._stop_typing()
        self._typing_stop_event.clear()

        def _typing_worker():
            while not self._typing_stop_event.is_set():
                if self.client:
                    self.client.send_chat_action(chat_id, "typing")
                self._typing_stop_event.wait(TYPING_INTERVAL)

        self._typing_thread = threading.Thread(target=_typing_worker, daemon=True, name="ArgentTelegramTyping")
        self._typing_thread.start()

    def _stop_typing(self) -> None:
        """Stop the repeating typing action."""
        self._typing_stop_event.set()
        if self._typing_thread and self._typing_thread.is_alive():
            self._typing_thread.join(timeout=1.0)
        self._typing_thread = None

    def _is_authorized(self, chat_id: str | int) -> bool:
        """Verify that the message sender matches the configured chat ID."""
        configured = get_telegram_chat_id()
        if not configured:
            return False
        return str(chat_id).strip() == str(configured).strip()

    # ─── small helpers ────────────────────────────────────────────────────────

    def _bg(self, fn: Callable[..., Any], *args: Any) -> None:
        threading.Thread(target=lambda: self._quiet(fn, *args), daemon=True, name="ArgentTelegramBg").start()

    @staticmethod
    def _quiet(fn: Callable[..., Any], *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as e:
            log.debug("telegram side call failed: %s", e)
            return None

    def _react(self, chat_id: str | int, message_id: Optional[int], emoji: Optional[str]) -> None:
        fn = getattr(self.client, "set_reaction", None)
        if message_id and callable(fn):
            self._bg(fn, chat_id, message_id, emoji)

    @staticmethod
    def _message_id(res: Any) -> Optional[int]:
        return res.get("message_id") if isinstance(res, dict) else None

    def _remember_card(self, chat_id: Any, res: Any, card_html: str) -> Optional[int]:
        mid = self._message_id(res)
        if mid is not None:
            if len(self._card_html) > 60:
                self._card_html.pop(next(iter(self._card_html)))
            self._card_html[(str(chat_id), mid)] = card_html
        return mid

    def _state(self) -> Dict[str, Any]:
        try:
            st = self.session.state()
            return st if isinstance(st, dict) else {}
        except Exception:
            return {}

    def _busy(self) -> bool:
        try:
            busy = bool(self.session.busy)
        except Exception:
            busy = False
        # A PC task whose end was never reported (a crash) must not hold the queue shut forever.
        pc_busy = self._term is not None and (self._turn_started_at is None or time.monotonic() - self._turn_started_at < 7200)
        return busy or pc_busy

    # ─── terminal companion: a PC task, seen from the phone ───────────────────

    def _pump(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` on the mirror thread, after everything queued before it."""
        with self._pump_lock:
            if self._pump_thread is None or not self._pump_thread.is_alive():
                self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True, name="ArgentTelegramMirror")
                self._pump_thread.start()
        self._pump_q.put((fn, args))

    def _pump_loop(self) -> None:
        while True:
            fn, args = self._pump_q.get()
            try:
                fn(*args)
            except Exception as e:
                log.debug("telegram mirror step failed: %s", e)

    def notify_terminal_turn_started(self, user_input: str, source: str = "repl", wait: bool = True) -> None:
        """A task started at the PC: show it on the phone, with a Stop button.

        ``wait=False`` (what the PC frontends use) sends the panel from the mirror thread."""
        chat_id = get_telegram_chat_id()
        if not chat_id or not self.client:
            return
        if self._term is not None:                       # the previous one never reported its end
            self._term.live.finish("done")
        clean_input = " ".join((user_input or "").split())
        live = LiveStatus(self.client, chat_id, title=clean_input, label="Выполняется на ПК")
        self._active_lives.append(live)
        self._term = _TerminalTurn(user_input, live, source)
        self._turn_started_at = time.monotonic()

        def _show() -> None:
            self._start_typing(chat_id)
            try:
                live.start()
            except Exception as e:
                log.warning("Failed to send terminal turn start notification: %s", e)
            if live.message_id is not None:
                self._active_terminal_msg_id = live.message_id

        if wait:
            _show()
        else:
            self._pump(_show)

    def notify_terminal_turn_completed(self, user_input: str, response_text: str, state: Optional[str] = None,
                                       wait: bool = True) -> None:
        """Send completion notification to Telegram when a terminal turn finishes."""
        term, self._term = self._term, None
        self._turn_started_at = None
        if wait:
            self._finish_terminal_turn(term, user_input, response_text, state)
        else:
            self._pump(self._finish_terminal_turn, term, user_input, response_text, state)

    def _finish_terminal_turn(self, term: Optional[_TerminalTurn], user_input: str, response_text: str,
                              state: Optional[str]) -> None:
        self._stop_typing()
        chat_id = get_telegram_chat_id()
        if not chat_id or not self.client:
            return
        try:
            snapshot = None
            if term is not None:
                snapshot = term.snapshot
                response_text = response_text or "".join(term.content).strip()
                state = state or ("stopped" if term.stopped else ("failed" if term.errors and not response_text else "done"))
                term.live.finish(state)
                if term.live in self._active_lives:
                    self._active_lives.remove(term.live)
            self._active_terminal_msg_id = None
            state = state or "done"

            clean_input = one_line(user_input, 80)
            head = {
                "done": "🔔 <b>[Терминал] Задача завершена</b>",
                "stopped": "⏹ <b>[Терминал] Задача остановлена</b>",
                "failed": "❌ <b>[Терминал] Задача завершилась ошибкой</b>",
            }.get(state, "🔔 <b>[Терминал] Задача завершена</b>")
            body = format_html(response_text) if response_text else "<i>(без текстового вывода)</i>"
            footer = turn_footer(snapshot)
            text = f"{head}\n<i>«{esc(clean_input)}»</i>\n\n{body}" + (f"\n\n{footer}" if footer else "")
            self.client.send_message(chat_id, text, parse_mode="HTML")
        except Exception as e:
            log.warning("Failed to send terminal turn completion to Telegram: %s", e)
        finally:
            self._drain_queue()

    def mirror_event(self, event: Dict[str, Any]) -> None:
        """One event of a task running at the PC. Cheap and non-blocking: the work happens on the mirror thread."""
        if self._pump_q.qsize() > 4000 and event.get("type") in ("content", "content_stream", "thinking"):
            return
        self._pump(self._mirror_event_now, event)

    def _mirror_event_now(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if etype in ("approval_request", "select_request"):
            if self.mirrors_terminal:
                (self._mirror_tui_approval if etype == "approval_request" else self._mirror_tui_select)(event)
            return
        term = self._term
        if term is None:
            return
        live = term.live
        if etype == "tool_start":
            live.begin(event.get("name", ""), event.get("args") if isinstance(event.get("args"), dict) else {})
        elif etype == "tool_end":
            live.end(event.get("name", ""), event.get("result", ""), event.get("duration_secs"))
        elif etype == "diff":
            live.diff(event.get("file", ""), event.get("diff", ""))
        elif etype == "thinking":
            live.phase("🧠 Размышляю…")
        elif etype in ("content", "content_stream"):
            term.content.append(event.get("text", ""))
        elif etype == "content_replace":
            term.content = [event.get("text", "")]
        elif etype == "notice":
            live.note(re.sub(r"^\[(?:Argent|System):?\s*|\]$", "", str(event.get("text", "")).strip()))
        elif etype == "error":
            text = str(event.get("text", ""))
            if "прервано" in text.lower():
                term.stopped = True
            else:
                term.errors.append(text)
        elif etype == "turn_snapshot":
            term.snapshot = event.get("data") if isinstance(event.get("data"), dict) else None
        elif etype in ("cancelling",):
            term.stopped = True
        elif etype == "done" and term.source == "tui":
            self.notify_terminal_turn_completed(term.user_input, "".join(term.content).strip(), wait=True)

    def tee_chunks(self, chunks: Any) -> Any:
        """Wrap the classic terminal's chunk stream so the phone sees the tools run."""
        for chunk in chunks:
            try:
                ev = to_event(chunk) if isinstance(chunk, dict) else None
                if ev is not None:
                    self.mirror_event(ev)
            except Exception:
                pass
            yield chunk

    # ─── approvals: terminal relay (classic REPL) ─────────────────────────────

    def relay_terminal_approval(
        self,
        action: str,
        destructive: bool,
        grant_key: Optional[str],
        terminal_fn: Callable[[str, bool, Optional[str]], str],
    ) -> str:
        """Simultaneously prompt in terminal AND send interactive buttons to Telegram.
        Whichever answers first determines the decision.
        """
        chat_id = get_telegram_chat_id()
        if not chat_id or not self.client or self._stop_event.is_set():
            return terminal_fn(action, destructive, grant_key)

        with self._lock:
            self._approval_counter += 1
            aid = self._approval_counter
            done_event = threading.Event()
            entry = {
                "event": done_event,
                "decision": "deny",
                "resolved": False,
                "msg_id": None,
                "action": action,
                "chat_id": chat_id,
                "destructive": destructive,
            }
            self._pending_terminal_approvals[aid] = entry

        card = approval_card(action, destructive, source="Терминал")
        tg_res = self.client.send_message(
            chat_id, card, reply_markup=approval_keyboard("appr", aid, grant_key, destructive), parse_mode="HTML",
        )
        mid = self._remember_card(chat_id, tg_res, card)
        if mid is not None:
            entry["msg_id"] = mid
            entry["html"] = card

        # Run terminal prompt in a daemon thread so it can race with Telegram
        def _run_terminal_prompt():
            try:
                dec = terminal_fn(action, destructive, grant_key)
                self._resolve_terminal_approval(aid, dec, source="terminal")
            except BaseException:
                self._resolve_terminal_approval(aid, "deny", source="terminal")

        threading.Thread(target=_run_terminal_prompt, daemon=True, name=f"ArgentTerminalApproval-{aid}").start()

        # Wait for either Telegram button click OR Terminal input
        done_event.wait()

        with self._lock:
            decision = entry.get("decision", "deny")
            self._pending_terminal_approvals.pop(aid, None)

        return decision

    def _resolve_terminal_approval(self, aid: int, decision: str, source: str = "telegram") -> None:
        """Resolve a pending terminal approval from either Telegram or Terminal."""
        with self._lock:
            entry = self._pending_terminal_approvals.get(aid)
            if not entry or entry["resolved"]:
                return
            entry["decision"] = decision if decision in ("once", "always", "deny") else "deny"
            entry["resolved"] = True
            entry["event"].set()

        chat_id = entry.get("chat_id")
        msg_id = entry.get("msg_id")
        label = DECISION_LABELS.get(decision, decision)

        # Update Telegram message to reflect who approved and remove buttons
        if self.client and chat_id and msg_id:
            who = "с телефона" if source == "telegram" else "на ПК"
            base = entry.get("html") or approval_card(entry.get("action", ""), bool(entry.get("destructive")), source="Терминал")
            self.client.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=f"{base}\n\n<b>Решение ({who}):</b> {esc(label)}",
                reply_markup={"inline_keyboard": []},
            )

        # If resolved from Telegram, print a notice in the terminal console
        if source == "telegram":
            _console_print(f"\n[bold green]✓ Решение получено из Telegram: {label}[/bold green]")

    # ─── approvals and questions of the TUI, mirrored to the phone ────────────

    def _mirror_tui_approval(self, event: Dict[str, Any]) -> None:
        chat_id = get_telegram_chat_id()
        if not chat_id or not self.client:
            return
        aid = event.get("id")
        destructive = bool(event.get("destructive"))
        card = approval_card(str(event.get("action", "")), destructive, source="Терминал")
        res = self.client.send_message(
            chat_id, card, reply_markup=approval_keyboard("tui:ap", aid, event.get("grant_key"), destructive), parse_mode="HTML",
        )
        mid = self._remember_card(chat_id, res, card)
        if mid is not None:
            self._cards[("ap", str(aid))] = {"chat_id": chat_id, "msg_id": mid, "html": card}

    def _mirror_tui_select(self, event: Dict[str, Any]) -> None:
        options = [str(o) for o in (event.get("options") or [])][:10]
        chat_id = get_telegram_chat_id()
        if not options or not chat_id or not self.client:
            return                                         # free text cannot be answered with buttons
        sid = str(event.get("id"))
        multi = bool(event.get("multi"))
        checked = [bool(b) for b in (event.get("initial_checked") or [])][: len(options)]
        checked += [False] * (len(options) - len(checked))
        card = f"❓ <b>{esc(one_line(event.get('title', ''), 300))}</b>" + ("\n<i>Отметьте нужное и нажмите «Готово»</i>" if multi else "")
        info = {"chat_id": chat_id, "options": options, "multi": multi, "checked": checked, "html": card}
        res = self.client.send_message(chat_id, card, reply_markup=self._select_keyboard(sid, info), parse_mode="HTML")
        mid = self._remember_card(chat_id, res, card)
        if mid is not None:
            info["msg_id"] = mid
            self._cards[("sel", sid)] = info

    @staticmethod
    def _select_keyboard(sid: str, info: Dict[str, Any]) -> Dict[str, Any]:
        rows = []
        for i, opt in enumerate(info["options"]):
            if info["multi"]:
                rows.append([{"text": f"{'☑' if info['checked'][i] else '☐'} {one_line(opt, 40)}", "callback_data": f"tui:mt:{sid}:{i}"}])
            else:
                rows.append([{"text": one_line(opt, 44), "callback_data": f"tui:sel:{sid}:{i}"}])
        if info["multi"]:
            rows.append([{"text": "✅ Готово", "callback_data": f"tui:md:{sid}"}, {"text": "✖ Отмена", "callback_data": f"tui:sc:{sid}"}])
        return {"inline_keyboard": rows}

    def mirror_resolved(self, kind: str, ident: Any, outcome: str = "") -> None:
        """The PC answered a mirrored card: close it on the phone (idempotent)."""
        card = self._cards.pop((kind, str(ident)), None)
        if not card or not self.client:
            return
        label = DECISION_LABELS.get(outcome, outcome) if kind == "ap" else outcome
        text = card["html"] + (f"\n\n<b>Решение (на ПК):</b> {esc(label)}" if label else "\n\n<i>Отвечено на ПК</i>")
        self.client.edit_message_text(chat_id=card["chat_id"], message_id=card["msg_id"], text=text, reply_markup={"inline_keyboard": []})

    def _handle_tui_callback(self, query_id: Optional[str], data: str) -> None:
        frontend = self._terminal_frontend
        parts = data.split(":")
        kind = parts[1] if len(parts) > 1 else ""

        def answer(text: Optional[str] = None) -> None:
            if self.client and query_id:
                self.client.answer_callback_query(query_id, text) if text else self.client.answer_callback_query(query_id)

        if frontend is None:
            answer("Терминал недоступен")
            return

        if kind == "ap" and len(parts) == 4:
            _, _, aid, decision = parts
            card = self._cards.pop(("ap", aid), None)
            if card is None:
                answer("Уже отвечено")
                return
            try:
                frontend.reply_approval(int(aid), decision)
            except Exception as e:
                log.warning("could not relay approval to the terminal: %s", e)
            label = DECISION_LABELS.get(decision, decision)
            answer(f"Выбор: {label}")
            self.client.edit_message_text(chat_id=card["chat_id"], message_id=card["msg_id"],
                                          text=f"{card['html']}\n\n<b>Решение (с телефона):</b> {esc(label)}",
                                          reply_markup={"inline_keyboard": []})
            _console_print(f"\n[bold green]✓ Решение получено из Telegram: {label}[/bold green]")
            return

        sid = parts[2] if len(parts) > 2 else ""
        card = self._cards.get(("sel", sid))
        if card is None:
            answer("Уже отвечено")
            return

        def close(summary: str) -> None:
            self.client.edit_message_text(chat_id=card["chat_id"], message_id=card["msg_id"],
                                          text=f"{card['html']}\n\n<b>Ответ (с телефона):</b> {esc(summary)}",
                                          reply_markup={"inline_keyboard": []})

        try:
            if kind == "sel" and len(parts) == 4:
                value = card["options"][int(parts[3])]
                self._cards.pop(("sel", sid), None)
                frontend.reply_select(sid, value)
                answer("Выбрано")
                close(value)
            elif kind == "mt" and len(parts) == 4:
                i = int(parts[3])
                card["checked"][i] = not card["checked"][i]
                answer()
                self.client.edit_message_reply_markup(card["chat_id"], card["msg_id"], self._select_keyboard(sid, card))
            elif kind == "md":
                chosen = [o for o, c in zip(card["options"], card["checked"]) if c]
                self._cards.pop(("sel", sid), None)
                frontend.reply_select(sid, json.dumps(chosen, ensure_ascii=False))
                answer("Отправлено")
                close(", ".join(chosen) or "ничего")
            elif kind == "sc":
                self._cards.pop(("sel", sid), None)
                frontend.reply_select(sid, None)
                answer("Отменено")
                close("отмена")
            else:
                answer()
        except (ValueError, IndexError) as e:
            log.debug("bad select callback %r: %s", data, e)
            answer()
        except Exception as e:
            log.warning("could not relay the answer to the terminal: %s", e)
            answer("Не удалось передать ответ")

    # ─── polling and update handling ──────────────────────────────────────────

    def _polling_loop(self) -> None:
        """Long polling, surviving outages: waits with growing pauses, says once when it is back."""
        offset = None
        backoff = 1.0
        down_since: Optional[float] = None
        told_conflict = False

        self._quiet(self.client.delete_webhook)
        self._quiet(publish_commands, self.client)

        while not self._stop_event.is_set():
            try:
                if not self.client:
                    break
                updates = self.client.get_updates(offset=offset, timeout=15)

                if getattr(self.client, "last_ok", True) is False:
                    code = getattr(self.client, "last_code", 0)
                    if down_since is None:
                        down_since = time.time()
                        log.warning("Telegram unreachable (%s %s); retrying with backoff",
                                    code or "network", getattr(self.client, "last_error", ""))
                    if code == 409 and not told_conflict:
                        told_conflict = True
                        log.error("Telegram: another process is polling this bot (409). Only one Argent can use one bot token.")
                    delay = 30.0 if code in (401, 403) else backoff
                    self._stop_event.wait(delay)
                    backoff = min(backoff * 2, 30.0)
                    continue

                if down_since is not None:
                    log.info("Telegram reachable again after %.0fs", time.time() - down_since)
                    down_since = None
                told_conflict = False
                backoff = 1.0

                for update in updates:
                    offset = update.get("update_id", 0) + 1
                    try:
                        self._handle_update(update)
                    except Exception as e:
                        log.error("Error processing update: %s", e)
                self._report_stale()

            except Exception as e:
                if self._stop_event.is_set():
                    break
                log.warning("Telegram polling error: %s (retrying in %.1fs)", e, backoff)
                self._stop_event.wait(backoff)
                backoff = min(backoff * 1.5, 10.0)

    def _report_stale(self) -> None:
        if not self._stale_skipped or not self.client:
            return
        skipped, self._stale_skipped = self._stale_skipped, []
        chat_id = get_telegram_chat_id()
        if not chat_id:
            return
        shown = "\n".join(f"• <i>{esc(one_line(t, 70))}</i>" for t in skipped[:5])
        more = f"\n… и ещё {len(skipped) - 5}" if len(skipped) > 5 else ""
        self.client.send_message(
            chat_id,
            f"⏱ <b>Пропущено сообщений: {len(skipped)}</b>\n"
            "Они пришли, пока Argent был недоступен (старше 15 минут), поэтому я их не выполнял — "
            f"вдруг задача уже неактуальна.\n\n{shown}{more}\n\nОтправьте заново, если нужно.",
        )

    def _handle_update(self, update: Dict[str, Any]) -> None:
        """Route an update to the appropriate handler."""
        # 1. Handle Inline Button Clicks (Callback Query)
        if "callback_query" in update:
            self._handle_callback_query(update["callback_query"])
            return

        # 2. Handle Messages
        if "message" in update:
            self._handle_message(update["message"])

    # ─── callbacks ────────────────────────────────────────────────────────────

    def _handle_callback_query(self, query: Dict[str, Any]) -> None:
        """Handle inline button clicks: approvals, Stop, the control panel."""
        query_id = query.get("id")
        user = query.get("from", {})
        sender_id = str(user.get("id", ""))
        message = query.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        msg_id = message.get("message_id")
        data = query.get("data", "")

        # Check authorization
        if not self._is_authorized(chat_id) and not self._is_authorized(sender_id):
            if self.client and query_id:
                self.client.answer_callback_query(query_id, "⛔ Доступ запрещен.")
            return

        # Handle inline "Прервать выполнение" button
        if data == STOP_CALLBACK:
            if self.client and query_id:
                self.client.answer_callback_query(query_id, "⏹️ Прерывание выполнения...")
            self._stop_work()
            for live in list(self._active_lives):
                live.request_stop_ui()
            if get_telegram_mode() == "shared":
                _console_print("\n[bold yellow]⏹️ [Telegram]: Выполнение прервано пользователем по кнопке (Ctrl+C).[/bold yellow]")
            return

        if data.startswith("ui:") or data.startswith("q:"):
            self._handle_panel_callback(query_id, chat_id, msg_id, data)
            return

        if data.startswith("tui:"):
            self._handle_tui_callback(query_id, data)
            return

        if not data.startswith("appr:"):
            if self.client and query_id:
                self.client.answer_callback_query(query_id)
            return

        # Parse appr:<id>:<decision>
        parts = data.split(":")
        if len(parts) != 3:
            return

        _, approval_id_str, decision = parts
        try:
            aid = int(approval_id_str)
        except ValueError:
            return

        label = DECISION_LABELS.get(decision, decision)

        # 1. Check if this is a pending terminal relay approval
        if aid in self._pending_terminal_approvals:
            if self.client and query_id:
                self.client.answer_callback_query(query_id, f"Выбор: {label}")
            self._resolve_terminal_approval(aid, decision, source="telegram")
            return

        # 2. Otherwise it is a turn within AgentSession
        self.session.reply_approval(aid, decision)

        if self.client:
            if query_id:
                self.client.answer_callback_query(query_id, f"Выбор: {label}")

            # Update original message to remove buttons and record decision
            base = self._card_html.pop((chat_id, msg_id), None) or esc(message.get("text") or "")
            self.client.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text=f"{base}\n\n<b>Решение:</b> {esc(label)}",
                reply_markup={"inline_keyboard": []},  # remove buttons
            )

    def _stop_work(self) -> bool:
        """Interrupt whatever Argent is doing. True if there was something to interrupt."""
        stopped = False
        if self._terminal_agent is not None:
            self._terminal_agent.request_stop()
            stopped = True
        if self.session and self.session.busy:
            self.session.cancel()
            stopped = True
        self._stop_typing()
        return stopped

    # ─── the control panel ────────────────────────────────────────────────────

    def _menu_card(self) -> Tuple[str, Dict[str, Any]]:
        st = self._state()
        ctx = st.get("context") or {}
        mode = get_telegram_mode()
        vibe = bool(st.get("vibe", getattr(self.session, "vibe", False)))
        busy = self._busy()
        model, provider = st.get("model", "?"), str(st.get("provider", "?")).upper()
        pct = ctx.get("percent", 0)
        lines = [
            "🤖 <b>Argent</b> · на связи",
            f"<i>{esc(model)} · {esc(provider)}</i>",
            "",
            f"{bar(pct)} контекст {pct:.0f}%" if isinstance(pct, (int, float)) else "",
            f"{MODE_LABEL.get(mode, mode)}  ·  ✨ Vibe: {'вкл' if vibe else 'выкл'}",
            ("⏳ Работаю над задачей" + (f" · {fmt_duration(time.monotonic() - self._turn_started_at)}" if self._turn_started_at else "")) if busy else "🟢 Свободен",
        ]
        if self._queue:
            lines.append(f"🕓 В очереди: {len(self._queue)}")
        keyboard = {"inline_keyboard": [
            [{"text": "📊 Статус", "callback_data": "ui:status"}, {"text": "⏹ Прервать", "callback_data": "ui:stop"}],
            [{"text": f"✨ Vibe: {'вкл' if vibe else 'выкл'}", "callback_data": "ui:vibe"},
             {"text": MODE_LABEL.get(mode, mode), "callback_data": "ui:mode"}],
            [{"text": "🆕 Новый диалог", "callback_data": "ui:new"},
             {"text": f"🕓 Очередь ({len(self._queue)})", "callback_data": "ui:queue"}],
            [{"text": "📖 Справка", "callback_data": "ui:help"}],
        ]}
        return "\n".join(l for l in lines if l is not None).replace("\n\n\n", "\n\n"), keyboard

    def _status_card(self) -> str:
        st = self._state()
        ctx = st.get("context") or {}
        tokens, max_tokens, pct = ctx.get("tokens", 0), ctx.get("max", 0), ctx.get("percent", 0)
        mode = get_telegram_mode()
        vibe = "вкл" if st.get("vibe") else "выкл"
        busy = self._busy()
        work = "⏳ Работает" + (f" · {fmt_duration(time.monotonic() - self._turn_started_at)}" if self._turn_started_at else "") if busy else "🟢 Свободен"
        lines = [
            "📊 <b>Состояние Argent</b>",
            "",
            f"<b>Модель</b>  <code>{esc(st.get('model', '?'))}</code>",
            f"<b>Провайдер</b>  <code>{esc(str(st.get('provider', '?')).upper())}</code>",
            f"<b>Категория</b>  <code>{esc(st.get('tier', '?'))}</code>",
            f"<b>Контекст</b>  {bar(pct)} {pct:.0f}%  <i>({tokens} / {max_tokens})</i>" if isinstance(pct, (int, float)) else f"<b>Контекст</b>  {tokens} / {max_tokens}",
            f"<b>Vibe</b>  {vibe}   <b>Режим</b>  {esc(MODE_LABEL.get(mode, mode))}",
            f"<b>Статус</b>  {work}",
            f"<b>Очередь</b>  {len(self._queue)}",
            f"<b>Папка</b>  <code>{esc(one_line(os.getcwd(), 60))}</code>",
        ]
        return "\n".join(lines)

    def _help_card(self) -> str:
        mode = get_telegram_mode()
        return (
            "🤖 <b>Argent — справка</b>\n\n"
            "Напишите задачу обычным сообщением. Пока она выполняется, я показываю ход работы одним обновляемым "
            "сообщением; ответ приходит отдельно. Если Argent занят, ваше сообщение встанет в очередь.\n"
            "Можно прислать скриншот или файл — он попадёт в контекст.\n\n"
            "<b>Команды</b>\n"
            "/menu — панель управления\n"
            "/status — модель, контекст, режим\n"
            "/stop — прервать задачу\n"
            "/new — новый диалог (контекст сбрасывается)\n"
            "/vibe [on|off|&lt;задача&gt;] — автоодобрение безопасных действий\n"
            "/mode [shared|isolated] — компаньон или изолированный\n"
            "/queue — очередь сообщений\n"
            "/file &lt;путь&gt; — получить файл из проекта\n\n"
            f"<b>Режим сейчас:</b> {MODE_LABEL.get(mode, mode)} — {esc(MODE_DESC.get(mode, ''))}"
        )

    def _queue_card(self) -> Tuple[str, Dict[str, Any]]:
        if not self._queue:
            return "🕓 <b>Очередь пуста.</b>\nСообщение, отправленное во время работы, встанет сюда.", {"inline_keyboard": [[{"text": "☰ Меню", "callback_data": "ui:menu"}]]}
        lines = ["🕓 <b>Очередь</b>", ""]
        rows = []
        for i, item in enumerate(self._queue, 1):
            lines.append(f"{i}. <i>{esc(one_line(item['text'], 70))}</i>")
            rows.append([{"text": f"🗑 Убрать №{i}", "callback_data": f"q:del:{i - 1}"}])
        rows.append([{"text": "🧹 Очистить всё", "callback_data": "q:clear"}, {"text": "☰ Меню", "callback_data": "ui:menu"}])
        return "\n".join(lines), {"inline_keyboard": rows}

    def _show(self, chat_id: Any, msg_id: Any, text: str, keyboard: Optional[Dict[str, Any]]) -> None:
        """Replace the panel message in place (or send it fresh if it cannot be edited)."""
        if not self.client:
            return
        if msg_id is not None:
            if self.client.edit_message_text(chat_id=chat_id, message_id=msg_id, text=text, reply_markup=keyboard):
                return
        self.client.send_message(chat_id, text, reply_markup=keyboard, parse_mode="HTML")

    def _handle_panel_callback(self, query_id: Optional[str], chat_id: str, msg_id: Any, data: str) -> None:
        if not self.client:
            return
        back = {"inline_keyboard": [[{"text": "☰ Меню", "callback_data": "ui:menu"}]]}
        refresh = {"inline_keyboard": [[{"text": "🔄 Обновить", "callback_data": "ui:status"}, {"text": "☰ Меню", "callback_data": "ui:menu"}]]}
        parts = data.split(":")
        answer = ""
        action = parts[1] if len(parts) > 1 else ""

        if parts[0] == "q":
            if action == "clear":
                self._queue.clear()
                answer = "Очередь очищена"
            elif action == "del" and len(parts) == 3 and parts[2].isdigit() and int(parts[2]) < len(self._queue):
                self._queue.pop(int(parts[2]))
                answer = "Убрано"
            text, kb = self._queue_card()
            self.client.answer_callback_query(query_id, answer) if answer else self.client.answer_callback_query(query_id)
            self._show(chat_id, msg_id, text, kb)
            return

        if action == "stop":
            answer = "Останавливаю…" if self._stop_work() else "Сейчас нечего останавливать"
            for live in list(self._active_lives):
                live.request_stop_ui()
        elif action == "vibe":
            current = bool(getattr(self.session, "vibe", False))
            new = self.session.set_vibe(not current)
            answer = "Vibe включён" if new else "Vibe выключен"
        elif action == "mode" and len(parts) == 3 and parts[2] in MODE_LABEL:
            set_telegram_mode(parts[2])
            answer = f"Режим: {MODE_LABEL[parts[2]]}"
        elif action == "new" and len(parts) == 3 and parts[2] == "yes":
            self._reset_dialog()
            answer = "Новый диалог начат"

        if query_id:
            self.client.answer_callback_query(query_id, answer) if answer else self.client.answer_callback_query(query_id)

        if action == "status":
            self._show(chat_id, msg_id, self._status_card(), refresh)
        elif action == "help":
            self._show(chat_id, msg_id, self._help_card(), back)
        elif action == "queue":
            text, kb = self._queue_card()
            self._show(chat_id, msg_id, text, kb)
        elif action == "mode" and len(parts) == 2:
            mode = get_telegram_mode()
            kb = {"inline_keyboard": [
                [{"text": ("● " if mode == m else "○ ") + MODE_LABEL[m], "callback_data": f"ui:mode:{m}"}] for m in ("shared", "isolated")
            ] + [[{"text": "☰ Меню", "callback_data": "ui:menu"}]]}
            text = ("<b>Режим работы</b>\n\n"
                    f"{MODE_LABEL['shared']} — {esc(MODE_DESC['shared'])}.\n\n"
                    f"{MODE_LABEL['isolated']} — {esc(MODE_DESC['isolated'])}.")
            self._show(chat_id, msg_id, text, kb)
        elif action == "new" and len(parts) == 2:
            kb = {"inline_keyboard": [[{"text": "🆕 Да, начать заново", "callback_data": "ui:new:yes"}, {"text": "Отмена", "callback_data": "ui:menu"}]]}
            self._show(chat_id, msg_id, "🆕 <b>Начать новый диалог?</b>\nКонтекст текущего будет сброшен.", kb)
        else:
            text, kb = self._menu_card()
            self._show(chat_id, msg_id, text, kb)

    def _reset_dialog(self) -> None:
        if self.session.busy:
            self.session.cancel()
            self._stop_typing()
        if get_telegram_mode() == "shared" and self._terminal_agent is not None:
            if hasattr(self._terminal_agent, "clear_history"):
                self._terminal_agent.clear_history()
            self._terminal_session = AgentSession(agent=self._terminal_agent)
        else:
            self._isolated_session = AgentSession()
        self._queue.clear()

    # ─── messages ─────────────────────────────────────────────────────────────

    def _save_upload(self, file_id: str, name_hint: str, suffix_default: str = "") -> Optional[Tuple[str, str]]:
        """Fetch a Telegram file into the project's upload folder. (local path, remote path) or None."""
        if not self.client:
            return None
        info = self.client.get_file(file_id)
        if not info or not info.get("file_path"):
            return None
        if int(info.get("file_size") or 0) > MAX_ATTACH:
            return None
        remote = info["file_path"]
        base = re.sub(r"[^\w.\- ]+", "_", name_hint or Path(remote).name)[:80].strip() or "file"
        if not Path(base).suffix:
            base += Path(remote).suffix or suffix_default
        local = Path(".argent") / "artifacts" / "telegram_uploads" / f"{int(time.time())}_{base}"
        if self.client.download_file(remote, str(local)):
            return str(local), remote
        return None

    def _ingest_attachments(self, message: Dict[str, Any], text: str) -> str:
        """Photos, images and documents the user sent: saved, shown to the model, mentioned in the prompt."""
        photos = message.get("photo")
        doc = message.get("document")
        notes: List[str] = []
        try:
            if photos and isinstance(photos, list):
                got = self._save_upload(photos[-1].get("file_id"), "", ".jpg")
                if got:
                    self._queue_image(got[0], "Telegram Photo")
                    if not text:
                        text = "Пользователь отправил скриншот/изображение. Проанализируй его и опиши, что там видно."
            elif doc:
                is_image = str(doc.get("mime_type", "")).startswith("image/")
                got = self._save_upload(doc.get("file_id"), doc.get("file_name") or "", ".bin")
                if got and is_image:
                    self._queue_image(got[0], "Telegram Photo")
                    if not text:
                        text = "Пользователь отправил скриншот/изображение. Проанализируй его и опиши, что там видно."
                elif got:
                    notes.append(f"[Пользователь прислал файл из Telegram: {got[0]}]")
                    if not text:
                        text = "Пользователь прислал файл. Посмотри его и скажи, что в нём."
                elif doc.get("file_size") and int(doc["file_size"]) > MAX_ATTACH:
                    self.client.send_message(message.get("chat", {}).get("id"), "⚠️ Файл больше 20 МБ — Telegram не отдаёт такие боту.")
        except Exception as e:
            log.error("Failed downloading telegram attachment: %s", e)
        return (text + ("\n\n" + "\n".join(notes) if notes else "")).strip()

    @staticmethod
    def _queue_image(local_file: str, label: str) -> None:
        from vision import encode_image, queue_image
        payload, media_type, _ = encode_image(local_file)
        if payload and media_type:
            queue_image(payload, media_type, f"{label} ({local_file})")

    def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle text message, photo, document, or slash command."""
        chat_id = str(message.get("chat", {}).get("id", ""))
        user = message.get("from", {})
        sender_id = str(user.get("id", ""))
        text = (message.get("text") or message.get("caption") or "").strip()
        has_attachment = bool(message.get("photo") or message.get("document"))
        is_media = any(k in message for k in ("voice", "audio", "video", "video_note", "sticker"))

        if not text and not has_attachment and not is_media:
            return

        # If not authorized, check if this is an initial pairing or reject.  This comes BEFORE any
        # download: a stranger must not be able to make the bot fetch files.
        if not self._is_authorized(chat_id) and not self._is_authorized(sender_id):
            configured_id = get_telegram_chat_id()
            if not configured_id:
                # No chat ID configured yet: prompt the user on how to link
                if self.client:
                    self.client.send_message(
                        chat_id,
                        f"🔐 <b>Argent Security</b>\n\n"
                        f"Ваш Chat ID: <code>{chat_id}</code>\n"
                        f"Бот ещё не привязан к вашей установке Argent.\n"
                        f"Выполните в терминале Argent команду: <code>/telegram setup</code>",
                    )
            else:
                log.warning("Rejected unauthorized message from chat_id %s (%s)", chat_id, user.get("username"))
            return

        msg_id = message.get("message_id")
        if message.get("date") and time.time() - float(message["date"]) > STALE_AFTER and not text.startswith(("/stop", "/cancel")):
            self._stale_skipped.append(text or "(вложение)")
            return

        if is_media and not text and not has_attachment:
            if self.client:
                self.client.send_message(chat_id, "🎙 Голосовые, аудио и видео я пока не разбираю — напишите задачу текстом.")
            return

        text = REPLY_BUTTONS.get(text, text)

        if has_attachment:
            text = self._ingest_attachments(message, text)
        if not text:
            return

        # Route slash commands
        if text.startswith("/"):
            self._handle_command(chat_id, text)
        else:
            # If in shared mode, mirror input to terminal console
            if get_telegram_mode() == "shared":
                _console_print(f"\n[bold #58a6ff]📱 [Telegram]:[/bold #58a6ff] {text}")
            self._react(chat_id, msg_id, "👀")
            self._run_prompt_turn(chat_id, text, message_id=msg_id)

    def _handle_command(self, chat_id: str | int, command_text: str) -> None:
        """Process slash commands sent from Telegram."""
        if not self.client:
            return

        parts = command_text.strip().split(None, 1)
        verb = parts[0].lower().split("@", 1)[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if verb == "/start":
            send_welcome(self.client, chat_id)
            text, kb = self._menu_card()
            self.client.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")
            return

        if verb == "/menu":
            text, kb = self._menu_card()
            self.client.send_message(chat_id, text, reply_markup=kb, parse_mode="HTML")
            return

        if verb == "/help":
            self.client.send_message(chat_id, self._help_card())
            return

        if verb == "/mode":
            if arg.lower() in ("shared", "companion"):
                set_telegram_mode("shared")
                self.client.send_message(
                    chat_id,
                    "🔗 <b>Режим переключен на: Компаньон (shared)</b>\n"
                    "Сессия объединена с терминалом на ПК. Ваши сообщения и ответы будут отображаться в консоли, "
                    "а подтверждения и завершение задач из терминала будут приходить на ваш телефон.",
                )
            elif arg.lower() in ("isolated", "standalone"):
                set_telegram_mode("isolated")
                self.client.send_message(
                    chat_id,
                    "🔒 <b>Режим переключен на: Изолированный (isolated)</b>\n"
                    "Бот работает в отдельной сессии и не выводит сообщения в терминал на ПК.",
                )
            else:
                curr = get_telegram_mode()
                kb = {"inline_keyboard": [
                    [{"text": ("● " if curr == m else "○ ") + MODE_LABEL[m], "callback_data": f"ui:mode:{m}"}] for m in ("shared", "isolated")
                ]}
                self.client.send_message(
                    chat_id,
                    f"ℹ️ <b>Текущий режим:</b> {esc(MODE_LABEL.get(curr, curr))} — {esc(MODE_DESC.get(curr, ''))}\n\n"
                    "Выберите режим кнопкой или командой <code>/mode shared</code> / <code>/mode isolated</code>.",
                    reply_markup=kb,
                )
            return

        if verb in ("/stop", "/cancel"):
            stopped = self._stop_work()
            if stopped:
                for live in list(self._active_lives):
                    live.request_stop_ui()
                self.client.send_message(chat_id, "⏹️ <b>Выполнение остановлено пользователем (Ctrl+C).</b>")
                if get_telegram_mode() == "shared":
                    _console_print("\n[bold yellow]⏹️ [Telegram]: Выполнение остановлено пользователем (Ctrl+C).[/bold yellow]")
            else:
                self.client.send_message(chat_id, "ℹ️ Сейчас нет активных задач для остановки.")
            return

        if verb in ("/clear", "/new", "/reset"):
            self._reset_dialog()
            self.client.send_message(chat_id, "🧹 <b>Контекст диалога очищен.</b> Начата новая сессия.")
            return

        if verb == "/new_confirm":
            kb = {"inline_keyboard": [[{"text": "🆕 Да, начать заново", "callback_data": "ui:new:yes"}, {"text": "Отмена", "callback_data": "ui:menu"}]]}
            self.client.send_message(chat_id, "🆕 <b>Начать новый диалог?</b>\nКонтекст текущего будет сброшен.", reply_markup=kb)
            return

        if verb == "/queue":
            text, kb = self._queue_card()
            self.client.send_message(chat_id, text, reply_markup=kb)
            return

        if verb == "/file":
            self._send_project_file(chat_id, arg)
            return

        if verb == "/vibe":
            if arg in ("on", "1", "true"):
                self.session.set_vibe(True)
                self.client.send_message(chat_id, "✨ <b>Vibe режим:</b> ВКЛЮЧЁН (безопасные действия одобряются автоматически)")
                return
            elif arg in ("off", "0", "false"):
                self.session.set_vibe(False)
                self.client.send_message(chat_id, "🔒 <b>Vibe режим:</b> ВЫКЛЮЧЕН (подтверждение для всех действий)")
                return
            elif arg:
                # /vibe <task>
                self.session.set_vibe(True)
                if get_telegram_mode() == "shared":
                    _console_print(f"\n[bold #58a6ff]📱 [Telegram /vibe]:[/bold #58a6ff] {arg}")
                self._run_prompt_turn(chat_id, arg)
                return
            else:
                # Toggle
                current = getattr(self.session, "vibe", False)
                new_vibe = self.session.set_vibe(not current)
                status = "ВКЛЮЧЁН (безопасные действия одобряются автоматически)" if new_vibe else "ВЫКЛЮЧЕН (подтверждение для всех действий)"
                self.client.send_message(chat_id, f"✨ <b>Vibe режим:</b> {status}")
                return

        if verb in ("/stats", "/status"):
            self.client.send_message(
                chat_id, self._status_card(),
                reply_markup={"inline_keyboard": [[{"text": "🔄 Обновить", "callback_data": "ui:status"}, {"text": "☰ Меню", "callback_data": "ui:menu"}]]},
            )
            return

        # Forward other commands (e.g. /doctor, /init, /research) directly to the agent
        if get_telegram_mode() == "shared":
            _console_print(f"\n[bold #58a6ff]📱 [Telegram]:[/bold #58a6ff] {command_text}")
        self._run_prompt_turn(chat_id, command_text)

    # ─── files back to the phone ──────────────────────────────────────────────

    @staticmethod
    def _safe_project_path(raw: str) -> Tuple[Optional[Path], str]:
        """Resolve a user-given path inside the project, refusing anything that could leak a secret."""
        root = Path.cwd().resolve()
        try:
            target = (root / raw.strip().strip('"').strip("'")).resolve()
        except (OSError, ValueError):
            return None, "Некорректный путь."
        try:
            target.relative_to(root)
        except ValueError:
            return None, "Могу отдать только файлы из папки проекта."
        rel_parts = target.relative_to(root).parts
        if any(p in _SENSITIVE_DIRS for p in rel_parts) or _SENSITIVE_NAMES.search(target.name):
            return None, "Этот файл похож на секрет или служебные данные — не отдаю."
        return target, ""

    def _send_project_file(self, chat_id: str | int, raw: str) -> None:
        if not self.client:
            return
        if not raw:
            self.client.send_message(chat_id, "📎 <b>Получить файл из проекта</b>\nНапример: <code>/file README.md</code> или <code>/file src/app.py</code>")
            return
        target, why = self._safe_project_path(raw)
        if target is None:
            self.client.send_message(chat_id, f"⛔ {esc(why)}")
            return
        if target.is_dir():
            names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir() if not _SENSITIVE_NAMES.search(p.name) and p.name not in _SENSITIVE_DIRS)
            shown = "\n".join(names[:40]) + (f"\n… и ещё {len(names) - 40}" if len(names) > 40 else "")
            self.client.send_message(chat_id, f"📂 <b>{esc(raw)}</b>\n<pre>{esc(shown or '(пусто)')}</pre>")
            return
        if not target.is_file():
            self.client.send_message(chat_id, f"❔ Не нашёл <code>{esc(raw)}</code>.")
            return
        caption = f"📎 <code>{esc(one_line(raw, 80))}</code>"
        if target.suffix.lower() in _IMAGE_SUFFIXES:
            res = self.client.send_photo(chat_id, str(target), caption=caption)
        else:
            res = self.client.send_document(chat_id, str(target), caption=caption)
        if res is None:
            self.client.send_message(chat_id, f"⚠️ Не удалось отправить файл: {esc(getattr(self.client, 'last_error', '') or 'ошибка Telegram')}")

    def _maybe_send_artifact(self, chat_id: str | int, name: str, args: Dict[str, Any], result: Any) -> None:
        """A screenshot the agent took on request from the phone goes back to the phone."""
        if name not in _CAPTURE_TOOLS or not self.client:
            return
        candidates = [str((args or {}).get("output_path") or "")]
        candidates += re.findall(r"([A-Za-z]:[\\/][^\s\"'<>|]+?\.(?:png|jpe?g|webp)|[\w./\\-]+?\.(?:png|jpe?g|webp))", str(result or ""))
        for cand in candidates:
            if not cand:
                continue
            p = Path(cand)
            try:
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES and p.stat().st_size <= 10 * 1024 * 1024:
                    self._bg(self.client.send_photo, chat_id, str(p), "📸 Снимок экрана")
                    return
            except OSError:
                continue

    # ─── running a task from the phone ────────────────────────────────────────

    def _run_prompt_turn(self, chat_id: str | int, text: str, message_id: Optional[int] = None) -> None:
        """Start a turn in the active AgentSession (or queue it) and consume events asynchronously."""
        if not self.client:
            return

        with self._lock:
            if self._busy():
                if len(self._queue) >= MAX_QUEUE:
                    self.client.send_message(
                        chat_id,
                        f"⏳ <b>Argent сейчас выполняет задачу, а очередь заполнена ({MAX_QUEUE}).</b>\n"
                        "Дождитесь завершения или отправьте <code>/stop</code> для отмены.",
                    )
                    return
                self._queue.append({"chat_id": chat_id, "text": text, "message_id": message_id})
                position = len(self._queue)
                queued = True
            else:
                queued = False
                # Start the turn in AgentSession worker thread
                self.session.start(text)
                self._turn_started_at = time.monotonic()

        if queued:
            self.client.send_message(
                chat_id,
                f"🕓 <b>В очереди №{position}</b>\n<i>«{esc(one_line(text, 80))}»</i>\n"
                "Argent сейчас занят — выполню сразу после текущей задачи.",
                reply_markup={"inline_keyboard": [[{"text": "🗑 Убрать из очереди", "callback_data": f"q:del:{position - 1}"}]]},
                disable_notification=True,
            )
            return

        # Start background event consumer thread
        consumer = threading.Thread(
            target=self._event_consumer_worker,
            args=(chat_id, text, message_id),
            daemon=True,
            name="ArgentTelegramConsumer",
        )
        consumer.start()

    def _drain_queue(self) -> None:
        """Start the next waiting message, if Argent is free."""
        with self._lock:
            if not self._queue or self._stop_event.is_set() or self._busy():
                return
            item = self._queue.pop(0)
        if not self.client:
            return
        self.client.send_message(
            item["chat_id"], f"▶️ <b>Из очереди:</b> <i>«{esc(one_line(item['text'], 80))}»</i>",
            disable_notification=True, reply_to=item.get("message_id"),
        )
        if get_telegram_mode() == "shared":
            _console_print(f"\n[bold #58a6ff]📱 [Telegram, из очереди]:[/bold #58a6ff] {item['text']}")
        self._run_prompt_turn(item["chat_id"], item["text"], message_id=item.get("message_id"))

    def _event_consumer_worker(self, chat_id: str | int, title: str = "", message_id: Optional[int] = None) -> None:
        """Consumes events from AgentSession until 'done': progress panel, approvals, and the final answer."""
        if not self.client:
            return

        live = LiveStatus(self.client, chat_id, title=title)
        self._active_lives.append(live)
        self._start_typing(chat_id)
        content: List[str] = []
        errors: List[str] = []
        snapshot: Optional[Dict[str, Any]] = None
        stopped = False
        failed = False
        pending_args: Dict[str, Dict[str, Any]] = {}
        wrote_phase = False

        try:
            while True:
                event = self.session.get_event()
                etype = event.get("type")

                if etype == "approval_request":
                    self._stop_typing()
                    self._send_approval_request(chat_id, event)
                    live.phase("🔐 Жду вашего решения")
                    # Resume typing while waiting or executing
                    self._start_typing(chat_id)

                elif etype == "tool_start":
                    name = event.get("name", "инструмент")
                    args = event.get("args") if isinstance(event.get("args"), dict) else {}
                    pending_args[name] = args
                    live.begin(name, args)
                    wrote_phase = False
                    if get_telegram_mode() == "shared":
                        _console_print(f"[dim]⚙️ [Telegram]: {name}...[/dim]")

                elif etype == "tool_end":
                    name = event.get("name", "")
                    live.end(name, event.get("result", ""), event.get("duration_secs"))
                    self._maybe_send_artifact(chat_id, name, pending_args.pop(name, {}), event.get("result", ""))

                elif etype == "diff":
                    live.diff(event.get("file", ""), event.get("diff", ""))

                elif etype == "thinking":
                    if not wrote_phase:
                        live.phase("🧠 Размышляю…")

                elif etype in ("content", "content_stream"):
                    content.append(event.get("text", ""))
                    if live.started and not wrote_phase:
                        wrote_phase = True
                        live.phase("✍️ Пишу ответ…")

                elif etype == "content_replace":
                    content = [event.get("text", "")]

                elif etype == "notice":
                    live.note(re.sub(r"^\[(?:Argent|System):?\s*|\]$", "", str(event.get("text", "")).strip()))

                elif etype == "error":
                    text = str(event.get("text", ""))
                    if "прервано пользователем" in text.lower():
                        stopped = True
                    else:
                        errors.append(text)

                elif etype == "turn_snapshot":
                    snapshot = event.get("data") if isinstance(event.get("data"), dict) else None

                elif etype == "cancelling":
                    stopped = True

                elif etype == "done":
                    break

        except Exception as e:
            log.error("Error in event consumer: %s", e)
            errors.append(f"Ошибка сессии: {e}")
            failed = True
        finally:
            self._stop_typing()

        final_text = "".join(content).strip()
        state = "stopped" if stopped else ("failed" if (failed or (errors and not final_text)) else "done")
        try:
            if live.started:
                live.finish(state)
            elif live in self._active_lives:
                live.drop()
            self._deliver_result(chat_id, final_text, errors, state, snapshot, message_id)
            self._react(chat_id, message_id, {"done": "👍", "stopped": None, "failed": "😢"}[state])
        except Exception as e:
            log.warning("could not deliver the result to Telegram: %s", e)
        finally:
            if live in self._active_lives:
                self._active_lives.remove(live)
            self._turn_started_at = None
            self._drain_queue()

    def _deliver_result(self, chat_id: str | int, final_text: str, errors: List[str], state: str,
                        snapshot: Optional[Dict[str, Any]], message_id: Optional[int]) -> None:
        """The answer as its own message: formatted, with a footer; a very long one goes as a file."""
        assert self.client is not None
        footer = turn_footer(snapshot)
        if final_text:
            body = final_text
            if errors:
                body += "\n\n⚠️ **Были ошибки:**\n" + "\n".join(f"- {one_line(e, 160)}" for e in errors[:3])
            html_body = format_html(body)
            if state == "stopped":
                html_body += "\n\n⏹ <i>Остановлено до завершения. Напишите «продолжай», чтобы вернуться к задаче.</i>"
            document: Optional[Tuple[str, bytes]] = None
            if len(html_body) > LONG_ANSWER:
                document = ("answer.md", final_text.encode("utf-8"))
                html_body = format_html(_cut_markdown(final_text, 2200)) + "\n\n<i>… полный ответ — во вложении.</i>"
            self.client.send_message(chat_id, html_body + (f"\n\n{footer}" if footer else ""), parse_mode="HTML")
            if document is not None:
                self.client.send_document_bytes(chat_id, document[0], document[1], caption="📎 Полный ответ")
        elif state == "stopped":
            self.client.send_message(chat_id, "⏹ <b>Остановлено.</b> Напишите «продолжай», чтобы вернуться к задаче." + (f"\n\n{footer}" if footer else ""))
        elif errors:
            detail = "\n".join(one_line(e, 400) for e in errors[:3])
            self.client.send_message(chat_id, f"❌ <b>Не получилось</b>\n<pre>{esc(detail)}</pre>" + (f"\n\n{footer}" if footer else ""))
        else:
            self.client.send_message(chat_id, "✅ <b>Готово.</b>" + (f"\n\n{footer}" if footer else ""))

        # If in shared mode, mirror response to terminal console cleanly
        if final_text and get_telegram_mode() == "shared":
            try:
                from rich.markdown import Markdown
                from rich.panel import Panel
                from ui import console
                console.print()
                console.print(Panel(Markdown(final_text), title="[bold #58a6ff]🤖 Argent → Telegram[/bold #58a6ff]", border_style="#58a6ff"))
            except Exception:
                _console_print(f"\n[bold #58a6ff]🤖 [Argent → Telegram]:[/bold #58a6ff]\n{final_text}\n")

    def _send_approval_request(self, chat_id: str | int, event: Dict[str, Any]) -> None:
        """Send an interactive approval card with Inline Keyboard."""
        if not self.client:
            return

        aid = event.get("id")
        destructive = bool(event.get("destructive", False))
        card = approval_card(str(event.get("action", "")), destructive)
        res = self.client.send_message(
            chat_id, card,
            reply_markup=approval_keyboard("appr", aid, event.get("grant_key"), destructive),
            parse_mode="HTML",
        )
        self._remember_card(chat_id, res, card)


# Global singleton instance
telegram_bridge = TelegramBridge()
