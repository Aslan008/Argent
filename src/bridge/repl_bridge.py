"""REPL Bridge connecting modern TUI frontends (e.g. Rust/Ratatui) with Argent's core.

Coordinates session lifecycle, slash commands dispatching, history repair,
file mentions expansion, persistent history, background process monitoring,
and stdout isolation.
"""

import sys
import os
import re
import json
import atexit
import threading
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List

from src.bridge.session import AgentSession
from src.cli.commands import dispatch_command, CommandContext, CommandResult
from src.cli.cli_prompt import expand_file_mentions
from argent.session import save_session
from argent.config import get_provider

_ANSI_STRIP_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

# Every CSI sequence EXCEPT SGR (final byte "m"). Colours and weights are kept —
# the TUI draws them — while cursor movement, erase and mode switches (what a
# Rich `Live` or a progress bar emits) would corrupt the alternate screen and
# must never reach it.
_NON_SGR_CSI_RE = re.compile(r'\x1b\[[0-9;?]*[@-ln-~]')


def _flatten_on_white(img):
    """A clipboard grab carries an alpha channel; a model needs opaque pixels."""
    from PIL import Image

    if img.mode != "RGBA":
        return img.convert("RGB")
    alpha = img.getchannel("A")
    if alpha.getextrema()[0] == 255:
        return img.convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=alpha)
    return bg


class StdoutTrap:
    """Thread-safe context manager / capture device for stdout/stderr during TUI activity.
    Redirects prints to a notice callback instead of letting them corrupt the alternate screen.
    """

    def __init__(self, on_output: Optional[Callable[[str], None]] = None):
        self._on_output = on_output
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        self._lock = threading.Lock()
        self._active = False

    def start(self):
        with self._lock:
            if not self._active:
                self._active = True
                sys.stdout = self
                sys.stderr = self
                try:
                    from argent.ui import console
                    self._orig_console_file = console.file
                    console.file = self
                except Exception:
                    pass

    def stop(self):
        with self._lock:
            if self._active:
                self._active = False
                sys.stdout = self._orig_stdout
                sys.stderr = self._orig_stderr
                try:
                    from argent.ui import console
                    if hasattr(self, "_orig_console_file"):
                        console.file = self._orig_console_file
                except Exception:
                    pass

    def write(self, s: str):
        if not s or not self._active:
            return
        # Newlines are KEPT. The REPL streams a command's output one write at a
        # time with `console.print(..., end="")`, so the next write continues the
        # line the previous one left open; dropping the newline here (as this used
        # to) told the frontend that every write ended a line, and a chunk boundary
        # in the middle of one became a hard break (`git status --short` printed
        # " M" and the path as two separate rows).
        clean = _NON_SGR_CSI_RE.sub('', s)
        if clean and self._on_output is not None:
            try:
                self._on_output(clean)
            except Exception:
                pass

    def flush(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def _clean_choice_item(c):
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return str(c.get("title") or c.get("name") or c.get("value") or c)
    if hasattr(c, "title") and c.title:
        return str(c.title)
    if hasattr(c, "name") and c.name:
        return str(c.name)
    if hasattr(c, "value") and c.value:
        return str(c.value)
    return str(c)


def _prepare_choices(raw_choices):
    """Clean choices into display titles, map title -> real value, and extract initial checks."""
    clean_choices = []
    mapping = {}
    initial_checked = []
    has_checked = False
    for c in (raw_choices or []):
        if isinstance(c, str):
            # Plain strings are the common case, and they must be handled before
            # the attribute probing below: `str` has a *method* called `title`,
            # so getattr(c, "title") is truthy and the option used to be shown
            # as "<built-in method title of str object at 0x...>".
            clean_choices.append(c)
            mapping[c] = c
            initial_checked.append(False)
            continue
        if isinstance(c, dict):
            if c.get("disabled") or str(type(c)).endswith("Separator'>"):
                continue
            title = str(c.get("title") or c.get("name") or c.get("value") or c)
            val = c.get("value", title)
            chk = bool(c.get("checked", False))
        else:
            if getattr(c, "disabled", False) or str(type(c)).endswith("Separator'>"):
                continue
            title = str(getattr(c, "title", None) or getattr(c, "name", None) or getattr(c, "value", None) or c)
            val = getattr(c, "value", title)
            chk = bool(getattr(c, "checked", False))
        if chk:
            has_checked = True
        clean_choices.append(title)
        mapping[title] = val
        initial_checked.append(chk)
    return clean_choices, mapping, (initial_checked if has_checked else None)


class _TuiPromptProxy:
    """Proxy object simulating questionary.Question for TUI mode."""
    def __init__(self, ask_fn):
        self._ask_fn = ask_fn

    def ask(self, **kwargs):
        return self._ask_fn()

    def unsafe_ask(self, **kwargs):
        return self._ask_fn()


def patch_questionary_for_tui() -> None:
    """Route questionary interactive prompts to native TUI modals when ARGENT_TUI=1."""
    try:
        import questionary
    except ImportError:
        return

    orig_select = getattr(questionary, "select", None)
    orig_confirm = getattr(questionary, "confirm", None)
    orig_text = getattr(questionary, "text", None)
    orig_password = getattr(questionary, "password", None)
    orig_checkbox = getattr(questionary, "checkbox", None)

    def _tui_select(message, choices=None, default=None, **kwargs):
        if os.environ.get("ARGENT_TUI") == "1" and ReplBridge.active_bridge is not None:
            clean_choices, mapping, _ = _prepare_choices(choices or [])
            safe_def = _clean_choice_item(default) if default is not None else None
            def _ask():
                res = ReplBridge.ask_active_select(str(message), clean_choices, safe_def)
                if res is None:
                    return None
                return mapping.get(res, res)
            return _TuiPromptProxy(_ask)
        if orig_select:
            return orig_select(message, choices=choices, default=default, **kwargs)
        return _TuiPromptProxy(lambda: None)

    def _tui_confirm(message, default=True, **kwargs):
        if os.environ.get("ARGENT_TUI") == "1" and ReplBridge.active_bridge is not None:
            opts = ["Да", "Нет"]
            safe_def = "Да" if default else "Нет"
            def _ask():
                res = ReplBridge.ask_active_select(str(message), opts, safe_def)
                if res is None:
                    return None
                return res == "Да"
            return _TuiPromptProxy(_ask)
        if orig_confirm:
            return orig_confirm(message, default=default, **kwargs)
        return _TuiPromptProxy(lambda: default)

    def _tui_text(message, default="", **kwargs):
        if os.environ.get("ARGENT_TUI") == "1" and ReplBridge.active_bridge is not None:
            return _TuiPromptProxy(lambda: ReplBridge.ask_active_select(str(message), [], default=str(default)))
        if orig_text:
            return orig_text(message, default=default, **kwargs)
        return _TuiPromptProxy(lambda: default)

    def _tui_password(message, **kwargs):
        if os.environ.get("ARGENT_TUI") == "1" and ReplBridge.active_bridge is not None:
            return _TuiPromptProxy(lambda: ReplBridge.ask_active_select(str(message), [], default=""))
        if orig_password:
            return orig_password(message, **kwargs)
        return _TuiPromptProxy(lambda: "")

    def _tui_checkbox(message, choices=None, **kwargs):
        if os.environ.get("ARGENT_TUI") == "1" and ReplBridge.active_bridge is not None:
            clean_choices, mapping, initial_checked = _prepare_choices(choices or [])
            def _ask():
                import json
                res = ReplBridge.ask_active_select(
                    str(message), clean_choices, None, multi=True, initial_checked=initial_checked
                )
                if res is None:
                    return None
                try:
                    chosen_titles = json.loads(res)
                    if not isinstance(chosen_titles, list):
                        chosen_titles = [res] if res else []
                except Exception:
                    chosen_titles = [res] if res else []
                return [mapping.get(t, t) for t in chosen_titles]
            return _TuiPromptProxy(_ask)
        if orig_checkbox:
            return orig_checkbox(message, choices=choices, **kwargs)
        return _TuiPromptProxy(lambda: [])

    questionary.select = _tui_select
    questionary.confirm = _tui_confirm
    questionary.text = _tui_text
    questionary.password = _tui_password
    questionary.checkbox = _tui_checkbox


def _telegram_bridge():
    """The Telegram bridge when it is running beside the TUI in companion mode, else None.

    Looked up through ``sys.modules`` so a TUI that never enabled Telegram does not import it, and the
    per-event check stays a dictionary lookup."""
    mod = sys.modules.get("src.telegram.bot")
    bridge = getattr(mod, "telegram_bridge", None) if mod is not None else None
    return bridge if bridge is not None and bridge.mirrors_terminal else None


def _mirrored(sink: Optional[Callable[[dict], None]]) -> Optional[Callable[[dict], None]]:
    """The session's sink, with a copy of every event going to the phone (progress, approvals, questions)."""
    if sink is None:
        return None

    def deliver(event: dict) -> None:
        try:
            tg = _telegram_bridge()
            if tg is not None:
                tg.mirror_event(event)
        except Exception:
            pass                      # the phone is a convenience: it must never break the TUI
        sink(event)

    return deliver


#: Unattended turns in a row that ran no tool at all before the run is stopped: the
#: agent is repeating itself, not working. There is no ceiling on how long real work
#: may take — vibe runs until it is switched off or the work stops moving.
AUTO_MAX_IDLE_TURNS = 3
AUTO_DELAY_S = 1.0
#: Wake-ups the agent may chain on its own (wait, wake, wait, wake…) before a person has to say something.
AUTO_MAX_WAKES = 40

#: Turn endings that mean "the agent stopped itself, nothing was achieved". An
#: unattended run must not treat any of them as progress and wake again.
_FORCED_STOP_REASONS = {"loop guard", "tool call written as text"}

#: Turn endings that mean "the work is not done — pick it up again". Everything else
#: ends the run: a final answer, a stop by the person, or a reason nobody anticipated.
#: The list is explicit rather than a blacklist on purpose — a mode that keeps poking
#: on an unknown ending is exactly the heartbeat nobody asked for.
_UNFINISHED_ENDINGS = (
    "heartbeat_signal",
    "tool call limit",
    "length limit hit repeatedly",
    "context overflow",
    "provider error",
    "provider could not be created",
    "provider config invalid",
    "network lost",
    "fallback stream failed",
    "salvage continuations exhausted",
    "turn raised",
    "unhandled error",
    "interrupted — the caller stopped reading",
)

_HEARTBEAT_FULL = re.compile(
    r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\s*\|\s*until=(.*?)\s*\|\s*timeout=(\d+)\s*\]"
)
_HEARTBEAT_SHORT = re.compile(r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\s*\]")


def parse_heartbeat(text: str) -> Optional[str]:
    """Extract a wake-up prompt from a tool result, or None.

    Mirrors `cli_ui._check_auto_signals`: the `|until=|timeout=` form is tried
    first (anchored on those keys so pipes inside the condition survive), then
    the shorter form.
    """
    if not text or "[HEARTBEAT_REQUEST:" not in text:
        return None
    m = _HEARTBEAT_FULL.search(text)
    if not m:
        m = _HEARTBEAT_SHORT.search(text)
        if not m:
            return None
        reason = m.group(2).strip()
        return f"[Heartbeat пробуждение] Причина: {reason}" if reason else "[Heartbeat пробуждение]"
    reason = m.group(2).strip()
    until = m.group(3).strip()
    return (
        f"[Heartbeat пробуждение] Причина: {reason} · условие: {until}"
        if reason or until
        else "[Heartbeat пробуждение]"
    )


def _spend_breakdown(usage, provider: str = "", model: str = "") -> str:
    """Hover text for the spend chip: what the last request took, and how it is known.

    Two different numbers live here and they must not be confused. A provider
    that reports a cost per request (OpenRouter) gives an exact figure; DeepSeek
    does not, so the only evidence is the prepaid balance moving, which another
    client on the same key can move too — hence "~". The ``est`` line is a third
    number: Argent-only tokens priced at the current public list price. It is an
    estimate, not an invoice.
    """
    def _tokens(n) -> str:
        n = int(n or 0)
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    def _money(value: float) -> str:
        return f"{value:.4f}" if value < 1 else f"{value:.2f}"

    lines = []
    if usage.last_prompt or usage.last_completion:
        lines.append(f"Последний запрос   {_tokens(usage.last_prompt)} -> "
                     f"{_tokens(usage.last_completion)} tok")
        if usage.last_cached:
            pct = int(usage.last_cached / usage.last_prompt * 100) if usage.last_prompt else 0
            lines.append(f"Из кэша            {_tokens(usage.last_cached)} tok ({pct}%)")
    if usage.last_spent is not None and usage.last_spent > 0:
        symbol = "$" if usage.last_spent_currency == "USD" else f"{usage.last_spent_currency} "
        if usage.last_spent_approx:
            lines.append(f"Списано с баланса  ~{symbol}{usage.last_spent:.4f}")
            lines.append("Оценка по изменению баланса: другой клиент")
            lines.append("с тем же ключом тоже мог его изменить.")
        else:
            lines.append(f"Стоимость запроса  {symbol}{usage.last_spent:.4f}")
            lines.append("Точная цифра от провайдера.")
    if usage.balance is not None:
        symbol = "$" if usage.balance_currency == "USD" else f"{usage.balance_currency} "
        lines.append(f"Баланс             {symbol}{usage.balance:.2f}")
    if usage.requests:
        total = usage.prompt + usage.completion
        lines.append("")
        lines.append(f"За сессию          {_tokens(total)} tok, {usage.requests} запр.")
        if usage.cost > 0:
            lines.append(f"Всего по счетам    ${usage.cost:.4f}")
        elif usage.session_spent > 0:
            symbol = "$" if usage.session_spent_currency == "USD" else f"{usage.session_spent_currency} "
            lines.append(f"Списано (аккаунт)  ~{symbol}{usage.session_spent:.2f}")

    try:
        from argent.model_pricing import PRICE_VERSION, estimate_cost, is_peak

        session_est = estimate_cost(
            provider, model, usage.prompt, usage.completion, usage.cached
        )
        last_est = estimate_cost(
            provider, model, usage.last_prompt, usage.last_completion, usage.last_cached
        )
        if session_est is not None:
            lines.append("")
            lines.append(f"Оценка Argent      est ${_money(session_est)}")
            if last_est is not None:
                lines.append(f"Оценка запроса     est ${_money(last_est)}")
            lines.append(
                "По токенам и прайсу DeepSeek; не биллинг."
            )
            lines.append(
                f"Тариф {PRICE_VERSION}, {'peak' if is_peak() else 'off-peak'}; "
                "праздники КНР не учтены."
            )
    except Exception:
        pass
    return "\n".join(lines)


class ReplBridge:
    """Central orchestration bridge for TUI frontends."""
    active_bridge: Optional['ReplBridge'] = None

    def __init__(self, agent=None):
        os.environ["ARGENT_TUI"] = "1"
        ReplBridge.active_bridge = self
        patch_questionary_for_tui()
        self.session = AgentSession(agent=agent)
        self.turn_counter = 0
        self.is_project_mode = False
        self.project_iterations = 0
        self.vibe_mode = False
        self._event_sink: Optional[Callable[[dict], None]] = None
        self.turn_rejected = False
        self._idle_turns = 0
        self._auto_timer: Optional[threading.Timer] = None
        self._shutting_down = False
        self._trap = StdoutTrap(on_output=self._on_trap_output)
        self._bg_monitor_started = False
        self._pending_wakes = []
        self._wakes_in_a_row = 0
        # EVENT_QUEUE is process-global and has one consumer at a time; the monitor
        # that drains it must be stoppable, or a finished bridge keeps eating events
        # that belong to whoever waits on the queue next.
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        atexit.register(self.cleanup)

    @property
    def agent(self):
        return self.session.agent

    def turn_was_rejected(self) -> bool:
        """True when the last start_turn() call refused to start a turn."""
        return self.turn_rejected

    def start_trap(self) -> None:
        self._trap.start()

    def stop_trap(self) -> None:
        self._trap.stop()

    def set_event_sink(self, sink: Optional[Callable[[dict], None]]) -> None:
        """Register the receiver for in-memory events from agent and commands."""
        self._event_sink = sink
        self.session.set_event_sink(_mirrored(sink))
        # A restored conversation is put on the screen as one `history` block (see history_replay).
        from src.agent import history_replay
        history_replay.set_sink((lambda event: self._emit(event)) if sink is not None else None)
        # "the model is busy, your request waits" / "background work waits for the PC" reach the screen.
        from src.agent import governor
        governor.set_notifier(
            (lambda text: self._emit({"type": "notice", "text": text})) if sink is not None else None)
        if sink is not None:
            # A frontend is attached, so a finished wait can wake the agent: wait_heartbeat
            # may return at once instead of ending the turn and freezing the prompt.
            from src.agent.watches import WATCHES
            WATCHES.set_handler(self._on_watch_fired)
            self._on_attached()

    def emit(self, event: dict) -> None:
        """Send an event to the attached frontend (for commands that must reach it)."""
        self._emit(event)

    def _emit(self, event: dict) -> None:
        """Emit to the frontend, attaching an actionable hint to failures."""
        if event.get("type") == "error" and "hint" not in event:
            try:
                from argent.error_help import explain_error
                hint = explain_error(str(event.get("text", "")))
                if hint:
                    event = dict(event, hint=hint)
            except Exception:
                pass
        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception:
                pass

    def _on_trap_output(self, text: str) -> None:
        # Route stray console prints to the frontend as live output. Distinct from
        # `notice`: output fragments belong to the same line until a write ends it,
        # and the frontend joins them (a notice is always a message of its own).
        self._emit({"type": "output", "text": text})

    def queue_clipboard_image(self, width: int, height: int, rgba: bytes) -> str:
        """Attach a picture from the clipboard to the next turn.

        The Rust frontend can only hand over raw RGBA (that is the shape a
        clipboard gives), so Pillow turns it into a PNG and `vision.queue_image`
        — the same queue `view_image` and the browser tools use — carries it into
        the turn. Returns the label to show, or the reason it did not work.
        """
        try:
            from PIL import Image
        except Exception:
            return "Pillow не установлен"
        try:
            w, h = int(width), int(height)
            if w <= 0 or h <= 0:
                return "пустое изображение"
            data = bytes(rgba)
            if len(data) < w * h * 4:
                return "повреждённые данные буфера"
            img = _flatten_on_white(Image.frombytes("RGBA", (w, h), data))

            import tempfile
            import time
            path = Path(tempfile.gettempdir()) / f"argent-paste-{int(time.time() * 1000)}.png"
            img.save(path, "PNG")

            from argent import vision
            payload, media, note = vision.encode_image(str(path))
            if not payload:
                return note or "не удалось подготовить изображение"
            vision.queue_image(payload, media, f"скриншот из буфера ({w}×{h})")
            return f"{w}×{h}"
        except Exception as exc:
            return f"ошибка: {exc}"

    def drop_pending_image(self, index: int) -> int:
        """Take an attached picture back out of the queue; returns how many remain."""
        from argent import vision

        return vision.drop_image(int(index))

    def pending_image_full(self, index: int) -> str:
        """One attached picture as a full-screen payload (the `image_preview` shape)."""
        from argent import vision

        return vision.image_full(int(index))

    def preview_full(self) -> str:
        """The last preview at screen resolution, built on demand.

        The state poll carries only a cheap frame (96 columns): a grid wide enough to
        fill a terminal is ~200 KB of base64, and the poll runs every couple of seconds.
        This builds the big one the moment the person opens the picture.
        """
        from argent import image_preview

        return image_preview.full_snapshot()

    def _on_attached(self) -> None:
        """Run startup sequences and emit onboarding banners once sink is ready."""
        try:
            from argent.version import __version__ as _ver
        except Exception:
            _ver = "?"
        prov = get_provider().upper()
        mod = getattr(self.agent, "model_name", "unknown")
        cwd = os.getcwd()

        # The same start screen the REPL prints (main.py): a rule, then the
        # system lines, so the two frontends open identically.
        self._emit({"type": "banner", "version": _ver})
        self._emit({"type": "system", "text": "Argent Coder — автономная среда разработки."})
        self._emit({"type": "system", "text": "Список команд: /help"})
        self._emit({"type": "system", "text": f"Провайдер: {prov}"})
        self._emit({"type": "system", "text": f"Модель: {mod}"})
        self._emit({"type": "system", "text": f"Рабочая директория: {cwd}"})

        if not Path(".argent/AGENTS.md").exists() and not Path("AGENTS.md").exists():
            self._emit({"type": "system", "text": "[dim]Подсказка: нет AGENTS.md — выполните /init, чтобы Argent изучил проект и создал память о нём.[/dim]"})

        def _start_services():
            # Everything below runs AFTER the start-up question has been answered:
            # a server's working directory is fixed when it is spawned, and
            # restoring a session may have changed directory (the REPL orders it
            # the same way).
            # MCP servers startup
            try:
                from argent.config import get_mcp_servers
                from argent.mcp_client import mcp_client
                mcp_servers = get_mcp_servers()
                if mcp_servers:
                    for srv in mcp_servers:
                        name = srv["name"]
                        cfg = {k: v for k, v in srv.items() if k != "name"}
                        try:
                            result = mcp_client.register_and_start(name, cfg)
                            self._emit({"type": "notice", "text": f"MCP: {result}"})
                        except Exception as e:
                            self._emit({"type": "notice", "text": f"MCP server '{name}': {e}"})
                # "Running" and "the model can use it" are independent switches that
                # both look like ON; say so when they disagree, as the REPL does.
                try:
                    from src.agent.mcp_prompt import model_access_warning
                    warning = model_access_warning()
                    if warning:
                        self._emit({"type": "notice", "text": f"{warning}"})
                except Exception:
                    pass
            except Exception:
                pass

            # Startup hook
            try:
                from argent.hook_manager import hook_manager
                hook_manager.call_hook("on_startup")
            except Exception:
                pass
            try:
                from src.agent.shell_hooks import session_started, SHELL_HOOKS
                printed = session_started()
                if printed:
                    self._emit({"type": "notice", "text": printed})
                waiting = SHELL_HOOKS.status()["untrusted"]
                if waiting:
                    self._emit({"type": "notice", "text": (
                        f"Хуки проекта ({waiting}) не запускаются, пока вы им не доверите: /hooks trust")})
            except Exception:
                pass

            # Scheduled automations (/tasks) fire while Argent is open, and the
            # GUI server that used to run the schedule is gone. Jobs run in child
            # processes, so nothing they print can reach this screen. The TUI
            # speaks only when there is something to say — the same rule the
            # desktop toast follows — or an every-15-minutes monitor would fill
            # the chat with "started / finished". A run you started yourself is
            # always answered here.
            try:
                from src.automation import notice
                from src.automation.scheduler import start_background_scheduler

                def _automation_event(ev):
                    text = notice.format_event_notice(ev)
                    if text:
                        self._emit({"type": "notice", "text": text})

                start_background_scheduler(on_event=_automation_event)
                digest = notice.startup_digest()
                if digest:
                    self._emit({"type": "notice", "text": digest})
            except Exception as exc:
                self._emit({"type": "notice", "text": f"Планировщик /tasks не запущен: {exc}"})

            # Telegram bridge attach
            try:
                from src.telegram.bot import telegram_bridge
                telegram_bridge.attach_terminal_agent(self.agent)
                telegram_bridge.attach_terminal_frontend(self)
                from argent.config import get_telegram_enabled, get_telegram_token
                if get_telegram_enabled() and get_telegram_token():
                    ok, msg = telegram_bridge.start()
                    if ok:
                        self._emit({"type": "notice", "text": f"Telegram-мост: ВКЛ ({msg})"})
            except Exception:
                pass

            # Auto-KB
            try:
                from argent.config import get_auto_kb
                if get_auto_kb():
                    from argent.rag_engine import init_external_kbs
                    kb_res = init_external_kbs()
                    if "Successfully" in kb_res:
                        self._emit({"type": "notice", "text": f"KB: {kb_res}"})
            except Exception:
                pass

        def _startup_sequence():
            """First-run setup, the "С чего начать?" question, THEN the services."""
            from src.startup import (
                StartupIO, offer_first_run_setup, offer_previous_context, previous_session,
            )

            # A Low-integrity process can read the project but cannot write to
            # the config, target/ or the project even when the ACL grants
            # FullControl (Windows no-write-up). Say it once instead of making
            # every permission error look like a new mystery.
            try:
                from argent.integrity import integrity_label

                if integrity_label() == "Low":
                    self._emit({"type": "notice", "text": (
                        "[yellow]Argent запущен с Low integrity: запись в конфиг, "
                        "проект и target будет запрещена. Запустите его из обычного "
                        "терминала/Проводника, а не из песочницы или low-integrity "
                        "окна.[/yellow]"
                    )})
            except Exception:
                pass

            io = StartupIO(
                system=lambda text: self._emit({"type": "system", "text": text}),
                error=lambda text: self._emit({"type": "error", "text": text}),
            )

            def run_slash(command: str) -> None:
                ctx = CommandContext(agent=self.agent, user_input=command)
                with self._trap:
                    dispatch_command(command, ctx)

            # Scripted and automated runs have nobody to answer: with
            # ARGENT_NO_STARTUP_PROMPTS=1 the questions are skipped and the
            # session starts clean, exactly as if "Начать с чистого листа" were
            # not offered at all.
            if os.environ.get("ARGENT_NO_STARTUP_PROMPTS") != "1":
                try:
                    offer_first_run_setup(run_slash, io)
                except Exception as exc:
                    self._emit({"type": "notice", "text": f"Первичная настройка не завершена: {exc}"})
                try:
                    offer_previous_context(self.agent, io)
                except Exception as exc:
                    self._emit({"type": "notice", "text": f"Не удалось предложить прошлую сессию: {exc}"})
            _start_services()

        threading.Thread(target=_startup_sequence, daemon=True, name="ArgentTuiStartup").start()

        # Background EVENT_QUEUE monitor
        if not self._bg_monitor_started:
            self._bg_monitor_started = True
            stop = self._monitor_stop

            def _event_queue_monitor():
                import queue
                # Only the active bridge listens: a superseded one must not keep
                # taking events from the queue it no longer owns.
                while not stop.is_set() and ReplBridge.active_bridge is self:
                    try:
                        from src.agent.events import EVENT_QUEUE
                        if EVENT_QUEUE is not None:
                            try:
                                ev = EVENT_QUEUE.get(timeout=0.5)
                                if stop.is_set():
                                    # Stopped while blocked in get(): this event is
                                    # no longer ours, hand it back to the next waiter.
                                    if ev is not None:
                                        EVENT_QUEUE.put(ev)
                                    break
                                if ev and ev.get("type") == "bg_done":
                                    pid = ev.get("pid", "?")
                                    code = ev.get("exit_code", 0)
                                    status = "успешно" if code == 0 else f"код {code}"
                                    self._emit({"type": "notice", "text": f"Фоновая команда {pid} завершена ({status})"})
                            except queue.Empty:
                                pass
                        else:
                            stop.wait(0.5)
                    except Exception:
                        stop.wait(1.0)
            self._monitor_thread = threading.Thread(
                target=_event_queue_monitor, daemon=True, name="ArgentTuiEventQueueMonitor")
            self._monitor_thread.start()

    def stop_event_monitor(self, join: bool = False) -> None:
        """Stop draining the global event queue (idempotent).

        With ``join`` the call returns only once the monitor thread has exited, so
        nothing can take an event from the queue after it."""
        self._monitor_stop.set()
        thread = getattr(self, "_monitor_thread", None)
        if join and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def get_history(self) -> List[str]:
        """Load persistent command history from .argent/input_history."""
        hist_path = Path(".argent") / "input_history"
        if not hist_path.exists():
            return []
        try:
            from prompt_toolkit.history import FileHistory
            fh = FileHistory(str(hist_path))
            # load_history_strings yields newest first -> reverse to chronological order
            entries = list(fh.load_history_strings())
            entries.reverse()
            return entries
        except Exception:
            return []

    def append_history(self, prompt: str) -> None:
        """Append submitted user prompt to .argent/input_history."""
        if not prompt or not prompt.strip():
            return
        try:
            hist_path = Path(".argent") / "input_history"
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            from prompt_toolkit.history import FileHistory
            fh = FileHistory(str(hist_path))
            fh.store_string(prompt.strip())
        except Exception:
            pass

    def set_vibe(self, enabled: bool) -> None:
        """Centralized setter for vibe mode ensuring session sync."""
        self.vibe_mode = bool(enabled)
        self.session.set_vibe(self.vibe_mode)
        self._idle_turns = 0
        if not self.vibe_mode:
            self._cancel_auto()

    def start_turn(self, raw_input: str, _wake: bool = False) -> None:
        """Process user input asynchronously on a worker thread so the TUI thread never hangs."""
        user_input = (raw_input or "").strip()
        if not user_input:
            self._emit({"type": "done"})
            return

        # Claim the turn slot. A previous worker may still be unwinding after a
        # cancel; refuse to run a second turn against the same session, and tell
        # the caller so it can put the prompt back on the queue.
        with self.session._lock:
            if self.session._turn_running:
                self._emit({
                    "type": "notice",
                    "text": "Предыдущий ход ещё завершается — сообщение возвращено в очередь.",
                })
                self.turn_rejected = True
                return
            self.session._turn_running = True
            self.turn_rejected = False

        # A new instruction from the person starts the run over; an unattended turn
        # does not. Re-arming on every turn is what made the old budget meaningless
        # and kept the "[Heartbeat завершен] Продолжай." loop alive.
        self._cancel_auto()
        if not _wake:
            self._wakes_in_a_row = 0
            self._idle_turns = 0

        self.session._active = True
        if not user_input.startswith(("/", "!")):
            try:
                tg = _telegram_bridge()
                if tg is not None:
                    tg.notify_terminal_turn_started(user_input, source="tui", wait=False)
            except Exception:
                pass
        threading.Thread(
            target=self._run_turn_worker,
            args=(user_input,),
            daemon=True,
            name="ArgentTuiTurnWorker"
        ).start()

    # ── waits that finished (wait_heartbeat registers a watch and returns) ───

    def _on_watch_fired(self, message: str, watch) -> None:
        """Runs on the watcher thread: wake the agent now if it is idle, else after its turn."""
        if self._shutting_down or ReplBridge.active_bridge is not self:
            return
        self._emit({"type": "notice", "text": f"Ожидание #{watch.id} закончилось: {watch.describe()}"})
        with self.session._lock:
            if self.session._turn_running:
                self._pending_wakes.append(message)
                return
        self._deliver_wakes([message])

    def _deliver_wakes(self, messages: list) -> None:
        if not messages or self._shutting_down:
            return
        if self._wakes_in_a_row >= AUTO_MAX_WAKES:
            self._emit({"type": "notice", "text": (
                f"[System: агент {AUTO_MAX_WAKES} раз подряд просыпался сам — остановлено. "
                f"Напишите что-нибудь, чтобы продолжить.]")})
            return
        self._wakes_in_a_row += 1
        self.start_turn("\n\n".join(messages), _wake=True)
        if self.turn_was_rejected():            # a person's turn got in first: try again when it ends
            with self.session._lock:
                self._pending_wakes[:0] = messages

    def _drain_wakes(self) -> None:
        with self.session._lock:
            if self.session._turn_running or not self._pending_wakes:
                return
            messages, self._pending_wakes = self._pending_wakes, []
        self._deliver_wakes(messages)

    # ── unattended turns (vibe mode + heartbeat protocol) ────────────────────

    def _cancel_auto(self) -> None:
        if self._auto_timer is not None:
            try:
                self._auto_timer.cancel()
            except Exception:
                pass
            self._auto_timer = None

    def _maybe_auto_continue(self) -> None:
        """Queue the next unattended turn while there is work, not while a clock allows.

        Vibe mode is an approval switch, not a one-shot: it stays on until the person
        turns it off, and this function never flips it. What ends a run is the agent
        standing still — a turn that called no tool changed nothing, and asking again
        only repeats the same answer. That repetition is the real defect the old budget
        was built around (and then hid).
        """
        if self._shutting_down:
            return
        wake = parse_heartbeat(self.session.last_tool_output)
        if wake is None and not self.vibe_mode:
            self._idle_turns = 0
            return

        ended = self.session.last_turn_end

        # A turn the agent itself cut short did no work, however many tools it waved
        # at the problem. Counting those as progress is how an unattended run spun:
        # the loop guard stopped the turn, and this function immediately started the
        # next one, forever. Its own message says why, so this comes first.
        if ended in _FORCED_STOP_REASONS:
            self._idle_turns = 0
            self._emit({"type": "notice", "text": (
                f"[System: ход остановлен принудительно ({ended}) — "
                f"автономный прогон прекращён, чтобы не крутиться вхолостую. "
                f"Vibe остаётся включён; напишите что-нибудь, чтобы продолжить.]")})
            return

        # The agent's own final answer means the work is finished. Poking it afterwards
        # is what made vibe feel broken: a heartbeat arrived after every completed turn,
        # while nothing was running. The same goes for a turn the person stopped — they
        # took control — and for an ending nobody anticipated. Vibe itself stays on:
        # only the auto-run stops, and the person is the one who turns the switch off.
        ends_the_run = not any(ended == r or ended.startswith(r) for r in _UNFINISHED_ENDINGS)
        if wake is None and ends_the_run:
            self._idle_turns = 0
            return

        prompt = wake or "[Heartbeat завершен] Продолжай."
        self._emit({
            "type": "notice",
            "text": f"[System: автоход через {AUTO_DELAY_S:g}с: {prompt}]",
        })
        self._auto_timer = threading.Timer(AUTO_DELAY_S, self._auto_turn, args=(prompt,))
        self._auto_timer.daemon = True
        self._auto_timer.start()

    def _auto_turn(self, prompt: str) -> None:
        if self._shutting_down:
            return
        with self.session._lock:
            if self.session._turn_running:
                # Something else started meanwhile; let that run and stop here.
                self._idle_turns = 0
                return
        self.start_turn(prompt, _wake=True)

    def _run_turn_worker(self, user_input: str) -> None:
        """Runs on a background worker thread: handles commands and agent execution safely."""
        # A person is looking at this frontend, so code on this thread may ask them a question.
        from src.agent.interaction import mark_interactive
        mark_interactive(True)
        try:
            # 0. pre_prompt modifier hook — the single extension point plugins use
            #    to rewrite or filter a prompt before anything else sees it.
            #    Mirrors the legacy CLI, which called this before @file expansion
            #    and slash dispatch (main.py).
            try:
                from argent.hook_manager import hook_manager
                user_input = hook_manager.call_modifier_hook("pre_prompt", user_input)
            except Exception as exc:
                self._emit({"type": "notice", "text": f"[System: pre_prompt hook failed: {exc}]"})
            if not (user_input or "").strip():
                self.session._active = False
                self._emit({"type": "done"})
                return

            # UserPromptSubmit (hooks.json): context for the agent, or a veto. Commands are not
            # prompts — a hook that decorates "what is in this repo" must not mangle "/model".
            if not user_input.startswith(("/", "!")):
                from src.agent.shell_hooks import prompt_submitted
                user_input, note = prompt_submitted(user_input)
                if user_input is None:
                    self._emit({"type": "notice", "text": f"Сообщение остановлено хуком: {note}"})
                    self.session._active = False
                    self._emit({"type": "done"})
                    return

            # 1. Handle Slash Commands & Shell Escapes
            if user_input.startswith("/") or user_input.startswith("!"):
                ctx = CommandContext(
                    agent=self.agent,
                    user_input=user_input,
                    turn_counter=self.turn_counter,
                    is_project_mode=self.is_project_mode,
                    project_iterations=self.project_iterations,
                    vibe_mode=self.vibe_mode,
                )
                with self._trap:
                    cmd_res = dispatch_command(user_input, ctx)

                if cmd_res.handled:
                    if cmd_res.vibe_mode is not None:
                        self.set_vibe(cmd_res.vibe_mode)
                    if cmd_res.turn_counter is not None:
                        self.turn_counter = cmd_res.turn_counter
                    if cmd_res.is_project_mode is not None:
                        self.is_project_mode = cmd_res.is_project_mode
                    if cmd_res.project_iterations is not None:
                        self.project_iterations = cmd_res.project_iterations

                    if cmd_res.new_input:
                        # Command transformed input and wants agent execution
                        user_input = cmd_res.new_input
                    else:
                        # Handled completely by command handler
                        self.session._active = False
                        if cmd_res.should_exit:
                            # The handler already ran its own side effects (e.g.
                            # /exit exports the chat). Tell the frontend to quit;
                            # without this the TUI kept running after /exit.
                            self._emit({"type": "exit"})
                        else:
                            self._emit({"type": "done"})
                        return
                else:
                    self._emit({"type": "error", "text": f"Неизвестная команда: {user_input}"})
                    self.session._active = False
                    self._emit({"type": "done"})
                    return

            # 2. Expand @file mentions
            expanded_input, attached_files = expand_file_mentions(user_input)
            if attached_files:
                files_str = ", ".join("@" + f for f in attached_files)
                self._emit({"type": "notice", "text": f"Прикреплены файлы: {files_str}"})
                user_input = expanded_input

            # 3. Auto-save session periodically
            self.turn_counter += 1
            if self.turn_counter == 1 or self.turn_counter % 5 == 0:
                self._auto_save_session()

            # 4. Run agent turn (runs inline inside this worker thread)
            # The agent and its tools may print from arbitrary threads. Without
            # the trap those writes land on the alternate screen and can shift
            # the toolbar/cards until a resize forces a full repaint.
            with self._trap:
                self.session._run_turn(user_input)
            try:
                from src.agent.shell_hooks import turn_stopped
                printed = turn_stopped()
                if printed:
                    self._emit({"type": "notice", "text": printed})
            except Exception:
                pass
            # A heartbeat signal or vibe mode may ask for another turn.
            self._maybe_auto_continue()
        except Exception as e:
            self._emit({"type": "error", "text": f"Ошибка выполнения: {e}"})
            self.session._active = False
            self._emit({"type": "done"})
        finally:
            mark_interactive(False)
            try:
                # A denied/aborted turn never reaches _snapshot_markup, so
                # refresh here as well: the balance follows actual spend even
                # when the turn ended early.
                from argent.provider_balance import prime_balance

                prime_balance(force=True)
            except Exception:
                pass
            with self.session._lock:
                self.session._turn_running = False
                self.session._active = False
            self._drain_wakes()

    def cancel_turn(self) -> None:
        """Cancel current execution and repair conversation history."""
        self.session.cancel()
        try:
            if hasattr(self.agent, "repair_history"):
                self.agent.repair_history()
        except Exception:
            pass

    def reply_approval(self, approval_id: int, decision: str) -> None:
        """Reply to a pending approval request."""
        self.session.reply_approval(approval_id, decision)
        try:
            tg = _telegram_bridge()
            if tg is not None:
                tg.mirror_resolved("ap", approval_id, decision)
        except Exception:
            pass

    def ask_select(
        self,
        title: str,
        options: list,
        default: Optional[str] = None,
        multi: bool = False,
        initial_checked: Optional[list] = None,
    ) -> Optional[str]:
        """Request an interactive option selection from the TUI frontend."""
        return self.session.ask_select(
            title, options, default, multi=multi, initial_checked=initial_checked
        )

    def reply_select(self, select_id: str, value: Optional[str] = None) -> None:
        """Reply to a pending select request from TUI."""
        self.session.reply_select(select_id, value)
        try:
            tg = _telegram_bridge()
            if tg is not None:
                tg.mirror_resolved("sel", select_id, "" if value is None else "ответ на ПК")
        except Exception:
            pass

    @classmethod
    def ask_active_select(
        cls,
        title: str,
        options: list,
        default: Optional[str] = None,
        multi: bool = False,
        initial_checked: Optional[list] = None,
    ) -> Optional[str]:
        """Convenience helper to ask for selection on the active TUI bridge."""
        if cls.active_bridge is not None:
            return cls.active_bridge.ask_select(
                title, options, default, multi=multi, initial_checked=initial_checked
            )
        return None

    def background_view(self, pid: str) -> str:
        """What the background window shows, as JSON.

        The TUI polls this while its window is open. Unlike `/jobs <pid>` it takes
        nothing away from the agent: the process's output tail is copied, not drained.
        """
        from argent.tools.command_ops import background_view

        try:
            view = background_view(pid)
        except Exception as e:                     # a window must never break a turn
            view = {"pid": str(pid), "status": "unknown", "command": "", "elapsed": 0,
                    "code": None, "output": "", "error": str(e)}
        return json.dumps(view, ensure_ascii=False)

    def get_state(self) -> Dict[str, Any]:
        """Get snapshot of current runtime state for TUI status/header."""
        state = self.session.state()
        try:
            # The REPL's toolbar names the strategy tier (tiny/local/cloud), not
            # the model-size category the session state reports.
            from src.cli.cli_prompt import _tier_label
            state["tier"] = _tier_label(self.agent)
        except Exception:
            pass
        try:
            # The toolbar shows "Claude Sonnet 5.5", the config keeps "claude-sonnet-5-5".
            from argent.claude_catalog import label_for
            state["model"] = label_for(state.get("provider"), state.get("model"))
        except Exception:
            pass
        state["turn_counter"] = self.turn_counter
        try:
            from src.agent.watches import WATCHES
            state["waits"] = len(WATCHES.pending())
        except Exception:
            pass
        state["vibe"] = self.vibe_mode
        state["is_project_mode"] = self.is_project_mode
        state["cwd"] = os.path.basename(os.getcwd()) or os.getcwd()
        try:
            import argent.config as _cfg
            configured_max = _cfg.get_context_window()
        except Exception:
            configured_max = 32768
        ctx_usage = state.get("context", {})
        state["ctx_tokens"] = ctx_usage.get("tokens", 0)
        state["ctx_max"] = ctx_usage.get("max") or configured_max
        state["ctx_percent"] = float(ctx_usage.get("percent", 0.0))
        try:
            import argent.config as _cfg
            state["thinking_level"] = _cfg.get_thinking_level()
        except Exception:
            state["thinking_level"] = "auto"
        try:
            state["snapshot_mode"] = _cfg.get_turn_snapshot_mode()
        except Exception:
            state["snapshot_mode"] = "compact"
        try:
            # The exact label the REPL's toolbar shows ("AUTO+rethink", ...).
            state["thinking_label"] = _cfg.get_thinking_display_label() or ""
        except Exception:
            state["thinking_label"] = ""
        state["mode"] = "VIBE" if self.vibe_mode else "CHAT"
        try:
            from src.agent.plan_mode import PLAN
            from src.agent.todos import TODOS
            if PLAN.active:
                state["mode"] = "PLAN"
            # The checklist rides as JSON text: the frontend's state map holds strings.
            state["todos"] = json.dumps(TODOS.snapshot("main"), ensure_ascii=False)
            # Delegated work has its own list. Without it the panel shows one item
            # ("delegate") while a sub-agent works through eight, unseen.
            state["todos_sub"] = json.dumps(TODOS.snapshot_subagents(), ensure_ascii=False)
        except Exception:
            state["todos"] = "[]"
        try:
            # Only the running ones ride here: the panel is a glance, and a finished
            # process can carry kilobytes of output that must not be polled twice a
            # second. `background_done` is just a number for the panel's button.
            from argent.tools.command_ops import (
                ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK, COMPLETED_PROCESSES,
            )
            import time as _time

            now = _time.time()
            with ACTIVE_PROCESSES_LOCK:
                # A process that has exited must leave the "running" list even when
                # the agent is idle: the bg_done event is drained only during a turn,
                # so a finished sleep(240) used to sit in the panel as "9:29" and keep
                # counting. poll() is free, so the panel heals itself.
                from argent.tools.command_ops import _close_proc_pipes, _record_completed

                for _pid in list(ACTIVE_PROCESSES.keys()):
                    _info = ACTIVE_PROCESSES[_pid]
                    _proc = _info.get("process")
                    if _proc is None or not hasattr(_proc, "poll"):
                        continue
                    try:
                        _code = _proc.poll()
                    except Exception:
                        continue
                    if _code is None:
                        continue
                    try:
                        _close_proc_pipes(_proc)
                    except Exception:
                        pass
                    del ACTIVE_PROCESSES[_pid]
                    _record_completed(_pid, _info, _code)

                running = [
                    {
                        "pid": str(pid),
                        "command": str(info.get("command", "")),
                        "elapsed": int(now - info["started"]) if info.get("started") else 0,
                        "pty": bool(info.get("pty")),
                    }
                    for pid, info in ACTIVE_PROCESSES.items()
                ]
                # The finished ones ride along as a short list: no output here, the
                # panel only names them. Their text comes from /jobs <pid> on a click.
                finished = [
                    {
                        "pid": str(pid),
                        "command": str(info.get("command", "")),
                        "code": info.get("exit_code"),
                        "secs": (
                            int((info.get("finished") or 0) - info["started"])
                            if info.get("finished") and info.get("started") else 0
                        ),
                    }
                    for pid, info in list(COMPLETED_PROCESSES.items())[-10:]
                ]
                done = len(COMPLETED_PROCESSES)
            state["background"] = json.dumps(running, ensure_ascii=False)
            state["background_finished"] = json.dumps(finished, ensure_ascii=False)
            state["background_done"] = str(done)
        except Exception:
            state["background"] = "[]"
            state["background_finished"] = "[]"
            state["background_done"] = "0"
        try:
            # The picture the last image tool brought back, as a grid the frontend
            # draws with half blocks. Only the newest one rides here: a card keeps
            # the copy it was given, so this never has to be polled per frame.
            from argent import image_preview

            state["image_preview"] = image_preview.snapshot()
        except Exception:
            state["image_preview"] = ""
        try:
            # Pictures waiting to be sent with the next message. Tools attach them
            # (view_image, a screenshot) as well as a paste, and until now the only
            # trace of them was a notice in the chat: there was no way to see what
            # was going with the message or to take it back.
            from argent import vision as _vision

            state["pending_images"] = _vision.pending_snapshot()
        except Exception:
            state["pending_images"] = "[]"
        try:
            from argent.usage_tracker import usage as _usage

            # Kick a background refresh on the first state read and let the
            # toolbar keep rendering from the last known value. The worker also
            # refreshes after every turn, so the number follows real spend.
            try:
                from argent.provider_balance import current_balance, prime_balance

                prime_balance()
                _bal = current_balance()
                if _bal is not None:
                    _usage.set_balance(_bal.amount, _bal.currency, provider=state.get("provider", ""))
                elif (
                    state.get("provider")
                    and _usage.balance_provider
                    and state.get("provider") != _usage.balance_provider
                ):
                    _usage.clear_balance(state.get("provider", ""))
            except Exception:
                pass

            state["session_prompt"] = float(_usage.prompt)
            state["session_completion"] = float(_usage.completion)
            state["session_cost"] = float(_usage.cost)
            state["session_requests"] = int(_usage.requests)
            # Same one-liner the REPL's toolbar prints. Balance alone is enough
            # to show before the first request (the provider was just probed).
            state["session_summary"] = (
                _usage.format_session()
                if (_usage.requests or _usage.balance is not None)
                else ""
            )
            # What the LAST request took, and the hover text that explains it.
            # Survives a restart because the session file carries it.
            state["last_spend"] = _usage.format_last_spend()
            state["spend_breakdown"] = _spend_breakdown(
                _usage, state.get("provider", ""), state.get("model", "")
            )
        except Exception:
            pass
        try:
            from argent.model_capabilities import inert_label
            from argent.config import get_current_model as _current_model

            # The raw provider/model, not the display label set above: the
            # Ollama capability cache is keyed by the real model name.
            _provider = state.get("provider", "") or ""
            state["inert_label"] = inert_label(_provider, _current_model(_provider or None))
        except Exception:
            state["inert_label"] = ""
        try:
            # Hover text for the context chip: where the next request's tokens go.
            # The gauge must count the SAME total: system prompt + tool schemas
            # + history. It used to show history only, so the chip said 16%
            # while its own hover said 18%.
            signature = (len(self.agent.messages), state.get("provider", ""), state.get("model", ""))
            cached = getattr(self, "_ctx_breakdown_cache", None)
            if cached and len(cached) >= 5 and cached[0] == signature:
                text = cached[1]
                total, ctx_max, percent = cached[2], cached[3], cached[4]
            else:
                breakdown = self.agent.get_context_breakdown()

                def _tokens(n) -> str:
                    n = int(n or 0)
                    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

                lines = [
                    f"Системный промпт   {_tokens(breakdown.get('system'))} tok",
                    f"Инструменты ({breakdown.get('tool_count', 0)})   {_tokens(breakdown.get('tools'))} tok",
                    f"История разговора   {_tokens(breakdown.get('history'))} tok",
                    "",
                    f"Всего   {_tokens(breakdown.get('total'))} / {_tokens(breakdown.get('max'))} tok"
                    f"  ·  {float(breakdown.get('percent', 0.0)):.0f}%",
                ]
                text = "\n".join(lines)
                total = int(breakdown.get("total", 0) or 0)
                ctx_max = int(breakdown.get("max", 0) or 0)
                percent = float(breakdown.get("percent", 0.0) or 0.0)
                self._ctx_breakdown_cache = (signature, text, total, ctx_max, percent)
            state["ctx_breakdown"] = text
            state["ctx_tokens"] = total
            state["ctx_max"] = ctx_max
            state["ctx_percent"] = percent
            state["context"] = {"tokens": total, "max": ctx_max, "percent": percent}
        except Exception:
            state["ctx_breakdown"] = ""
        return state

    def render_block(self, kind: str, payload_json: str, width: int) -> str:
        """Render one visual block to ANSI at `width` columns (see bridge.render).

        The frontend draws what the REPL would print, using the REPL's own
        renderer, rather than a second implementation of it.
        """
        import json
        from src.bridge.render import render_block
        return render_block(kind, json.loads(payload_json or "{}"), int(width))

    def get_ui_config(self) -> str:
        """JSON of the theme/look settings the frontend's live widgets need."""
        import json
        from src.bridge.render import ui_config
        return json.dumps(ui_config(), ensure_ascii=False)

    def set_thinking_level(self, level: str) -> str:
        """Change the reasoning effort level. Returns the level actually applied.

        This writes straight to the config the agent reads on every request, so
        it is a real change, not a display toggle. Raises ValueError on an
        unknown level, which surfaces to the TUI as an error message.
        """
        import argent.config as _cfg
        applied = _cfg.set_thinking_level(level)
        try:
            from src.cli.sound import play_sound
            play_sound("done")
        except Exception:
            pass
        return applied

    def get_thinking_levels(self) -> List[str]:
        """The cycle of levels the frontend may step through."""
        import argent.config as _cfg
        return list(_cfg.THINKING_LEVELS)

    def get_command_completions(self) -> List[Dict[str, str]]:
        """Return commands, aliases and documented subcommands for TUI completion.

        The legacy CLI completed far more than the registry: it also offered
        every `aliases=[...]` entry and the subcommand strings catalogued in
        `cli_prompt.COMMAND_HELP` (e.g. `/sound set done`). Without those the
        TUI popup could only ever show the ~60 registry names.
        """
        from src.cli.commands import completion_catalog

        res = completion_catalog()
        seen = {c["name"] for c in res}
        try:
            from argent.hook_manager import hook_manager
            for name in hook_manager.get_custom_commands():
                if f"/{name}" not in seen:
                    res.append({"name": f"/{name}", "help": "(команда плагина)",
                                "category": "plugin"})
        except Exception:
            pass
        return res

    def _session_state(self) -> dict:
        """What belongs to the conversation but is not a message.

        Today that is the checklist. The messages came back on restore while the list
        did not, so the agent's own notes vanished under a history that still talked
        about them, and the panel went blank mid-plan.
        """
        state = {}
        try:
            from src.agent.todos import TODOS

            snap = TODOS.snapshot("main")
            if snap:
                state["todos"] = json.dumps(snap, ensure_ascii=False)
        except Exception:
            pass
        return state

    def _auto_save_session(self) -> None:
        try:
            # The spend rides along: a session restored without it starts its
            # counters at zero, and the last task's cost is then unknowable.
            try:
                from argent.usage_tracker import usage as _usage

                usage_snapshot = _usage.snapshot()
            except Exception:
                usage_snapshot = {}
            saved_id = save_session(
                getattr(self.agent, "messages", []),
                {
                    "model": getattr(self.agent, "model_name", "unknown"),
                    "provider": get_provider(),
                },
                session_id=getattr(self.agent, "session_id", None),
                usage=usage_snapshot,
                state=self._session_state(),
            )
            if saved_id:
                self.agent.session_id = saved_id
                try:
                    self.session.link_trace_session(saved_id)
                except Exception:
                    pass
        except Exception:
            pass

    def cleanup(self) -> None:
        """Complete application shutdown handler matching the old CLI."""
        self._shutting_down = True
        self.stop_event_monitor()
        self._cancel_auto()
        try:
            from src.agent.watches import WATCHES
            if WATCHES._handler == self._on_watch_fired:
                WATCHES.set_handler(None)
        except Exception:
            pass
        self.stop_trap()
        # Shut down browser engine
        try:
            from argent.browser_engine import browser_engine
            browser_engine.run(browser_engine.shutdown(), timeout=5)
        except Exception:
            pass
        # Stop background command processes
        try:
            from argent.tools.command_ops import ACTIVE_PROCESSES, _kill_process_tree, _close_proc_pipes
            items = list(ACTIVE_PROCESSES.items()) if isinstance(ACTIVE_PROCESSES, dict) else list(ACTIVE_PROCESSES)
            for item in items:
                try:
                    proc_info = item[1] if isinstance(item, tuple) else item
                    proc = proc_info.get("process")
                    if proc:
                        _kill_process_tree(proc)
                        _close_proc_pipes(proc)
                except Exception:
                    pass
        except Exception:
            pass
        # Stop MCP servers
        try:
            from argent.mcp_client import mcp_client
            mcp_client.stop_all()
        except Exception:
            pass
        # Stop Telegram bridge
        try:
            from src.telegram.bot import telegram_bridge
            if telegram_bridge.is_running:
                telegram_bridge.stop()
        except Exception:
            pass
        # Save session
        self._auto_save_session()
        try:
            self.session.close_trace()
        except Exception:
            pass
