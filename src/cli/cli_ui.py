import re
import time
from rich.live import Live
from rich.console import Group
from rich.text import Text
from rich.panel import Panel
from rich.spinner import Spinner as RichSpinner
from rich.markup import escape
from ui import (
    console, print_system, print_error, print_tool_start, print_tool_end,
    s, c, create_content_panel,
    create_final_panel, safe_print, print_context_usage, print_assistant_header,
    print_committed_markdown, print_checkpoint,
)


# System recovery/status messages travel on the same "error" stream channel as
# genuine failures, but they are NOT errors — they are Argent narrating its own
# self-healing (auto-continue, salvage, loop-guard, no-action nudge, context
# trim). They are tagged with a bracketed source. Render those as muted system
# notices instead of a red "Error:", which alarms and confuses the user.
_NOTICE_PREFIXES = ("[System:", "[Loop Guard]", "[Argent:")


def _render_stream_error(content: str) -> None:
    """Route an 'error' stream chunk: system notices to print_system, genuine
    failures to print_error."""
    if content.lstrip().startswith(_NOTICE_PREFIXES):
        print_system(content.strip())
    else:
        print_error(content)


def _time_observe(source, timing: dict, start_time: float):
    """Pass chunks through unchanged while measuring the turn: first-chunk time
    (≈ prompt prefill/TTFT) and total tool time (the span between each tool_start
    and its tool_end — the agent blocks producing tool_end until the tool
    finishes). Everything else is model generation. Kept a pure generator so the
    accounting is testable without the renderer."""
    for ch in source:
        if timing["first"] is None:
            timing["first"] = time.time() - start_time
        kind = ch.get("type")
        if kind == "tool_start":
            timing["_ts"] = time.time()
        elif kind == "tool_end" and timing["_ts"] is not None:
            timing["tools"] += time.time() - timing["_ts"]
            timing["_ts"] = None
        yield ch


def queue_panel(queue_items):
    """The `Очередь сообщений (N)` box for pending messages, or None when empty."""
    if not queue_items:
        return None
    lines = []
    for i, item in enumerate(queue_items, 1):
        short_item = item.replace("\n", " ")
        if len(short_item) > 65:
            short_item = short_item[:62] + "…"
        lines.append(f" [dim]{i}.[/dim] [white]{escape(short_item)}[/white]")
    return Panel(
        Text.from_markup("\n".join(lines)),
        title=f"[bold cyan]Очередь сообщений ({len(queue_items)})[/bold cyan]",
        border_style="cyan dim",
        padding=(0, 1),
    )


def _render_queue_input_display():
    """Renders the queued messages box and the live interactive input line."""
    try:
        from src.cli.input_queue import get_live_input_state
        state = get_live_input_state()
        text, cursor_pos, queue_items = state.snapshot()
    except Exception:
        return None

    if not text and not queue_items:
        return None

    elements = []
    panel = queue_panel(queue_items)
    if panel is not None:
        elements.append(panel)

    if text:
        before = text[:cursor_pos]
        if cursor_pos < len(text):
            cur_char = text[cursor_pos]
            after = text[cursor_pos + 1:]
        else:
            cur_char = " "
            after = ""

        rendered_before = escape(before).replace("\n", "\n  │ ")
        cur_markup = "[black on white]" + escape(cur_char if cur_char != "\n" else "↵\n  │ ") + "[/black on white]"
        rendered_after = escape(after).replace("\n", "\n  │ ")

        input_markup = (
            f"[bold #58a6ff]❯ [Ввод на лету]:[/bold #58a6ff] {rendered_before}{cur_markup}{rendered_after}\n"
            f"[dim](Enter: в очередь · Shift+Enter: новая строка · ↑: вернуть из очереди · Esc: стереть)[/dim]"
        )
        elements.append(Text.from_markup(input_markup))

    if len(elements) == 1:
        return elements[0]
    return Group(*elements)


# ── Dynamic renderables ─────────────────────────────────────────────
# These classes implement __rich_console__, which Rich calls on EVERY
# Live refresh frame. This means the elapsed timer updates
# automatically even while next(chunk_iterator) is blocked.

def _format_time_compact(seconds: float) -> str:
    """Format seconds into readable compact timer string, e.g. '0:08', '1:24', '1:12:05'."""
    secs = int(max(0, seconds))
    mins, sec = divmod(secs, 60)
    hours, m = divmod(mins, 60)
    if hours > 0:
        return f"{hours}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _format_time_human(seconds: float) -> str:
    """Format seconds into friendly summary string, e.g. '8.2s', '1m 24s', '1h 12m'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    rem_secs = int(seconds % 60)
    hours = int(mins // 60)
    if hours > 0:
        rem_mins = mins % 60
        return f"{hours}h {rem_mins}m"
    return f"{mins}m {rem_secs}s"


class _WaitingDisplay:
    """Shows live animated spinner with ticking timer while waiting for initial response/tokens (TTFT/prefill)."""

    def __init__(self, start_turn_time: float):
        self.start_turn_time = start_turn_time
        self._spinner = RichSpinner("dots")

    def __rich_console__(self, console, options):
        elapsed = _format_time_compact(time.time() - self.start_turn_time)
        self._spinner.text = Text(f" Ожидание ответа... ({elapsed})", style="dim cyan")
        yield self._spinner
        q_disp = _render_queue_input_display()
        if q_disp is not None:
            yield q_disp


class _ThinkingDisplay:
    """Shows thinking text + animated spinner with elapsed timer (phase and turn total)."""

    def __init__(self, label_style, text_style, start_turn_time: float = None, title: str = "Анализ:", spinner_label: str = "Анализ задачи..."):
        self.thinking_text = ""
        self.phase_start_time = time.time()
        self.start_turn_time = start_turn_time or self.phase_start_time
        self._label_style = label_style
        self._text_style = text_style
        self._title = title
        self._spinner_label = spinner_label
        self._spinner = RichSpinner("dots")

    def __rich_console__(self, console, options):
        now = time.time()
        phase_elapsed = _format_time_compact(now - self.phase_start_time)
        total_elapsed = _format_time_compact(now - self.start_turn_time)

        yield Text.from_markup(
            f"\n[{self._label_style}]{self._title}[/{self._label_style}]\n"
        )

        max_lines = max(3, (console.size.height or 24) - 6)
        width = console.size.width or 80
        display = self.thinking_text[-(max_lines * width):]
        display = "\n".join(display.splitlines()[-max_lines:]) or "..."
        yield Text(display, style=self._text_style)

        yield Text("")

        if (now - self.phase_start_time) < (now - self.start_turn_time) - 1:
            time_label = f"этап {phase_elapsed} · всего {total_elapsed}"
        else:
            time_label = f"{total_elapsed}"

        self._spinner.text = Text(
            f" {self._spinner_label} ({time_label})", style="dim cyan"
        )
        yield self._spinner
        q_disp = _render_queue_input_display()
        if q_disp is not None:
            yield q_disp


class _ToolGenDisplay:
    """Shows streaming tool argument text in a panel with elapsed timer (phase and total)."""

    def __init__(self, start_turn_time: float = None):
        self.tool_name = "?"
        self.tool_text = ""
        self.total_bytes = 0
        self.phase_start_time = time.time()
        self.start_turn_time = start_turn_time or self.phase_start_time

    def __rich_console__(self, console, options):
        now = time.time()
        phase_elapsed = _format_time_compact(now - self.phase_start_time)
        total_elapsed = _format_time_compact(now - self.start_turn_time)
        if (now - self.phase_start_time) < (now - self.start_turn_time) - 1:
            time_label = f"этап {phase_elapsed} · всего {total_elapsed}"
        else:
            time_label = f"{total_elapsed}"

        display_text = self.tool_text[-1000:]
        if len(self.tool_text) > 1000:
            display_text = "...\n" + display_text

        yield Panel(
            Text(display_text or "...", style="dim green"),
            title=(
                f"[dim cyan]Генерация: {self.tool_name} "
                f"({self.total_bytes} B | {time_label})[/dim cyan]"
            ),
            border_style="cyan",
        )
        q_disp = _render_queue_input_display()
        if q_disp is not None:
            yield q_disp


class _ToolWaitDisplay:
    """Shows dynamic spinner with elapsed timer while a tool is executing."""

    def __init__(self, tool_name: str, start_turn_time: float, tool_args: dict = None):
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.phase_start = time.time()
        self.start_turn_time = start_turn_time
        self._spinner = RichSpinner("dots")

    def __rich_console__(self, console, options):
        now = time.time()
        phase_elapsed = _format_time_compact(now - self.phase_start)
        from ui import _format_tool_header_markup
        header_markup = _format_tool_header_markup(self.tool_name, self.tool_args)

        self._spinner.text = Text.from_markup(
            f"  [bold cyan]⠋[/bold cyan] {header_markup} [dim]({phase_elapsed})[/dim]"
        )
        yield self._spinner
        q_disp = _render_queue_input_display()
        if q_disp is not None:
            yield q_disp


class _HeartbeatWaitDisplay:
    """Shows dynamic spinner with elapsed/timeout timer and active background process status during heartbeat wait."""

    def __init__(self, condition: str = "", timeout: float = 60, target_pids: set = None, is_timer: bool = False):
        self.condition = condition
        self.timeout = float(timeout or 60)
        self.target_pids = set(target_pids) if target_pids else set()
        self.is_timer = is_timer
        self.start_time = time.monotonic()
        self._spinner = RichSpinner("dots")

    def __rich_console__(self, console, options):
        now = time.monotonic()
        elapsed = now - self.start_time
        waited_str = _format_time_compact(elapsed)
        timeout_str = _format_time_compact(self.timeout)

        proc_parts = []
        try:
            from tools.command_ops import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
            with ACTIVE_PROCESSES_LOCK:
                if self.target_pids:
                    pids_to_check = sorted(self.target_pids, key=lambda x: int(x) if str(x).isdigit() else str(x))
                else:
                    pids_to_check = sorted(ACTIVE_PROCESSES.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
                for pid in pids_to_check:
                    pinfo = ACTIVE_PROCESSES.get(str(pid))
                    if pinfo:
                        proc = pinfo.get("process")
                        if proc and proc.poll() is None:
                            started = pinfo.get("started")
                            p_elapsed = int(time.time() - started) if started else 0
                            proc_parts.append(f"PID {pid}: [bold green]RUNNING[/bold green] ({_format_time_compact(p_elapsed)})")
                        elif proc:
                            proc_parts.append(f"PID {pid}: [bold yellow]EXITED ({proc.poll()})[/bold yellow]")
        except Exception:
            pass

        extra = f" · {' · '.join(proc_parts)}" if proc_parts else ""

        if self.is_timer:
            rem = max(0.0, self.timeout - elapsed)
            msg = f"[bold magenta]Heartbeat таймер[/bold magenta]: осталось [bold white]{_format_time_compact(rem)}[/bold white] / {timeout_str}"
        else:
            cond_disp = self.condition
            if len(cond_disp) > 45:
                cond_disp = cond_disp[:42] + "…"
            msg = f"[bold magenta]Heartbeat[/bold magenta]: [cyan]{escape(cond_disp)}[/cyan] [dim]({waited_str} / {timeout_str})[/dim]"

        hint = "[dim](Ctrl+C: продолжить диалог)[/dim]"
        self._spinner.text = Text.from_markup(f"  [bold magenta]⠋[/bold magenta] {msg}{extra}  {hint}")
        yield self._spinner
        q_disp = _render_queue_input_display()
        if q_disp is not None:
            yield q_disp


# ── Helpers ──────────────────────────────────────────────────────────

def _handle_tool_generating(first_chunk, chunk_iterator, start_turn_time: float = None):
    """Process tool_generating events with a Live panel.
    Returns (next_chunk, next_type, is_done)."""
    display = _ToolGenDisplay(start_turn_time=start_turn_time)
    display.tool_name = first_chunk.get("name", "?")
    display.tool_text = first_chunk.get("delta", "")
    display.total_bytes = first_chunk.get("bytes", 0)

    done = False
    chunk = first_chunk
    type_ = "tool_generating"

    with Live(display, console=console, refresh_per_second=8, transient=True):
        while True:
            try:
                chunk = next(chunk_iterator)
            except StopIteration:
                done = True
                break
            type_ = chunk.get("type")
            if type_ == "tool_generating":
                display.tool_name = chunk.get("name", display.tool_name)
                delta = chunk.get("delta", "")
                if delta:
                    display.tool_text += delta
                display.total_bytes = chunk.get("bytes", display.total_bytes)
            else:
                break

    return chunk, type_ if not done else None, done


def _check_auto_signals(res, is_auto, sleep_t, wake_ctx):
    """Check tool result for heartbeat wait control signals."""
    res_str = str(res)
    if "[HEARTBEAT_REQUEST:" in res_str:
        # Two forms: a delay plus a condition to wait on, or a plain timer delay.
        # Check condition wait first (anchored on |until= and |timeout= to allow pipes inside expressions)
        m_until = re.search(
            r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\s*\|\s*until=(.*?)\s*\|\s*timeout=(\d+)\s*\]",
            res_str,
            re.DOTALL,
        )
        if m_until:
            sleep_t = int(m_until.group(1))
            wake_ctx = {
                "reason": m_until.group(2),
                "until": m_until.group(3),
                "timeout": int(m_until.group(4) or 600),
            }
        else:
            m_plain = re.search(
                r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\s*\]",
                res_str,
                re.DOTALL,
            )
            if m_plain:
                sleep_t = int(m_plain.group(1))
                wake_ctx = f"[Heartbeat пробуждение] Причина: {m_plain.group(2)}"
    return is_auto, sleep_t, wake_ctx


def _handle_tool_end(chunk, is_auto, sleep_t, wake_ctx, duration: float = None, args: dict = None):
    """Process a tool_end event."""
    res = chunk.get("result", "")
    tool_name = chunk.get("name", "")
    print_tool_end(tool_name, res, duration=duration, args=args)
    if tool_name == "wait_heartbeat":
        return _check_auto_signals(res, is_auto, sleep_t, wake_ctx)
    return is_auto, sleep_t, wake_ctx


def turn_snapshot_lines(agent, elapsed: float, timing: dict = None,
                        usage: dict = None, tool_calls: int = 0,
                        mode: str = None) -> list:
    """Rich-markup lines of the post-response turn snapshot.

    Pure: nothing is printed and the session totals are only READ (callers add
    the turn's usage to `usage_tracker.usage` first, exactly once). Shared by the
    REPL and the TUI bridge so both frontends show the same text.

    Modes (configured via /snapshot):
      - 'compact' (default): single line with time, tokens, cache, tools, ctx, cost
      - 'verbose': multi-line breakdown of stages, context anatomy, and cache stats
      - 'off': minimal classic time/token display
    Also warns when the context window exceeds 85%.
    """
    if mode is None:
        from config import get_turn_snapshot_mode
        mode = get_turn_snapshot_mode()
    timing = timing or {}
    usage = usage or {}
    lines: list = []

    from usage_tracker import usage as session_usage, _short, _fmt_cost

    # ── OFF mode: minimal classic line ──
    if mode == "off":
        total_fmt = _format_time_human(elapsed)
        usage_str = ""
        if usage:
            try:
                usage_str = " · " + session_usage.format_last(usage)
            except Exception:
                usage_str = ""
        tools_t = timing.get("tools", 0.0)
        model_t = max(0.0, elapsed - tools_t)
        prefill_t = timing.get("first") or 0.0
        breakdown_parts = []
        if prefill_t > 0.4:
            breakdown_parts.append(f"ожидание: {_format_time_human(prefill_t)}")
        if model_t > 0.1:
            breakdown_parts.append(f"модель: {_format_time_human(model_t)}")
        if tools_t > 0.1:
            breakdown_parts.append(f"инструменты: {_format_time_human(tools_t)}")
        breakdown_str = " · ".join(breakdown_parts)
        if breakdown_str:
            breakdown_str = f" ({breakdown_str})"
        lines.append(f"[dim](⏱ {total_fmt}{breakdown_str}{usage_str})[/dim]")
        return lines

    # ── VERBOSE mode: multi-line detailed breakdown ──
    if mode == "verbose":
        tools_t = timing.get("tools", 0.0)
        model_t = max(0.0, elapsed - tools_t)
        prefill_t = timing.get("first") or 0.0

        # Line 1: Timing breakdown
        t_parts = [f"ожидание: {_format_time_human(prefill_t)}"]
        if model_t > 0.05:
            t_parts.append(f"модель: {_format_time_human(model_t)}")
        if tools_t > 0.05:
            t_parts.append(f"инструменты: {_format_time_human(tools_t)}")
        lines.append(f"  [dim]┄ {_format_time_human(elapsed)} ({' · '.join(t_parts)})[/dim]")

        # Line 2: Tokens & Cache & Cost
        prompt_tok = usage.get("prompt", 0) or 0
        compl_tok = usage.get("completion", 0) or 0
        cached_tok = usage.get("cached", 0) or 0
        cost = usage.get("cost", 0.0) or 0.0

        tok_parts = []
        if prompt_tok or compl_tok:
            tok_parts.append(f"{_short(prompt_tok)}→{_short(compl_tok)} tok")
            if cached_tok > 0:
                pct = int(cached_tok / prompt_tok * 100) if prompt_tok else 0
                tok_parts.append(f"cached: {_short(cached_tok)} ({pct}%)")

        session_total = session_usage.prompt + session_usage.completion
        if session_total > 0:
            tok_parts.append(f"sum {_short(session_total)} tok")
        if cost > 0:
            tok_parts.append(_fmt_cost(cost))
        elif session_usage.cost > 0:
            tok_parts.append(f"session {_fmt_cost(session_usage.cost)}")

        if tok_parts:
            lines.append(f"  [dim]┄ {' · '.join(tok_parts)}[/dim]")

        # Line 3: Context breakdown
        try:
            bd = agent.get_context_breakdown() if agent and hasattr(agent, "get_context_breakdown") else {}
            ctx = agent.get_context_usage() if agent and hasattr(agent, "get_context_usage") else {}
            if ctx:
                ctx_str = f"ctx {_short(ctx['tokens'])}/{_short(ctx['max'])} ({ctx['percent']:.0f}%)"
                bd_parts = [ctx_str]
                if bd.get("system"):
                    bd_parts.append(f"sys {_short(bd['system'])}")
                if bd.get("tools"):
                    bd_parts.append(f"tools {_short(bd['tools'])}")
                if bd.get("history"):
                    bd_parts.append(f"history {_short(bd['history'])}")
                lines.append(f"  [dim]┄ {' · '.join(bd_parts)}[/dim]")
        except Exception:
            pass

        # Line 4: Tools count
        tool_parts = []
        if tool_calls > 0:
            tool_parts.append(f"{tool_calls} tools")
        try:
            local_calls = int(getattr(agent, "_turn_local_agent_calls", 0) or 0)
        except (TypeError, ValueError):
            local_calls = 0
        if local_calls > 0:
            tool_parts.append(f"{local_calls} local agents")
        if tool_parts:
            lines.append(f"  [dim]┄ {' · '.join(tool_parts)}[/dim]")

    else:
        # ── COMPACT mode (default): elegant single line ──
        parts = []

        # 1. Total time
        parts.append(_format_time_human(elapsed))

        # 2. Tokens & Cache
        prompt_tok = usage.get("prompt", 0) or 0
        compl_tok = usage.get("completion", 0) or 0
        if prompt_tok or compl_tok:
            tok_str = f"{_short(prompt_tok)}→{_short(compl_tok)} tok"
            cached = usage.get("cached", 0) or 0
            if cached > 0:
                pct = int(cached / prompt_tok * 100) if prompt_tok else 0
                tok_str += f" · cached {_short(cached)} ({pct}%)"
            parts.append(tok_str)

        # 3. Tool calls
        if tool_calls > 0:
            parts.append(f"{tool_calls} tools")

        # 4. Context usage
        try:
            if agent and hasattr(agent, "get_context_usage"):
                ctx = agent.get_context_usage()
                if ctx and ctx.get("max", 0) > 0:
                    parts.append(f"ctx {_short(ctx['tokens'])}/{_short(ctx['max'])} ({ctx['percent']:.0f}%)")
        except Exception:
            pass

        # 5. Cost
        cost = usage.get("cost", 0.0) or 0.0
        if cost > 0:
            parts.append(_fmt_cost(cost))

        # 6. Session total (if multiple requests)
        session_total = session_usage.prompt + session_usage.completion
        if session_usage.requests > 1 and session_total > 0:
            sess_str = f"sum {_short(session_total)}"
            if session_usage.cost > 0:
                sess_str += f" · {_fmt_cost(session_usage.cost)}"
            parts.append(sess_str)

        lines.append(f"  [dim]┄ {' · '.join(parts)}[/dim]")

    # ── Context overflow warning (>85%) ──
    try:
        if agent and hasattr(agent, "get_context_usage"):
            ctx = agent.get_context_usage()
            if ctx and ctx.get("percent", 0) > 85:
                lines.append(
                    f"  [bold yellow]▲ Контекст заполнен на {ctx['percent']:.0f}% — рекомендуется /clear или /compact[/bold yellow]"
                )
    except Exception:
        pass

    return lines


def _print_turn_snapshot(agent, elapsed: float, timing: dict = None,
                         usage: dict = None, tool_calls: int = 0):
    """Prints a turn snapshot after the model response (see turn_snapshot_lines)."""
    from usage_tracker import usage as session_usage
    if usage:
        try:
            session_usage.add(usage)
        except Exception:
            pass
    for line in turn_snapshot_lines(agent, elapsed, timing=timing, usage=usage,
                                    tool_calls=tool_calls):
        safe_print(line)
    safe_print("")


def content_tail_renderable(pending: str, now: float, phase_start: float,
                            turn_start: float, height: int, queue_display=None):
    """The live region under a streaming answer: the incomplete trailing block
    as plain text, then the `Генерация...` spinner (cli_ui keeps it on screen only
    until the block completes and is committed). Pure, so a frontend that draws
    the same region can be checked against it."""
    max_lines = max(3, height - 8)
    body = "\n".join(pending.splitlines()[-max_lines:]).strip("\n")

    phase_elapsed = _format_time_compact(now - phase_start)
    total_elapsed = _format_time_compact(now - turn_start)
    if (now - phase_start) < (now - turn_start) - 1:
        time_label = f"этап {phase_elapsed} · всего {total_elapsed}"
    else:
        time_label = f"{total_elapsed}"

    spinner = RichSpinner(
        "dots", text=Text(f" Генерация... ({time_label})", style="dim cyan")
    )
    items = []
    if body:
        items.append(Text(body, style=c.get("assistant_text", "default")))
    items.append(spinner)
    if queue_display is not None:
        items.append(queue_display)
    return Group(*items)


# ── Main stream renderer ────────────────────────────────────────────

def render_response_stream(
    agent, response_chunks, is_auto_mode=False
) -> tuple[str, bool, int, str]:
    """
    Renders the streaming response from the agent in real time.
    Handles thinking, content panels with Rich Live, tool execution, and errors.
    Returns (streamed_text, final_is_auto_mode, auto_sleep_time, auto_wake_context).
    """
    # Capture usage chunks in a side-channel so the streaming state machine
    # below doesn't have to handle them at every next() call site.
    captured_usage = {}

    def _capture_usage(source):
        for ch in source:
            if ch.get("type") == "usage":
                captured_usage.update(ch.get("data", {}))
                continue
            yield ch

    # Per-turn timing, observed on the chunk stream (outside the state machine).
    start_total_time = time.time()
    _timing = {"first": None, "tools": 0.0, "_ts": None}
    chunk_iterator = _time_observe(_capture_usage(iter(response_chunks)), _timing, start_total_time)
    streamed_text = ""
    full_streamed_text = ""
    is_tool_executing = False
    current_tool_name = ""
    current_tool_args = {}
    current_tool_start_ts = None

    executed_tool_calls = 0
    auto_sleep_time = 0
    auto_wake_context = ""
    final_is_auto_mode = is_auto_mode
    pending_chunk = None
    assistant_header_printed = False

    while True:
        try:
            done = False
            if not is_tool_executing:
                # ====================================================
                # PHASE 1: TTFT + Thinking + Tool generation
                # ====================================================

                # ── Step 1: TTFT wait ──
                if pending_chunk is not None:
                    chunk = pending_chunk
                    pending_chunk = None
                else:
                    try:
                        waiting_display = _WaitingDisplay(start_total_time)
                        with Live(waiting_display, console=console, refresh_per_second=10, transient=True):
                            chunk = next(chunk_iterator)
                    except StopIteration:
                        done = True
                        break

                type_ = chunk.get("type")

                # ── Step 2: Thinking / Reasoning stream ──
                # Uses a Live display with a dynamic renderable
                # (_ThinkingDisplay). The renderable recalculates
                # elapsed time on every Live refresh frame, so the
                # timer keeps ticking even when next() is blocked.
                if type_ == "thinking_stream":
                    thinking_display = _ThinkingDisplay(
                        label_style=c.get("reasoning_label", "bold #607d8b"),
                        text_style=c.get("reasoning_text", "dim white"),
                        start_turn_time=start_total_time,
                    )
                    current_thinking_block = chunk["content"]
                    thinking_display.thinking_text = current_thinking_block
                    thinking_start_time = time.time()

                    with Live(
                        thinking_display,
                        console=console,
                        refresh_per_second=6,
                        transient=True,
                    ):
                        while True:
                            try:
                                chunk = next(chunk_iterator)
                            except StopIteration:
                                done = True
                                break
                            type_ = chunk.get("type")
                            if type_ == "thinking_stream":
                                current_thinking_block += chunk["content"]
                                thinking_display.thinking_text += chunk["content"]
                            else:
                                break

                    # Re-render thinking as static text or compact summary
                    from ui import print_reasoning
                    if current_thinking_block:
                        thinking_duration = time.time() - thinking_start_time
                        print_reasoning(current_thinking_block, duration=thinking_duration)

                    if done:
                        break

                if type_ == "rethink_stream":
                    rethink_display = _ThinkingDisplay(
                        label_style="bold #d18f52",
                        text_style="italic #ffe0b2",
                        start_turn_time=start_total_time,
                        title="Перепроверка (Rethink):",
                        spinner_label="Стресс-тест решения...",
                    )
                    current_rethink_block = chunk["content"]
                    rethink_display.thinking_text = current_rethink_block
                    rethink_start_time = time.time()

                    with Live(
                        rethink_display,
                        console=console,
                        refresh_per_second=6,
                        transient=True,
                    ):
                        while True:
                            try:
                                chunk = next(chunk_iterator)
                            except StopIteration:
                                done = True
                                break
                            type_ = chunk.get("type")
                            if type_ == "rethink_stream":
                                current_rethink_block += chunk["content"]
                                rethink_display.thinking_text += chunk["content"]
                            else:
                                break

                    from ui import print_rethink
                    if current_rethink_block:
                        rethink_duration = time.time() - rethink_start_time
                        print_rethink(current_rethink_block, duration=rethink_duration)

                    if done:
                        break

                    if type_ == "thinking_stream":
                        pending_chunk = chunk
                        continue

                # ── Step 3: Tool argument generation ──
                if type_ == "tool_generating":
                    chunk, type_, done = _handle_tool_generating(
                        chunk, chunk_iterator, start_turn_time=start_total_time
                    )
                    if done:
                        break

                # ── Step 4: Transition ──
                if type_ == "tool_start":
                    assistant_header_printed = False
                    current_tool_name = chunk.get("name", "")
                    current_tool_args = chunk.get("args", {})
                    current_tool_start_ts = time.time()
                    print_tool_start(current_tool_name, current_tool_args, silent=True)
                    is_tool_executing = True
                elif type_ == "tool_end":
                    executed_tool_calls += 1
                    duration = time.time() - current_tool_start_ts if current_tool_start_ts else None
                    final_is_auto_mode, auto_sleep_time, auto_wake_context = (
                        _handle_tool_end(
                            chunk, final_is_auto_mode,
                            auto_sleep_time, auto_wake_context,
                            duration=duration, args=current_tool_args,
                        )
                    )
                    is_tool_executing = False
                    current_tool_name = ""
                    current_tool_args = {}
                    current_tool_start_ts = None
                elif type_ == "diff":
                    from ui import print_inline_diff
                    print_inline_diff(chunk.get("file", ""), chunk.get("diff", ""))
                elif type_ == "error":
                    _render_stream_error(chunk["content"])
                elif type_ == "checkpoint":
                    print_checkpoint(chunk.get('sha', ''))

                # ====================================================
                # PHASE 2: Main content stream
                # ====================================================
                if not done and not is_tool_executing and type_ in (
                    "content_stream", "content", "content_replace",
                ):
                    if not assistant_header_printed:
                        assistant_header_printed = True
                        print_assistant_header()
                    from src.cli.stream_markdown import MarkdownStreamSplitter
                    splitter = MarkdownStreamSplitter()
                    content_phase_start = time.time()

                    def _commit(live, md):
                        print_committed_markdown(md, out=live.console)

                    def _tail():
                        return content_tail_renderable(
                            splitter.pending(), time.time(), content_phase_start,
                            start_total_time, console.size.height or 24,
                            _render_queue_input_display(),
                        )

                    with Live(
                        _tail(),
                        console=console,
                        refresh_per_second=10,
                        transient=True,
                    ) as live:
                        while True:
                            if type_ in ("content_stream", "content"):
                                committed = splitter.feed(chunk["content"])
                                full_streamed_text += chunk["content"]
                                if committed:
                                    _commit(live, committed)
                            elif type_ == "content_replace":
                                splitter.replace(chunk["content"])
                                full_streamed_text = chunk["content"]
                            elif type_ == "tool_generating":
                                break
                            elif type_ not in ("thinking_stream", "rethink_stream"):
                                break


                            live.update(_tail())

                            try:
                                chunk = next(chunk_iterator)
                                type_ = chunk.get("type")
                            except StopIteration:
                                done = True
                                break

                        # Commit the final, complete block before the transient
                        # tail is cleared on exit.
                        _commit(live, splitter.finalize())

                    # Handle what caused Phase 2 to exit
                    if not done and not is_tool_executing:
                        if type_ == "tool_start":
                            assistant_header_printed = False
                            current_tool_name = chunk.get("name", "")
                            current_tool_args = chunk.get("args", {})
                            current_tool_start_ts = time.time()
                            print_tool_start(
                                current_tool_name, current_tool_args, silent=True
                            )
                            is_tool_executing = True
                        elif type_ == "tool_generating":
                            pass  # next outer-loop iteration
                        elif type_ == "tool_end":
                            duration = time.time() - current_tool_start_ts if current_tool_start_ts else None
                            (
                                final_is_auto_mode,
                                auto_sleep_time,
                                auto_wake_context,
                            ) = _handle_tool_end(
                                chunk, final_is_auto_mode,
                                auto_sleep_time, auto_wake_context,
                                duration=duration, args=current_tool_args,
                            )
                            is_tool_executing = False
                            current_tool_name = ""
                            current_tool_args = {}
                            current_tool_start_ts = None
                        elif type_ == "diff":
                            from ui import print_inline_diff
                            print_inline_diff(chunk.get("file", ""), chunk.get("diff", ""))
                        elif type_ == "error":
                            _render_stream_error(chunk["content"])
                        elif type_ == "checkpoint":
                            print_checkpoint(chunk.get('sha', ''))

            else:
                # ====================================================
                # Tool execution — wait for tool_end
                # ====================================================
                from tools.schemas import INTERACTIVE_TOOLS
                use_spinner = current_tool_name not in INTERACTIVE_TOOLS

                if use_spinner:
                    tool_wait_display = _ToolWaitDisplay(
                        current_tool_name or "инструмента",
                        start_total_time,
                        tool_args=current_tool_args,
                    )
                    with Live(tool_wait_display, console=console, refresh_per_second=8, transient=True):
                        try:
                            chunk = next(chunk_iterator)
                        except StopIteration:
                            done = True
                else:
                    esc_listener = getattr(agent, "_esc_listener", None)
                    if esc_listener:
                        esc_listener.pause()
                    try:
                        chunk = next(chunk_iterator)
                    except StopIteration:
                        done = True
                    finally:
                        if esc_listener:
                            esc_listener.resume()

                if not done:
                    type_ = chunk.get("type")
                    if type_ == "tool_end":
                        duration = time.time() - current_tool_start_ts if current_tool_start_ts else None
                        (
                            final_is_auto_mode,
                            auto_sleep_time,
                            auto_wake_context,
                        ) = _handle_tool_end(
                            chunk, final_is_auto_mode,
                            auto_sleep_time, auto_wake_context,
                            duration=duration, args=current_tool_args,
                        )
                        is_tool_executing = False
                        current_tool_name = ""
                        current_tool_args = {}
                        current_tool_start_ts = None
                    elif type_ == "diff":
                        from ui import print_inline_diff
                        print_inline_diff(chunk.get("file", ""), chunk.get("diff", ""))
                    elif type_ == "tool_start":
                        assistant_header_printed = False
                        current_tool_name = chunk.get("name", "")
                        current_tool_args = chunk.get("args", {})
                        current_tool_start_ts = time.time()
                        print_tool_start(
                            current_tool_name, current_tool_args, silent=True
                        )
                    elif type_ == "error":
                        _render_stream_error(chunk["content"])

            if done:
                break

        except StopIteration:
            break
        except Exception as e:
            try:
                from logger import get_logger
                get_logger("ui").exception("Stream exception in render_response_stream: %s", e)
            except Exception:
                pass
            print_error(f"Ошибка потока: {e}")
            break

    # No final re-render: content was committed block-by-block during the
    # stream (append-only), so re-printing it here would duplicate it.

    elapsed_time = time.time() - start_total_time
    _print_turn_snapshot(
        agent=agent,
        elapsed=elapsed_time,
        timing=_timing,
        usage=captured_usage,
        tool_calls=executed_tool_calls or getattr(agent, "_turn_tool_calls", 0),
    )

    return full_streamed_text, final_is_auto_mode, auto_sleep_time, auto_wake_context
