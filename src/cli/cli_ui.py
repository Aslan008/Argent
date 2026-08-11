import re
import time
from rich.live import Live
from rich.console import Group
from rich.text import Text
from rich.panel import Panel
from rich.spinner import Spinner as RichSpinner
from ui import (
    console, print_system, print_error, print_tool_start, print_tool_end,
    s, c, create_content_panel,
    create_final_panel, safe_print, print_context_usage
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


# ── Dynamic renderables ─────────────────────────────────────────────
# These classes implement __rich_console__, which Rich calls on EVERY
# Live refresh frame.  This means the elapsed timer updates
# automatically even while next(chunk_iterator) is blocked.

class _ThinkingDisplay:
    """Shows thinking text + animated spinner with elapsed timer."""

    def __init__(self, label_style, text_style):
        self.thinking_text = ""
        self.start_time = time.time()
        self._label_style = label_style
        self._text_style = text_style
        self._spinner = RichSpinner("dots")

    def __rich_console__(self, console, options):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}:{secs:02d}"

        yield Text.from_markup(
            f"\n[{self._label_style}]Анализ:[/{self._label_style}]\n"
        )

        # Bound the live preview to the terminal height so this transient Live
        # never scrolls off-screen — otherwise Rich can't erase the scrolled
        # part on exit and it gets stranded in the scrollback (the full reasoning
        # is committed once afterwards via print_reasoning). Cap by chars first
        # (covers a single very long, wrapping line), then by visible lines.
        max_lines = max(3, (console.size.height or 24) - 6)
        width = console.size.width or 80
        display = self.thinking_text[-(max_lines * width):]
        display = "\n".join(display.splitlines()[-max_lines:]) or "..."
        yield Text(display, style=self._text_style)

        yield Text("")

        self._spinner.text = Text(
            f" Генерация... ({time_str})", style="dim cyan"
        )
        yield self._spinner


class _ToolGenDisplay:
    """Shows streaming tool argument text in a panel with elapsed timer."""

    def __init__(self):
        self.tool_name = "?"
        self.tool_text = ""
        self.total_bytes = 0
        self.start_time = time.time()

    def __rich_console__(self, console, options):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}:{secs:02d}"

        display_text = self.tool_text[-1000:]
        if len(self.tool_text) > 1000:
            display_text = "...\n" + display_text

        yield Panel(
            Text(display_text or "...", style="dim green"),
            title=(
                f"[dim cyan]Генерация: {self.tool_name} "
                f"({self.total_bytes} B | {time_str})[/dim cyan]"
            ),
            border_style="cyan",
        )


# ── Helpers ──────────────────────────────────────────────────────────

def _handle_tool_generating(first_chunk, chunk_iterator):
    """Process tool_generating events with a Live panel.
    Returns (next_chunk, next_type, is_done)."""
    display = _ToolGenDisplay()
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
    """Check tool result for auto-mode control signals."""
    if "[END_AUTO_MODE]" in str(res):
        if is_auto:
            is_auto = False
            print_system("🏁 Автоматический режим завершен агентом.")
    elif "[HEARTBEAT_REQUEST:" in str(res):
        # Two forms: a plain delay, or a delay plus a condition to wait on.
        # The optional tail keeps the original signal readable by itself.
        match = re.search(
            r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*([^|\]]*?)\s*"
            r"(?:\|\s*until=(.*?)\s*\|\s*timeout=(\d+)\s*)?\]",
            str(res)
        )
        if match:
            sleep_t = int(match.group(1))
            wake_ctx = f"[Heartbeat пробуждение] Причина: {match.group(2)}"
            if match.group(3):
                wake_ctx = {
                    "reason": match.group(2),
                    "until": match.group(3),
                    "timeout": int(match.group(4) or 600),
                }
    return is_auto, sleep_t, wake_ctx


def _handle_tool_end(chunk, is_auto, sleep_t, wake_ctx):
    """Process a tool_end event."""
    res = chunk.get("result", "")
    print_tool_end(chunk["name"], res)
    return _check_auto_signals(res, is_auto, sleep_t, wake_ctx)


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

    auto_sleep_time = 0
    auto_wake_context = ""
    final_is_auto_mode = is_auto_mode

    while True:
        try:
            done = False
            if not is_tool_executing:
                # ====================================================
                # PHASE 1: TTFT + Thinking + Tool generation
                # ====================================================

                # ── Step 1: TTFT wait ──
                # console.status is safe here because no inline text
                # has been printed yet.
                try:
                    with console.status(
                        "[dim cyan]Ожидание ответа...[/dim cyan]",
                        spinner="dots",
                    ):
                        chunk = next(chunk_iterator)
                except StopIteration:
                    done = True
                    break

                type_ = chunk.get("type")

                # ── Step 2: Thinking / Reasoning stream ──
                # Uses a Live display with a dynamic renderable
                # (_ThinkingDisplay).  The renderable recalculates
                # elapsed time on every Live refresh frame, so the
                # timer keeps ticking even when next() is blocked
                # (i.e. Ollama is silently generating tool-call JSON).
                if type_ == "thinking_stream":
                    thinking_display = _ThinkingDisplay(
                        label_style=c.get("reasoning_label", "bold #607d8b"),
                        text_style=c.get("reasoning_text", "dim white"),
                    )
                    current_thinking_block = chunk["content"]
                    thinking_display.thinking_text = current_thinking_block

                    with Live(
                        thinking_display,
                        console=console,
                        refresh_per_second=4,
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
                                # No live.update() needed — the display
                                # object is re-rendered on each refresh.
                            else:
                                break

                    # Re-render thinking as static text (the Live
                    # panel was transient and is now gone).
                    from ui import print_reasoning
                    if current_thinking_block:
                        print_reasoning(current_thinking_block)

                    if done:
                        break

                # ── Step 3: Tool argument generation ──
                if type_ == "tool_generating":
                    chunk, type_, done = _handle_tool_generating(
                        chunk, chunk_iterator
                    )
                    if done:
                        break

                # ── Step 4: Transition ──
                if type_ == "tool_start":
                    print_tool_start(chunk["name"], chunk.get("args", {}))
                    is_tool_executing = True
                    current_tool_name = chunk["name"]
                elif type_ == "tool_end":
                    final_is_auto_mode, auto_sleep_time, auto_wake_context = (
                        _handle_tool_end(
                            chunk, final_is_auto_mode,
                            auto_sleep_time, auto_wake_context,
                        )
                    )
                elif type_ == "error":
                    _render_stream_error(chunk["content"])
                elif type_ == "checkpoint":
                    console.print(f"[dim]🕰 Чекпоинт {chunk.get('sha', '')} создан — откат: /rewind[/dim]")

                # ====================================================
                # PHASE 2: Main content stream
                # ====================================================
                if not done and not is_tool_executing and type_ in (
                    "content_stream", "content", "content_replace",
                ):
                    # Append-only streaming (à la Claude Code): commit COMPLETE
                    # markdown blocks to the terminal as they finalize — they
                    # scroll naturally and are never redrawn — and keep only the
                    # incomplete trailing block in a small, bounded live tail.
                    # No whole-message redraw and no final re-render, so long
                    # answers can't strand a partial copy in the scrollback.
                    from src.cli.stream_markdown import MarkdownStreamSplitter
                    from ui import _render_code_blocks
                    splitter = MarkdownStreamSplitter()

                    def _commit(live, md):
                        md = md.strip("\n")
                        if not md:
                            return
                        for el in _render_code_blocks(md):
                            live.console.print(el)
                        live.console.print("")

                    def _tail():
                        height = console.size.height or 24
                        max_lines = max(3, height - 8)
                        body = "\n".join(
                            splitter.pending().splitlines()[-max_lines:]
                        ).strip("\n")
                        spinner = RichSpinner(
                            "dots", text=Text(" Генерация...", style="dim cyan")
                        )
                        if not body:
                            return spinner
                        return Group(
                            Text(body, style=c.get("assistant_text", "default")),
                            spinner,
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
                            elif type_ not in ("thinking_stream",):
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
                            print_tool_start(
                                chunk["name"], chunk.get("args", {})
                            )
                            is_tool_executing = True
                            current_tool_name = chunk["name"]
                        elif type_ == "tool_generating":
                            pass  # next outer-loop iteration
                        elif type_ == "tool_end":
                            (
                                final_is_auto_mode,
                                auto_sleep_time,
                                auto_wake_context,
                            ) = _handle_tool_end(
                                chunk, final_is_auto_mode,
                                auto_sleep_time, auto_wake_context,
                            )
                        elif type_ == "error":
                            _render_stream_error(chunk["content"])
                        elif type_ == "checkpoint":
                            console.print(f"[dim]🕰 Чекпоинт {chunk.get('sha', '')} создан — откат: /rewind[/dim]")

            else:
                # ====================================================
                # Tool execution — wait for tool_end
                # ====================================================
                from tools.schemas import INTERACTIVE_TOOLS
                use_spinner = current_tool_name not in INTERACTIVE_TOOLS

                if use_spinner:
                    with console.status(
                        "[dim cyan]Выполнение...[/dim cyan]",
                        spinner="dots",
                    ):
                        try:
                            chunk = next(chunk_iterator)
                        except StopIteration:
                            done = True
                else:
                    try:
                        chunk = next(chunk_iterator)
                    except StopIteration:
                        done = True

                if not done:
                    type_ = chunk.get("type")
                    if type_ == "tool_end":
                        (
                            final_is_auto_mode,
                            auto_sleep_time,
                            auto_wake_context,
                        ) = _handle_tool_end(
                            chunk, final_is_auto_mode,
                            auto_sleep_time, auto_wake_context,
                        )
                        is_tool_executing = False
                        current_tool_name = ""
                    elif type_ == "tool_start":
                        print_tool_start(
                            chunk["name"], chunk.get("args", {})
                        )
                        current_tool_name = chunk["name"]
                    elif type_ == "error":
                        _render_stream_error(chunk["content"])

            if done:
                break

        except StopIteration:
            break
        except Exception as e:
            print_error(f"Ошибка потока: {e}")
            break

    # No final re-render: content was committed block-by-block during the
    # stream (append-only), so re-printing it here would duplicate it.

    elapsed_time = time.time() - start_total_time
    usage_str = ""
    if captured_usage:
        try:
            from usage_tracker import usage as session_usage
            session_usage.add(captured_usage)
            usage_str = " · " + session_usage.format_last(captured_usage)
        except Exception:
            usage_str = ""

    tools_t = _timing["tools"]
    model_t = max(0.0, elapsed_time - tools_t)
    prefill_t = _timing["first"] or 0.0
    safe_print(
        f"[dim](⏱ всего {elapsed_time:.1f}s · модель {model_t:.1f}s "
        f"[prefill≈{prefill_t:.1f}s] · инструменты {tools_t:.1f}s{usage_str})[/dim]"
    )
    safe_print("")

    return full_streamed_text, final_is_auto_mode, auto_sleep_time, auto_wake_context
