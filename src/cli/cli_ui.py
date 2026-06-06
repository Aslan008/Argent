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

        display = self.thinking_text
        if len(display) > 3000:
            display = "...\n" + display[-3000:]
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
    if is_auto:
        if "[END_AUTO_MODE]" in str(res):
            is_auto = False
            print_system("🏁 Автоматический режим завершен агентом.")
        elif "[HEARTBEAT_REQUEST:" in str(res):
            match = re.search(
                r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\]", str(res)
            )
            if match:
                sleep_t = int(match.group(1))
                wake_ctx = f"[Heartbeat пробуждение] Причина: {match.group(2)}"
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
    chunk_iterator = iter(response_chunks)
    start_total_time = time.time()
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
                    print_error(chunk["content"])

                # ====================================================
                # PHASE 2: Main content stream
                # ====================================================
                if not done and not is_tool_executing and type_ in (
                    "content_stream", "content", "content_replace",
                ):
                    streamed_text = ""
                    with Live(
                        create_content_panel(""),
                        console=console,
                        refresh_per_second=10,
                        transient=True,
                    ) as live:
                        while True:
                            if type_ in ("content_stream", "content"):
                                streamed_text += chunk["content"]
                                full_streamed_text += chunk["content"]
                                live.update(create_content_panel(streamed_text))
                            elif type_ == "content_replace":
                                streamed_text = chunk["content"]
                                full_streamed_text = chunk["content"]
                                live.update(create_content_panel(streamed_text))
                            elif type_ == "tool_generating":
                                break
                            elif type_ not in ("thinking_stream",):
                                break

                            if streamed_text:
                                live.update(
                                    Group(
                                        create_content_panel(streamed_text),
                                        RichSpinner(
                                            "dots",
                                            text=Text(
                                                " Генерация...",
                                                style="dim cyan",
                                            ),
                                        ),
                                    )
                                )

                            try:
                                chunk = next(chunk_iterator)
                                type_ = chunk.get("type")
                                if type_ in (
                                    "content_stream", "content",
                                    "content_replace",
                                ):
                                    pass
                                elif streamed_text:
                                    live.update(
                                        create_content_panel(streamed_text)
                                    )
                            except StopIteration:
                                done = True
                                break

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
                            print_error(chunk["content"])

            else:
                # ====================================================
                # Tool execution — wait for tool_end
                # ====================================================
                interactive_tools = {
                    "ask_user_questions", "run_command",
                    "run_admin_command", "start_background_command",
                    "delete_file", "plan_work_changes",
                }
                use_spinner = current_tool_name not in interactive_tools

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
                        print_error(chunk["content"])

            if done:
                break

        except StopIteration:
            break
        except Exception as e:
            print_error(f"Streaming error: {e}")
            break

    # Final static render
    if full_streamed_text:
        for el in create_final_panel(full_streamed_text):
            safe_print(el)
            safe_print("")

    elapsed_time = time.time() - start_total_time
    safe_print(f"[dim](Время ответа: {elapsed_time:.1f}s)[/dim]")
    safe_print("")

    return full_streamed_text, final_is_auto_mode, auto_sleep_time, auto_wake_context
