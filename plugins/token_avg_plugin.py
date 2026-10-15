"""Reports the average token generation rate for the current Argent session.

Reads the process-wide usage singleton (usage_tracker.usage) that every LLM
request accumulates, then averages the model's generated tokens (completion)
across all requests in the session.

Usage:
  /token_avg          Show average token generation (and prompt) for the session.
  /token_avg --reset  Reset the session token counters (new session from here).
"""

from ui import console
from rich.table import Table
from rich.panel import Panel


def _short(n) -> str:
    """Human-compact number: 1234 -> '1.2k', 1_234_567 -> '1.2M'."""
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "0"
    if n < 1000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


def _fmt_cost(cost: float) -> str:
    return f"${cost:.4f}" if cost < 0.1 else f"${cost:.2f}"


def _avg(value: int, requests: int) -> float:
    return value / requests if requests else 0.0


def command_token_avg(*args):
    """Average token generation for the current session."""
    # Optional flags: /token_avg --reset
    if any(a in ("--reset", "-r", "reset") for a in args):
        try:
            from usage_tracker import usage as session_usage
            session_usage.reset()
            console.print("[green]Session token counters reset.[/green]")
            return
        except Exception as e:
            console.print(f"[red]Failed to reset counters: {e}[/red]")
            return

    try:
        from usage_tracker import usage as session_usage
    except Exception as e:
        console.print(f"[red]Could not read session usage: {e}[/red]")
        return

    requests = session_usage.requests
    prompt = int(session_usage.prompt or 0)
    completion = int(session_usage.completion or 0)

    # A session with no LLM requests yet: nothing to average.
    if requests <= 0:
        console.print(
            Panel(
                "[yellow]No LLM requests recorded yet.[/yellow]\n"
                "Token counters start accumulating from the first model response.",
                title="[bold cyan]Token Average[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )
        return

    avg_completion = _avg(completion, requests)
    avg_prompt = _avg(prompt, requests)
    avg_total = _avg(prompt + completion, requests)
    total = prompt + completion

    table = Table(
        title="[bold cyan]Average Token Generation[/bold cyan]",
        expand=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="dim")
    table.add_column("Per request", justify="right")
    table.add_column("Total", justify="right")

    table.add_row("Generation (completion)", f"{_short(avg_completion)} tok", f"{_short(completion)} tok")
    table.add_row("Prompt (input)", f"{_short(avg_prompt)} tok", f"{_short(prompt)} tok")
    table.add_row("Total", f"{_short(avg_total)} tok", f"{_short(total)} tok")

    if session_usage.cost:
        table.add_row(
            "Cost / request",
            f"{_fmt_cost(session_usage.cost / requests)}",
            f"{_fmt_cost(session_usage.cost)}",
        )

    console.print(
        Panel(
            table,
            subtitle=(
                f"[dim]{requests} request(s) · "
                f"avg generation {_short(avg_completion)} tok/request[/dim]"
            ),
            title="[bold cyan]Token Average[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )
