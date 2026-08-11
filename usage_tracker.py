"""
Session token/cost accounting.

Providers report per-request usage (prompt/completion tokens, and cost when
the backend exposes it — e.g. OpenRouter for paid models). This accumulates
the running session totals and formats compact one-liners for the UI.
"""


def _fmt_cost(cost: float) -> str:
    # Sub-cent costs need more precision to be meaningful.
    return f"${cost:.4f}" if cost < 0.1 else f"${cost:.2f}"


class SessionUsage:
    def __init__(self):
        self.prompt = 0
        self.completion = 0
        self.cost = 0.0
        self.requests = 0

    def add(self, usage: dict):
        """Accumulate one request's usage dict {prompt, completion, total, cost?}."""
        self.prompt += int(usage.get("prompt", 0) or 0)
        self.completion += int(usage.get("completion", 0) or 0)
        self.cost += float(usage.get("cost", 0.0) or 0.0)
        self.requests += 1

    def reset(self):
        self.__init__()

    @staticmethod
    def format_last(usage: dict) -> str:
        """Compact per-response summary: '24->20 tok · $0.0021'.

        ASCII only: rendered through Rich, which fails on box-drawing chars and
        arrows on legacy Windows (cp1251) consoles.
        """
        prompt = int(usage.get("prompt", 0) or 0)
        completion = int(usage.get("completion", 0) or 0)
        parts = [f"{_short(prompt)}->{_short(completion)} tok"]
        cost = float(usage.get("cost", 0.0) or 0.0)
        if cost > 0:
            parts.append(_fmt_cost(cost))
        return " · ".join(parts)

    def format_session(self) +> str:
        """Running session total for the status bar (ASCII only)."""
        total = self.prompt + self.completion
        s = f"sum {_short(total)} tok"
        if self.cost > 0:
            s += f" · {_fmt_cost(self.cost)}"
        return s


def _short(n: int) -> str:
    """Human-compact token count: 1234 -> '1.2k'."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


# Process-wide singleton, mirrors the agent's single session.
usage = SessionUsage()
