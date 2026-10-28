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
        self.cached = 0
        self.cost = 0.0
        self.requests = 0
        # Provider balance (when the API exposes it), so the toolbar can show
        # what is left, not only what this session's tokens cost.
        self.balance: float | None = None
        self.balance_currency = "USD"
        self.balance_provider = ""
        # The LAST request's own numbers. The session totals answer "what has
        # this run cost so far"; after a restart they start at zero, and the
        # question people actually ask is "what did the last task take off my
        # balance". Kept separately so a restored session can answer it.
        self.last_prompt = 0
        self.last_completion = 0
        self.last_cached = 0
        self.last_cost = 0.0
        self.last_spent: float | None = None
        self.last_spent_currency = "USD"
        self.last_spent_approx = False
        # Account-wide spend accumulated from balance drops during this session.
        # It is shown as ``spent~`` because another client on the same account
        # can move the balance too.
        self.session_spent = 0.0
        self.session_spent_currency = "USD"
        self.session_spent_approx = False

    def add(self, usage: dict):
        """Accumulate one request's usage dict {prompt, completion, total, cost?, cached?}."""
        prompt = int(usage.get("prompt", 0) or 0)
        completion = int(usage.get("completion", 0) or 0)
        cached = int(usage.get("cached", 0) or 0)
        cost = float(usage.get("cost", 0.0) or 0.0)
        self.prompt += prompt
        self.completion += completion
        self.cached += cached
        self.cost += cost
        self.requests += 1
        self.last_prompt = prompt
        self.last_completion = completion
        self.last_cached = cached
        self.last_cost = cost
        # A provider that reports the cost per request makes the balance drop
        # redundant: the exact number wins over the approximate one.
        if cost > 0:
            self.last_spent = cost
            self.last_spent_approx = False
            self.last_spent_currency = self.balance_currency

    def set_balance(self, amount: float | None, currency: str = "USD", provider: str = "",
                    record: bool = True):
        """Record the provider's current prepaid balance.

        This is the **account-wide** balance reported by the provider. DeepSeek
        does not expose per-request cost, so the difference between two readings
        cannot be attributed to Argent alone: another client using the same
        account (DSH, a browser, a script) moves it too. The UI therefore shows
        the exact remaining balance and Argent's own token/cost counters, and
        labels the account delta as approximate (``last~``) rather than as a
        measured spend.

        ``record=False`` only updates the reading. A restored session uses it to
        seed the baseline from the balance saved with the session: recording
        that as a spend would charge the previous run's money to this one.
        """
        if amount is None:
            return
        try:
            value = float(amount)
        except (TypeError, ValueError):
            return
        provider = (provider or "").strip()
        if provider:
            self.balance_provider = provider
        if record and self.balance is not None and value < self.balance:
            drop = self.balance - value
            # Below a tenth of a cent the difference is rounding, not spending.
            if drop > 0.0005:
                self.last_spent = drop
                self.last_spent_approx = True
                self.last_spent_currency = (currency or "USD").upper()
                self.session_spent += drop
                self.session_spent_currency = (currency or "USD").upper()
                self.session_spent_approx = True
        self.balance = value
        self.balance_currency = (currency or "USD").upper()

    def clear_balance(self, provider: str = ""):
        """Forget a balance that belonged to the previous provider."""
        self.balance = None
        self.balance_currency = "USD"
        self.balance_provider = (provider or "").strip()

    def reset(self):
        self.__init__()

    def snapshot(self) -> dict:
        """What a saved session keeps about spending, so a restart can show it."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "cached": self.cached,
            "cost": self.cost,
            "requests": self.requests,
            "last_prompt": self.last_prompt,
            "last_completion": self.last_completion,
            "last_cached": self.last_cached,
            "last_cost": self.last_cost,
            "last_spent": self.last_spent,
            "last_spent_currency": self.last_spent_currency,
            "last_spent_approx": self.last_spent_approx,
            "session_spent": self.session_spent,
            "session_spent_currency": self.session_spent_currency,
            "session_spent_approx": self.session_spent_approx,
            "balance": self.balance,
            "balance_currency": self.balance_currency,
            "balance_provider": self.balance_provider,
        }

    def restore(self, data: dict | None) -> bool:
        """Put back what ``snapshot`` saved. Returns False for nothing to restore.

        The balance comes back as a READING, not as a spend: the next real
        reading is compared against it, so only this run's drop is recorded.
        """
        if not isinstance(data, dict) or not data:
            return False

        def _num(key, default=0.0):
            try:
                value = data.get(key, default)
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        self.prompt = int(_num("prompt"))
        self.completion = int(_num("completion"))
        self.cached = int(_num("cached"))
        self.cost = _num("cost")
        self.requests = int(_num("requests"))
        self.last_prompt = int(_num("last_prompt"))
        self.last_completion = int(_num("last_completion"))
        self.last_cached = int(_num("last_cached"))
        self.last_cost = _num("last_cost")
        spent = data.get("last_spent")
        self.last_spent = None if spent is None else _num("last_spent")
        self.last_spent_currency = str(data.get("last_spent_currency") or "USD").upper()
        self.last_spent_approx = bool(data.get("last_spent_approx"))
        self.session_spent = _num("session_spent")
        self.session_spent_currency = str(data.get("session_spent_currency") or "USD").upper()
        self.session_spent_approx = bool(data.get("session_spent_approx"))
        balance = data.get("balance")
        self.balance = None if balance is None else _num("balance")
        self.balance_currency = str(data.get("balance_currency") or "USD").upper()
        self.balance_provider = str(data.get("balance_provider") or "")
        return True

    def format_last_spend(self) -> str:
        """What the last request took: 'last $0.0021' or 'last~ $0.03'.

        Empty when nothing is known — a provider that reports neither a cost
        nor a balance (a local model) has nothing to say here, and inventing a
        zero would read as "this turn was free".
        """
        if self.last_spent is None or self.last_spent <= 0:
            return ""
        symbol = "$" if self.last_spent_currency == "USD" else f"{self.last_spent_currency} "
        mark = "~" if self.last_spent_approx else ""
        # Precision follows the AMOUNT, not the provenance: a few cents printed as
        # "$0.03" hides what the turn actually cost, while "$0.2500" claims digits a
        # cent-quoted balance never had. Below a dime keep four places either way.
        places = 4 if self.last_spent < 0.1 else 2
        return f"last{mark} {symbol}{self.last_spent:.{places}f}"

    @staticmethod
    def format_last(usage: dict) -> str:
        """Compact per-response summary: '24->20 tok · $0.0021'.

        ASCII only: rendered through Rich, which fails on box-drawing chars and
        arrows on legacy Windows (cp1251) consoles.
        """
        prompt = int(usage.get("prompt", 0) or 0)
        completion = int(usage.get("completion", 0) or 0)
        cached = int(usage.get("cached", 0) or 0)
        tok_str = f"{_short(prompt)}->{_short(completion)} tok"
        if cached > 0:
            pct = int(cached / prompt * 100) if prompt > 0 else 0
            tok_str += f" (cached {_short(cached)} {pct}%)"
        parts = [tok_str]
        cost = float(usage.get("cost", 0.0) or 0.0)
        if cost > 0:
            parts.append(_fmt_cost(cost))
        return " · ".join(parts)

    def format_session(self) -> str:
        """Running session total plus the provider's account-wide balance.

        The cost part is only what the provider reported for Argent's own
        requests. ``bal`` is account-wide and is intentionally not converted
        into a ``spent`` number: with DSH and Argent on the same DeepSeek
        account, balance movement is not attributable to one client.
        """
        total = self.prompt + self.completion
        parts = []
        if self.requests or self.balance is None:
            parts.append(f"sum {_short(total)} tok")
        if self.cost > 0:
            parts.append(_fmt_cost(self.cost))
        elif self.session_spent > 0.0005:
            symbol = "$" if self.session_spent_currency == "USD" else f"{self.session_spent_currency} "
            parts.append(f"spent~ {symbol}{self.session_spent:.2f}")
        if self.balance is not None:
            symbol = "$" if self.balance_currency == "USD" else f"{self.balance_currency} "
            parts.append(f"bal {symbol}{self.balance:.2f}")
        return " · ".join(parts) if parts else f"sum {_short(total)} tok"


def _short(n: int) -> str:
    """Human-compact token count: 1234 -> '1.2k'."""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


# Process-wide singleton, mirrors the agent's single session.
usage = SessionUsage()
