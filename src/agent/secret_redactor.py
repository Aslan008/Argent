"""Secrets never reach the model, the log or the person's screen.

A shell command prints what it was asked to: `env`, `cat .env`, `git remote -v` with a token in the URL,
a stack trace that includes a request header. Whatever it printed was handed to the model (and, with a
cloud provider, to somebody else's server), written to the log and kept in the saved session. Nobody asked
for that: the model needed the *shape* of the output, not the key.

This module replaces the secret with a short marker that still says what was there
(`[секрет скрыт: openai-key]`), so the model knows a value existed and does not go looking for it.

Two entry points:

* `redact(text)` — one finished text.
* `StreamRedactor` — a stream of chunks. A secret can be cut by the chunk boundary (`sk-ab` | `cdef...`),
  and the second half alone matches no pattern, so the trailing run of non-blank characters (and a partial
  line that may open a PEM block) is held back until the next chunk. `release_stale` lets a prompt such as
  `Password:` through after a short silence, so the interactive-prompt detector still sees it.

What it recognises: vendor key formats, JWT, `Authorization: Bearer`, `NAME=value` and `"name": "value"` for
names that spell key/secret/password/token, `--password x` flags, passwords inside URLs, PEM private keys,
and the exact values of every key the person stored in Argent's config or exported to the environment.

Switch: config `redact_secrets` (on by default). Pure Python, no network.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Callable, Iterable, List, Optional, Set

#: A held-back tail longer than this is not a token any more; it is let through (patterns still apply to it).
MAX_HOLD = 1024
#: The longest PEM block that is swallowed without an END line before the filter gives up and lets text through.
MAX_PEM_CHARS = 64 * 1024

_MARK = "[секрет скрыт: {}]"


def _mark(kind: str) -> str:
    return _MARK.format(kind)


# ── patterns ────────────────────────────────────────────────────────────────

#: (kind, regex): the WHOLE match is the secret.
_TOKEN_RULES = [
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openrouter-key", re.compile(r"\bsk-or-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{24,}")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}")),
    ("aws-key-id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("telegram-token", re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
]

_BEARER = re.compile(r"(?i)(\bauthorization\s*[:=]\s*(?:bearer|basic|token)\s+)([A-Za-z0-9._~+/=\-]{12,})")
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^/\s:@]+:)([^@\s/]+)(?=@)", re.I)
_CLI_FLAG = re.compile(
    r"(?i)(--?(?:password|passwd|pass|token|api[-_]?key|secret|access[-_]?key)(?:\s+|=))(?![-<$%{\[])([^\s\"']{4,})")

# NAME=value / "name": "value" for names that spell a secret
_KV = re.compile(
    r"""(?ix)
    (?P<name>\b[A-Z0-9_.\-]*
        (?:API[_\-]?KEY|APIKEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE[_\-]?KEY|ACCESS[_\-]?KEY|CREDENTIAL)
        [A-Z0-9_.\-]*)
    (?P<nq>["']?)
    (?P<sep>\s*[=:]\s*)
    (?P<q>["']?)
    (?P<val>[^\s"'&;,]{6,})
    """
)
# a name that merely contains "token" (max_tokens, token_count) is a secret only if the value looks like one
_WEAK_NAME = re.compile(r"(?i)token|credential")
_PLACEHOLDER = re.compile(r"(?i)^(?:none|null|true|false|undefined|redacted|changeme|example|your[_\-].*|x+|\*+|\.+|<.*|\$.*|%.*|\{.*|\[.*)$")

_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
_PEM_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
_PEM_BLOCK = re.compile(r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----.*?(?:-----END \1-----|\Z)", re.S)

_TRAILING_RUN = re.compile(r"\S+\Z")
_SECRET_NAME = re.compile(r"(?i)(?:api[_\-]?key|token|secret|password|passwd)$")


def _looks_secret(value: str, weak: bool) -> bool:
    if _PLACEHOLDER.match(value):
        return False
    if len(set(value)) <= 2:
        return False
    if weak:
        return (len(value) >= 16 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value))
    return True


# ── known values ───────────────────────────────────────────────────────────

class _Known:
    """The exact secrets this installation owns (config + environment), refreshed once in a while."""

    TTL = 30.0

    def __init__(self, collector: Optional[Callable[[], Set[str]]] = None):
        self._collector = collector or self._collect
        self._lock = threading.Lock()
        self._at = 0.0
        self._rx: Optional[re.Pattern] = None
        self._extra: Set[str] = set()

    @staticmethod
    def _collect() -> Set[str]:
        values: Set[str] = set()
        try:
            import config
            for key, value in (config.load_config() or {}).items():
                if isinstance(value, str) and len(value) >= 8 and _SECRET_NAME.search(str(key)):
                    values.add(value)
        except Exception:
            pass
        for key, value in os.environ.items():
            if len(value) >= 8 and _SECRET_NAME.search(key):
                values.add(value)
        return values

    def add(self, *values: str) -> None:
        """Values known to be secret for the life of this process (a key typed with /key, a test value)."""
        with self._lock:
            self._extra.update(v for v in values if v and len(v) >= 6)
            self._rx = None
            self._at = 0.0

    def pattern(self) -> Optional[re.Pattern]:
        now = time.monotonic()
        with self._lock:
            if self._rx is not None and now - self._at < self.TTL:
                return self._rx
            try:
                values = set(self._collector()) | self._extra
            except Exception:
                values = set(self._extra)
            values = {v for v in values if v and len(v) >= 6}
            self._at = now
            if not values:
                self._rx = re.compile(r"(?!x)x")        # never matches
            else:
                self._rx = re.compile("|".join(re.escape(v) for v in sorted(values, key=len, reverse=True)))
            return self._rx


KNOWN = _Known()


_enabled_cache = (0.0, True)


def enabled() -> bool:
    """The `redact_secrets` setting, read at most every few seconds: the log filter asks on every line."""
    global _enabled_cache
    at, value = _enabled_cache
    now = time.monotonic()
    if now - at < 5.0:
        return value
    try:
        import config
        value = bool(config.is_secret_redaction_enabled())
    except Exception:
        value = True
    _enabled_cache = (now, value)
    return value


# ── the one-shot filter ────────────────────────────────────────────────────

def _redact_plain(text: str, known: Optional[_Known] = None) -> str:
    """Patterns only (no PEM state)."""
    if not text or len(text) < 6:
        return text
    for kind, rx in _TOKEN_RULES:
        if rx.search(text):
            text = rx.sub(_mark(kind), text)
    text = _BEARER.sub(lambda m: m.group(1) + _mark("auth-header"), text)
    text = _URL_PASSWORD.sub(lambda m: m.group(1) + _mark("url-password"), text)
    text = _CLI_FLAG.sub(lambda m: m.group(1) + _mark("cli-secret"), text)

    def _kv(m: "re.Match") -> str:
        value = m.group("val")
        if _MARK.split("{}")[0] in value:                 # already replaced above
            return m.group(0)
        if not _looks_secret(value, bool(_WEAK_NAME.search(m.group("name")) and not re.search(
                r"(?i)api[_\-]?key|secret|password|passwd|private", m.group("name")))):
            return m.group(0)
        return f"{m.group('name')}{m.group('nq')}{m.group('sep')}{m.group('q')}{_mark('value')}"

    text = _KV.sub(_kv, text)
    rx = (known or KNOWN).pattern()
    if rx is not None:
        text = rx.sub(_mark("ключ из конфигурации"), text)
    return text


def redact(text: str) -> str:
    """One finished text, secrets replaced. Returns `text` itself when nothing was found or the filter is off."""
    if not text or not isinstance(text, str) or not enabled():
        return text
    if "-----BEGIN" in text:
        text = _PEM_BLOCK.sub(lambda m: _mark("private-key"), text)
    return _redact_plain(text)


def redact_all(items: Iterable[str]) -> List[str]:
    return [redact(x) for x in items]


# ── the chunk filter ───────────────────────────────────────────────────────

class StreamRedactor:
    """Feed chunks as they arrive; what comes back is safe to show.

    `feed` may return less than it was given (the tail that could be the start of a secret waits for the
    next chunk) and `flush` returns what is left."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._tail = ""
        self._in_pem = False
        self._pem_chars = 0
        self._fed_at = 0.0
        self._clock = clock

    @property
    def pending(self) -> str:
        return self._tail

    def _process(self, body: str) -> str:
        out: List[str] = []
        pos = 0
        while pos < len(body):
            if self._in_pem:
                end = _PEM_END.search(body, pos)
                if end is None:
                    self._pem_chars += len(body) - pos
                    if self._pem_chars > MAX_PEM_CHARS:       # no END line is coming: stop swallowing
                        self._in_pem = False
                        self._pem_chars = 0
                    return "".join(out)
                self._in_pem = False
                self._pem_chars = 0
                pos = end.end()
                continue
            begin = _PEM_BEGIN.search(body, pos)
            if begin is None:
                out.append(_redact_plain(body[pos:]))
                break
            out.append(_redact_plain(body[pos:begin.start()]))
            out.append(_mark("private-key"))
            self._in_pem = True
            pos = begin.end()
        return "".join(out)

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if not enabled():
            out, self._tail = self._tail + chunk, ""
            return out
        self._fed_at = self._clock()
        data = self._tail + chunk
        self._tail = ""
        cut = data.rfind("\n") + 1
        partial = data[cut:]
        hold = ""
        if partial:
            if "-----" in partial and len(partial) <= MAX_HOLD:      # may be opening a PEM block
                hold = partial
            else:
                m = _TRAILING_RUN.search(partial)
                if m and len(m.group()) <= MAX_HOLD:
                    hold = m.group()
        body = data[: len(data) - len(hold)]
        self._tail = hold
        return self._process(body)

    def release_stale(self, age: float = 0.25) -> str:
        """The held tail after `age` seconds of silence: a prompt (`Password:`), not a half secret."""
        if not self._tail or self._clock() - self._fed_at < age:
            return ""
        out = _redact_plain(self._tail)
        self._tail = ""
        return out

    def flush(self) -> str:
        out = self._process(self._tail) if self._tail else ""
        self._tail = ""
        return out
