"""Quizzes in a group: the model writes the question, the bridge runs the round.

The model's whole contribution is one compact line at the end of its answer
(``[[quiz:{...}]]``). Everything after that — buttons, the clock, who answered, the
scoreboard — happens locally, so a round costs no requests and does not grow the
conversation the model has to read on every turn.

A round is timer-driven. The model may add ``n`` when it knows how many people are
playing; then the round also closes as soon as that many have answered, which is what
"wait for all of you" means in practice. Nobody can be waited for before they press a
button — the bot cannot see who intends to play.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

MAX_QUESTION = 300
MAX_OPTION = 60
MAX_OPTIONS = 6
MIN_OPTIONS = 2
DEFAULT_SECONDS = 30
MAX_SECONDS = 300

#: ``[[quiz:{...}]]`` — flat JSON only, so the braces cannot be fooled by nesting.
TAG = re.compile(r"\[\[\s*quiz\s*:\s*(\{[^{}]*\})\s*\]\]", re.IGNORECASE | re.DOTALL)

#: One line of the eventual result: who got it.
MAX_NAME = 24


@dataclass
class Round:
    id: str
    chat_id: str
    question: str
    options: List[str]
    answer: int
    seconds: int
    expected: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    answers: Dict[str, int] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    message_id: Optional[int] = None
    closed: bool = False
    #: No clock: the question waits for whoever opens the chat later.
    open_ended: bool = False

    @property
    def left(self) -> float:
        if self.open_ended:
            return float("inf")
        return max(0.0, self.seconds - (time.time() - self.started_at))


def parse(raw: str, default_seconds: int = DEFAULT_SECONDS) -> Optional[dict]:
    """Validate one spec. None means "not a game", never a half-broken one."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    question = str(data.get("q") or "").strip()
    options = data.get("o")
    if not question or not isinstance(options, list):
        return None
    options = [str(o).strip()[:MAX_OPTION] for o in options if str(o).strip()][:MAX_OPTIONS]
    if len(options) < MIN_OPTIONS:
        return None

    try:
        answer = int(data.get("a"))
    except (TypeError, ValueError):
        return None
    if not 0 <= answer < len(options):
        return None

    try:
        raw_seconds = int(data.get("t") if data.get("t") is not None else default_seconds)
    except (TypeError, ValueError):
        raw_seconds = default_seconds
    # `t: 0` means "no clock": the question stays open for whoever comes later.
    open_ended = raw_seconds <= 0
    seconds = 0 if open_ended else max(5, min(MAX_SECONDS, raw_seconds))

    try:
        expected = int(data["n"]) if data.get("n") is not None else None
    except (TypeError, ValueError):
        expected = None
    if expected is not None and expected < 1:
        expected = None

    return {"q": question[:MAX_QUESTION], "o": options, "a": answer,
            "t": seconds, "n": expected, "open": open_ended}


def extract(text: str, default_seconds: int = DEFAULT_SECONDS) -> Tuple[str, Optional[dict]]:
    """Return ``(text without the spec, spec or None)``.

    The spec is removed whether or not it parses: a broken spec must not reach the chat
    as raw JSON, and a round that cannot be played is better skipped than half-drawn.
    """
    if not text:
        return text, None
    match = TAG.search(text)
    if not match:
        return text, None
    return TAG.sub("", text).rstrip(), parse(match.group(1), default_seconds)


class Games:
    """Rounds and scores per chat. In memory: a restart drops them (documented).

    Two kinds of round live here. A **timed** one blocks the chat — there is one, it has a
    clock, and its answer is revealed to everybody at the end. An **open-ended** one
    (``t: 0``) is a question left on the table: many may be unanswered, it never blocks a
    new one, its buttons stay alive, and a late player gets the verdict privately so
    nothing is spoiled for anybody still thinking.
    """

    #: How many open-ended questions a chat may keep; the oldest is forgotten first.
    MAX_OPEN = 10

    def __init__(self) -> None:
        self._rounds: Dict[str, Round] = {}
        self._open: Dict[str, List[Round]] = {}
        self._scores: Dict[str, Dict[str, int]] = {}
        self._names: Dict[str, Dict[str, str]] = {}

    def active(self, chat_id: str | int) -> Optional[Round]:
        """The timed round running in this chat, if any."""
        round_ = self._rounds.get(str(chat_id))
        return round_ if round_ and not round_.closed else None

    def find(self, chat_id: str | int, round_id: str) -> Optional[Round]:
        """Any round of this chat — running or left open — by its id."""
        round_ = self._rounds.get(str(chat_id))
        if round_ and round_.id == round_id:
            return round_
        for open_round in self._open.get(str(chat_id), []):
            if open_round.id == round_id:
                return open_round
        return None

    def start(self, chat_id: str | int, spec: dict) -> Optional[Round]:
        """Open a round. None only when a timed one is already running."""
        chat = str(chat_id)
        round_ = Round(id=uuid.uuid4().hex[:6], chat_id=chat, question=spec["q"],
                       options=list(spec["o"]), answer=int(spec["a"]),
                       seconds=int(spec["t"]), expected=spec.get("n"),
                       open_ended=bool(spec.get("open")))
        if round_.open_ended:
            shelf = self._open.setdefault(chat, [])
            shelf.append(round_)
            del shelf[:-self.MAX_OPEN]
            return round_
        if self.active(chat):
            return None
        self._rounds[chat] = round_
        return round_

    def answer(self, chat_id: str | int, round_id: str, user_id: str | int,
               name: str, option: int) -> str:
        """Register one press. Returns a status word the caller shows to that person."""
        round_ = self.find(chat_id, round_id)
        if not round_ or round_.closed:
            return "closed"
        if round_.left <= 0:
            return "timeout"
        uid = str(user_id)
        if uid in round_.answers:
            return "already"
        if not 0 <= option < len(round_.options):
            return "bad"
        round_.answers[uid] = option
        round_.names[uid] = (name or "Участник")[:MAX_NAME]
        if round_.open_ended:
            # Nobody waits for a verdict here: this person gets it now, and only they see it.
            self._credit(round_, uid, option)
            return "correct" if option == round_.answer else "wrong"
        if round_.expected and len(round_.answers) >= round_.expected:
            return "complete"
        return "ok"

    def _credit(self, round_: Round, uid: str, option: int) -> None:
        names = self._names.setdefault(round_.chat_id, {})
        names.setdefault(uid, round_.names.get(uid, "Участник"))
        if option == round_.answer:
            scores = self._scores.setdefault(round_.chat_id, {})
            scores[uid] = scores.get(uid, 0) + 1

    def close(self, chat_id: str | int) -> Optional[Round]:
        """Close the round and score it. None when nothing was open (idempotent)."""
        round_ = self._rounds.get(str(chat_id))
        if not round_ or round_.closed:
            return None
        round_.closed = True
        chat = str(chat_id)
        scores = self._scores.setdefault(chat, {})
        names = self._names.setdefault(chat, {})
        for uid, option in round_.answers.items():
            names.setdefault(uid, round_.names.get(uid, "Участник"))
            if option == round_.answer:
                scores[uid] = scores.get(uid, 0) + 1
        return round_

    def cancel(self, chat_id: str | int) -> Optional[Round]:
        """Close without scoring (a `/reset` in the middle of a round)."""
        round_ = self._rounds.get(str(chat_id))
        if not round_ or round_.closed:
            return None
        round_.closed = True
        return round_

    def scoreboard(self, chat_id: str | int, limit: int = 5) -> List[Tuple[str, int]]:
        chat = str(chat_id)
        names = self._names.get(chat, {})
        rows = [(names.get(uid, "Участник"), points)
                for uid, points in self._scores.get(chat, {}).items()]
        rows.sort(key=lambda pair: -pair[1])
        return rows[:limit]


# ─── rendering ───────────────────────────────────────────────────────────────

def question_text(round_: Round) -> str:
    from src.telegram.render import esc

    return (f"❓ <b>Вопрос</b>\n\n{esc(round_.question)}\n\n"
            f"<i>Выберите вариант — {round_.seconds} с. Ответы не видны другим.</i>")


def keyboard(round_: Round) -> dict:
    """Two options per row: longer labels still fit on a phone."""
    rows, row = [], []
    for index, option in enumerate(round_.options):
        row.append({"text": option[:48], "callback_data": f"qz:{round_.id}:{index}"})
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return {"inline_keyboard": rows}


def result_text(round_: Round, scoreboard: List[Tuple[str, int]]) -> str:
    from src.telegram.render import esc

    right = [round_.names.get(uid, "Участник")
             for uid, option in round_.answers.items() if option == round_.answer]
    wrong = [round_.names.get(uid, "Участник")
             for uid, option in round_.answers.items() if option != round_.answer]

    lines = [f"❓ {esc(round_.question)}", "",
             f"✅ <b>Правильный ответ:</b> {esc(round_.options[round_.answer])}"]
    if right:
        lines.append("🎯 Угадали: " + esc(", ".join(right)))
    if wrong:
        lines.append("❌ Не угадали: " + esc(", ".join(wrong)))
    if not round_.answers:
        lines.append("<i>Никто не ответил.</i>")
    if scoreboard:
        lines.append("")
        lines.append("🏆 " + esc(" · ".join(f"{name} {points}" for name, points in scoreboard)))
    return "\n".join(lines)


def score_text(scoreboard: List[Tuple[str, int]]) -> str:
    from src.telegram.render import esc

    if not scoreboard:
        return "🏆 <b>Счёт</b>\n\nПока пусто — сыграйте раунд."
    lines = ["🏆 <b>Счёт</b>", ""]
    for place, (name, points) in enumerate(scoreboard, 1):
        lines.append(f"{place}. {esc(name)} — {points}")
    return "\n".join(lines)
