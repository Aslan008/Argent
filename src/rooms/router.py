"""Pick the room for a task — deterministic, free, explainable.

`/rooms <task>` without a room name used to start `analyze` every time. A
recipe library is only useful if the right recipe is found from the person's own
words, so this scores a task against each room's name, tags and description
(Russian and English, with endings tolerated) and says how sure it is. The caller
runs the room when the pick is clear and asks the person when it is not — no model
call is spent on routing, and the same task always gives the same room.
"""

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")
_STOP = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "all", "any", "are", "not",
    "для", "это", "что", "как", "при", "или", "его", "все", "всё", "мне", "нам", "надо", "нужно",
    "пожалуйста", "можно", "через", "после", "перед", "чтобы", "если", "только", "также",
})

#: A task word counts for a tag / a room-name word / a description word.
W_TAG, W_NAME, W_DESC = 3, 2, 1
#: A pick is "clear" when its score is at least this (one tag word alone is not enough: "сделай" is in
#: half the sentences people type) and it beats the runner-up by at least MARGIN.
CLEAR_SCORE = 4
CLEAR_MARGIN = 2


def words(text: str) -> list:
    out = []
    for w in _WORD.findall((text or "").lower().replace("ё", "е")):
        if len(w) >= 3 and w not in _STOP:
            out.append(w)
    return out


def _same_word(a: str, b: str) -> bool:
    """Whether two words are inflections of one stem: they share a prefix of 4 letters
    (or all of the shorter one, when it is 3 letters)."""
    n = min(len(a), len(b))
    if n < 3:
        return False
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k >= min(n, 4)


@dataclass
class Pick:
    name: str = ""                       # the best room, "" when nothing matched at all
    score: int = 0
    clear: bool = False                  # safe to run without asking
    ranked: list = field(default_factory=list)   # [(name, score, why)] best first, score > 0 only


def score_room(task_words: list, room) -> tuple:
    """(score, matched words) of a task against one room."""
    tags = [w for t in room.tags for w in words(t)]
    names = words(room.room.replace("_", " "))
    desc = words(room.description)
    score = 0
    matched = []
    for tw in dict.fromkeys(task_words):
        for bucket, weight in ((tags, W_TAG), (names, W_NAME), (desc, W_DESC)):
            if any(_same_word(tw, w) for w in bucket):
                score += weight
                matched.append(tw)
                break
    return score, matched


def pick_room(task: str, library, exclude=("triage",)) -> Pick:
    """Rank the library's active rooms for `task`."""
    tw = words(task)
    ranked = []
    active = library.active_rooms() if hasattr(library, "active_rooms") else library.rooms
    for name, room in active.items():
        if name in exclude:
            continue
        s, matched = score_room(tw, room)
        if s > 0:
            ranked.append((name, s, ", ".join(matched)))
    ranked.sort(key=lambda r: (-r[1], r[0]))
    if not ranked:
        return Pick()
    top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    clear = top[1] >= CLEAR_SCORE and top[1] - second >= CLEAR_MARGIN
    return Pick(name=top[0], score=top[1], clear=clear, ranked=ranked)
