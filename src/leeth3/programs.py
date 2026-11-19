"""The program an R001 record's action runs, and the programs the corpus already has too much of.

The exam asks about commands of programs the training never saw, so the safety corpus needs many programs a little each. Asked in
words ("one program - one pair per batch", rounds 10-14), the teacher drifted back to the familiar ones: programs new to the corpus
62 -> 40 -> 25 -> 25 per round, git 14 times in round 14. So the queue names the most frequent programs of the corpus in every
R001 batch, and the ingest refuses a twin pair whose action runs one of them.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

_SKIP = {"sudo", "doas", "env", "nohup", "time", "exec", "command", "xargs", "nice", "ionice", "stdbuf", "timeout"}
# "Агент: ", "Agent executes the terminal command: ", "Action de l'agent : ", "代理：", "エージェント：". An ASCII colon needs a space
# after it (so "https://..." or "C:\..." of a label-less command are not taken for a label); a full-width one does not.
_LABEL = re.compile(r"^[^\n`\"$|;&=/\\]{0,80}?(?::\s+|：\s*)")
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def action_program(state: str) -> str:
    """The program of the last line of a safety state ("" when there is none): the first word after the speaker label, past
    ``sudo``/``env`` and ``VAR=value``, as a lowercase base name ("/usr/bin/rsync" -> "rsync", "Remove-Item" -> "remove-item",
    ``click("#buy")`` -> "click")."""
    lines = [l for l in state.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    line = lines[-1].strip()
    m = _LABEL.match(line)
    if m:
        line = line[m.end():]
    for tok in line.strip().strip("`'\"").split():
        tok = tok.strip("`'\"()")
        if not tok or tok in _SKIP or _ASSIGN.match(tok) or tok.startswith("-"):
            continue
        tok = re.split(r"[(\[{]", tok, 1)[0]
        tok = tok.replace("\\", "/").rsplit("/", 1)[-1].lower()
        return tok
    return ""


def top_programs(states: Iterable[str], n: int = 25, min_count: int = 20) -> List[str]:
    """The ``n`` most frequent programs (each at least ``min_count`` times), most frequent first."""
    c = Counter(p for p in (action_program(s) for s in states) if p)
    return [p for p, k in c.most_common(n) if k >= min_count]
