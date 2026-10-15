"""Procedural, expert-labelled domains: games, NPC policies, sorting, browser pages, tool selection.

Every generator returns ``Sample`` lists whose labels are computed by a rule or a perfect solver, never guessed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ...schema import Sample
from ..common import stable_hash
from . import browser, games, npc, npc2, sorting, tools

# (module, generator name, function, training calls)
PLAN: List[Tuple[str, str, Callable[..., Optional[object]], int]] = [
    ("games", "snake", games.gen_snake, 24000),
    ("games", "maze", games.gen_maze, 24000),
    ("games", "ttt", games.gen_ttt, 12000),
    ("games", "treasure", games.gen_treasure, 20000),
    ("npc", "policy", npc.gen_policy, 30000),
    ("npc", "persona", npc.gen_persona, 12000),
    ("npc2", "policy2", npc2.gen_policy2, 60000),
    ("sorting", "files", sorting.gen_files, 15000),
    ("sorting", "tickets", sorting.gen_tickets, 15000),
    ("sorting", "order", sorting.gen_order, 20000),
    ("sorting", "compare", sorting.gen_compare_numbers, 8000),
    ("browser", "click", browser.gen_click, 45000),
    ("browser", "risk", browser.gen_risk, 8000),
    ("tools", "toolpick", tools.gen_toolpick, 40000),
]
EVAL_CAPS = {"dev": 120, "calib": 250, "test": 500}
ROLE_SEED = {"train": 0, "dev": 101, "calib": 202, "test": 303}


def _as_list(x) -> List[Sample]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def generate(fn: Callable, n_calls: int, seed: int, **kw) -> List[Sample]:
    rng = random.Random(seed)
    out: List[Sample] = []
    tries = 0
    done = 0
    while done < n_calls and tries < n_calls * 6:
        tries += 1
        got = _as_list(fn(rng, **kw))
        if not got:
            continue
        ok = []
        for s in got:
            try:
                s.validate()
            except ValueError:
                continue
            s.meta = {**(s.meta or {}), "lic": "own", "synth": True}
            ok.append(s)
        if ok:
            out.extend(ok)
            done += 1
    return out


def build(out_dir: Path, seed: int = 2026, scale: float = 1.0) -> Dict[str, List[Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res: Dict[str, List[Path]] = {"train": [], "dev": [], "calib": [], "test": []}

    def write(name: str, role: str, ss: List[Sample]) -> None:
        if not ss:
            return
        p = out_dir / f"synth_{name}.{role}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for s in ss:
                f.write(s.to_json() + "\n")
        res[role].append(p)

    for mod, name, fn, n_train in PLAN:
        n_tr = max(20, int(n_train * scale))
        base = stable_hash(mod, name) & 0xFFFF
        write(f"{mod}_{name}", "train", generate(fn, n_tr, seed + base + ROLE_SEED["train"]))
        for role, cap in EVAL_CAPS.items():
            n = max(10, int(cap * (1.0 if scale >= 1.0 else max(scale * 4, 0.2))))
            write(f"{mod}_{name}", role, generate(fn, n, seed + base + ROLE_SEED[role]))
    # unseen tools: evaluation only
    ss = generate(tools.gen_toolpick, 200, seed + 777 + ROLE_SEED["test"], holdout=True)
    for s in ss:
        s.task = "tool_pick_unseen"
        s.meta = {**(s.meta or {}), "heldout": True}
    write("tools_unseen", "test", ss)
    # NPC rules written with a vocabulary the model never trained on: evaluation only
    ss = generate(npc2.gen_policy2_unseen, 300, seed + 778 + ROLE_SEED["test"])
    for s in ss:
        s.meta = {**(s.meta or {}), "heldout": True}
    write("npc2_unseen", "test", ss)
    return res
