"""Tool libraries of 6 .. 255 entries: pick the tool for a request when most of the list is a near-duplicate of the right one.

An agent attached to a few MCP servers sees hundreds of tools whose names differ by one word (``archive_invoice`` / ``export_invoice`` /
``archive_receipt``).  The older tool-picking data (tools.py) had 3..25 tools out of 33; the pool-inflation step (longctx.py) adds real
alternatives of corpus tasks.  This is the direct version: a library built from verbs x objects, requests that paraphrase the verb of
the tool's name (the object noun stays), and a list that holds the right tool's neighbours (same object with another verb, same verb
with another object) for 30-70 % of its entries.

The vocabulary is disjoint from the exam's (scripts/leeth21_exam.py: ten verbs and thirty objects): a model that scores on the exam has
learnt to read near-duplicate lists, not the words.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from ...schema import Sample

VERBS: Dict[str, List[str]] = {
    "list": ["show all", "enumerate", "give me every", "what are the"], "rename": ["change the name of", "give a new name to", "retitle", "relabel"],
    "duplicate": ["make a copy of", "clone", "copy", "replicate"], "restore": ["bring back", "recover", "undelete", "roll back to an earlier version of"],
    "approve": ["sign off on", "give the green light to", "okay", "accept"], "reject": ["turn down", "decline", "refuse", "send back"],
    "sync": ["bring up to date", "synchronise", "refresh from the server", "pull the latest of"], "merge": ["combine", "join", "fold together", "consolidate"],
    "tag": ["put a label on", "mark with a tag", "categorise", "annotate"], "pin": ["keep at the top", "stick to the top", "make sticky", "highlight"],
    "mute": ["silence", "stop notifications from", "turn off alerts for", "hush"], "download": ["save locally", "fetch a copy of", "grab", "pull down"],
    "upload": ["push to the cloud", "put online", "send up", "store remotely"], "print": ["send to the printer", "make a hard copy of", "produce paper from", "output on paper"],
    "schedule": ["set a time for", "plan", "put on the timetable", "line up"], "cancel": ["call off", "abort", "stop", "withdraw"],
    "assign": ["hand over to a teammate", "give ownership of", "delegate", "allocate"], "preview": ["take a look at a draft of", "show a draft of", "render a sample of", "peek at"],
    "validate": ["check the correctness of", "verify", "make sure it is valid:", "lint"], "translate": ["put into another language", "localise", "convert to French", "render in Spanish"],
    "favorite": ["add to my favourites", "star", "save for later", "keep handy"], "flag": ["raise a flag on", "mark as suspicious", "report", "call attention to"],
    "escalate": ["send up the chain", "raise the priority of", "bump to management", "make urgent"], "close": ["finish off", "wrap up", "mark as done", "shut"],
    "reopen": ["bring back to life", "open again", "revive", "start over on"], "pause": ["hold", "put on hold", "freeze for now", "suspend"],
    "resume": ["continue", "pick up again", "carry on with", "unfreeze"], "publish": ["make public", "release", "put live", "announce"],
    "unpublish": ["take down", "pull offline", "hide from the public", "retract"],
}
OBJECTS: List[str] = [
    "timesheet", "expense claim", "wiki page", "pull request", "build", "deployment", "timetable", "mailbox", "chat channel", "dashboard", "survey", "shipment",
    "warehouse order", "customer", "lead", "campaign", "podcast episode", "document", "folder", "backup", "license", "device", "alert", "incident", "release",
    "branch", "milestone", "sprint", "workspace", "account", "template", "webhook", "certificate", "invitation", "form", "snippet", "feedback item", "quote",
    "claim", "appointment",
]
SUFFIX = ["", "", " from last month", " for the Madrid office", " that I created on Monday", " for the Q3 review", " that Dana mentioned", " belonging to the vendor"]
TEMPLATES = ["{syn} the {obj}{suf}", "please {syn} my {obj}{suf}", "I need you to {syn} that {obj}{suf}", "could you {syn} the {obj}{suf}?", "{syn} this {obj}{suf} now", "hey, {syn} the {obj}{suf}"]
STATES = ["User request: “{r}”", "Request: {r}", "The user asks: {r}", "Incoming message:\n{r}", "{r}"]
QUESTIONS = ["Which tool should be called?", "Pick the tool that fits this request.", "Which function matches the request?", "Select the correct tool.", "What should be run for this?"]
NONE_OPTION = ["none of these — answer directly", "no suitable tool", "no tool needed"]
CHAT = ["what is the capital of France", "tell me a joke", "thanks, that helps", "how are you today", "explain what a mutex is", "who wrote Hamlet"]


def library() -> List[Tuple[str, str]]:
    """(tool name, description) for every verb x object."""
    out = []
    for v in VERBS:
        for o in OBJECTS:
            out.append((f"{v}_{o.replace(' ', '_')}", f"{v.capitalize()} the {o}" if v != "list" else f"List all {o}s"))
    return out


_LIB = library()


def _request(rng: random.Random, verb: str, obj: str) -> str:
    return rng.choice(TEMPLATES).format(syn=rng.choice(VERBS[verb]), obj=obj, suf=rng.choice(SUFFIX))


def gen_bigtools(rng: random.Random, k_range: Optional[Tuple[int, int]] = None) -> Optional[List[Sample]]:
    r = rng.random()
    if k_range is not None:
        K = rng.randint(*k_range)
    elif r < 0.35:
        K = rng.randint(6, 30)
    elif r < 0.65:
        K = rng.randint(31, 100)
    else:
        K = rng.randint(101, 255)
    none_case = rng.random() < 0.08
    ti = rng.randrange(len(_LIB))
    name, desc = _LIB[ti]
    verb, obj = name.split("_", 1)[0], name.split("_", 1)[1].replace("_", " ")
    near = [j for j, (n, _) in enumerate(_LIB) if j != ti and (n.split("_", 1)[1] == name.split("_", 1)[1] or n.split("_", 1)[0] == verb)]
    rng.shuffle(near)
    rest = [j for j in range(len(_LIB)) if j != ti and j not in near]
    rng.shuffle(rest)
    n_near = min(len(near), int((K - 1) * rng.choice([0.3, 0.5, 0.7])))
    others = near[:n_near] + rest[: max(0, K - 1 - n_near)]
    ids = ([] if none_case else [ti]) + others[: K - (0 if none_case else 1)]
    rng.shuffle(ids)
    style = rng.choice(["both", "both", "name", "desc"])
    fmt = {"both": lambda j: f"{_LIB[j][0]} — {_LIB[j][1]}", "name": lambda j: _LIB[j][0], "desc": lambda j: _LIB[j][1]}[style]
    options = [fmt(j) for j in ids]
    if len(set(options)) != len(options):
        return None
    if none_case:
        options.insert(rng.randrange(len(options) + 1), rng.choice(NONE_OPTION))
        target = options.index(next(o for o in options if o in NONE_OPTION))
        req = rng.choice(CHAT)
    else:
        target = ids.index(ti)
        req = _request(rng, verb, obj)
    state = rng.choice(STATES).format(r=req)
    approx = sum(len(o) for o in options) / 2.6 + len(options) + len(state) / 2.4 + 80
    meta = {"lic": "own", "synth": True}
    if len(options) > 40:
        meta["max_len"] = int(min(8192, max(4096, approx + 200)))
        meta["big"] = len(options)
    s = Sample("bigtools", "tools", "en", "choice", state, rng.choice(QUESTIONS), options, target, meta=meta)
    return [s]


def gen_bigtools_small(rng: random.Random) -> Optional[List[Sample]]:
    """The same task with 6..12 options: the rows that may sit in dev / calib / test, which the trainer evaluates with a 640-token window."""
    return gen_bigtools(rng, k_range=(6, 12))


GENERATORS = [("bigtools", gen_bigtools, 15000)]
