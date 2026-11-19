"""The honest arena: Leeth against Laya on the same items, each model asked through its own interface (2026-10-20).

Why a second arena.  The first one (``.argent/kaggle/leeth2/kernel_arena/arena.py``) built its questions with Leeth's own
corpus builder and handed exactly those to Laya: 38 % of the questions on English tasks were in Russian (``banks.LANG_MIX_SAME``),
the options were bare label keys, an exception from Laya counted as a wrong answer (Laya-en scored 0.000 on CLINC150 because 151
options do not fit its 512-token window), Leeth answered big lists in two rounds (``split_k=128``) while Laya got one question,
Leeth's probabilities were temperature-fitted and Laya's were not, and Laya's score was the best of three checkpoints picked on
the test sample itself.  Our Laya numbers missed its published ones by 0.19-0.27 on MASSIVE and XNLI.

Here:
* one item = one text, one option list (same order) and one gold answer, shared by every model; the id list is written once;
* each model reads the item the way its own documentation asks: Leeth gets its English training question, Laya gets the state
  dict and instructions of its own benchmark scripts (``research/scripts/build_benchmark_nb.py``); option texts are identical;
* each model has a few documented ways to answer (Leeth: one question / parts of 16; Laya: three checkpoints x one question /
  wide option budget / tournament of 16) and the way is chosen per task on the dev split, never on the test split;
* errors are counted, not hidden; calibration is compared after the same one-temperature refit on dev for everybody;
* a reproduction check comes first: Laya's own protocol must give Laya's published numbers, or the harness is not trusted.

Pure functions only (no network, no models): ``scripts/leeth_arena_fair.py`` fetches, runs and reports.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SEED = 20261020
N_TEST = 500
N_DEV = 200


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    config: str
    test_split: str
    dev_split: str
    group: str                      # "heldout": Leeth never trained on it | "home": Leeth trained on its train split
    kind: str = "cls"               # "cls" | "nli"
    text: Tuple[str, ...] = ("text",)
    label: str = "label"
    drop: Tuple[str, ...] = ()      # labels left out of the task (CLINC's out-of-scope, as in Laya's tournament benchmark)
    ordinal: bool = False           # options keep their natural order (SST-5), others are shuffled per item
    lang: str = "en"
    what: str = "category"          # Leeth's question: "Which {what} fits best?"
    state_key: str = "text"         # Laya's state: {state_key: text}
    instructions: str = ""          # Laya's instructions
    note: str = ""


TASKS: Dict[str, Task] = {t.name: t for t in [
    Task("clinc150", "clinc/clinc_oos", "plus", "test", "validation", "heldout", label="intent", drop=("oos",), what="user intent",
         state_key="message", instructions="Which intent does `message` express?", note="150 in-scope intents (Laya's tournament benchmark)"),
    Task("banking77", "mteb/banking77", "default", "test", "train", "heldout", label="label_text", what="banking intent",
         state_key="message", instructions="Which banking intent does `message` express?", note="77 intents; Laya publishes 0.425"),
    Task("dbpedia14", "fancyzhx/dbpedia_14", "dbpedia_14", "test", "train", "heldout", text=("title", "content"), what="category of the entity",
         state_key="article", instructions="Which kind of entity does `article` describe?"),
    Task("emotion", "dair-ai/emotion", "split", "test", "validation", "heldout", what="emotion",
         state_key="text", instructions="Which emotion is most strongly expressed in `text`?", note="held out for Laya too; Laya publishes 0.573"),
    Task("sst5", "SetFit/sst5", "default", "test", "validation", "heldout", label="label_text", ordinal=True, what="sentiment (five grades)",
         state_key="text", instructions="What is the sentiment of `text`?", note="held out for Laya too; Laya publishes 0.372 (score)"),
    Task("massive_en", "mteb/amazon_massive_intent", "en", "test", "validation", "home", label="label", what="user intent",
         state_key="utterance", instructions="What is the user asking for in `utterance`?", note="all 60 intents (Laya publishes 20-option figures)"),
    Task("massive_ru", "mteb/amazon_massive_intent", "ru", "test", "validation", "home", label="label", lang="ru", what="user intent",
         state_key="utterance", instructions="What is the user asking for in `utterance`?"),
    Task("xnli_en", "facebook/xnli", "en", "test", "validation", "home", kind="nli", ordinal=True,
         state_key="premise", instructions="Does `premise` entail `hypothesis`, contradict it, or neither?"),
    Task("xnli_ru", "facebook/xnli", "ru", "test", "validation", "home", kind="nli", ordinal=True, lang="ru",
         state_key="premise", instructions="Does `premise` entail `hypothesis`, contradict it, or neither?"),
]}
HELDOUT = [n for n, t in TASKS.items() if t.group == "heldout"]
HOME = [n for n, t in TASKS.items() if t.group == "home"]
NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_DESCRIPTIONS = {"entailment": "the hypothesis follows from the premise", "neutral": "the premise neither confirms nor refutes the hypothesis",
                    "contradiction": "the hypothesis contradicts the premise"}
SST5_LEVELS = ("very negative", "negative", "neutral", "positive", "very positive")
LEETH_NLI_Q = "Does the text above entail the following statement, contradict it, or neither?\nStatement: {h}"

# Laya's own protocol, reproduced before anything else (README "Multilingual"/"English tasks" tables, BENCHMARKS.md, the tournament README).
# rows: the first N test rows (or N random ones for the tournament benchmark); options: "all" or the gold plus 19 seeded distractors.
REPRO = {
    "repro_massive20_en": dict(task="massive_en", rows=300, first=True, k=20, published={"en": 0.783, "multilingual": 0.657}),
    "repro_xnli_en": dict(task="xnli_en", rows=300, first=True, published={"en": 0.860, "multilingual": 0.843}),
    # 0.425 is the multilingual default (head_max_len 256, README); no number is published for the English checkpoint.
    "repro_banking77": dict(task="banking77", rows=500, first=True, published={"multilingual": 0.425}),
    "repro_emotion": dict(task="emotion", rows=600, first=True, published={"en": 0.573, "multilingual": 0.513}),
    "repro_sst5": dict(task="sst5", rows=600, first=True, score=True, published={"en": 0.372, "multilingual": 0.282}),
    "repro_clinc150": dict(task="clinc150", rows=500, first=False, published={"en": 0.604, "en/tournament": 0.894}),
}
REPRO_TOL = 0.03


def rng_for(*parts: Any) -> random.Random:
    return random.Random(int.from_bytes(hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest(), "little"))


def humanize(label: Any) -> str:
    """'alarm_set' -> 'alarm set' (both Laya's benchmark scripts and Leeth's builder do this)."""
    return re.sub(r"_+", " ", str(label).strip())


def row_text(task: Task, row: Dict[str, Any]) -> str:
    return "\n".join(str(row.get(k) or "").strip() for k in task.text if str(row.get(k) or "").strip())


def label_of(task: Task, row: Dict[str, Any], names: Optional[Sequence[str]]) -> Optional[str]:
    v = row.get(task.label)
    if v is None or v == "":
        return None
    if names is not None and isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return names[int(v)] if 0 <= int(v) < len(names) else None
    return str(v)


def label_set(task: Task, rows: Iterable[Dict[str, Any]], names: Optional[Sequence[str]]) -> List[str]:
    """The task's labels: the ClassLabel names, or every label seen in the full split; ``drop`` removed. SST-5 in its order."""
    if task.kind == "nli":
        return list(NLI_LABELS)
    if task.ordinal and task.name == "sst5":
        return list(SST5_LEVELS)
    labs = list(names) if names is not None else sorted({lab for r in rows if (lab := label_of(task, r, None))})
    return [lab for lab in labs if lab not in task.drop]


def make_item(task: Task, split: str, idx: int, row: Dict[str, Any], labels: Sequence[str], names: Optional[Sequence[str]],
              k: int = 0, seed: int = SEED) -> Optional[Dict[str, Any]]:
    """One shared item, or None when the row is out of the task (a dropped label, empty text).  ``k`` > 0: the gold plus ``k - 1``
    distractors (Laya's 20-option MASSIVE protocol).  Options are shuffled per item with a seed of the item id, except ordinal tasks."""
    if task.kind == "nli":
        p, h = str(row.get("premise") or "").strip(), str(row.get("hypothesis") or "").strip()
        v = row.get(task.label)
        if not p or not h or v is None or not (0 <= int(v) <= 2):
            return None
        gold, body = NLI_LABELS[int(v)], {"premise": p, "hypothesis": h}
    else:
        text, gold = row_text(task, row), label_of(task, row, names)
        if not text or gold is None or gold not in labels:
            return None
        body = {"text": text}
    iid = f"{task.name}:{split}:{idx}"
    opts = list(labels)
    if k and len(opts) > k:
        r = rng_for(seed, iid, "distractors")
        opts = [gold] + r.sample([o for o in opts if o != gold], k - 1)
    if not task.ordinal:
        rng_for(seed, iid, "order").shuffle(opts)
    elif k:
        opts = [o for o in labels if o in set(opts)]
    shown = [humanize(o) for o in opts]
    if len(set(shown)) != len(shown):
        return None
    return {"id": iid, "task": task.name, "split": split, "lang": task.lang, **body, "options": shown, "gold": opts.index(gold)}


def sample_indices(n: int, want: int, *parts: Any) -> List[int]:
    idx = list(range(n))
    rng_for(SEED, *parts).shuffle(idx)
    return sorted(idx[:want])


# ------------------------------------------------------------------------------------------------ rendering, one per model
def leeth_question(task: Task, item: Dict[str, Any]) -> Tuple[str, str]:
    """(state, question) the way Leeth was trained: the text, then an English question (never the Russian half of the bank)."""
    if task.kind == "nli":
        return item["premise"], LEETH_NLI_Q.format(h=item["hypothesis"])
    return item["text"], f"Which {task.what} fits best?"


def laya_request(task: Task, item: Dict[str, Any], score: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(state, questions) the way Laya's benchmark scripts ask: a state dict and instructions naming its field."""
    if task.kind == "nli":
        state = {"premise": item["premise"], "hypothesis": item["hypothesis"]}
        crit = {o: NLI_DESCRIPTIONS.get(o) for o in item["options"]}
        return state, {"q": {"type": "choice", "instructions": task.instructions, "criteria": crit}}
    state = {task.state_key: item["text"]}
    if score:
        return state, {"q": {"type": "score", "instructions": task.instructions, "criteria": list(item["options"])}}
    return state, {"q": {"type": "choice", "instructions": task.instructions, "criteria": {o: None for o in item["options"]}}}


def laya_probs(item: Dict[str, Any], answer: Dict[str, Any]) -> List[float]:
    """Laya's answer as probabilities over the item's options, in the item's order.  A tournament answers over its finalists only:
    the eliminated options get 0.  A score answer has probabilities by level index."""
    probs = answer.get("probabilities") or {}
    if answer.get("type") == "score":
        v = [float(probs.get(str(i), 0.0)) for i in range(len(item["options"]))]
    else:
        v = [float(probs.get(o, 0.0)) for o in item["options"]]
    s = sum(v)
    return [x / s for x in v] if s > 0 else v


# ------------------------------------------------------------------------------------------------ statistics
def accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    return float(np.mean([r["ok"] for r in rows])) if rows else float("nan")


def outcome(item: Dict[str, Any], probs: Optional[Sequence[float]], error: Optional[str] = None) -> Dict[str, Any]:
    """One answer: ok / confidence / probabilities.  An error is a wrong answer AND is counted as an error."""
    if error is not None or not probs or not np.isfinite(np.asarray(probs, float)).all() or sum(probs) <= 0:
        return {"id": item["id"], "ok": 0.0, "conf": 0.0, "probs": None, "error": error or "no probabilities"}
    p = np.asarray(probs, float)
    return {"id": item["id"], "ok": float(int(np.argmax(p)) == item["gold"]), "conf": float(p.max()), "probs": [round(float(x), 6) for x in p], "error": None}


def ece(conf: Sequence[float], ok: Sequence[float], bins: int = 15) -> float:
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if not len(conf):
        return float("nan")
    tot = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins) if b else (conf >= 0) & (conf <= 1 / bins)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(tot)


def _tempered(p: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(p, 1e-6, 1.0)) / t
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def fit_temperature(dev: Sequence[Tuple[Sequence[float], int]]) -> float:
    """One temperature on the dev answers by NLL over a log grid (Laya's ``fit_temperature`` grid: 0.2..10, 160 points).
    It works on probabilities, so it treats both models alike: softmax(logits / T) = p ** (1 / T), renormalised."""
    pairs = [(np.asarray(p, float), g) for p, g in dev if p]
    if len(pairs) < 25:
        return 1.0
    grid = np.exp(np.linspace(math.log(0.2), math.log(10.0), 160))
    nll = [-sum(math.log(max(_tempered(p, t)[g], 1e-12)) for p, g in pairs) for t in grid]
    return float(grid[int(np.argmin(nll))])


def refit_ece(dev: Sequence[Dict[str, Any]], test: Sequence[Dict[str, Any]], gold: Dict[str, int]) -> Tuple[float, float]:
    """(temperature, ECE on test) after one temperature fitted on dev.  Errors stay in with confidence 0."""
    t = fit_temperature([(r["probs"], gold[r["id"]]) for r in dev])
    conf, ok = [], []
    for r in test:
        if r["probs"]:
            p = _tempered(np.asarray(r["probs"], float), t)
            conf.append(float(p.max()))
            ok.append(float(int(np.argmax(p)) == gold[r["id"]]))
        else:
            conf.append(0.0)
            ok.append(0.0)
    return t, ece(conf, ok)


def select_config(dev_acc: Dict[str, float], order: Sequence[str]) -> str:
    """The config with the best dev accuracy; a tie goes to the earlier (simpler) config in ``order``."""
    best = max(dev_acc.values())
    return next(c for c in order if c in dev_acc and dev_acc[c] >= best - 1e-12)


def paired_bootstrap(a: Dict[str, Sequence[float]], b: Dict[str, Sequence[float]], n_boot: int = 10000, seed: int = SEED) -> Dict[str, Any]:
    """Accuracy difference a - b per task and averaged over tasks, with 95 % intervals.  ``a[task]`` and ``b[task]`` are 0/1 outcomes
    on the same items in the same order; items are resampled within each task (the pairs stay together)."""
    rng = np.random.default_rng(seed)
    tasks = sorted(a)
    diffs = {t: np.asarray(a[t], float) - np.asarray(b[t], float) for t in tasks}
    boots = {t: np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]) for t, d in diffs.items()}
    out = {"tasks": {}}
    for t in tasks:
        lo, hi = np.percentile(boots[t], [2.5, 97.5])
        out["tasks"][t] = {"diff": float(diffs[t].mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diffs[t]))}
    mean = np.mean([boots[t] for t in tasks], axis=0)
    lo, hi = np.percentile(mean, [2.5, 97.5])
    out["mean"] = {"diff": float(np.mean([diffs[t].mean() for t in tasks])), "lo": float(lo), "hi": float(hi)}
    return out


def verdict(boot: Dict[str, Any]) -> str:
    """The rule written down before any number (reports/leeth_laya_fair_arena_2026-10-20.md, "Правила"):
    "выше"     -- the mean difference's interval is above 0 and no task's interval is entirely below 0;
    "ниже"     -- the mean difference's interval is entirely below 0;
    "на уровне" -- anything else (the mean interval covers 0, or the mean is ahead with a task significantly behind)."""
    m = boot["mean"]
    behind = [t for t, v in boot["tasks"].items() if v["hi"] < 0]
    if m["lo"] > 0 and not behind:
        return "выше"
    if m["hi"] < 0:
        return "ниже"
    return "на уровне"


def repro_ok(measured: float, published: float, n: int) -> bool:
    """A reproduction holds when it is within 0.03 of the published number or within two standard errors of it."""
    se = math.sqrt(max(published * (1 - published), 1e-9) / max(1, n))
    return abs(measured - published) <= max(REPRO_TOL, 2 * se)


# ------------------------------------------------------------------------------------------------ contamination
def _norm_words(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def shingles(s: str, k: int = 8) -> set:
    w = _norm_words(s)
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)} if len(w) >= k else ({" ".join(w)} if w else set())


def contaminated(text: str, corpus_shingles: set, k: int = 8, share: float = 0.5) -> bool:
    """True when at least ``share`` of the text's word ``k``-grams occur in the corpus (a short text: the whole text)."""
    sh = shingles(text, k)
    return bool(sh) and sum(s in corpus_shingles for s in sh) / len(sh) >= share


@dataclass
class Table:
    """Accumulates per (model, config, task, split) outcomes -- the report reads only this."""
    rows: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = field(default_factory=dict)

    def add(self, model: str, config: str, task: str, split: str, rs: List[Dict[str, Any]]) -> None:
        self.rows.setdefault((model, config, task, split), []).extend(rs)

    def configs(self, model: str, task: str) -> List[str]:
        return sorted({c for (m, c, t, s) in self.rows if m == model and t == task})

    def get(self, model: str, config: str, task: str, split: str) -> List[Dict[str, Any]]:
        return self.rows.get((model, config, task, split), [])
