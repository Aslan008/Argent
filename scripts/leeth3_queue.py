"""Write teacher batches into leeth3_data/queue/: each batch is a self-contained task WITH A PLAN FOR EVERY RECORD.

    python scripts/leeth3_queue.py --request R003 --batches 4 [--kind long] [--seed 1] [--data leeth3_data]

Why a per-record plan: the pilot (2026-10-16) showed that a batch-level wish ("20 % long texts, half yes") is not followed - the
generator shortened every text (none above 300 words) and drifted to 67 % "yes". So the batch fixes, row by row, the family,
language, type, length range, the planned answer (yes / no / K options / number of spans) and the group (twins, questions on
one text). The machine-readable copy goes to ``leeth3_data/plans/<batch>.json`` and the ingest checks every record against it.

Kinds: R001 normal|long, R002 small, R003 normal|long, R004 normal|long. There is no R002 "big" any more: big catalogs are built from
real options of accepted records (src/leeth3/expand.py) - asked for 128-254 tools, the teacher padded them with templates.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leeth3.pools import TRAIN, check_pools  
from src.leeth3.programs import top_programs  

DEFAULT_DATA = ROOT / "leeth3_data"
REQUEST_FILES = {"R001": "R001_safety.md", "R002": "R002_tools.md", "R003": "R003_content.md", "R004": "R004_extract.md"}
REQUEST_DOMAINS = {"R001": "safety", "R002": "tools", "R003": "content", "R004": "extract"}
OTHER_LANGS = ["de", "es", "fr", "tr", "uk", "kk", "pt", "it", "pl", "zh", "ja"]
LANG_NAMES = {"ru": "русский", "en": "английский", "de": "немецкий", "es": "испанский", "fr": "французский", "tr": "турецкий",
              "uk": "украинский", "kk": "казахский", "pt": "португальский", "it": "итальянский", "pl": "польский", "zh": "китайский",
              "ja": "японский"}
# length ranges in words: (min, max)
LEN = {
    "R001": {"short": (2, 15), "context": (20, 80), "long": (150, 400)},
    "R002": {"short": (3, 30)},
    "R003": {"short": (15, 45), "medium": (60, 150), "long": (300, 600)},
    "R004": {"short": (20, 80), "medium": (80, 400), "long": (400, 1500)},
}
KINDS = {"R001": ("normal", "long"), "R002": ("small",), "R003": ("normal", "long"), "R004": ("normal", "long")}


class Planner:
    """Fills the per-record plan of one batch."""

    def __init__(self, rng: random.Random, families: List[Tuple[str, str]], langs: List[str]):
        self.rng, self.families, self.langs = rng, families, langs
        self.rows: List[Dict[str, Any]] = []
        self.groups = 0
        self.units = 0                                       # one family per unit: a group (pair) or a single row

    def lang(self) -> str:
        x = self.rng.random()
        return "ru" if x < 0.45 else "en" if x < 0.80 else self.rng.choice(self.langs)

    def difficulty(self) -> str:
        x = self.rng.random()
        return "easy" if x < 0.25 else "medium" if x < 0.70 else "hard"

    def family(self) -> str:
        """The next family in rotation. Called once per unit: rows of a pair share the family their group drew (counting rows
        instead would give a pair-only batch just every other family)."""
        f = self.families[self.units % len(self.families)][0]
        self.units += 1
        return f

    def group(self) -> str:
        self.groups += 1
        return f"g{self.groups:02d}"

    def add(self, **row: Any) -> Dict[str, Any]:
        row.setdefault("family", self.family())
        row.setdefault("lang", self.lang())
        row.setdefault("difficulty", self.difficulty())
        row.setdefault("group", "")
        self.rows.append(row)
        return row


# "A minor setting without an explicit request" was read as "the user asked to look, the agent changed a setting" in 12 of 24
# records of rounds 14-19 (`wrangler deploy` for "check the status") - a "no", not a 0.5. The setting is now what was asked for.
SOFT_REASONS = ("условие из слов пользователя в тексте не проверено", "размытая жалоба без условия («что-то зависло»)",
                "неизвестно, нужны ли ещё старые данные (логи, кэш, временные файлы)",
                "просили настроить в общих словах, значение выбрал агент (второстепенная обратимая настройка)")


# What kind of fact decides a fact pair. Asked in words for "a fact, not an assessment", the teacher made it "production or test" in
# 85 of 108 fact pairs of rounds 15-17 - one axis, learnt as a word ("prod" -> no), not as reading the situation.
FACT_KINDS = ("есть ли другая копия (единственная копия / свежая проверенная копия)",
              "несохранённая работа (незакоммиченные правки, незавершённая выгрузка / всё сохранено и отправлено)",
              "кто сейчас пользуется (активные сессии, процесс держит файл / никого)",
              "чьё это (общий ресурс других команд / своя временная папка)",
              "сколько затронет (один объект / тысячи объектов, весь каталог)",
              "что показал предыдущий шаг (вывод команды: проверка прошла / нашла ошибки)",
              "что идёт сейчас (миграция, сборка, запись в процессе / всё завершено)",
              "боевая или тестовая среда")


def plan_r001(p: Planner, kind: str) -> None:
    lens = LEN["R001"]
    facts = list(FACT_KINDS)
    p.rng.shuffle(facts)
    if kind == "long":
        # Long twins: the same request and action in both, the answer decided by ONE fact placed in the middle of the context.
        # Round 6 long records were decided by the last two lines alone (request vs. a flagrant opposite action), so the context
        # was filler - and reading the middle of a long state is exactly where the model fails (s70: ~0.55 at 16-32k).
        for _ in range(4):
            g, fam, lang = p.group(), p.family(), p.lang()
            fact = facts.pop()
            for target in p.rng.sample(["да", "нет"], 2):
                p.add(ptype="noul", length=lens["long"], answer=target, group=g, family=fam, lang=lang, twist="факт в середине контекста",
                      words=True, fact=fact)
        return
    # Twins: two outcomes for almost the same situation. WHAT differs is planned too - the X3S exam (2026-10-18) showed the model
    # cannot tell a command from the request around it, so half the pairs change the user's words or the scope, not the command.
    # Shares follow what the model fails (scripts/leeth3_probe.py, 2026-10-20, pilot-100 on unseen batches): twins differing in the
    # command 0.90, in the user's words 0.89, in scope 0.69, in mode (plan/apply, -n/-f, -WhatIf) 0.59 - chance. Mode up, command down.
    # "в условии": the same conditional request and action, the context shows the condition met in one record and not in the other.
    # Up to round 13 a condition in the user's words stood in 56 % of the "спорно" records and in 3 % of the yes/no ones: "if" alone
    # gave the answer. The words twins (0.89 on the probe) give two of their pairs to it.
    # "в факте": the same words and action, ONE short line of the situation differs (the only copy / a fresh temporary copy). The
    # pilots (2026-10-22) do not use such a fact at all - not in the middle of a long context, not right before the action, not
    # with the rest of the context removed: the skill is missing, not the reach. Short pairs teach it; the command twins (0.92) pay.
    twists = (["в команде"] * 1 + ["в словах пользователя"] * 3 + ["в масштабе"] * 3 + ["в режиме"] * 5 + ["в условии"] * 2
              + ["в факте"] * 2)
    p.rng.shuffle(twists)
    for twist in twists:
        g, fam, lang = p.group(), p.family(), p.lang()
        ln = lens["context"] if twist in ("в словах пользователя", "в условии", "в факте") or p.rng.random() < 0.3 else lens["short"]
        extra = {"fact": facts.pop()} if twist == "в факте" else {}
        for target in p.rng.sample(["да", "нет"], 2):
            p.add(ptype="noul", length=ln, answer=target, group=g, family=fam, lang=lang, twist=twist, words=True, **extra)
    # Whether the user's words are there is planned per row (the ingest checks it): asked for "at least two thirds with the user's
    # words, about 10 % the action alone", the teacher wrote no bare action in rounds 11-15 (0 of 1216) and then 99 of 288 in
    # round 16 - every short row, twins included. Now the yes/no single rows (4 of 40) are the bare actions, everything else has words.
    # Why each soft record is soft is planned too: asked in words, the teacher went back to "if ..." for 17 of 24 soft records in
    # round 15 (yes/no: 12 of 132 each), and a condition marked "спорно" again.
    reasons = list(SOFT_REASONS)
    p.rng.shuffle(reasons)
    for _ in range(4):
        p.add(ptype="noul", length=lens["short"] if p.rng.random() < 0.6 else lens["context"], answer="спорно", words=True,
              why=reasons.pop())
    for target in ["да", "нет", "да", "нет"]:
        p.add(ptype="noul", length=lens["short"], answer=target, words=False)


def plan_r002(p: Planner, kind: str) -> None:
    # Paraphrase groups: one need said three ways over the SAME tool set, one answer. The independent exam (2026-10-16) found the
    # same request in other words keeps its answer only 67 % of the time - the model keys on wording, not on the need.
    units: List[Any] = ["null"] * 4 + ["два шага"] * 3 + ["инструмент"] * 12 + ["перефразы"] * 2
    p.rng.shuffle(units)
    for a in units:
        if a == "перефразы":
            g, fam, lang, k = p.group(), p.family(), p.lang(), p.rng.randint(6, 24)
            for _ in range(3):
                p.add(ptype="choice", length=LEN["R002"]["short"], options=k, answer="инструмент", group=g, family=fam, lang=lang,
                      twist="перефраз")
        else:
            # Up to 40 real tools: every option written here is also a distractor of the big catalogs built from the corpus.
            k = p.rng.randint(4, 24) if p.rng.random() < 0.7 else p.rng.randint(25, 40)
            p.add(ptype="choice", length=LEN["R002"]["short"], options=k, answer=a)


PAIR_METHODS = ("два события", "без слов текста")


def plan_r003(p: Planner, kind: str) -> None:
    lens = LEN["R003"]
    pairs = 4 if kind == "long" else 14
    # Both questions of a pair share one style, and the refusal / negative event sits in the YES question half the time. B0087-B0088
    # wrote every YES question as an abstract positive criterion and every NO question as a concrete reversal (banned, demolished,
    # refused): the style alone gave the answer (question-only 0.62). Fixing it per pair removes the cue by construction.
    # HOW the pair is built is planned too. Rounds 7-8 (B0172/B0173, B0205/B0206) kept the subject but copied the text's verb into the
    # YES question and put its antonym, absent from the text, into the NO question ("отменили" / "обеспечили"): the word-overlap rule
    # answered 0.68-0.89. The batches that passed used a text with two events and one verb about two subjects.
    styles = [(s, neg, m) for s in ("оба обобщённые", "оба конкретные") for neg in ("да", "нет") for m in PAIR_METHODS] * (pairs // 8 + 1)
    p.rng.shuffle(styles)
    for style, neg, method in styles[:pairs]:                  # one text, a YES question and a NO question about the same subject
        g, fam, lang = p.group(), p.family(), p.lang()
        ln = lens["long"] if kind == "long" else (lens["short"] if p.rng.random() < 0.4 else lens["medium"])
        pair_style = f"{style}; отказ или отрицательное событие — в вопросе «{neg}»"
        for target in p.rng.sample(["да", "нет"], 2):
            p.add(ptype="noul", length=ln, answer=target, group=g, family=fam, lang=lang, pair_style=pair_style, pair_method=method)
    if kind == "long":
        return
    for _ in range(8):
        k = p.rng.randint(3, 8)
        p.add(ptype="multi", length=lens["short"] if p.rng.random() < 0.4 else lens["medium"], options=k,
              answer=f"{p.rng.randint(0, k)} из {k}")
    for _ in range(4):
        lv = p.rng.randint(3, 5)
        p.add(ptype="score", length=lens["medium"], levels=lv, answer=f"{p.rng.randint(1, lv)} из {lv}")


def plan_r004(p: Planner, kind: str) -> None:
    lens = LEN["R004"]
    n = 8 if kind == "long" else 30
    counts = ["0"] * round(n * 0.15) + ["1"] * round(n * 0.35)
    counts += [str(p.rng.randint(2, 8)) for _ in range(n - len(counts))]
    p.rng.shuffle(counts)
    for c in counts:
        ln = lens["long"] if kind == "long" else (lens["short"] if p.rng.random() < 0.5 else lens["medium"])
        p.add(ptype="extract", length=ln, answer=c)


PLANNERS = {"R001": plan_r001, "R002": plan_r002, "R003": plan_r003, "R004": plan_r004}
FAMILIES_PER_BATCH = {"R001": 4, "R002": 5, "R003": 4, "R004": 3}


def next_number(data: Path) -> int:
    nums = [int(m.group(1)) for d in (data / "queue", data / "queue" / "done", data / "incoming", data / "plans")
            if d.exists() for p in d.iterdir() if (m := re.match(r"B(\d{4})", p.name))]
    return max(nums, default=0) + 1


def spec_cell(row: Dict[str, Any]) -> str:
    parts = []
    if row.get("twist") == "перефраз":
        parts.append("перефраз (группа)")
    elif "twist" in row:
        parts.append(f"близнец: {row['twist']}")
    if "pair_style" in row:
        parts.append(f"пара: {row['pair_style']}")
    if "pair_method" in row:
        parts.append(f"приём: {row['pair_method']}")
    if "options" in row:
        parts.append(f"вариантов {row['options']}")
    if "levels" in row:
        parts.append(f"levels {row['levels']}")
    if row["answer"] == "два шага":
        # "ответ: два шага" was read as "the answer is two tools": all 42 such records of round 24 had a list answer.
        parts.append("запрос из двух шагов, ответ — **один** инструмент первого шага, если второй шаг лишь сообщает его результат, "
                     "иначе `null`; не список")
    else:
        parts.append(f"ответ: {row['answer']}")
    if "why" in row:
        parts.append(f"почему спорно: {row['why']}")
    if "fact" in row:
        parts.append(f"вид факта: {row['fact']} — команда та, чей исход от него зависит")
    if "words" in row:
        parts.append("слова пользователя: есть" if row["words"] else "без слов пользователя — только действие")
    return ", ".join(parts)


def batch_text(name: str, req: str, kind: str, fams: List[Tuple[str, str]], rows: List[Dict[str, Any]], avoid: Sequence[str] = ()) -> str:
    fam_lines = "\n".join(f"- `{c}` — {d}" for c, d in fams)
    avoid_text = "" if req != "R001" else (
        "\n## Программы\n\n**Каждая программа — не больше одного раза в пакете:** одна пара близнецов или одна одиночная запись. "
        "Приём отклоняет вторую и следующие пары с той же программой.\n" + ("" if not avoid else
        "\nЭти программы в корпусе уже встречаются слишком часто, в этом пакете действие агента их **не запускает** (приём "
        "отклоняет такую пару целиком): " + ", ".join(f"`{p}`" for p in avoid) + ". Бери другие программы той же области — "
        "менее известные утилиты, их подкоманды, командлеты PowerShell, вызовы API облаков.\n"))
    def length_cell(r: Dict[str, Any]) -> str:
        lo, hi = r["length"]
        if r["lang"] in ("zh", "ja"):            # no spaces between words: the ingest counts ~2 characters per word
            return f"{2 * lo}–{2 * hi} иероглифов, без пробелов между словами"
        if lo >= 300:                            # rounds 22-24 wrote ~250 words for "400-1500": a count of paragraphs is easier to keep
            return f"{lo}–{hi} (не меньше {lo // 50} абзацев по ~60 слов)"
        return f"{lo}–{hi}"

    table = "\n".join(
        f"| {r['id']} | {r['group'] or '—'} | {r['family']} | {r['lang']} | {r['ptype']} | {length_cell(r)} | "
        f"{r['difficulty']} | {spec_cell(r)} |" for r in rows)
    # The fields are named in the batch itself, not only in FORMAT.md: late in a long session the teacher wrote whole batches
    # without `request`, `domain`, `teacher` (rounds 19 and 23: 16 and 150 records refused).
    extra = {"R002": ', `"abstain_ok": true`', "R003": ", у `score` ещё `levels`"}.get(req, "")
    return f"""# Пакет {name} ({req}{', ' + kind if kind else ''})

- Задание: **{req}** — прочитай `requests/{REQUEST_FILES[req]}`, `FORMAT.md`, `AGENT.md`.
- Сколько примеров: **{len(rows)}** — ровно по строкам плана ниже, в том же порядке, с теми же `id`.
- Результат: `incoming/{name}.jsonl`. После записи и самопроверки перенеси этот файл в `queue/done/`.
- **Каждая строка — все поля:** `id`, `"request": "{req}"`, `family`, `"domain": "{REQUEST_DOMAINS[req]}"`, `lang`, `ptype`,
  `state`, `question`, `answer` (и `options` у `choice`/`multi`), `difficulty`, `"teacher": "gemini-3.8-flash"`, `group`{extra}.
  Строка без поля отклоняется целиком.

## Подтемы

{fam_lines}
{avoid_text}
## План (обязателен построчно)

Длина — число слов в `state`: не короче и не длиннее указанного. «Ответ» — какой ответ должен получиться у этого примера:
пиши пример так, чтобы правильный ответ был именно таким. Одинаковое значение в колонке «группа» — записи связаны (см. задание).

| id | группа | family | lang | ptype | длина (слов) | сложность | параметры |
|---|---|---|---|---|---|---|---|
{table}
"""


def overused_programs(data: Path, n: int = 25) -> List[str]:
    """The most frequent programs of the accepted safety records (src/leeth3/programs.py); none in an empty corpus."""
    path = data / "accepted" / "records.jsonl"
    if not path.exists():
        return []
    rows = (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    return top_programs([d["state"] for d in rows if d.get("domain") == "safety"], n)


def make(data: Path, req: str, batches: int, seed: int, kind: str = "", families: Sequence[str] = ()) -> List[Path]:
    """``families`` narrows the rotation to those train families - a round aimed at the cells an exam found weak."""
    check_pools()
    kind = kind or KINDS[req][0]
    if kind not in KINDS[req]:
        raise SystemExit(f"{req}: unknown kind {kind!r}; known: {KINDS[req]}")
    pool = list(TRAIN[req])
    if families:
        unknown = sorted(set(families) - {c for c, _ in pool})
        if unknown:
            raise SystemExit(f"{req}: not train families: {unknown}")
        pool = [f for f in pool if f[0] in set(families)]
    rng = random.Random(f"{req}/{kind}/{seed}")
    avoid = overused_programs(data) if req == "R001" else []
    rng.shuffle(pool)
    for d in ("queue", "queue/done", "incoming", "plans"):
        (data / d).mkdir(parents=True, exist_ok=True)
    num = next_number(data)
    k = FAMILIES_PER_BATCH[req] if kind != "long" else 2
    out: List[Path] = []
    for b in range(batches):
        fams = [pool[(b * k + i) % len(pool)] for i in range(k)]
        name = f"B{num + b:04d}"
        p = Planner(rng, fams, rng.sample(OTHER_LANGS, 2))
        PLANNERS[req](p, kind)
        for i, row in enumerate(p.rows, 1):
            row["id"] = f"{name}-{i:03d}"
        (data / "queue" / f"{name}.md").write_text(batch_text(name, req, kind, fams, p.rows, avoid), encoding="utf-8")
        plan = {"batch": name, "request": req, "kind": kind, "rows": p.rows}
        if avoid:
            plan["avoid_programs"] = avoid
        if req == "R001":
            plan["max_units_per_program"] = 1          # one pair (or one single record) per program; the ingest refuses the rest
        (data / "plans" / f"{name}.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
        out.append(data / "queue" / f"{name}.md")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, choices=sorted(TRAIN))
    ap.add_argument("--kind", default="")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--families", default="", help="comma-separated train family codes to rotate over (default: all)")
    a = ap.parse_args(argv)
    fams = [f.strip() for f in a.families.split(",") if f.strip()]
    for p in make(Path(a.data), a.request, a.batches, a.seed, a.kind, fams):
        print(p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
