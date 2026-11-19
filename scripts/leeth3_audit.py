"""Audit of a teacher round BEYOND the ingest gates - the checks that found the leaks of rounds 2-5 (2026-10-19).

    python scripts/leeth3_audit.py FIRST LAST        # batch numbers, e.g. 119 150

Prints numbers and record ids only. What to look for: question-only accuracy ~0.5 (R003), no hint phrases in safety states,
"words" twins sharing the command, paraphrase groups with little shared wording, planned empty answers. Needs scikit-learn.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import leeth3_import_x3s as x
import leeth3_ingest as ing

first, last = int(sys.argv[1]), int(sys.argv[2])
data = ROOT / "leeth3_data"
new = []
plans = {}
unreadable = {}                                          # the ingest refuses these lines; the audit measures the rest
for b in range(first, last + 1):
    name = f"B{b:04d}"
    plans.update(ing.load_plan(data, name) or {})
    for line in (data / "incoming" / f"{name}.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            try:
                new.append(ing.neutralise(ing.parse_record(line)))
            except ing.RecordError:
                unreadable[name] = unreadable.get(name, 0) + 1
if unreadable:
    print(f"unreadable records left out: {unreadable}")
new_ids = {r.id for r in new}                            # an already-ingested round must not be counted twice
acc = [r for r in (ing.parse_record(l) for l in (data / "accepted" / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
       if r.id not in new_ids]


def leak(rows, field):
    rows = [r for r in rows if r.ptype == "noul" and (r.answer >= 0.7 or r.answer <= 0.3)]
    y = np.array([int(r.answer >= 0.7) for r in rows])
    g = [r.id[:5] for r in rows]
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2).fit_transform([getattr(r, field) for r in rows])
    return y, g, cross_val_predict(LogisticRegression(max_iter=2000, C=3), X, y, cv=GroupKFold(min(8, len(set(g)))), groups=g)


print("=== R001 safety")
from src.leeth3.hints import verdict_in_state  
s_new = [r for r in new if r.domain == "safety"]
print("verdict in state (the ingest refuses these):", [r.id for r in s_new if verdict_in_state(r.state)][:10])
pats = {"hint": x.HINT_WORDS, "narration": x.NARRATION, "meta": x.META_HINT}
# The user's words: a speaker label with a colon or a quote after it - at a line start (rounds 1-8) or inside a one-line state
# ("... Пользователь спросил: «...» Агент выполняет: ...", round 9).
user = re.compile(r"(?i)(пользовател|user|користувач|utilisateur|usuario|benutzer|użytkownik|kullanıcı|қолданушы|пайдаланушы|utente|usuário|"
                  r"ユーザー|用户|разработчик|инженер|developer|engineer|admin|оператор|operator)[^\n:：«\"]{0,60}[:：«\"]")
hits = collections.Counter()
ex = []
for r in s_new:
    for k, p in pats.items():
        for m in p.finditer(r.state):
            line = r.state[r.state.rfind("\n", 0, m.start()) + 1:m.start()]
            if m.group(0).lower().startswith("безопасност"):             # the noun names an object ("журнал безопасности")
                continue
            if user.search(line):                                         # inside the user's own words: allowed by the request
                continue
            hits[k] += 1
            ex.append((k, r.id, m.group(0)))
            break
print("hint phrases in state (outside the user's words):", dict(hits), ex[:10])
print("states with user words:", round(np.mean([bool(user.search(r.state)) for r in s_new]), 2), "of", len(s_new),
      "| one-line states:", sum("\n" not in r.state.strip() for r in s_new))
tw = collections.defaultdict(list)
for r in s_new:
    t = plans.get(r.id, {}).get("twist")
    if t == "в словах пользователя":
        tw[(r.id[:5], r.group)].append(r)
action = lambda st: st.strip().splitlines()[-1].rsplit(": ", 1)[-1].strip()    # noqa: E731  (the text after the last "runs: ")
same_cmd = sum(len({action(r.state) for r in rs}) == 1 for rs in tw.values())
print("'words' twins with the same action:", same_cmd, "of", len(tw))
# A prohibition in the user's words ("Не трогай…", "Never…") answers "no" by itself (round 12: 22 of 30 "no" twins). Rough, ru/uk/en.
ban = re.compile(r"(?i)(\bни в коем\b|\bзапрещ|\bне (трогай|удаляй|запускай|выкатывай|отправляй|сбрасывай|меняй|закрывай|нажимай|чіпай)|"
                 r"\bdo not\b|\bdon't\b|\bnever\b|\bat all costs\b|\bunder no circumstances\b)")
user_line = lambda st: " ".join(l for l in st.splitlines() if user.search(l))  # noqa: E731
by_ans = {k: [r for r in s_new if user.search(r.state) and (r.answer >= 0.7 if k == "yes" else r.answer <= 0.3)] for k in ("yes", "no")}
# A condition in the user's words ("если…", "if…") must not mark "спорно" by itself (rounds 1-12: 56 % of soft records, 3 % of yes/no).
cond = re.compile(r"(?i)(\bесли\b|\bif\b|\bfalls\b|\bse\b|\bsi\b|\bjeśli\b|\beğer\b|\bякщо\b|もし|如果|қажет болса|\bпри необходимости\b)")
soft = [r for r in s_new if user.search(r.state) and 0.3 < r.answer < 0.7]
print("condition in the user's words: yes", sum(bool(cond.search(user_line(r.state))) for r in by_ans["yes"]), "of", len(by_ans["yes"]),
      "| no", sum(bool(cond.search(user_line(r.state))) for r in by_ans["no"]), "of", len(by_ans["no"]),
      "| soft", sum(bool(cond.search(user_line(r.state))) for r in soft), "of", len(soft))
print("prohibition in the user's words: no", sum(bool(ban.search(user_line(r.state))) for r in by_ans["no"]), "of", len(by_ans["no"]),
      "| yes", sum(bool(ban.search(user_line(r.state))) for r in by_ans["yes"]), "of", len(by_ans["yes"]))
from src.leeth3.programs import action_program  
prog = collections.Counter()
for r in s_new:
    if action_program(r.state):
        prog[(r.id[:5], action_program(r.state))] += 1
per_batch = collections.Counter(b for b, _ in prog)
print("distinct programs per batch (pairs ~20):", round(len(prog) / max(1, len(per_batch)), 1),
      "| records of a program repeated > 2 times in its batch:", sum(c for c in prog.values() if c > 2))
# Long twins "факт в середине контекста": ONE fact differs, in the middle. Round 8 kept to it (1 line), round 12 drifted to ~4
# lines near the end - no shortcut, but no longer a test of reading the middle.
long_tw = collections.defaultdict(list)
for r in s_new:
    if plans.get(r.id, {}).get("twist") == "факт в середине контекста":
        long_tw[(r.id[:5], r.group)].append(r)
mid = []
for rs in long_tw.values():
    if len(rs) != 2:
        continue
    a_, b_ = rs[0].state.splitlines(), rs[1].state.splitlines()
    d = [i for i, (p, q) in enumerate(zip(a_, b_)) if p != q]
    mid.append((len(d) + abs(len(a_) - len(b_)), max(d) / max(1, len(a_) - 1) if d else 0.0))
if mid:
    print("long twins:", len(mid), "| one or two differing lines:", sum(n <= 2 for n, _ in mid),
          "| a differing line after 85 % of the text:", sum(p > 0.85 for _, p in mid))
pool = [r for r in acc if r.domain == "safety"] + s_new
y, g, pred = leak(pool, "state")
isnew = np.array([r.id[:5] >= f"B{first:04d}" for r in pool if r.ptype == "noul" and (r.answer >= 0.7 or r.answer <= 0.3)])
print("state-only acc (sanity, should be high): new", round((pred[isnew] == y[isnew]).mean(), 3))

print("=== R002 tools")
t_new = [r for r in new if r.domain == "tools"]
groups = collections.defaultdict(list)
for r in t_new:
    if plans.get(r.id, {}).get("twist") == "перефраз":
        groups[(r.id[:5], r.group)].append(r)
st = lambda w: w.lower()[:5]  # noqa: E731
W = lambda s: {st(w) for w in re.findall(r"\w{3,}", s)}  # noqa: E731
jac = [len(W(a.state) & W(b.state)) / max(1, len(W(a.state) | W(b.state))) for rs in groups.values() for i, a in enumerate(rs) for b in rs[i + 1:]]
print("paraphrase groups:", len(groups), "mean jaccard", round(float(np.mean(jac)), 2) if jac else None)
nulls = sum(r.answer is None for r in t_new)
print("null answers:", nulls, "of", len(t_new))
name_in_req = sum(1 for r in t_new if r.answer and r.answer.lower().replace("_", " ") in r.state.lower().replace("_", " "))
print("tool name written in the request:", name_in_req)
from src.leeth3.padding import padding_options  
pad = [(r.id, len(padding_options(r.options, [r.answer] if r.answer else []))) for r in t_new]
print("template filler options (the ingest removes them):", sum(n for _, n in pad), [i for i, n in pad if n][:10])

print("=== R003 content")
c_new = [r for r in new if r.domain == "content" and r.ptype == "noul"]
pool = [r for r in acc if r.domain == "content"] + [r for r in new if r.domain == "content"]
y, g, pred = leak(pool, "question")
isnew = np.array([r.id[:5] >= f"B{first:04d}" for r in pool if r.ptype == "noul" and (r.answer >= 0.7 or r.answer <= 0.3)])
print("question-only acc: new", round((pred[isnew] == y[isnew]).mean(), 3), "n", int(isnew.sum()), "| old", round((pred[~isnew] == y[~isnew]).mean(), 3))
print("pairs of differing form:", len(ing.pairs_of_differing_form(c_new)))
print("overlap rule on new pairs:", ing.overlap_rule_accuracy(c_new))
neg = {}
for r in c_new:
    ps = plans.get(r.id, {}).get("pair_style", "")
    neg.setdefault(ps.split("«")[-1].rstrip("»") if ps else "-", []).append(r)
print("pairs by planned negative side:", {k: len(v) // 2 for k, v in neg.items()})

print("=== R004 extract")
e_new = [r for r in new if r.domain == "extract"]
print("empty answers:", sum(len(r.answer) == 0 for r in e_new), "of", len(e_new),
      "| planned 0:", sum(1 for r in e_new if str(plans.get(r.id, {}).get("answer")) == "0"))
