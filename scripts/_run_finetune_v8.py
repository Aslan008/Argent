"""Run finetune v10: feedback + rebalanced compound injection + hand-written
destructive UNFAMILIAR rescue + clean browser_open contrast set.

v9 lesson: stratified 15-states/tool let the UNFAMILIAR class outvote
calculate at the meta+math boundary (conf 0.99 -> 0.54), and the corpus
UNFAMILIAR family has almost no destructive queries (they live in the
security family with a different question type), so the rescue was thin.
v10: calculate 40 states, others 20; all compound negatives; hand-written
destructive UNFAMILIAR set; browser contrast incl. bare "открой {url}".
"""
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.leeth.dataset_generator import LeethDatasetGenerator, LeethSample
from src.leeth.finetune import finetune_from_feedback

DATA = Path(__file__).resolve().parent.parent / "src/leeth/data"

CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR"
]
WORKDIRS = ["C:/Users/user/Desktop/Argent", "D:/Projects/GameServer", "/home/user/repo"]
CHOICE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
NOUL_Q = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"
EXTRACT_Q = "Извлеките ключевой целевой параметр (путь к файлу, запрос или команду)"


def make_samples(query, target, noul, param=None, workdirs=None):
    out = []
    for wd in (workdirs or WORKDIRS):
        state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: {wd}'
        out.append(LeethSample(state=state, question_type="choice", question=CHOICE_Q,
                               choices=CHOICES, target_choice=target))
        out.append(LeethSample(state=state, question_type="noul", question=NOUL_Q,
                               target_noul=noul))
        if param is not None:
            out.append(LeethSample(state=state, question_type="extract", question=EXTRACT_Q,
                                   target_span_text=param))
    return out


# ── 1. Compound curriculum (stratified, calculate boosted) ───────────────────
samples = []
with open(DATA / "leeth_compound_preview.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        samples.append(LeethSample(**json.loads(line)))

by_state = {}
for s in samples:
    by_state.setdefault(s.state, []).append(s)


def state_target(ss):
    for s in ss:
        if s.question_type == "choice":
            return s.target_choice
    return None


neg_states = [st for st, ss in by_state.items() if state_target(ss) in ("NO_TOOL", "UNFAMILIAR")]
pos_by_tool = defaultdict(list)
for st, ss in by_state.items():
    t = state_target(ss)
    if t not in ("NO_TOOL", "UNFAMILIAR", None):
        pos_by_tool[t].append(st)

rng = random.Random(7)
injection = []
for st in neg_states:                        # ALL hard negatives (boundary)
    injection.extend(by_state[st])
PER_TOOL = {"calculate": 60, "read_file": 20, "run_command": 20,
            "find_by_name": 20, "grep_search": 20, "list_directory": 20,
            "browser_open": 20}
for tool, cap in PER_TOOL.items():
    for st in rng.sample(pos_by_tool[tool], min(cap, len(pos_by_tool[tool]))):
        injection.extend(by_state[st])

# ── 2. Hand-written destructive UNFAMILIAR rescue ────────────────────────────
DESTRUCTIVE_QUERIES = [
    "удали папку build", "удали файл test.txt", "удали всё содержимое папки logs",
    "очисти папку dist", "rm -rf node_modules", "rm -rf /",
    "снеси базу данных", "форматни диск D", "убей процесс на порту 8080",
    "сотри историю git", "del /f /s /q temp", "стереть все логи",
]
rescue = []
for q in DESTRUCTIVE_QUERIES:
    rescue.extend(make_samples(q, "UNFAMILIAR", 1.0))

# ── 3. Clean browser_open contrast set (incl. bare "открой {url}") ──────────
BROWSER_URLS = ["example.com", "docs.python.org", "github.com", "habr.com",
                "stackoverflow.com", "google.com", "localhost:3000", "gitlab.com"]
browser = []
for u in BROWSER_URLS:
    for tpl in ["зайди на {u}", "открой сайт {u}", "открой {u}"]:
        q = tpl.format(u=u)
        browser.extend(make_samples(q, "browser_open", 1.0, param=u,
                                    workdirs=rng.sample(WORKDIRS, 2)))
# ── 4. Conversational destructive-mention negatives (imperative vs question) ─
CONVERSATIONAL_DESTRUCTIVE = [
    "Расскажи, как устроена команда rm -rf в операционных системах Linux",
    "Объясни, чем опасна команда rm -rf и когда её применяют",
    "Как удалить папку в Windows через командную строку?",
    "Что делает команда del /f /s /q?",
]
conversational_neg = []
for q in CONVERSATIONAL_DESTRUCTIVE:
    conversational_neg.extend(make_samples(q, "NO_TOOL", 0.0,
                                           workdirs=rng.sample(WORKDIRS, 2)))
# ── 4b. Conceptual-question negatives (NO_TOOL vs UNFAMILIAR boundary) ───────
CONCEPTUAL_QUESTIONS = [
    "Объясни мне, чем отличается async/await от Task в C#",
    "Объясни, чем отличается REST от GraphQL",
    "Расскажи, как работает garbage collector в .NET",
    "Что такое event loop в Node.js?",
]
conceptual_neg = []
for q in CONCEPTUAL_QUESTIONS:
    conceptual_neg.extend(make_samples(q, "NO_TOOL", 0.0,
                                       workdirs=rng.sample(WORKDIRS, 3)))

# ── 4c. Conversational family subset (NO_TOOL mass balance) ─────────────────
conv_family = LeethDatasetGenerator(seed=321).generate_conversational_negative_samples()
conv_by_state = {}
for s in conv_family:
    conv_by_state.setdefault(s.state, []).append(s)
conv_states = rng.sample(sorted(conv_by_state), min(25, len(conv_by_state)))
conv_subset = []
for st in conv_states:
    conv_subset.extend(conv_by_state[st])

# ── 5. Signature pattern: chatter + transition + math (primary use case) ─────
SIGNATURE_META = [
    "pytest крутит другой агент, не обращай на него внимания. А теперь проверька.",
    "параллельно идёт сборка, её не трогай. А теперь:",
]
signature = []
for e in ["2 + 2 * 22", "128 * 64", "1024 / 8", "2 ** 10",
          "512 + 512", "60 * 60 * 24", "100 - 37", "25000 / 32"]:
    for meta in SIGNATURE_META:
        q = f"{meta} Сколько будет {e}?"
        signature.extend(make_samples(q, "calculate", 1.0, param=e,
                                      workdirs=rng.sample(WORKDIRS, 2)))

injection.extend(rescue)
injection.extend(browser)
injection.extend(conversational_neg)
injection.extend(conceptual_neg)
injection.extend(conv_subset)
injection.extend(signature)

# ── 6. Multi-intent decomposition curriculum ────────────────────────────────
# count=2 pairs + signature read+calc pairs + count=1 negatives (score head)
# + first_intent spans (extract head). The choice contract is unchanged
# (multi-intent -> UNFAMILIAR); the decomposition branch reads the NEW heads.
mi_samples = LeethDatasetGenerator(seed=777).generate_multiintent_samples()
injection.extend(mi_samples)

n_neg = sum(1 for s in injection if s.question_type == "choice" and s.target_choice in ("NO_TOOL", "UNFAMILIAR"))
logging.info("Injection: %d samples (%d neg-choice, %d destructive-rescue, %d browser) | per-tool states: %s",
             len(injection), n_neg, len(rescue), len(browser),
             {t: min(c, len(pos_by_tool[t])) for t, c in PER_TOOL.items()})

result = finetune_from_feedback(replay_ratio=1.0, extra_replay=injection)
print(json.dumps(result, ensure_ascii=False, indent=2, default=str))