"""A/B: chat routing on v6 (pre-UNFAMILIAR) vs v7 (current) checkpoints."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.leeth.model import LeethModel
from src.leeth.runtime import LeethRuntime
from src.leeth.primitives import ChoiceQuestion, NoulQuestion

DATA = Path("src/leeth/data")
CHOICES_14 = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL"
]

QUESTIONS = [
    ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=CHOICES_14),
    NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
]

PROMPTS = [
    "Расскажи, как устроена команда rm -rf в операционных системах Linux",
    "Объясни мне, чем отличается async/await от Task в C#",
    "Привет! Как твои дела и чем ты можешь мне помочь?",
]


def probe(ckpt_name, label):
    """Build a temp model_dir with the given checkpoint as leeth_best.pt."""
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix=f"leeth_ab_{label}_"))
    shutil.copy(DATA / ckpt_name, tmp / "leeth_best.pt")
    for fname in ["leeth_vocab.json", "choice_classes.json"]:
        src = DATA / fname
        if src.exists():
            shutil.copy(src, tmp / fname)
    rt = LeethRuntime(model_dir=str(tmp))
    out = []
    for q in PROMPTS:
        state = f'Incoming user request in Argent session:\n{q}\nWorking directory: C:/Users/user/Desktop/Argent'
        res = rt.decide(state, QUESTIONS)
        out.append((res.results["tool"].choice, round(res.results["tool"].confidence, 3)))
    return out


print("v6 (pre-UNFAMILIAR):")
for q, (choice, conf) in zip(PROMPTS, probe("leeth_best_v6.pt", "v6")):
    print(f"  {choice:14s} conf={conf}  {q[:60]}")
print("v7 (current best):")
for q, (choice, conf) in zip(PROMPTS, probe("leeth_best.pt", "v7")):
    print(f"  {choice:14s} conf={conf}  {q[:60]}")