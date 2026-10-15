"""A/B: destructive + browser routing on v7 vs v8 checkpoints."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.leeth.runtime import LeethRuntime
from src.leeth.primitives import ChoiceQuestion, NoulQuestion

DATA = Path("src/leeth/data")
CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR"
]
QUESTIONS = [
    NoulQuestion(id="need_tool", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
    ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?", choices=CHOICES),
]

PROMPTS = [
    "удали папку build",
    "удали файл test.txt",
    "очисти папку logs",
    "rm -rf node_modules",
    "снеси базу данных",
    "открой example.com",
    "зайди на docs.python.org",
    "найди файл main.py",
]


def probe(ckpt_name, label):
    tmp = Path(tempfile.mkdtemp(prefix=f"leeth_ab_{label}_"))
    shutil.copy(DATA / ckpt_name, tmp / "leeth_best.pt")
    for fname in ["leeth_vocab.json", "leeth_merges.json", "choice_classes.json"]:
        src = DATA / fname
        if src.exists():
            shutil.copy(src, tmp / fname)
    rt = LeethRuntime(model_dir=str(tmp))
    out = []
    for q in PROMPTS:
        state = f'Incoming user request in Argent session:\n"{q}"\nWorking directory: C:/Users/user/Desktop/Argent'
        res = rt.decide(state, QUESTIONS)
        tool = res.results["tool"]
        need = res.results["need_tool"]
        out.append((tool.choice, round(tool.confidence, 3), round(need.probability, 3)))
    return out


v7 = probe("leeth_best_v7.pt", "v7")
v8 = probe("leeth_best.pt", "v8")
print(f"{'query':30s} | {'v7':32s} | v8")
print("-" * 100)
for q, a, b in zip(PROMPTS, v7, v8):
    print(f"{q:30s} | {a[0]:12s} conf={a[1]:<6} need={a[2]:<5} | {b[0]:12s} conf={b[1]:<6} need={b[2]:<5}")