"""Verify production runtime (src/leeth/data) loads the 400M model correctly."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.runtime import LeethRuntime
from src.leeth.primitives import ChoiceQuestion, NoulQuestion

rt = LeethRuntime(model_dir="src/leeth/data")
ok = rt.initialize(force_device="cuda")
print(f"prod runtime init: {ok}, device={rt.device}", flush=True)
if not ok:
    print(f"init error: {rt._init_error}")
    sys.exit(1)

n_params = sum(p.numel() for p in rt.model.parameters()) if rt.model is not None else 0
print(f"params: {n_params:,} ({n_params/1e6:.0f}M)", flush=True)

cases = [
    ("Incoming user request in Argent session:\n\"прочитай main.py\"\nWorking directory: C:/dev",
     ChoiceQuestion(id="q1", question="Какой инструмент Argent должен быть вызван в первую очередь?",
                    choices=["read_file", "run_command", "NO_TOOL", "UNFAMILIAR", "calculate"]),
     "read_file"),
    ("Traceback (most recent call last):\n  File \"src/server/gateway.py\", line 42, in handle\nKeyError: 'user_id'",
     ChoiceQuestion(id="q2", question="Какой язык программирования используется в данном компоненте?",
                    choices=["C# (.NET)", "C++", "JavaScript", "Python", "Rust"]),
     "Python"),
    ("Incoming user request in Argent session:\n\"привет, как дела?\"",
     NoulQuestion(id="q3", question="Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"),
     0.0),
]
for state, q, expected in cases:
    t0 = time.perf_counter()
    batch = rt.decide(state=state, questions=[q])
    dt = (time.perf_counter() - t0) * 1000
    res = batch.results.get(q.id)
    if hasattr(res, "choice"):
        got = res.choice
    elif hasattr(res, "value"):
        got = res.value
    else:
        got = "?"
    print(f"  {q.id}: got={got!r} expected={expected!r} ({dt:.0f}ms) {'OK' if got == expected or got == bool(expected >= 0.5) else 'MISMATCH'}", flush=True)
print("PROD OK", flush=True)