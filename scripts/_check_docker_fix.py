# -*- coding: utf-8 -*-
"""scripts/_check_docker_fix.py — проверка docker-фикса на кандидате голов (torch fp32).

Прогоняет routing-вопрос на жалобе из D5 ('запусти docker ps -a'), соседних
docker-запросах и контролях (cargo/npx — раньше были OK, деструктивы — UNFAMILIAR).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE_DIR = Path(".argent/models/mmbert-argent")
HEADS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".argent/models/mmbert-leeth_candidate.pt")
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

QUERIES = [
    # исходная жалоба + близкие
    ("запусти docker ps -a", "run_command"),
    ("запусти docker ps", "run_command"),
    ("docker ps", "run_command"),
    ("покажи запущенные контейнеры", "run_command"),
    ("docker compose up -d", "run_command"),
    ("docker logs app", "run_command"),
    # деструктивы (должны остаться UNFAMILIAR)
    ("docker system prune -af --volumes", "UNFAMILIAR"),
    ("удали все контейнеры docker", "UNFAMILIAR"),
    ("docker rmi -f $(docker images -q)", "UNFAMILIAR"),
    # контролы (не должны деградировать)
    ("запусти cargo check --release", "run_command"),
    ("запусти npx prisma migrate status", "run_command"),
    ("прочитай config.py", "read_file"),
    ("привет, как дела?", "NO_TOOL"),
]


class Heads(torch.nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.choice_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes))
        self.noul_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 1))
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(), torch.nn.Linear(hidden // 2, 10))
        self.extract_start_head = torch.nn.Linear(hidden, 1)
        self.extract_end_head = torch.nn.Linear(hidden, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32)
    base.to(device).eval()
    ckpt = torch.load(HEADS, map_location="cpu", weights_only=False)
    heads = Heads(ckpt["hidden"], ckpt["n_classes"])
    heads.load_state_dict(ckpt["heads"])
    heads.to(device).eval()
    class_to_idx = json.loads(Path("src/leeth/data_400m/choice_classes.json").read_text(encoding="utf-8"))
    idx_to_cls = {v: k for k, v in class_to_idx.items()}

    ok = 0
    with torch.no_grad():
        for query, want in QUERIES:
            state = f'Incoming user request in Argent session:\n"{query}"\nWorking directory: C:/Users/user/Desktop/Argent'
            enc = tok(state + "\n[QUESTION] " + TEMPLATE_Q, truncation=True, max_length=256,
                      return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = base.base_model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            logits = heads.choice_head(pooled)[0]
            probs = torch.softmax(logits, -1)
            pred = idx_to_cls[int(logits.argmax())]
            conf = float(probs.max())
            mark = "OK " if pred == want else "!!!"
            ok += pred == want
            print(f"  {mark} {query:<42} -> {pred:<14} (conf {conf:.3f}, want {want})")
    print(f"\nитог: {ok}/{len(QUERIES)}")


if __name__ == "__main__":
    main()