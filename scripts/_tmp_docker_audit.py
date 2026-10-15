# -*- coding: utf-8 -*-
"""scripts/_tmp_docker_audit.py — аудит docker-строк в routing-корпусе (read-only)."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"

for name in ("leeth_400m_train.jsonl", "leeth_400m_val.jsonl", "leeth_400m_train_extra.jsonl"):
    p = ROOT / "src" / "leeth" / "data_400m" / name
    routing, docker = [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if "question_type" not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("question_type") == "choice" and r.get("question") == TEMPLATE_Q:
                routing.append(r)
                if "docker" in (r.get("state") or "").lower():
                    docker.append(r)
    print(f"== {name}: routing={len(routing)}, docker={len(docker)}")
    if docker:
        print("   targets:", dict(Counter(r["target_choice"] for r in docker)))
        for label in ("run_command", "UNFAMILIAR"):
            ex = [r for r in docker if r["target_choice"] == label][:4]
            print(f"   --- примеры target={label}: {len(ex)} ---")
            for r in ex:
                print("     ", (r["state"] or "").replace("\n", " | ")[:100])