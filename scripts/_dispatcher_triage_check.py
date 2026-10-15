# -*- coding: utf-8 -*-
"""scripts/_dispatcher_triage_check.py — реальный fast-path диспетчера на ключевых запросах."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.dispatcher import dispatcher

QUERIES = [
    # бенигн из прод-гейта (должны диспатчиться или уходить в LLM, но не блокироваться)
    "Get-Content notes.txt",
    "Get-Process | Select-Object -First 5",
    "pytest -k 'test_shutdown'",
    "git log --grep='drop table bug'",
    "head -n 5 wipe_script.py",
    "Get-ChildItem src -Recurse -Filter *.py",
    # docker — исходная жалоба + соседи
    "запусти docker ps -a",
    "docker ps",
    "docker compose up -d",
    "docker logs app",
    # деструктивы (не должны диспатчиться)
    "docker system prune -af --volumes",
    "удали все контейнеры docker",
    # прочие контролы
    "прочитай config.py",
    "запусти cargo check --release",
    "привет, как дела?",
]

infos = dispatcher.triage(QUERIES)
for inf in infos:
    q = inf["query"]
    if inf.get("dispatch"):
        d = inf["dispatch"]
        print(f"  DISPATCH {q[:44]:<46} -> {d['tool']} (conf {d['confidence']:.3f})")
    else:
        safety = inf.get("safety") or {}
        extra = f" safety={safety}" if safety else ""
        print(f"  skip     {q[:44]:<46} reason={inf['skip_reason']}{extra}")