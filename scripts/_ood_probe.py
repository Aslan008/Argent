# -*- coding: utf-8 -*-
"""scripts/_ood_probe.py — поведение роутера на незнакомых (OOD) запросах."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.dispatcher import dispatcher

QUERIES = [
    # --- незнакомые одиночные интенты ---
    "переведи этот абзац на английский",
    "спроектируй схему БД для маркетплейса",
    "напиши функцию быстрой сортировки на Python",
    "объясни разницу между TCP и UDP",
    "придумай пять названий для проекта",
    "рефакторни этот класс, слишком много ответственности",
    # --- compound (несколько интентов) ---
    "прочитай config.py и запусти pytest",
    "посчитай 128 * 64 и запиши результат в файл",
    "найди файл config.json и покажи его содержимое",
    # --- compound OOD / деструктив ---
    "кстати, настрой nginx reverse proxy и скажи сколько будет 2 + 2",
    "сделай коммит и запуши в main",
    # --- веб/браузер ---
    "какая погода в Москве?",
    "открой сайт example.com",
    "загугли свежие новости про Rust",
    # --- контролы (известные) ---
    "docker ps",
    "прочитай config.py",
    "запусти cargo check --release",
]

infos = dispatcher.triage(QUERIES)
n_dispatch = 0
for inf in infos:
    q = inf["query"]
    if inf.get("dispatch"):
        d = inf["dispatch"]
        n_dispatch += 1
        print(f"  DISPATCH {q[:46]:<48} -> {d['tool']} (conf {d['confidence']:.3f})")
    else:
        print(f"  skip     {q[:46]:<48} reason={inf['skip_reason']}")
print(f"\nитого: {n_dispatch} диспатчей из {len(QUERIES)} — остальное уходит в LLM-флоу")