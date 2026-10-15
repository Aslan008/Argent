# -*- coding: utf-8 -*-
"""scripts/_smoke_md6.py — smoke-тест рантайма после перезапуска (md6, 42 класса)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth import leeth
from src.leeth.primitives import ChoiceQuestion

ok = leeth.initialize(force_device="cuda")
print("init:", ok, "| device:", leeth.device, "| классов:", len(leeth.class_to_idx), flush=True)

state = 'Incoming user request in Argent session:\n"прочитай файл src/leeth/runtime.py"\nWorking directory: C:/Users/user/Desktop/Argent'
q = ChoiceQuestion(id="tool", question="Какой инструмент Argent должен быть вызван в первую очередь?",
                   choices=["read_file", "run_command", "UNFAMILIAR"])
b = leeth.decide(state, [q])
r = b.results["tool"]
print("smoke:", r.choice, round(r.confidence, 3))