# -*- coding: utf-8 -*-
"""scripts/mine_shadow.py — Фаза A1: shadow-лог арбитра -> обучающие семплы.

Зачем: арбитр (CDCL) уже работает в проде и логирует каждое расхождение
«модель предложила X, решётор сказал что (X, param) противоречит жёстким
клаузам». Эти записи — бесплатный учебник с гарантированно верными метками:
решётор не ошибается в домене безопасности по построению.

Семантика меток (важно!):
  - UNSAT с destructive=True в core -> команда ОПАСНА:
      noul по вопросу БЕЗОПАСНОСТИ (Q_NOUL_SAFE) = 0.0, shell-state с param.
      НЕ need_tool=0: запрос может легитимно требовать инструмент.
  - UNSAT по форме param (calculate с неарифметикой и т.п.) -> выбор/параметр
      неверны, но потребность в инструменте неизвестна -> ТОЛЬКО
      choice = UNFAMILIAR (безопасный делегат), без noul-семпла.
  - SAT -> 10% положительный replay (модель была права).
  - Записи с error (сбой солвера) — НЕ клаузы, пропускаем.

Запуск: python scripts/mine_shadow.py
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.leeth.dataset_generator import LeethSample
from src.leeth.dispatcher import dispatcher
from src.leeth.diverse_generator import Q_NOUL_SAFE

SHADOW_LOG = Path(".argent/artifacts/arbitrator_shadow.jsonl")
SHADOW_LOG_OLD = Path(".argent/artifacts/arbitrator_shadow.old.jsonl")
OUT_PATH = Path("src/leeth/data/shadow_mined.jsonl")

POSITIVE_REPLAY_RATE = 0.10
SEED = 42

CHOICES = [
    "read_file", "write_file", "replace_in_file", "run_command",
    "grep_search", "find_by_name", "list_directory", "browser_open",
    "browser_click", "browser_type", "browser_upload_file", "calculate",
    "ask_user_questions", "NO_TOOL", "UNFAMILIAR",
]


def load_records() -> list:
    records = []
    for path in (SHADOW_LOG_OLD, SHADOW_LOG):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def make_noul_safe_sample(param: str) -> dict:
    """Safety-семпл: shell-state с командой из param, вопрос безопасности."""
    param_clean = (param or "").strip()
    state = f"User shell execution request:\nCommand: {param_clean}\nCaller: Argent SubAgent"
    return {
        "state": state,
        "question_type": "noul",
        "question": Q_NOUL_SAFE,
        "target_noul": 0.0,
    }


def make_choice_sample(query: str, target_choice: str) -> dict:
    state = dispatcher._build_state(query)
    return {
        "state": state,
        "question_type": "choice",
        "question": "Какой инструмент Argent должен быть вызван в первую очередь?",
        "choices": CHOICES,
        "target_choice": target_choice,
    }


def main() -> None:
    records = load_records()
    if not records:
        print("Shadow log is empty or missing — nothing to mine yet.")
        print("The arbitrator populates it automatically during live dispatch.")
        return

    rng = random.Random(SEED)
    mined: list = []
    stats = {
        "total": len(records), "sat": 0, "unsat": 0, "errors_skipped": 0,
        "mined_destructive": 0, "mined_param_shape": 0, "mined_replay": 0,
    }

    for rec in records:
        query = rec.get("query") or ""
        if not query or len(query) < 3:
            continue
        if rec.get("error"):
            stats["errors_skipped"] += 1
            continue
        if rec.get("sat"):
            stats["sat"] += 1
            # Positive replay: reinforce decisions the solver confirmed.
            if rng.random() < POSITIVE_REPLAY_RATE:
                tool = rec.get("model_tool") or rec.get("tool")
                if tool and tool in CHOICES and tool != "UNFAMILIAR":
                    mined.append(make_choice_sample(query, tool))
                    stats["mined_replay"] += 1
        else:
            stats["unsat"] += 1
            core = rec.get("core") or []
            is_destructive = any("destructive=True" in str(c) for c in core)
            # Choice: безопасный делегат в обоих случаях.
            choice_sample = make_choice_sample(query, "UNFAMILIAR")
            choice_sample["_core"] = core
            mined.append(choice_sample)
            if is_destructive:
                # Опасная команда: учим Noul-безопасность на shell-state с param.
                # БЕЗ param — пропускаем: натуральный язык запроса в роли
                # «Command:» с меткой unsafe = отравленная метка.
                param = (rec.get("param") or "").strip()
                if param:
                    safety_sample = make_noul_safe_sample(param)
                    safety_sample["_core"] = core
                    mined.append(safety_sample)
                    stats["mined_destructive"] += 1
                else:
                    stats["mined_destructive_skipped_no_param"] = (
                        stats.get("mined_destructive_skipped_no_param", 0) + 1
                    )
            else:
                # Форма param неверна, но потребность в инструменте неизвестна —
                # need_tool=0 здесь был бы ЛОЖНЫМ сигналом (under-dispatching).
                stats["mined_param_shape"] += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for sample in mined:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Written {len(mined)} samples -> {OUT_PATH}")


if __name__ == "__main__":
    main()