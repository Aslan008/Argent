"""Локальный smoke кернела dopt: генераторы + токенизатор mmBERT (БЕЗ модели — лёгкий запуск).

Проверяет: retention фильтра длин, распределение токен-длин, корректность
поиска позиций маркеров [OPT] на реальном токенизаторе.
"""

import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # dataclass annotation resolution требует модуль в sys.modules
    spec.loader.exec_module(m)
    return m


k = load(ROOT / ".argent/kaggle/kernel_dopt/leeth_dopt_kernel.py", "kern")
td = load(ROOT / "src/leeth/task_domains.py", "td")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(ROOT / ".argent/models/mmbert-argent"))
marker_ids = tok.encode("[OPT]", add_special_tokens=False)
print("marker_ids:", marker_ids)

rng = random.Random(1)
for domain, gen in td.GENERATORS.items():
    samples = k.generate_split(gen, 200, rng)
    kept = k.filter_by_tokens(samples, tok, marker_ids)
    texts = [
        f"[STATE] {s.state_text} [QUESTION] {s.question} " + " ".join(f"[OPT] {o}" for o in s.options)
        for s in kept[:50]
    ]
    lens = sorted(len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"])
    print(f"{domain}: kept {len(kept)}/{len(samples)}; token_len p50={lens[len(lens) // 2]} max={lens[-1]}")
    ids = tok(texts[0], add_special_tokens=True)["input_ids"]
    pos = k.find_marker_positions(ids, marker_ids)
    s = kept[0]
    assert len(pos) == len(s.options), (len(pos), len(s.options))
    assert pos == sorted(pos) and len(set(pos)) == len(pos)
    print(f"  пример: {len(s.options)} опций, маркеры @ {pos}, correct='{s.options[s.correct_idx]}'")

# OOD-генераторы с параметрами
ood_snake = k.generate_split(td.GENERATORS["snake"], 20, rng, min_size=12, max_size=12)
ood_maze = k.generate_split(td.GENERATORS["maze"], 20, rng, cell_choices=[(6, 6), (7, 7)])
assert all(f"{12}x{12}" in s.state_text.split("\n")[0] for s in ood_snake)
assert all(int(s.state_text.split("\n")[0].split()[1].rstrip(",").split("x")[0]) >= 13 for s in ood_maze)
print(f"OOD: snake 12x12 ok ({len(ood_snake)}), maze 13x13+ ok ({len(ood_maze)})")
print("SMOKE OK")