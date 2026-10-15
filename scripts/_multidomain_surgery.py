# -*- coding: utf-8 -*-
"""scripts/_multidomain_surgery.py — рост choice-головы 400M: 26 -> 42 класса.

Новые классы (16): 5 отделов поддержки + 6 типов писем + 5 категорий модерации
(spam общий для email/violation). Существующие 26 индексов НЕ меняются —
обученные веса зависят от них. Новые строки последнего Linear инициализируются
N(0, 0.02). Выход: data_400m/leeth_best.pt (surgery на месте, бэкап leeth_best_md26_backup.pt)
+ обновлённый choice_classes.json.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

DATA = ROOT / "src/leeth/data_400m"

NEW_CLASSES = [
    # support departments (5)
    "billing", "technical", "sales", "account", "other",
    # email types (6)
    "question", "complaint", "meeting_request", "newsletter", "personal", "spam",
    # violation categories (5 новых; spam общий)
    "none", "harassment", "violence", "adult", "fraud",
]


def main():
    classes_path = DATA / "choice_classes.json"
    best_path = DATA / "leeth_best.pt"

    class_to_idx = json.loads(classes_path.read_text(encoding="utf-8"))
    n_old = len(class_to_idx)
    to_add = [c for c in NEW_CLASSES if c not in class_to_idx]
    print(f"текущих классов: {n_old}; добавляем: {len(to_add)} -> {n_old + len(to_add)}")
    if not to_add:
        print("все классы уже на месте — surgery не нужна")
        return

    # 1. Бэкап
    backup = DATA / "leeth_best_md26_backup.pt"
    if not backup.exists():
        shutil.copy(best_path, backup)
        print(f"бэкап: {backup.name}")

    # 2. Грузим чекпоинт
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    n_old_ckpt = getattr(config, "num_choices", None)
    print(f"config.num_choices = {n_old_ckpt}")

    sd = checkpoint["state_dict"]
    w_key = "choice_head.3.weight"
    b_key = "choice_head.3.bias"
    w = sd[w_key]
    b = sd[b_key]
    assert w.shape[0] == n_old, f"choice_head.3.weight {tuple(w.shape)} != num_choices {n_old}"

    n_new = n_old + len(to_add)
    hidden = w.shape[1]
    gen = torch.Generator().manual_seed(42)

    new_w = torch.randn(n_new, hidden, generator=gen) * 0.02
    new_w[:n_old] = w
    new_b = torch.zeros(n_new)
    new_b[:n_old] = b
    sd[w_key] = new_w
    sd[b_key] = new_b
    config.num_choices = n_new
    checkpoint["state_dict"] = sd
    if "class_to_idx" in checkpoint:
        checkpoint["class_to_idx"] = {**class_to_idx, **{c: n_old + i for i, c in enumerate(to_add)}}

    torch.save(checkpoint, best_path)
    print(f"choice_head: {n_old} -> {n_new}, сохранено в {best_path.name}")

    # 3. choice_classes.json (старые индексы не трогаем)
    for i, c in enumerate(to_add):
        class_to_idx[c] = n_old + i
    classes_path.write_text(json.dumps(class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"choice_classes.json: {n_old} -> {len(class_to_idx)} классов")

    # 4. Верификация: модель грузится и даёт 42 логита
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from src.leeth.model import LeethConfig, LeethModel

    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        cfg = LeethConfig(**cfg)
    model = LeethModel(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    with torch.no_grad():
        dummy = torch.randint(0, 100, (1, 16))
        outs = model(dummy, attention_mask=torch.ones_like(dummy))
    assert outs["choice_logits"].shape[-1] == n_new, outs["choice_logits"].shape
    print(f"верификация: choice_logits {tuple(outs['choice_logits'].shape)} OK")


if __name__ == "__main__":
    main()