# -*- coding: utf-8 -*-
"""scripts/validate_mmbert_battery.py — Фаза E4: валидация mmBERT на прод-батарее.

Зачем: честное сравнение новой mmBERT-модели (mmbert-argent + mmbert-leeth.pt)
со старой базой Leeth 400M на ТЕХ ЖЕ 109 кейсах прод-батареи.
Старые точки: baseline 55.0% (B3), после финетюнов 54.1% (B4).

Формат входа как в тренировке port_heads.py: state + "\\n[QUESTION] " + question.
У батареи state пустой -> текст = "\\n[QUESTION] " + query.

Запуск: python scripts/validate_mmbert_battery.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

BASE_DIR = Path(".argent/models/mmbert-argent")
HEADS_PATH = Path(".argent/models/mmbert-leeth.pt")
CLASSES_PATH = Path("src/leeth/data_400m/choice_classes.json")
BATTERY_PATH = Path("scripts/prod_battery.json")
OUT_PATH = Path(".argent/artifacts/mmbert_battery_e4.md")

# Старые точки отсчёта (B3/B4, старая модель Leeth 400M)
OLD_BASELINE_ACC = 0.550
OLD_FINETUNED_ACC = 0.541


class Heads(nn.Module):
    """Точная копия Heads из port_heads.py (веса совпадают с mmbert-leeth.pt)."""

    def __init__(self, hidden: int, n_classes: int):
        super().__init__()
        self.choice_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
        self.noul_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.score_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 10))
        self.extract_start_head = nn.Linear(hidden, 1)
        self.extract_end_head = nn.Linear(hidden, 1)


def main() -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    battery = json.loads(BATTERY_PATH.read_text(encoding="utf-8"))
    cases = battery["cases"]
    class_to_idx = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32).to(device).eval()
    ckpt = torch.load(HEADS_PATH, map_location=device, weights_only=False)
    heads = Heads(ckpt["hidden"], ckpt["n_classes"]).to(device)
    heads.load_state_dict(ckpt["heads"])
    heads.eval()

    # Канонический прод-формат (dispatcher._build_state): запрос в state,
    # question — фиксированный шаблон (весь сигнал в state, как в тренировке).
    template_q = "Какой инструмент Argent должен быть вызван в первую очередь?"

    def build_state(query: str) -> str:
        normalized = query[:1].lower() + query[1:] if query[:1].isalpha() else query
        return f'Incoming user request in Argent session:\n"{normalized}"\nWorking directory: C:/Users/user/Desktop/Argent'

    texts = [build_state(c["query"]) + "\n[QUESTION] " + template_q for c in cases]
    expected = [c["expected_tool"] for c in cases]

    t0 = time.perf_counter()
    preds: list = []
    confs: list = []
    bs = 16
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i + bs], truncation=True, max_length=256,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = base.base_model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            probs = torch.softmax(heads.choice_head(pooled), dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(idx_to_class[int(p)] for p in pred.tolist())
            confs.extend(float(c) for c in conf.tolist())
    wall = time.perf_counter() - t0

    correct = 0
    per_tool: dict = {}
    corrs: list = []
    misses: list = []
    for case, pred, exp, conf in zip(cases, preds, expected, confs):
        ok = pred == exp
        correct += ok
        corrs.append(1.0 if ok else 0.0)
        bucket = per_tool.setdefault(exp, {"n": 0, "ok": 0})
        bucket["n"] += 1
        bucket["ok"] += ok
        if not ok:
            misses.append({"query": case["query"], "expected": exp, "pred": pred, "conf": round(conf, 3)})

    n = len(cases)
    acc = correct / n
    n_bins = 10
    ece = 0.0
    brier = sum((c - o) ** 2 for c, o in zip(confs, corrs)) / n
    bins: list = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corrs):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    for b in bins:
        if b:
            avg_c = sum(c for c, _ in b) / len(b)
            acc_b = sum(o for o, _ in b) / len(b)
            ece += len(b) / n * abs(acc_b - avg_c)

    lines = [
        "# E4: mmBERT на прод-батарее",
        "",
        f"- Cases: {n} (те же, что в B3/B4)",
        f"- **Accuracy: {acc:.1%}** ({correct}/{n})",
        f"- Старая модель (Leeth 400M): baseline 55.0%, после финетюнов 54.1%",
        f"- Дельта к старому baseline: {acc - OLD_BASELINE_ACC:+.1%}",
        f"- ECE (max-softmax): {ece:.4f}",
        f"- Brier: {brier:.4f}",
        f"- Wall time: {wall:.1f}s ({wall / n * 1000:.0f} мс/кейс, device={device})",
        "",
        "## Per-tool accuracy",
        "",
        "| Expected tool | n | ok |",
        "|---|---|---|",
    ]
    for tool, b in sorted(per_tool.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"| {tool} | {b['n']} | {b['ok']} ({b['ok'] / b['n']:.0%}) |")
    lines += ["", "## Misses", ""]
    for m in misses:
        lines.append(f"- {m['query']!r}: expected **{m['expected']}**, pred **{m['pred']}** (conf {m['conf']})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Accuracy: {acc:.1%} ({correct}/{n}) | ECE: {ece:.4f} | Brier: {brier:.4f}")
    print(f"Старая база: 55.0% | дельта: {acc - OLD_BASELINE_ACC:+.1%}")
    print(f"Misses: {len(misses)} (см. {OUT_PATH})")


if __name__ == "__main__":
    main()