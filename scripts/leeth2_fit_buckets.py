# -*- coding: utf-8 -*-
"""Подобрать температуру головы выбора по корзинам размера каталога — на данных, которых модель не видела.

Калибровочный набор берётся из корпуса фабрики (`leeth3_data/accepted/records.jsonl`, запрос R002 — выбор инструмента из списка),
и НЕ из замороженного экзамена: иначе подгонка и проверка были бы на одном и том же, и число ничего бы не значило. Установленная
s70 (2026-10-13) корпус не видела вообще, так что утечки нет ни для неё, ни для будущих выпусков, которые обучены на его
отложенной половине.

Что делает: гоняет модель по калибровочным записям, берёт сырые логиты выбора, подбирает температуру на каждую корзину размера
каталога (по NLL), проверяет выигрыш двухфолдово и печатает готовый список `temp_choice_buckets` для конфига.

    python scripts/leeth2_fit_buckets.py <model_dir> --calib leeth3_data/accepted/records.jsonl --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                       # noqa: E402
from scipy.optimize import minimize_scalar               # noqa: E402

BOUNDS = ((8, 8), (32, 32), (128, 128), (255, 255))      # the catalogue sizes the exam and the product actually use


def _logt_nll(logits: Sequence[np.ndarray], labels: Sequence[int], logt: float) -> float:
    t = float(np.exp(logt))
    tot = 0.0
    for lg, y in zip(logits, labels):
        z = lg / t
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        tot += -float(np.log(max(p[y], 1e-12)))
    return tot / max(1, len(logits))


def fit_one(logits: Sequence[np.ndarray], labels: Sequence[int]) -> float:
    if len(logits) < 30:
        return 1.0
    r = minimize_scalar(lambda x: _logt_nll(logits, labels, x), bounds=(np.log(0.2), np.log(8.0)), method="bounded",
                        options={"xatol": 1e-3})
    return float(np.exp(r.x))


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    if not conf:
        return float("nan")
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - avg)
    return err


def raw_logits(rt, state, question, texts):
    """The uncalibrated choice logits and the option order the caller sees: what a temperature is fitted on."""
    from src.leeth import primitives as P
    from src.leeth2.evaluate import predict

    sample = rt._to_sample(rt.fit_state(state, question), P.ChoiceQuestion("q", question, list(texts)))
    p = predict(rt.model, rt.encoder, [sample], device=rt.device, amp=rt.amp, calibrated=False)[0]
    return p.raw["logits_shown"], p.raw["perm"]


def load_calibration(path: Path, per_bucket: int, seed: int = 5) -> Dict[int, List[Tuple[str, str, List[str], str]]]:
    """R002 records are exactly the shape the choice head sees: a state, a question, the option texts and the right one."""
    by_k: Dict[int, List[Tuple[str, str, List[str], str]]] = defaultdict(list)
    rng = random.Random(seed)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("request") != "R002" or not r.get("options") or r.get("answer") in (None, ""):
            continue
        k = len(r["options"])
        bucket = next((b for b, _ in BOUNDS if k <= b), BOUNDS[-1][0])
        if len(by_k[bucket]) >= per_bucket:
            continue
        texts = [f"{n}: {d}" if d else n for n, d in r["options"].items()]
        if r.get("abstain_ok"):
            texts.append({"ru": "ни один из вариантов не подходит", "en": "none of these fits"}.get(r.get("lang", "en"),
                       "none of these fits"))
        labels = list(r["options"]) + ([None] if r.get("abstain_ok") else [])
        by_k[bucket].append((r["state"], r["question"], texts, labels, r["answer"]))
    rng.shuffle(by_k[BOUNDS[0][0]])
    return by_k


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--calib", default=str(ROOT / "leeth3_data" / "accepted" / "records.jsonl"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-bucket", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "scratch" / "temps_buckets.json"))
    a = ap.parse_args(argv)

    from src.leeth2.runtime import Leeth2Runtime

    rt = Leeth2Runtime.load(a.model_dir, device=a.device, max_len=4096)
    data = load_calibration(Path(a.calib), a.per_bucket)
    for b, rows in sorted(data.items()):
        print('корзина %3d: %d записей' % (b, len(rows)), flush=True)

    # raw logits per bucket: one predict() call per record, keeping the shown logits and the index of the right option
    raw: Dict[int, Tuple[List[np.ndarray], List[int]]] = {}
    for b, rows in sorted(data.items()):
        logits, labels = [], []
        for state, question, texts, labels_in, answer in rows:
            lg, perm = raw_logits(rt, state, question, texts)      # perm[shown_position] = original index in `texts`
            if answer not in labels_in:
                continue
            orig = labels_in.index(answer)
            if orig not in perm:
                continue
            logits.append(np.asarray(lg, dtype=np.float64))
            labels.append(list(perm).index(orig))
        raw[b] = (logits, labels)
        print('корзина %3d: логитов %d' % (b, len(labels)), flush=True)

    # two-fold: fit on one half, score the other, then swap — the printed gain is the held-out one
    out: Dict[str, Any] = {"model": a.model_dir, "buckets": []}
    buckets_cfg: List[List[float]] = []
    for b, (logits, labels) in sorted(raw.items()):
        if not logits:
            continue
        idx = list(range(len(logits)))
        halves = (idx[0::2], idx[1::2])
        before, after, temps = [], [], []
        for fit_ids, eval_ids in ((halves[0], halves[1]), (halves[1], halves[0])):
            t = fit_one([logits[i] for i in fit_ids], [labels[i] for i in fit_ids])
            temps.append(t)
            for i in eval_ids:
                lg, y = logits[i], labels[i]
                def conf(temp: float) -> Tuple[float, bool]:
                    z = lg / temp
                    z = z - z.max()
                    p = np.exp(z)
                    p /= p.sum()
                    j = int(np.argmax(p))
                    return float(p[j]), j == y
                c0, ok0 = conf(1.0)
                c1, ok1 = conf(t)
                before.append((c0, ok0))
                after.append((c1, ok1))
        t_mean = sum(temps) / len(temps)
        e0 = ece([c for c, _ in before], [o for _, o in before])
        e1 = ece([c for c, _ in after], [o for _, o in after])
        acc = sum(o for _, o in before) / len(before)
        out["buckets"].append({"upper": b, "n": len(before), "acc": round(acc, 4), "temperature": round(t_mean, 3),
                               "ece_before": round(e0, 4), "ece_after": round(e1, 4)})
        buckets_cfg.append([b, round(t_mean, 3)])
        print('  до %.3f, после %.3f, T=%.2f (n=%d, точность %.3f)' % (e0, e1, t_mean, len(before), acc), flush=True)

    out["temp_choice_buckets"] = buckets_cfg
    worst_before = max(b["ece_before"] for b in out["buckets"])
    worst_after = max(b["ece_after"] for b in out["buckets"])
    out["worst_before"], out["worst_after"] = round(worst_before, 4), round(worst_after, 4)
    print('\nхудшая корзина: %.4f -> %.4f' % (worst_before, worst_after))
    print('для конфига: "temp_choice_buckets": %s' % json.dumps(buckets_cfg))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
