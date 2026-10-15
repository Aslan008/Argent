# -*- coding: utf-8 -*-
"""scripts/_kaggle_precompute_feats.py — предрасчёт pooled-фичей замороженной базы.

Реплицирует pooled_forward из _kaggle_train_choice.py ТОЧНО:
  text = state + "\\n[QUESTION] " + question
  tokenizer(truncation=256, padding=True)
  base.base_model(**enc, output_hidden_states=True); hs = hidden_states[-1]
  pooled = (hs*mask).sum(1) / mask.sum(1).clamp(min=1)
База в eval (без dropout) -> фичи детерминированы, обучение голов на фичах
математически идентично рецепту D4. Хранение fp16 (шум ~1e-3 пренебрежим
на фоне int8-квантизации прода ~0.4%).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / ".argent/models/mmbert-argent"
DATA = ROOT / "src/leeth/data_400m"
OUT = ROOT / ".argent/kaggle/dataset_v2"
TEMPLATE_Q = "Какой инструмент Argent должен быть вызван в первую очередь?"
P = "Incoming user request in Argent session:"
MAX_LEN = 256
BATCH = 32
DOCKER_ONLY = os.environ.get("DOCKER_ONLY") == "1"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def is_choice(r):
    return r.get("question_type") == "choice" and r.get("target_choice")


@torch.no_grad()
def embed(rows, tokenizer, base, device, class_to_idx, desc):
    hidden = None
    feats, targets = None, None
    confs = np.full(len(rows), -100, dtype=np.int16)
    t0 = time.time()
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        texts = [r["state"] + "\n[QUESTION] " + r["question"] for r in chunk]
        enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True,
                        return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = base.base_model(**enc, output_hidden_states=True)
        hs = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        if feats is None:
            hidden = pooled.shape[1]
            feats = np.empty((len(rows), hidden), dtype=np.float16)
            targets = np.empty(len(rows), dtype=np.int16)
        feats[i:i + len(chunk)] = pooled.cpu().numpy().astype(np.float16)
        for j, r in enumerate(chunk):
            targets[i + j] = class_to_idx.get(r["target_choice"], -100)
            confs[i + j] = class_to_idx.get(r.get("confuser"), -100)
        if (i // BATCH) % 200 == 0:
            print(f"  {desc}: {i + len(chunk)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {desc}: DONE {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    return feats, targets, confs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))
    base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), torch_dtype=torch.float32)
    base.to(device).eval()

    class_to_idx = json.loads((DATA / "choice_classes.json").read_text(encoding="utf-8"))
    assert len(class_to_idx) == 26, class_to_idx

    corpus = [r for r in load_rows(DATA / "leeth_400m_train.jsonl") if is_choice(r)]
    extra = [r for r in load_rows(DATA / "leeth_400m_train_extra.jsonl") if is_choice(r)]
    docker = load_rows(DATA / "docker_train.jsonl")
    val_all = [r for r in load_rows(DATA / "leeth_400m_val.jsonl") if is_choice(r)]
    docker_val = load_rows(DATA / "docker_val.jsonl")
    print(f"corpus={len(corpus)} extra={len(extra)} docker={len(docker)} "
          f"val={len(val_all)} docker_val={len(docker_val)}", flush=True)

    todo = (("docker", docker), ("docker_val", docker_val), ("val", val_all)) \
        if DOCKER_ONLY else (("corpus", corpus), ("extra", extra), ("docker", docker),
                             ("docker_val", docker_val), ("val", val_all))
    for name, rows in todo:
        f, t, c = embed(rows, tokenizer, base, device, class_to_idx, name)
        payload = {"feats": f, "targets": t, "confs": c}
        if name == "val":
            payload["routing_mask"] = np.array(
                [r["question"] == TEMPLATE_Q and r["state"].startswith(P) for r in val_all],
                dtype=bool)
        np.savez_compressed(OUT / f"feat_{name}.npz", **payload)
        print(f"saved feat_{name}.npz: {f.shape}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()