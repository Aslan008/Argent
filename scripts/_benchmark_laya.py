"""Phase 3 (benchmark): Leeth 400M vs Laya on OUR val + battery, same questions/hardware.

Runs both models over the same labeled samples (choice tool-routing + noul need-tool),
computes accuracy / ECE / latency for each, and writes a comparison table artifact.
Laya API: agent.predict(state, {qid: {"type": "choice", "instructions": ..., "criteria": {...}}})
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch  # noqa: E402

VAL_PATH = ROOT / "src/leeth/data_400m/leeth_400m_val.jsonl"
OUT_ARTIFACT = ROOT / ".argent/artifacts/phase3_laya_benchmark.md"
BATCH = 32
N_CHOICE = 400  # subset for Laya (their predict is per-sample; keep runtime sane)
N_NOUL = 300

Q_TOOL = "Какой инструмент Argent должен быть вызван в первую очередь?"
Q_NOUL = "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?"


def ece(confs, corrects, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corrects):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    total = len(confs)
    e = 0.0
    for b in bins:
        if not b:
            continue
        avg_c = sum(c for c, _ in b) / len(b)
        acc = sum(o for _, o in b) / len(b)
        e += len(b) / total * abs(acc - avg_c)
    return e


def load_val():
    samples = [json.loads(l) for l in open(VAL_PATH, encoding="utf-8")]
    choice = [
        s for s in samples
        if s["question_type"] == "choice" and s.get("choices") and s.get("target_choice") is not None
    ]
    noul = [s for s in samples if s["question_type"] == "noul"]
    return choice[:N_CHOICE], noul[:N_NOUL]


# ---------------- Leeth side ----------------
def run_leeth(choice_samples, noul_samples):
    from src.leeth.runtime import leeth
    ok = leeth.initialize()
    if not ok:
        print(f"leeth init error: {leeth._init_error}")
        sys.exit(1)
    device = leeth.device
    model, tok, c2i = leeth.model, leeth.tokenizer, leeth.class_to_idx

    def forward(texts, type_token):
        built = []
        for state, q in texts:
            text = f"[STATE] {state} [SEP] [QUESTION] [{type_token}] {q}"
            built.append(tok.encode(text, max_length=512, padding=True, truncation=True, add_special_tokens=True))
        max_len = max(len(x) for x in built)
        pad = tok.pad_token_id
        ids = torch.tensor([x + [pad] * (max_len - len(x)) for x in built], dtype=torch.long, device=device)
        mask = torch.tensor([[1 if t != pad else 0 for t in row] for row in ids.tolist()], dtype=torch.long, device=device)
        with torch.no_grad():
            return model(ids, attention_mask=mask)

    # choice
    confs, corrects, lat = [], [], []
    for i in range(0, len(choice_samples), BATCH):
        chunk = choice_samples[i : i + BATCH]
        t0 = time.perf_counter()
        out = forward([(s["state"], s["question"]) for s in chunk], "CHOICE")
        lat.append((time.perf_counter() - t0) * 1000 / len(chunk))
        probs = torch.softmax(out["choice_logits"].float().cpu(), dim=-1)
        for j, s in enumerate(chunk):
            idxs = [c2i[c] for c in s["choices"] if c in c2i]
            sub = probs[j, idxs] if idxs else probs[j]
            sub = sub / sub.sum()
            confs.append(float(sub.max()))
            pred = s["choices"][int(sub.argmax())] if idxs else c2i.get(int(probs[j].argmax()), "?")
            corrects.append(1.0 if pred == s["target_choice"] else 0.0)
    ch = {"acc": sum(corrects) / len(corrects), "ece": ece(confs, corrects), "ms": sum(lat) / len(lat), "n": len(corrects)}

    # noul
    confs, corrects, lat = [], [], []
    for i in range(0, len(noul_samples), BATCH):
        chunk = noul_samples[i : i + BATCH]
        t0 = time.perf_counter()
        out = forward([(s["state"], s["question"]) for s in chunk], "NOUL")
        lat.append((time.perf_counter() - t0) * 1000 / len(chunk))
        p = torch.sigmoid(out["noul_logits"].float().cpu())
        for j, s in enumerate(chunk):
            confs.append(float(max(p[j].item(), 1 - p[j].item())))
            corrects.append(1.0 if (p[j].item() >= 0.5) == bool(s["target_noul"]) else 0.0)
    nu = {"acc": sum(corrects) / len(corrects), "ece": ece(confs, corrects), "ms": sum(lat) / len(lat), "n": len(corrects)}
    return {"choice": ch, "noul": nu}


# ---------------- Laya side ----------------
def run_laya(choice_samples, noul_samples, english=False):
    import laya
    agent = laya.load(device="cuda")
    print(f"laya loaded: {agent.model_id} (english={english})", flush=True)
    instr_choice_en = "Which tool should Argent call first?"
    instr_noul_en = "Does answering this request require calling at least one system tool?"

    # choice
    confs, corrects, lat = [], [], []
    for i, s in enumerate(choice_samples):
        instr = instr_choice_en if english else s["question"]
        qs = {"tool": {"type": "choice", "instructions": instr, "criteria": {c: c for c in s["choices"]}}}
        t0 = time.perf_counter()
        r = agent.predict(s["state"], qs)
        lat.append((time.perf_counter() - t0) * 1000)
        res = r.get("answers", {}).get("tool", r)
        # Laya choice result: {'choice': ..., 'probabilities': {...}, 'confidence': ...}
        got, conf = extract_laya_choice(res, s["choices"])
        confs.append(conf)
        corrects.append(1.0 if got == s["target_choice"] else 0.0)
        if (i + 1) % 50 == 0:
            print(f"  laya choice {i + 1}/{len(choice_samples)}  acc={sum(corrects) / len(corrects):.3f}", flush=True)
    ch = {"acc": sum(corrects) / len(corrects), "ece": ece(confs, corrects), "ms": sum(lat) / len(lat), "n": len(corrects)}

    # noul
    confs, corrects, lat = [], [], []
    for i, s in enumerate(noul_samples):
        instr = instr_noul_en if english else s["question"]
        qs = {"need": {"type": "noul", "instructions": instr}}
        t0 = time.perf_counter()
        r = agent.predict(s["state"], qs)
        lat.append((time.perf_counter() - t0) * 1000)
        res = r.get("answers", {}).get("need", r)
        prob, val = extract_laya_noul(res)
        confs.append(float(max(prob, 1 - prob)))
        corrects.append(1.0 if val == bool(s["target_noul"]) else 0.0)
        if (i + 1) % 50 == 0:
            print(f"  laya noul {i + 1}/{len(noul_samples)}  acc={sum(corrects) / len(corrects):.3f}", flush=True)
    nu = {"acc": sum(corrects) / len(corrects), "ece": ece(confs, corrects), "ms": sum(lat) / len(lat), "n": len(corrects)}
    return {"choice": ch, "noul": nu}


def extract_laya_choice(res, s):
    """Laya choice result -> (choice, confidence=max probability among offered criteria)."""
    if isinstance(res, dict):
        probs = res.get("probabilities") or res.get("probs") or {}
        if isinstance(probs, dict) and probs:
            best = max(probs, key=probs.get)
            return best, float(probs[best])
        if "choice" in res:
            return res["choice"], float(res.get("confidence") or 0.0)
    return str(res), 0.0


def extract_laya_noul(res):
    """Laya noul result -> (P(true), bool). Actual shape: {'noul': 0.0001, 'confidence': ...}"""
    if isinstance(res, dict):
        for k in ("noul", "prob", "probability", "p_true"):
            if k in res and isinstance(res[k], (int, float)):
                return float(res[k]), float(res[k]) >= 0.5
    return 0.5, bool(res)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    choice_samples, noul_samples = load_val()
    print(f"benchmark: {len(choice_samples)} choice, {len(noul_samples)} noul", flush=True)
    results = {}
    if which in ("both", "leeth"):
        results["leeth"] = run_leeth(choice_samples, noul_samples)
        print("leeth:", json.dumps(results["leeth"], ensure_ascii=False), flush=True)
    if which in ("both", "laya"):
        english = len(sys.argv) > 2 and sys.argv[2] == "en"
        key = "laya_en" if english else "laya"
        results[key] = run_laya(choice_samples, noul_samples, english=english)
        print(key + ":", json.dumps(results[key], ensure_ascii=False), flush=True)
    raw_path = ROOT / ".argent/artifacts/_phase3_raw.json"
    merged = json.loads(raw_path.read_text(encoding="utf-8")) if raw_path.exists() else {}
    merged.update(results)
    raw_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print("saved raw results")


if __name__ == "__main__":
    main()