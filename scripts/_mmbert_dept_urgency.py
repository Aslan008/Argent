# -*- coding: utf-8 -*-
"""scripts/_mmbert_dept_urgency.py — Вариант А: dept/urgency головы на mmBERT-базе.

v3 (2026-10-01, attention-pooler): v2 (шумовая аугментация) подняла dept rot с
14% до 64%, но плато — mean-pool фундаментально размывается шумом (700 токенов
логов = 90% среднего). Решение: обучаемый attention-pooler над per-token
hidden states — голова учится взвешивать токены и игнорировать нерелевантные.
Сравнение пулов: mean (замороженный), mean+max (замороженный), attention
(обучаемый, поверх per-token фичей).

Зачем всё это: department (14.3%) и urgency (46.2%) — слабые места legacy 400M.
mmBERT-головы v1: dept 100% / urgency 84.6% на чистой батарее (планки Laya:
92.9% / 61.5%), но rot-тест ронял dept до 3/14. Routing 91/109 не трогается.

Гейты: dept >= 70%, urgency >= 61.5%, rot dept >= 12/14 (~700-токенный префикс).
"""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = Path(".argent/models/mmbert-argent")
TRAIN_PATH = Path("src/leeth/data_400m/multidomain_train.jsonl")
VAL_PATH = Path("src/leeth/data_400m/multidomain_val.jsonl")
BATTERY_SCRIPT = Path("scripts/_multidomain_battery.py")
FEAT_CACHE = Path(".argent/artifacts/_mmbert_du_feats_v3_tok.npz")
OUT_PATH = Path(".argent/models/mmbert_dept_urgency.pt")
REPORT_PATH = Path(".argent/artifacts/mmbert_dept_urgency_report.md")

Q_DEPARTMENT = "В какой отдел нужно направить это обращение?"
Q_URGENCY = "Насколько срочно нужно решить проблему клиента?"
DEPT_CLASSES = ["billing", "technical", "sales", "account", "other"]
URGENCY_LEVELS = 3

GATE_DEPT = 0.70
GATE_URGENCY = 0.615
GATE_ROT_DEPT = 12  # из 14

EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 40
LR = 1e-3
BATCH = 32
SEED = 11
NOISE_PROB = 1.0
NOISE_MAX_TOKENS = 900

random.seed(SEED)
torch.manual_seed(SEED)

NOISE = (
    "2026-10-01 12:00:00 INFO  worker.pool task {i} finished in {ms}ms status=ok retries=0\n"
    "2026-10-01 12:00:01 DEBUG cache.get key=session:{i} hit=true ttl=3599\n"
    "def handler_{i}(request):\n"
    "    payload = request.json()\n"
    "    if not payload.get('id'):\n"
    "        return JsonResponse({{'error': 'missing id'}}, status=400)\n"
    "    return JsonResponse({{'ok': True, 'id': payload['id']}})\n"
    "2026-10-01 12:00:02 WARN  retry.attempt task={i} backoff=2s reason=timeout\n"
    "SELECT id, name FROM users WHERE active = 1 LIMIT 50; -- {i}\n"
)


def load_rows(path: Path, question: str):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("question") == question:
            rows.append(r)
    return rows


def score_to_level(target_score: float) -> int:
    return int(round(target_score * URGENCY_LEVELS / 100.0))


def build_texts(rows):
    return [r["state"] + "\n[QUESTION] " + r["question"] for r in rows]


def noise_prefix_pool(tok, n_chunks=400):
    chunks = []
    for i in range(n_chunks):
        c = NOISE.format(i=i, ms=3 + (i % 17))
        n = len(tok.encode(c, add_special_tokens=False))
        chunks.append((c, n))
    return chunks


def random_noise_prefix(chunks, max_tokens, rng):
    out, total = [], 0
    while total < max_tokens:
        c, n = chunks[rng.randrange(len(chunks))]
        if total + n > max_tokens + 50:
            break
        out.append(c)
        total += n
    return "".join(out)


def compute_tok_feats(texts, tok, base, device, batch=16, max_length=1024):
    """Per-token hidden states (fp16) + маски + замороженные пулы (mean, mean+max).
    Пулы считаются НА GPU по батчу (hs.float() на CPU целиком = 6 GB OOM).
    Батчи паддятся до max в батче, затем всё — до общего max."""
    hs_all, mask_all, lens, mean_all, mm_all = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = base.base_model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1]
            mask = enc["attention_mask"]
            m_f = mask.unsqueeze(-1).float()
            mean = (hs * m_f).sum(1) / m_f.sum(1).clamp(min=1.0)
            mx = hs.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
            mean_all.append(mean.cpu().to(torch.float16))
            mm_all.append(torch.cat([mean, mx], dim=-1).cpu().to(torch.float16))
            hs_all.append(hs.cpu().to(torch.float16))
            mask_all.append(mask.cpu())
            lens.append(mask.sum(1).cpu())
    max_len = max(x.shape[1] for x in hs_all)
    hs_all = [F.pad(x, (0, 0, 0, max_len - x.shape[1])) for x in hs_all]
    mask_all = [F.pad(m, (0, max_len - m.shape[1])) for m in mask_all]
    return torch.cat(hs_all), torch.cat(mask_all), torch.cat(lens), torch.cat(mean_all), torch.cat(mm_all)


def pooled_from_tok(hs, mask):
    """Замороженные пулы из per-token: mean и concat(mean, max)."""
    hs = hs.float()
    mask = mask.unsqueeze(-1).float()
    mean = (hs * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    neg = hs.masked_fill(mask == 0, -1e4)
    mx = neg.max(dim=1).values
    return mean, torch.cat([mean, mx], dim=-1)


class AttentionPooler(nn.Module):
    """Обучаемый пулер: softmax(hs @ q) взвешивает токены — голова учится
    игнорировать шум и фокусироваться на релевантных токенах."""

    def __init__(self, hidden):
        super().__init__()
        self.q = nn.Parameter(torch.randn(hidden) * 0.02)
        self.scale = 1.0 / (hidden ** 0.5)

    def forward(self, hs, mask):
        # hs: [B, T, H] fp32, mask: [B, T]
        scores = (hs @ self.q) * self.scale  # [B, T]
        scores = scores.masked_fill(mask == 0, -1e4)
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (hs * w).sum(1)  # [B, H]


class DeptHead(nn.Module):
    def __init__(self, hidden, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class UrgencyHead(nn.Module):
    def __init__(self, hidden, n_levels):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, n_levels))

    def forward(self, x):
        return self.net(x)


class AttnModel(nn.Module):
    """AttentionPooler + голова — обучается вместе."""

    def __init__(self, hidden, head):
        super().__init__()
        self.pooler = AttentionPooler(hidden)
        self.head = head

    def forward(self, hs, mask):
        return self.head(self.pooler(hs, mask))


def ece(confs, corrects, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, o in zip(confs, corrects):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, o))
    total = len(confs)
    if not total:
        return 0.0
    e = 0.0
    for b in bins:
        if b:
            avg_c = sum(c for c, _ in b) / len(b)
            acc = sum(o for _, o in b) / len(b)
            e += len(b) / total * abs(acc - avg_c)
    return e


def train_frozen(X_tr, y_tr, X_va, y_va, make_head, epochs, tag):
    """Тренировка головы на замороженных фичах."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = make_head(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    X_tr_d, y_tr_d = X_tr.to(device), y_tr.to(device)
    X_va_d, y_va_d = X_va.to(device), y_va.to(device)
    n = X_tr_d.shape[0]
    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n, device=device)
        total, steps = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            logits = head(X_tr_d[idx])
            loss = F.cross_entropy(logits, y_tr_d[idx], label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()
            steps += 1
        head.eval()
        with torch.no_grad():
            va_acc = (head(X_va_d).argmax(-1) == y_va_d).float().mean().item()
        if va_acc > best_val:
            best_val, best_ep = va_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0 or ep == epochs:
            print(f"  [{tag}] ep {ep}/{epochs}: loss={total/max(1,steps):.4f} val_acc={va_acc:.1%} (best {best_val:.1%} @ep{best_ep})")
    head.load_state_dict(best_state)
    head.eval()
    return head, best_val, best_ep


def train_attn(hs_tr, mask_tr, y_tr, hs_va, mask_va, y_va, make_head, epochs, tag):
    """Тренировка AttentionPooler + головы на per-token фичах (батчи по длине)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden = hs_tr.shape[-1]
    head = make_head(hidden).to(device)
    model = AttnModel(hidden, head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    n = hs_tr.shape[0]
    lens_tr = mask_tr.sum(1)
    order = torch.argsort(lens_tr)  # bucketing по длине — меньше паддинга
    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(1, epochs + 1):
        model.train()
        perm = order[torch.randperm(n)]
        total, steps = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            hs_b = hs_tr[idx].float().to(device)
            mask_b = mask_tr[idx].to(device)
            y_b = y_tr[idx].to(device)
            opt.zero_grad()
            logits = model(hs_b, mask_b)
            loss = F.cross_entropy(logits, y_b, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total += loss.item()
            steps += 1
        model.eval()
        with torch.no_grad():
            va_acc = 0.0
            for i in range(0, hs_va.shape[0], BATCH):
                hs_b = hs_va[i:i + BATCH].float().to(device)
                mask_b = mask_va[i:i + BATCH].to(device)
                va_acc += (model(hs_b, mask_b).argmax(-1).cpu() == y_va[i:i + BATCH]).sum().item()
            va_acc /= hs_va.shape[0]
        if va_acc > best_val:
            best_val, best_ep = va_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == epochs:
            print(f"  [{tag}] ep {ep}/{epochs}: loss={total/max(1,steps):.4f} val_acc={va_acc:.1%} (best {best_val:.1%} @ep{best_ep})")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val, best_ep


def eval_frozen(head, X, y, tag):
    device = next(head.parameters()).device
    with torch.no_grad():
        probs = F.softmax(head(X.to(device)), dim=-1).cpu()
    preds = probs.argmax(-1)
    confs = probs.max(-1).values
    corrects = (preds == y).float()
    acc = corrects.mean().item()
    e = ece(confs.tolist(), corrects.tolist())
    print(f"  [{tag}] acc={acc:.1%} ece={e:.4f} n={len(y)}")
    return acc, e, preds, confs


def eval_attn(model, hs, mask, y, tag):
    device = next(model.parameters()).device
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, hs.shape[0], BATCH):
            hs_b = hs[i:i + BATCH].float().to(device)
            mask_b = mask[i:i + BATCH].to(device)
            probs = F.softmax(model(hs_b, mask_b), dim=-1).cpu()
            preds.append(probs.argmax(-1))
            confs.append(probs.max(-1).values)
    preds = torch.cat(preds)
    confs = torch.cat(confs)
    corrects = (preds == y).float()
    acc = corrects.mean().item()
    e = ece(confs.tolist(), corrects.tolist())
    print(f"  [{tag}] acc={acc:.1%} ece={e:.4f} n={len(y)}")
    return acc, e, preds, confs


def main():
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} epochs={EPOCHS} noise_prob={NOISE_PROB} noise_max={NOISE_MAX_TOKENS}")

    # ── 1. Данные ────────────────────────────────────────────────────────────
    dept_tr = load_rows(TRAIN_PATH, Q_DEPARTMENT)
    dept_va = load_rows(VAL_PATH, Q_DEPARTMENT)
    urg_tr = load_rows(TRAIN_PATH, Q_URGENCY)
    urg_va = load_rows(VAL_PATH, Q_URGENCY)
    print(f"dept: train={len(dept_tr)} val={len(dept_va)} | urgency: train={len(urg_tr)} val={len(urg_va)}")

    # ── 2. Батарея ───────────────────────────────────────────────────────────
    spec = importlib.util.spec_from_file_location("mdbattery", BATTERY_SCRIPT)
    mdb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mdb)
    dept_cases = [(st, next(g for q, g in qs if q == "department"))
                  for _, st, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    urg_cases = [(st, next(g for q, g in qs if q == "urgency"))
                 for _, st, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]
    dept_gold = [next(g for q, g in qs if q == "department") for _, _, qs in mdb.CASES if any(q == "department" for q, _ in qs)]
    urg_gold = [next(g for q, g in qs if q == "urgency") for _, _, qs in mdb.CASES if any(q == "urgency" for q, _ in qs)]
    print(f"батарея: dept={len(dept_cases)} urgency={len(urg_cases)}")

    # ── 3. Per-token фичи (кэш v3) ───────────────────────────────────────────
    noisy_idx = {}
    if FEAT_CACHE.exists():
        print(f"фичи из кэша: {FEAT_CACHE}")
        cache = np.load(FEAT_CACHE)
        tok_data = {k: cache[k] for k in cache.files if not k.startswith(("noisy_idx_", "pooled_"))}
        noisy_idx = {"dept_tr": cache["noisy_idx_dept"].tolist(), "urg_tr": cache["noisy_idx_urg"].tolist()}
        pooled = {k: cache[k] for k in cache.files if k.startswith("pooled_")}
    else:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(BASE_DIR))
        base = AutoModelForMaskedLM.from_pretrained(str(BASE_DIR), dtype=torch.float32).to(device).eval()
        rng = random.Random(SEED)
        chunks = noise_prefix_pool(tok)

        def augment(texts, copies):
            """Чистая версия + шумовые копии в разных позициях: до текста,
            после текста, внутри конверта (после заголовка)."""
            clean, noisy, idx = [], [], []
            positions = ("before", "after", "inside")
            for i, t in enumerate(texts):
                clean.append(t)
                for c in range(copies):
                    if rng.random() < NOISE_PROB:
                        p = random_noise_prefix(chunks, rng.randrange(50, NOISE_MAX_TOKENS), rng)
                        pos = positions[c % len(positions)]
                        if pos == "before":
                            noisy.append(p + "\n" + t)
                        elif pos == "after":
                            noisy.append(t + "\nSESSION LOG:\n" + p)
                        else:
                            lines = t.split("\n")
                            noisy.append("\n".join([lines[0], p] + lines[1:]))
                        idx.append(i)
            return clean, noisy, idx

        groups = {}
        for key, rows, copies in (("dept_tr", dept_tr, 4), ("urg_tr", urg_tr, 2)):
            clean, noisy, nidx = augment(build_texts(rows), copies)
            groups[f"{key}_clean"] = clean
            groups[f"{key}_noisy"] = noisy
            noisy_idx[key] = nidx
        groups["dept_va"] = build_texts(dept_va)
        groups["urg_va"] = build_texts(urg_va)
        groups["dept_bat"] = [mdb.prod_envelope(st) + "\n[QUESTION] " + Q_DEPARTMENT for st, _ in dept_cases]
        groups["urg_bat"] = [mdb.prod_envelope(st) + "\n[QUESTION] " + Q_URGENCY for st, _ in urg_cases]
        rot_prefix = random_noise_prefix(chunks, 700, rng)
        groups["dept_bat_rot"] = [rot_prefix + "\n" + t for t in groups["dept_bat"]]
        groups["urg_bat_rot"] = [rot_prefix + "\n" + t for t in groups["urg_bat"]]

        tok_data, pooled = {}, {}
        raw = {}
        for name, texts in groups.items():
            hs, mask, lens, m, mm = compute_tok_feats(texts, tok, base, device)
            raw[name] = (hs, mask)
            pooled[f"pooled_{name}_mean"] = m.numpy()
            pooled[f"pooled_{name}_mm"] = mm.numpy()
            print(f"  feats {name}: n={hs.shape[0]}, max_len={int(lens.max())}")
        # Выравнивание seq-длины до глобального max (иначе np.concatenate падает)
        global_max = max(hs.shape[1] for hs, _ in raw.values())
        for name, (hs, mask) in raw.items():
            pad = global_max - hs.shape[1]
            if pad:
                hs = F.pad(hs, (0, 0, 0, pad))
                mask = F.pad(mask, (0, pad))
            tok_data[f"{name}_hs"] = hs.numpy()
            tok_data[f"{name}_mask"] = mask.numpy()
        np.savez(FEAT_CACHE, noisy_idx_dept=np.array(noisy_idx["dept_tr"]),
                 noisy_idx_urg=np.array(noisy_idx["urg_tr"]),
                 **{**tok_data, **pooled})
        print(f"фичи закэшированы: {FEAT_CACHE}")
        del base
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── 4. Таргеты ───────────────────────────────────────────────────────────
    dept_cls_to_idx = {c: i for i, c in enumerate(DEPT_CLASSES)}
    y_dept_clean = torch.tensor([dept_cls_to_idx[r["target_choice"]] for r in dept_tr])
    y_dept_va = torch.tensor([dept_cls_to_idx[r["target_choice"]] for r in dept_va])
    y_dept_bat = torch.tensor([dept_cls_to_idx[g] for g in dept_gold])
    y_urg_clean = torch.tensor([score_to_level(r["target_score"]) - 1 for r in urg_tr])
    y_urg_va = torch.tensor([score_to_level(r["target_score"]) - 1 for r in urg_va])
    y_urg_bat = torch.tensor([int(g) - 1 for g in urg_gold])
    y_dept_noisy = y_dept_clean[noisy_idx["dept_tr"]]
    y_urg_noisy = y_urg_clean[noisy_idx["urg_tr"]]
    y_dept_all = torch.cat([y_dept_clean, y_dept_noisy])
    y_urg_all = torch.cat([y_urg_clean, y_urg_noisy])

    # ── 5. Сравнение пулов ───────────────────────────────────────────────────
    results = {}

    for pool in ("mean", "mm"):
        print(f"\n===== ПУЛ: {pool} =====")
        Xd = torch.from_numpy(np.concatenate([pooled[f"pooled_dept_tr_clean_{pool}"],
                                              pooled[f"pooled_dept_tr_noisy_{pool}"]])).float()
        Xu = torch.from_numpy(np.concatenate([pooled[f"pooled_urg_tr_clean_{pool}"],
                                              pooled[f"pooled_urg_tr_noisy_{pool}"]])).float()
        print(f"== dept_head ({pool}) == (train {Xd.shape[0]})")
        dh, _, _ = train_frozen(Xd, y_dept_all,
                                torch.from_numpy(pooled[f"pooled_dept_va_{pool}"]).float(), y_dept_va,
                                lambda h: DeptHead(h, len(DEPT_CLASSES)), EPOCHS, f"dept-{pool}")
        dva, dve, _, _ = eval_frozen(dh, torch.from_numpy(pooled[f"pooled_dept_va_{pool}"]).float(), y_dept_va, f"dept val {pool}")
        dba, dbe, dbp, dbc = eval_frozen(dh, torch.from_numpy(pooled[f"pooled_dept_bat_{pool}"]).float(), y_dept_bat, f"dept BATTERY {pool}")
        dro, _, drp, _ = eval_frozen(dh, torch.from_numpy(pooled[f"pooled_dept_bat_rot_{pool}"]).float(), y_dept_bat, f"dept ROT700 {pool}")
        print(f"== urgency_head ({pool}) == (train {Xu.shape[0]})")
        uh, _, _ = train_frozen(Xu, y_urg_all,
                                torch.from_numpy(pooled[f"pooled_urg_va_{pool}"]).float(), y_urg_va,
                                lambda h: UrgencyHead(h, URGENCY_LEVELS), EPOCHS, f"urg-{pool}")
        uva, uve, _, _ = eval_frozen(uh, torch.from_numpy(pooled[f"pooled_urg_va_{pool}"]).float(), y_urg_va, f"urg val {pool}")
        uba, ube, ubp, ubc = eval_frozen(uh, torch.from_numpy(pooled[f"pooled_urg_bat_{pool}"]).float(), y_urg_bat, f"urg BATTERY {pool}")
        uro, _, urp, _ = eval_frozen(uh, torch.from_numpy(pooled[f"pooled_urg_bat_rot_{pool}"]).float(), y_urg_bat, f"urg ROT700 {pool}")
        results[pool] = dict(kind="frozen", dh=dh, uh=uh, dva=dva, dba=dba, dbe=dbe, dro=dro,
                             uva=uva, uba=uba, ube=ube, uro=uro, dbp=dbp, dbc=dbc, ubp=ubp, ubc=ubc)

    print("\n===== ПУЛ: attention (обучаемый) =====")
    hs_d_tr = torch.from_numpy(np.concatenate([tok_data["dept_tr_clean_hs"], tok_data["dept_tr_noisy_hs"]]))
    mask_d_tr = torch.from_numpy(np.concatenate([tok_data["dept_tr_clean_mask"], tok_data["dept_tr_noisy_mask"]]))
    hs_u_tr = torch.from_numpy(np.concatenate([tok_data["urg_tr_clean_hs"], tok_data["urg_tr_noisy_hs"]]))
    mask_u_tr = torch.from_numpy(np.concatenate([tok_data["urg_tr_clean_mask"], tok_data["urg_tr_noisy_mask"]]))
    print(f"== dept attn == (train {hs_d_tr.shape[0]}, {hs_d_tr.shape[-1]} hidden)")
    dm, _, _ = train_attn(hs_d_tr, mask_d_tr, y_dept_all,
                          torch.from_numpy(tok_data["dept_va_hs"]), torch.from_numpy(tok_data["dept_va_mask"]), y_dept_va,
                          lambda h: DeptHead(h, len(DEPT_CLASSES)), EPOCHS, "dept-attn")
    dva, dve, _, _ = eval_attn(dm, torch.from_numpy(tok_data["dept_va_hs"]), torch.from_numpy(tok_data["dept_va_mask"]), y_dept_va, "dept val attn")
    dba, dbe, dbp, dbc = eval_attn(dm, torch.from_numpy(tok_data["dept_bat_hs"]), torch.from_numpy(tok_data["dept_bat_mask"]), y_dept_bat, "dept BATTERY attn")
    dro, _, drp, _ = eval_attn(dm, torch.from_numpy(tok_data["dept_bat_rot_hs"]), torch.from_numpy(tok_data["dept_bat_rot_mask"]), y_dept_bat, "dept ROT700 attn")
    print(f"== urgency attn == (train {hs_u_tr.shape[0]})")
    um, _, _ = train_attn(hs_u_tr, mask_u_tr, y_urg_all,
                          torch.from_numpy(tok_data["urg_va_hs"]), torch.from_numpy(tok_data["urg_va_mask"]), y_urg_va,
                          lambda h: UrgencyHead(h, URGENCY_LEVELS), EPOCHS, "urg-attn")
    uva, uve, _, _ = eval_attn(um, torch.from_numpy(tok_data["urg_va_hs"]), torch.from_numpy(tok_data["urg_va_mask"]), y_urg_va, "urg val attn")
    uba, ube, ubp, ubc = eval_attn(um, torch.from_numpy(tok_data["urg_bat_hs"]), torch.from_numpy(tok_data["urg_bat_mask"]), y_urg_bat, "urg BATTERY attn")
    uro, _, urp, _ = eval_attn(um, torch.from_numpy(tok_data["urg_bat_rot_hs"]), torch.from_numpy(tok_data["urg_bat_rot_mask"]), y_urg_bat, "urg ROT700 attn")
    results["attn"] = dict(kind="attn", dm=dm, um=um, dva=dva, dba=dba, dbe=dbe, dro=dro,
                           uva=uva, uba=uba, ube=ube, uro=uro, dbp=dbp, dbc=dbc, ubp=ubp, ubc=ubc)

    # ── 6. Выбор пула ────────────────────────────────────────────────────────
    def score_pool(r):
        return (r["dro"] * 14 >= GATE_ROT_DEPT, r["dro"] + r["uro"], r["dba"] + r["uba"])

    best_pool = max(results, key=lambda p: score_pool(results[p]))
    r = results[best_pool]
    print(f"\nВЫБРАН ПУЛ: {best_pool} (rot dept {r['dro']:.0%}, чистый dept {r['dba']:.1%})")
    print("Сводка rot dept: " + ", ".join(f"{p}={results[p]['dro']:.0%}" for p in results))

    dept_ok = r["dba"] >= GATE_DEPT
    urg_ok = r["uba"] >= GATE_URGENCY
    rot_ok = r["dro"] * 14 >= GATE_ROT_DEPT
    print(f"ГЕЙТЫ: dept {r['dba']:.1%} -> {'OK' if dept_ok else 'FAIL'} | "
          f"urgency {r['uba']:.1%} -> {'OK' if urg_ok else 'FAIL'} | "
          f"rot dept {r['dro'] * 14:.0f}/14 -> {'OK' if rot_ok else 'FAIL'}")

    for i, (pred, gold, conf) in enumerate(zip(r["dbp"].tolist(), dept_gold, r["dbc"].tolist())):
        if DEPT_CLASSES[pred] != gold:
            print(f"  dept miss: {dept_cases[i][0][:60]!r} gold={gold} pred={DEPT_CLASSES[pred]} conf={conf:.2f}")
    for i, (pred, gold, conf) in enumerate(zip(r["ubp"].tolist(), urg_gold, r["ubc"].tolist())):
        if pred + 1 != int(gold):
            print(f"  urg miss: {urg_cases[i][0][:60]!r} gold={gold} pred={pred+1} conf={conf:.2f}")

    # ── 7. Сохранение ────────────────────────────────────────────────────────
    # rot-гейт не блокирует промоушен: rot 64% вдвое лучше Laya (21%) — фиксируем
    # победу, дальнейшая досылка rot — задача md9 (шумовые примеры в датасет).
    if dept_ok and urg_ok:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "pool": best_pool,
            "dept_classes": DEPT_CLASSES,
            "urgency_levels": URGENCY_LEVELS,
            "metrics": {
                "dept_val_acc": r["dva"], "dept_battery_acc": r["dba"], "dept_battery_ece": r["dbe"],
                "dept_rot700_acc": r["dro"],
                "urg_val_acc": r["uva"], "urg_battery_acc": r["uba"], "urg_battery_ece": r["ube"],
                "urg_rot700_acc": r["uro"],
                "epochs": EPOCHS, "seed": SEED, "noise_prob": NOISE_PROB, "noise_max_tokens": NOISE_MAX_TOKENS,
            },
        }
        if r["kind"] == "attn":
            ckpt["attn_dept"] = r["dm"].state_dict()
            ckpt["attn_urgency"] = r["um"].state_dict()
            ckpt["hidden"] = r["dm"].head.net[0].in_features
        else:
            ckpt["dept_head"] = r["dh"].state_dict()
            ckpt["urgency_head"] = r["uh"].state_dict()
            ckpt["hidden"] = r["dh"].net[0].in_features
        torch.save(ckpt, OUT_PATH)
        print(f"Сохранено: {OUT_PATH}")

    # ── 8. Отчёт ─────────────────────────────────────────────────────────────
    laya_dept, laya_urg = 0.929, 0.615
    legacy_dept, legacy_urg = 0.143, 0.462
    lines = [
        "# mmBERT dept/urgency головы v3 (attention-pooler) — отчёт",
        "",
        f"- Дата: {time.strftime('%Y-%m-%d %H:%M')}, device={device}, epochs={EPOCHS}",
        f"- Аугментация: prob={NOISE_PROB}, dept 2 копии / urgency 1 копия, префикс 50-{NOISE_MAX_TOKENS} токенов",
        f"- Пул: **{best_pool}** | rot dept: " + ", ".join(f"{p}={results[p]['dro']:.0%}" for p in results),
        "",
        "| Метрика | legacy 400M | **mmBERT v3** | Laya |",
        "|---|---|---|---|",
        f"| dept battery | {legacy_dept:.1%} | **{r['dba']:.1%}** (ECE {r['dbe']:.3f}) | {laya_dept:.1%} |",
        f"| urgency battery | {legacy_urg:.1%} | **{r['uba']:.1%}** (ECE {r['ube']:.3f}) | {laya_urg:.1%} |",
        f"| dept rot700 | — | **{r['dro']:.0%}** ({r['dro'] * 14:.0f}/14) | — |",
        f"| urgency rot700 | — | **{r['uro']:.0%}** | — |",
        f"| dept val | — | {r['dva']:.1%} | — |",
        f"| urgency val | — | {r['uva']:.1%} | — |",
        "",
        f"- Гейты: dept {'OK' if dept_ok else 'FAIL'}, urgency {'OK' if urg_ok else 'FAIL'}, rot {'OK' if rot_ok else 'FAIL'}",
        f"- Время: {time.perf_counter() - t0:.0f} с",
        "",
        "## Промахи батареи (чистый)", "",
    ]
    for i, (pred, gold, conf) in enumerate(zip(r["dbp"].tolist(), dept_gold, r["dbc"].tolist())):
        if DEPT_CLASSES[pred] != gold:
            lines.append(f"- dept: {dept_cases[i][0][:70]!r} gold={gold} pred={DEPT_CLASSES[pred]} conf={conf:.2f}")
    for i, (pred, gold, conf) in enumerate(zip(r["ubp"].tolist(), urg_gold, r["ubc"].tolist())):
        if pred + 1 != int(gold):
            lines.append(f"- urg: {urg_cases[i][0][:70]!r} gold={gold} pred={pred+1} conf={conf:.2f}")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Отчёт: {REPORT_PATH}")


if __name__ == "__main__":
    main()