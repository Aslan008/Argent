"""
src/leeth/dopt_export.py — ONNX-экспорт Universal Task Solver (DynamicOptionHead на mmBERT).

Чекпоинт v6/v7 (opt_head + unfrozen_layers layer_10/11) + база mmbert-argent →
ЕДИНЫЙ ONNX-граф:
    входы:  input_ids [B,L], attention_mask [B,L], state_mask [B,L], opt_positions [B,K]
    выход:  option_logits [B,K] (fp32, замаскированные -1e9 на пустых слотах)
- base: mmBERT encoder (hidden 768) с диффом размороженных слоёв из чекпоинта
- head: DynamicOptHead (state_mlp→512, opt_proj→512, dot/sqrt(512))
- mean-pool префикса до 1-го маркера и gather маркеров — В ГРАФЕ; поиск позиций
  маркеров — Python ([OPT] = 3 BPE-токена, ищется как подпоследовательность)

Формат промпта — байт-в-байт как в кернеле kernel_dopt (иначе модель вне распределения):
    "[STATE] {state} [QUESTION] {question} [OPT] {opt1} [OPT] {opt2} ..."

CLI:
    python -m src.leeth.dopt_export --ckpt .argent/models/mmbert_dynamic_opt_v1.pt \
        --base .argent/models/mmbert-argent --out .argent/models/dopt/dopt_v1.onnx
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Формат промпта и маркеры (консистентность с кернелом — критично)
# ---------------------------------------------------------------------------

def build_text(state_text: str, question: str, options: List[str]) -> str:
    """Единый формат промпта (байт-в-байт как build_text в kernel_dopt)."""
    return f"[STATE] {state_text} [QUESTION] {question} " + " ".join(f"[OPT] {o}" for o in options)


def find_marker_positions(ids: List[int], marker_ids: List[int]) -> List[int]:
    """Все вхождения подпоследовательности маркера [OPT] в токенах (как в кернеле)."""
    L = len(marker_ids)
    return [i for i in range(len(ids) - L + 1) if ids[i : i + L] == marker_ids]


# ---------------------------------------------------------------------------
# Модель (копия кернела — байт-в-байт семантика)
# ---------------------------------------------------------------------------

class DynamicOptHead(nn.Module):
    """Bi-encoder scorer: logit_i = (state_mlp(s) . opt_proj(o_i)) / sqrt(proj)."""

    def __init__(self, hidden_size: int = 768, proj_dim: int = 512):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim),
        )
        self.opt_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.scale = math.sqrt(proj_dim)

    def forward(self, s: torch.Tensor, o: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        s_p = self.state_mlp(s.float())   # [B, P]
        o_p = self.opt_proj(o.float())    # [B, K, P]
        logits = torch.einsum("bp,bkp->bk", s_p, o_p) / self.scale
        return logits.masked_fill(~valid, -1e9)


def load_dopt_torch(ckpt_path: Path, base_dir: Path, device: str = "cpu"):
    """Чекпоинт v6/v7 → (base, head, tok, marker_ids, ck). Дифф слоёв применяется к базе."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = AutoModelForMaskedLM.from_pretrained(str(base_dir), dtype=torch.float32).eval()
    layers = base.base_model.layers
    for name, sd in ck["unfrozen_layers"].items():
        layers[int(name.split("_")[1])].load_state_dict(sd)
    head = DynamicOptHead(ck.get("hidden_size", 768), ck.get("proj_dim", 512))
    head.load_state_dict(ck["opt_head"])
    head.eval()
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    marker_ids = list(ck.get("marker_ids") or tok.encode("[OPT]", add_special_tokens=False))
    return base, head, tok, marker_ids, ck


class DoptOnnxWrapper(nn.Module):
    """Единый граф для экспорта: encoder + pooling + gather + голова."""

    def __init__(self, base, head: DynamicOptHead):
        super().__init__()
        self.base = base.base_model  # encoder без MLM-головы (last_hidden_state)
        self.head = head

    def forward(
        self,
        input_ids: torch.Tensor,      # [B, L] long
        attention_mask: torch.Tensor,  # [B, L] long (1=токен, 0=паддинг)
        state_mask: torch.Tensor,      # [B, L] float (1=префикс до 1-го маркера)
        opt_positions: torch.Tensor,   # [B, K] long (позиции маркеров, -1=пусто)
    ) -> torch.Tensor:
        hs = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, L, H]
        sm = state_mask.unsqueeze(-1).to(hs.dtype)
        state_repr = (hs * sm).sum(1) / sm.sum(1).clamp(min=1.0)  # [B, H]
        idx = opt_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hs.size(-1))
        opt_reprs = hs.gather(1, idx)  # [B, K, H]
        valid = opt_positions >= 0
        return self.head(state_repr, opt_reprs, valid)  # [B, K] fp32


def encode_batch(
    texts: List[str],
    tok,
    marker_ids: List[int],
    max_length: int = 256,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Тексты → тензоры графа. Паддинг до макс. длины батча; K = max опций в батче."""
    enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    B, L = enc["input_ids"].shape
    ids_list = enc["input_ids"].tolist()
    pos_lists = [find_marker_positions(ids, marker_ids) for ids in ids_list]
    K = max(len(p) for p in pos_lists)
    state_mask = torch.zeros(B, L, dtype=torch.float32)
    opt_positions = torch.full((B, K), -1, dtype=torch.long)
    for j, pos in enumerate(pos_lists):
        if not pos:
            raise ValueError(f"маркер [OPT] не найден в промпте #{j}")
        state_mask[j, : pos[0]] = 1.0
        opt_positions[j, : len(pos)] = torch.tensor(pos, dtype=torch.long)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "state_mask": state_mask.to(device),
        "opt_positions": opt_positions.to(device),
    }


# ---------------------------------------------------------------------------
# Экспорт
# ---------------------------------------------------------------------------

def export_dopt_onnx(
    ckpt_path: Path,
    base_dir: Path,
    out_path: Path,
    opset_version: int = 17,
) -> Path:
    """Экспорт единого графа (encoder+pooling+голова) в ONNX."""
    base, head, tok, marker_ids, ck = load_dopt_torch(ckpt_path, base_dir)
    wrapper = DoptOnnxWrapper(base, head).eval()

    # Dummy-входы: L=64, K=4 (змейка) — формы динамические
    dummy = {
        "input_ids": torch.randint(100, 1000, (1, 64), dtype=torch.long),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
        "state_mask": torch.ones(1, 64, dtype=torch.float32),
        "opt_positions": torch.tensor([[10, 20, 30, 40]], dtype=torch.long),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"], dummy["state_mask"], dummy["opt_positions"]),
        str(out_path),
        input_names=["input_ids", "attention_mask", "state_mask", "opt_positions"],
        output_names=["option_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "state_mask": {0: "batch", 1: "seq_len"},
            "opt_positions": {0: "batch", 1: "num_options"},
            "option_logits": {0: "batch", 1: "num_options"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"экспортирован {out_path} ({size_mb:.0f} MB) за {time.time() - t0:.0f}s")
    return out_path


# ---------------------------------------------------------------------------
# Верификация: torch vs ONNX на реальных промптах всех доменов
# ---------------------------------------------------------------------------

def verify_dopt_onnx(
    ckpt_path: Path,
    base_dir: Path,
    onnx_path: Path,
    n_per_domain: int = 8,
    atol: float = 0.02,
) -> Dict[str, float]:
    """Граф ONNX == torch (экспорт корректен) И == кернел-путь (pooling в Python).
    Промпты — реальные сэмплы генераторов (snake/maze/ttt + OOD-размеры).

    Порог 0.02, а не 1e-4: fp32-reduction порядки ORT и torch различаются, на
    12-слойном энкодере наблюдаемый max-дифф логитов ~1.6e-3 при 100% совпадении
    argmax (28/28) — выбор хода численно идентичен, абсолютный дифф — артефакт
    порядка суммирования. Жёсткое требование: argmax_mismatches == 0."""
    import onnxruntime as ort

    sys.path.insert(0, str(Path(__file__).parent))
    from task_domains import GENERATORS  # src/leeth/task_domains.py

    import random

    base, head, tok, marker_ids, ck = load_dopt_torch(ckpt_path, base_dir)
    wrapper = DoptOnnxWrapper(base, head).eval()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_names = [i.name for i in sess.get_inputs()]

    rng = random.Random(20261003)
    texts: List[str] = []
    for domain, gen in GENERATORS.items():
        made, attempts = 0, 0
        while made < n_per_domain and attempts < n_per_domain * 5:
            attempts += 1
            s = gen(rng)
            if s is not None:
                texts.append(build_text(s.state_text, s.question, s.options))
                made += 1
    # OOD: змейка 12x12 (zero-shot размер)
    made, attempts = 0, 0
    while made < 4 and attempts < 40:
        attempts += 1
        s = GENERATORS["snake"](rng, min_size=12, max_size=12)
        if s is not None:
            texts.append(build_text(s.state_text, s.question, s.options))
            made += 1

    max_diff_graph, max_diff_kernel, argmax_mismatch = 0.0, 0.0, 0
    for i in range(0, len(texts), 4):
        chunk = texts[i : i + 4]
        t = encode_batch(chunk, tok, marker_ids)
        with torch.no_grad():
            ref_graph = wrapper(**t)  # torch-исполнение экспортируемого графа
            # кернел-путь: pooling/gather в Python (эталон обучения)
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            hs = base.base_model(**enc).last_hidden_state
            ref_kernel = []
            for j, text in enumerate(chunk):
                ids = enc["input_ids"][j].tolist()
                pos = find_marker_positions(ids, marker_ids)
                pm = torch.zeros(hs.shape[1])
                pm[: pos[0]] = 1.0
                sr = (hs[j] * pm.unsqueeze(-1)).sum(0) / pm.sum().clamp(min=1.0)
                o = torch.zeros(len(pos), hs.shape[-1])
                o[:] = hs[j, pos]
                v = torch.ones(len(pos), dtype=torch.bool)
                ref_kernel.append(head(sr.unsqueeze(0), o.unsqueeze(0), v.unsqueeze(0))[0])
            ref_kernel = torch.stack(ref_kernel)
        ort_out = sess.run(None, {
            "input_ids": t["input_ids"].numpy(),
            "attention_mask": t["attention_mask"].numpy(),
            "state_mask": t["state_mask"].numpy(),
            "opt_positions": t["opt_positions"].numpy(),
        })[0]
        d_graph = float(np.max(np.abs(ref_graph.numpy() - ort_out)))
        d_kernel = float(np.max(np.abs(ref_kernel.numpy() - ort_out)))
        max_diff_graph = max(max_diff_graph, d_graph)
        max_diff_kernel = max(max_diff_kernel, d_kernel)
        argmax_mismatch += int((ref_graph.argmax(-1) != torch.from_numpy(ort_out).argmax(-1)).sum())

    metrics = {
        "max_diff_onnx_vs_torch_graph": max_diff_graph,
        "max_diff_onnx_vs_kernel_path": max_diff_kernel,
        "argmax_mismatches": argmax_mismatch,
        "n_prompts": len(texts),
        "atol": atol,
        "pass": bool(max_diff_graph < atol and max_diff_kernel < atol and argmax_mismatch == 0),
    }
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Export DoPT task solver to ONNX")
    parser.add_argument("--ckpt", default=".argent/models/mmbert_dynamic_opt_v1.pt")
    parser.add_argument("--base", default=".argent/models/mmbert-argent")
    parser.add_argument("--out", default=".argent/models/dopt/dopt_v1.onnx")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    out = export_dopt_onnx(Path(args.ckpt), Path(args.base), Path(args.out))
    if not args.skip_verify:
        metrics = verify_dopt_onnx(Path(args.ckpt), Path(args.base), out)
        if not metrics["pass"]:
            raise SystemExit(f"VERIFY FAIL: {metrics}")
        print("VERIFY OK")


if __name__ == "__main__":
    main()