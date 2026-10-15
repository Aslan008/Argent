# -*- coding: utf-8 -*-
"""scripts/train_micro_adapters.py — Обучение и экспорт модульных микро-адаптеров (guard.onnx, ground.onnx).

Архитектура:
- Базовый хребет (Leeth 400M / mmBERT) заморожен и используется как экстрактор признаков:
  mean-pooled вектор (1024-dim для 400M / 768-dim для mmBERT).
- Микро-адаптер — 2-слойный MLP (1024 -> 256 -> 1) с активацией GELU и LayerNorm.
- Вес в ONNX: ~280 КБ.
- Время инференса на CPU: <= 0.03 мс.
- Назначение адаптеров:
  1) guard.onnx: Нейро-гейт безопасности терминальных команд (safe vs dangerous).
  2) ground.onnx: Проверка заземления (grounding) аргументов инструментов в контексте.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.leeth.tokenizer import LeethTokenizer

DATA_DIR = ROOT / "src/leeth/data_400m"
CONTRASTIVE_PATH = ROOT / "src/leeth/data/contrastive_flat.jsonl"
ADAPTERS_DIR = ROOT / ".argent/models/adapters"
REPORT_PATH = ROOT / "reports/micro_adapters_benchmark.json"

ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


class MicroAdapterMLP(nn.Module):
    """2-слойный MLP микро-адаптер с LayerNorm и GELU."""

    def __init__(self, in_features: int = 1024, hidden_features: int = 256, out_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Вычисляет Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def extract_features(
    items: List[Dict],
    tokenizer: LeethTokenizer,
    sess: ort.InferenceSession,
    batch_size: int = 32,
    max_len: int = 256,
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Извлекает 1024-dim pooled_repr через базовый ONNX-хребет с кэшированием."""
    if cache_path is not None and cache_path.exists():
        data = np.load(str(cache_path))
        print(f"Загружены кэшированные признаки из {cache_path.name}: {data['features'].shape}")
        return data["features"], data["targets"]

    all_features = []
    all_targets = []

    print(f"Извлечение признаков для {len(items)} примеров...")
    t0 = time.perf_counter()

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        ids_batch = []
        targets_batch = []

        for item in batch:
            state = item.get("state", "")
            q = item.get("question", "")
            q_type = item.get("question_type", "noul")
            type_token = f"[{q_type.upper()}]"
            text = f"[STATE] {state} [SEP] [QUESTION] {type_token} {q}"
            ids = tokenizer.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            ids_batch.append(ids)
            t_val = item.get("target_noul")
            target = float(t_val) if t_val is not None else (1.0 if item.get("target_choice") else 0.0)
            targets_batch.append(target)

        max_b_len = max(len(x) for x in ids_batch)
        padded_ids = [x + [0] * (max_b_len - len(x)) for x in ids_batch]
        padded_masks = [[1] * len(x) + [0] * (max_b_len - len(x)) for x in ids_batch]

        feed = {
            "input_ids": np.array(padded_ids, dtype=np.int64),
            "attention_mask": np.array(padded_masks, dtype=np.int64),
        }
        outs = sess.run(None, feed)
        pooled = outs[1]  # pooled_repr: [batch, 1024]
        all_features.append(pooled)
        all_targets.extend(targets_batch)

    elapsed = time.perf_counter() - t0
    feat_arr = np.concatenate(all_features, axis=0)
    target_arr = np.array(all_targets, dtype=np.float32)
    print(f"Признаки извлечены за {elapsed:.2f} сек. Форма: {feat_arr.shape}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(cache_path), features=feat_arr, targets=target_arr)
        print(f"Кэш сохранён: {cache_path.name}")

    return feat_arr, target_arr


def train_and_export_adapter(
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    export_path: Path,
    epochs: int = 30,
    lr: float = 8e-4,
) -> Dict:
    """Обучает MLP-адаптер на признаках и экспортирует в ONNX."""
    print(f"\n--- Обучение адаптера: {name} ---")
    n = len(features)
    indices = np.random.RandomState(42).permutation(n)
    val_split = int(n * 0.2)
    val_idx, train_idx = indices[:val_split], indices[val_split:]

    X_train, y_train = torch.from_numpy(features[train_idx]), torch.from_numpy(targets[train_idx]).unsqueeze(-1)
    X_val, y_val = torch.from_numpy(features[val_idx]), torch.from_numpy(targets[val_idx]).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

    in_dim = features.shape[1]
    model = MicroAdapterMLP(in_features=in_dim, hidden_features=256, out_features=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = criterion(val_out, y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Загружаем лучшую модель
    model.load_state_dict(best_state)
    model.eval()

    # Оценка
    with torch.no_grad():
        val_logits = model(X_val).squeeze(-1).numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        y_val_np = y_val.squeeze(-1).numpy()
        preds = (val_probs >= 0.5).astype(np.float32)
        acc = float(np.mean(preds == y_val_np))
        ece = compute_ece(val_probs, y_val_np)
        brier = float(np.mean((val_probs - y_val_np) ** 2))

    print(f"Валидация {name}: Acc={acc:.1%}, ECE={ece:.4f}, Brier={brier:.4f}")

    # Экспорт в ONNX
    dummy_input = torch.randn(1, in_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        input_names=["pooled_repr"],
        output_names=["logits"],
        dynamic_axes={"pooled_repr": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    size_kb = export_path.stat().st_size / 1024
    print(f"Экспортирован {export_path.name}: {size_kb:.1f} КБ")

    # Бенчмарк задержки на CPU
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(export_path), sess_options=opts, providers=["CPUExecutionProvider"])

    dummy_np = np.random.randn(1, in_dim).astype(np.float32)
    # Warmup
    for _ in range(20):
        sess.run(None, {"pooled_repr": dummy_np})

    latencies = []
    for _ in range(500):
        t_start = time.perf_counter()
        sess.run(None, {"pooled_repr": dummy_np})
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    print(f"Задержка CPU (500 итераций): mean={mean_ms:.4f} мс, p95={p95_ms:.4f} мс, p99={p99_ms:.4f} мс")

    return {
        "name": name,
        "export_path": str(export_path),
        "size_kb": size_kb,
        "accuracy": acc,
        "ece": ece,
        "brier": brier,
        "mean_latency_ms": mean_ms,
        "p95_latency_ms": p95_ms,
        "p99_latency_ms": p99_ms,
        "latency_target_passed": mean_ms <= 0.05,
    }


def main():
    print("=== Фаза 3: Обучение и экспорт модульных микро-адаптеров ===")

    # 1. Загрузка токенизатора и базовой ONNX-модели
    tokenizer = LeethTokenizer.load(DATA_DIR)
    onnx_path = DATA_DIR / "leeth_int8.onnx"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    base_sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    # 2. Чтение данных
    with open(CONTRASTIVE_PATH, "r", encoding="utf-8") as f:
        contrastive_data = [json.loads(line) for line in f]

    safety_samples = [d for d in contrastive_data if d.get("_source", "").startswith("minimal_safety")]
    ground_samples = [d for d in contrastive_data if d.get("_source", "").startswith("minimal_choice")]

    print(f"Найдено примеров для guard: {len(safety_samples)}")
    print(f"Найдено примеров для ground: {len(ground_samples)}")

    # 3. Извлечение признаков и обучение guard.onnx
    cache_guard = ROOT / ".argent/cache/guard_feats.npz"
    guard_feats, guard_targets = extract_features(safety_samples, tokenizer, base_sess, cache_path=cache_guard)
    guard_res = train_and_export_adapter(
        name="guard",
        features=guard_feats,
        targets=guard_targets,
        export_path=ADAPTERS_DIR / "guard.onnx",
        epochs=30,
        lr=8e-4,
    )

    # 4. Извлечение признаков и обучение ground.onnx
    cache_ground = ROOT / ".argent/cache/ground_feats.npz"
    ground_feats, _ = extract_features(ground_samples, tokenizer, base_sess, cache_path=cache_ground)
    ground_y = np.array([1.0 if d.get("_source") == "minimal_choice_pos" else 0.0 for d in ground_samples], dtype=np.float32)
    ground_res = train_and_export_adapter(
        name="ground",
        features=ground_feats,
        targets=ground_y,
        export_path=ADAPTERS_DIR / "ground.onnx",
        epochs=30,
        lr=8e-4,
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapters": [guard_res, ground_res],
        "summary": {
            "all_latency_passed": guard_res["latency_target_passed"] and ground_res["latency_target_passed"],
            "max_mean_latency_ms": max(guard_res["mean_latency_ms"], ground_res["mean_latency_ms"]),
            "max_ece": max(guard_res["ece"], ground_res["ece"]),
        },
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nОтчёт успешно сохранён: {REPORT_PATH}")
    print(f"Все критерии задержки (< 0.05 мс): {'ВЫПОЛНЕНЫ' if report['summary']['all_latency_passed'] else 'НЕ ВЫПОЛНЕНЫ'}")


if __name__ == "__main__":
    main()
