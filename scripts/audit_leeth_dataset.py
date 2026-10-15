# -*- coding: utf-8 -*-
"""scripts/audit_leeth_dataset.py — Типо-безопасный анти-шорткат аудит датасетов Leeth.

Реализует методологию аудита качества данных из Jeff (jeff-kit) и Bespoke Labs (Nimble):
1. Для 'choice':
   - Point-Biserial correlation (r_pb) между длиной строки варианта и истинностью.
   - Критерий согласия Пирсона (chi-squared) для проверки равномерности позиций правильного ответа.
2. Для 'noul':
   - Проверка баланса классов (доля 1.0 vs 0.0).
3. Для 'extract':
   - Проверка точного вхождения спана в текст состояния (state).
   - Проверка границ слов (отсутствие срезов в середине слова).
4. Для 'score':
   - Анализ распределения баллов и уровней рубрики.
5. Кросс-сплит аудит (при наличии train и val):
   - Поиск лексических утечек и точных дубликатов по полю state (3-gram Jaccard).

Запуск:
    python scripts/audit_leeth_dataset.py \
        --val src/leeth/data_400m/leeth_400m_val.jsonl \
        --train src/leeth/data_400m/leeth_400m_train.jsonl \
        --output reports/dataset_audit_report.json
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def compute_point_biserial_r(lengths_target: List[int], lengths_distractor: List[int]) -> float:
    """Вычисляет двурядно-точечный коэффициент корреляции (Point-Biserial r)

    для проверки гипотезы: «Правильный ответ систематически длиннее/короче дистракторов».
    0.0 = нет корреляции; > 0.15 = выраженный Length Shortcut.
    """
    n1 = len(lengths_target)
    n0 = len(lengths_distractor)
    n = n1 + n0
    if n1 == 0 or n0 == 0:
        return 0.0

    mean1 = sum(lengths_target) / n1
    mean0 = sum(lengths_distractor) / n0

    all_vals = lengths_target + lengths_distractor
    mean_all = sum(all_vals) / n
    variance = sum((x - mean_all) ** 2 for x in all_vals) / n
    std_all = math.sqrt(variance) if variance > 0 else 0.0

    if std_all == 0.0:
        return 0.0

    r_pb = ((mean1 - mean0) / std_all) * math.sqrt((n1 * n0) / (n * n))
    return float(r_pb)


def compute_chi_square_uniformity(counts: List[int]) -> Tuple[float, float]:
    """Критерий согласия хи-квадрат Пирсона против равномерного распределения.

    Возвращает: (chi2_stat, approximate_p_value).
    """
    k = len(counts)
    total = sum(counts)
    if k <= 1 or total == 0:
        return 0.0, 1.0

    expected = total / k
    chi2 = sum(((obs - expected) ** 2) / expected for obs in counts)

    # Приближенная оценка p-value через выживание хи-квадрат для степеней свободы df = k - 1
    df = k - 1
    try:
        from scipy.stats import chi2 as scipy_chi2
        p_val = float(1.0 - scipy_chi2.cdf(chi2, df))
    except Exception:
        # Аналитическая аппроксимация (Вильсон-Хильферти) при отсутствии scipy
        z = ((chi2 / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
        p_val = float(0.5 * math.erfc(z / math.sqrt(2.0))) if z > 0 else 1.0

    return float(chi2), float(max(0.0, min(1.0, p_val)))


def compute_jaccard_3gram(s1: str, s2: str) -> float:
    """Коэффициент сходства Жаккара по символьным 3-граммам."""
    if s1 == s2:
        return 1.0
    if len(s1) < 3 or len(s2) < 3:
        return 1.0 if s1 == s2 else 0.0

    grams1 = {s1[i : i + 3] for i in range(len(s1) - 2)}
    grams2 = {s2[i : i + 3] for i in range(len(s2) - 2)}
    inter = len(grams1 & grams2)
    union = len(grams1 | grams2)
    return inter / union if union > 0 else 0.0


def audit_file(filepath: Path) -> Dict[str, Any]:
    """Проводит типобезопасный аудит одного файла датасета .jsonl."""
    if not filepath.exists():
        return {"error": f"File {filepath} not found"}

    type_counts = Counter()
    choice_target_lengths: List[int] = []
    choice_distractor_lengths: List[int] = []
    choice_position_counts = Counter()
    choice_invalid_target_count = 0
    choice_total = 0

    noul_labels = Counter()
    noul_total = 0

    extract_total = 0
    extract_missing_span = 0
    extract_mid_word = 0

    score_total = 0
    score_values = []

    unique_states = set()
    duplicate_states = 0
    total_rows = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                continue

            total_rows += 1
            state = row.get("state", "")
            if state:
                if state in unique_states:
                    duplicate_states += 1
                else:
                    unique_states.add(state)

            q_type = row.get("question_type", "unknown")
            type_counts[q_type] += 1

            # 1. CHOICE
            if q_type == "choice":
                choice_total += 1
                choices = row.get("choices") or []
                target = row.get("target_choice")
                target_idx = row.get("target_choice_idx")

                if target is None and target_idx is not None and 0 <= target_idx < len(choices):
                    target = choices[target_idx]

                if target not in choices:
                    choice_invalid_target_count += 1
                else:
                    idx = choices.index(target)
                    choice_position_counts[idx] += 1
                    choice_target_lengths.append(len(str(target)))
                    for i, c in enumerate(choices):
                        if i != idx:
                            choice_distractor_lengths.append(len(str(c)))

            # 2. NOUL
            elif q_type == "noul":
                noul_total += 1
                target_noul = row.get("target_noul")
                if target_noul is not None:
                    # Нормализуем 1 / 0 / 1.0 / 0.0 / True / False
                    noul_labels[float(target_noul)] += 1

            # 3. EXTRACT
            elif q_type == "extract":
                extract_total += 1
                span_text = row.get("target_span_text")
                if span_text:
                    if span_text not in state:
                        extract_missing_span += 1
                    else:
                        idx = state.find(span_text)
                        # Проверка середины слова: символ перед спаном
                        if idx > 0 and state[idx - 1].isalnum() and span_text[0].isalnum():
                            extract_mid_word += 1

            # 4. SCORE
            elif q_type == "score":
                score_total += 1
                target_score = row.get("target_score")
                if target_score is not None:
                    score_values.append(float(target_score))

    # Сводные метрики по choice
    r_pb = compute_point_biserial_r(choice_target_lengths, choice_distractor_lengths)
    max_choice_pos = max(choice_position_counts.keys()) if choice_position_counts else 0
    pos_counts_list = [choice_position_counts.get(i, 0) for i in range(max_choice_pos + 1)]
    chi2_stat, chi2_pval = compute_chi_square_uniformity(pos_counts_list) if len(pos_counts_list) > 1 else (0.0, 1.0)

    # Оценка качества noul
    noul_pos_ratio = (noul_labels.get(1.0, 0) / noul_total) if noul_total > 0 else 0.0

    return {
        "file": str(filepath),
        "total_rows": total_rows,
        "type_counts": dict(type_counts),
        "state_duplicates_within_file": duplicate_states,
        "choice": {
            "total": choice_total,
            "invalid_targets": choice_invalid_target_count,
            "length_correlation_r_pb": round(r_pb, 4),
            "length_shortcut_detected": bool(abs(r_pb) >= 0.15),
            "position_counts": dict(choice_position_counts),
            "position_chi2_stat": round(chi2_stat, 2),
            "position_uniformity_p_val": round(chi2_pval, 4),
            "position_bias_detected": bool(chi2_pval < 0.05),
        },
        "noul": {
            "total": noul_total,
            "label_distribution": {str(k): v for k, v in noul_labels.items()},
            "positive_ratio": round(noul_pos_ratio, 4),
            "imbalance_detected": bool(noul_total > 50 and not (0.40 <= noul_pos_ratio <= 0.60)),
        },
        "extract": {
            "total": extract_total,
            "missing_span_in_state": extract_missing_span,
            "mid_word_extractions": extract_mid_word,
            "span_fidelity_rate": round((extract_total - extract_missing_span) / max(1, extract_total), 4),
        },
        "score": {
            "total": score_total,
            "mean": round(sum(score_values) / len(score_values), 2) if score_values else 0.0,
            "min": min(score_values) if score_values else 0.0,
            "max": max(score_values) if score_values else 0.0,
        },
    }


def audit_cross_split_leakage(train_path: Path, val_path: Path, sample_limit: int = 1000) -> Dict[str, Any]:
    """Проверяет лексические утечки и совпадения между train и val сплитами."""
    if not train_path.exists() or not val_path.exists():
        return {"status": "skipped", "reason": "train or val file does not exist"}

    train_states: Set[str] = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                s = row.get("state")
                if s:
                    train_states.add(s.strip())
            except Exception:
                continue

    val_states: List[str] = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                s = row.get("state")
                if s:
                    val_states.append(s.strip())
            except Exception:
                continue

    exact_matches = 0
    near_duplicates = 0

    # Проверка точных совпадений
    for v in val_states:
        if v in train_states:
            exact_matches += 1

    # Выборочная проверка близких дубликатов (Jaccard > 0.85) на подвыборке
    sub_val = val_states[:sample_limit]
    train_sample = list(train_states)[:sample_limit]
    for v in sub_val:
        if v in train_states:
            continue
        for t in train_sample:
            if compute_jaccard_3gram(v, t) >= 0.85:
                near_duplicates += 1
                break

    exact_leak_ratio = exact_matches / len(val_states) if val_states else 0.0

    return {
        "status": "completed",
        "train_unique_states": len(train_states),
        "val_total_states": len(val_states),
        "exact_state_leaks": exact_matches,
        "exact_leak_ratio": round(exact_leak_ratio, 4),
        "near_duplicate_states_sample": near_duplicates,
        "leak_risk_detected": bool(exact_leak_ratio > 0.05),
    }


def main():
    parser = argparse.ArgumentParser(description="Типобезопасный анти-шорткат аудит датасетов Leeth")
    parser.add_argument("--val", type=str, default="src/leeth/data_400m/leeth_400m_val.jsonl", help="Путь к валидационному датасету")
    parser.add_argument("--train", type=str, default="src/leeth/data_400m/leeth_400m_train.jsonl", help="Путь к обучающему датасету")
    parser.add_argument("--output", type=str, default="reports/dataset_audit_report.json", help="Путь для сохранения отчета")
    args = parser.parse_args()

    val_path = Path(args.val)
    train_path = Path(args.train)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  АНТИ-ШОРТКАТ АУДИТ ДАТАСЕТОВ LEETH (ПО МЕТОДОЛОГИИ JEFF & NIMBLE)")
    print("=" * 80)

    val_report = audit_file(val_path)
    train_report = audit_file(train_path) if train_path.exists() else None
    cross_report = audit_cross_split_leakage(train_path, val_path) if train_path.exists() else None

    full_report = {
        "validation_dataset": val_report,
        "training_dataset": train_report,
        "cross_split_leakage": cross_report,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    # Красивый вывод в консоль
    print(f"\n[1] ВАЛИДАЦИОННЫЙ ДАТАСЕТ: {val_path.name}")
    print(f"    Всего записей: {val_report.get('total_rows', 0)}")
    print(f"    Распределение типов: {val_report.get('type_counts')}")
    c = val_report.get("choice", {})
    print(f"    Choice: r_pb (длина) = {c.get('length_correlation_r_pb')} (Шорткат: {c.get('length_shortcut_detected')})")
    print(f"    Choice: p-val (позиция) = {c.get('position_uniformity_p_val')} (Смещение: {c.get('position_bias_detected')})")
    n = val_report.get("noul", {})
    print(f"    Noul: доля True = {n.get('positive_ratio')} (Дисбаланс: {n.get('imbalance_detected')})")
    e = val_report.get("extract", {})
    print(f"    Extract: точность вхождения спана = {e.get('span_fidelity_rate', 0) * 100:.2f}% (Срезы середины слов: {e.get('mid_word_extractions')})")

    if cross_report and cross_report.get("status") == "completed":
        print(f"\n[2] КРОСС-СПЛИТ АУДИТ УТЕЧЕК (Train vs Val):")
        print(f"    Точных совпадений состояний: {cross_report.get('exact_state_leaks')} ({cross_report.get('exact_leak_ratio', 0) * 100:.2f}%)")
        print(f"    Близких дубликатов в выборке: {cross_report.get('near_duplicate_states_sample')}")
        print(f"    Риск утечки данных: {cross_report.get('leak_risk_detected')}")

    print(f"\nПолный детальный отчет сохранен в: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
