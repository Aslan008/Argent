# -*- coding: utf-8 -*-
"""scripts/lmtrust_mutate_contrastive.py — Step 4: мутационная валидация.

mutmut требует WSL (недоступен) — компактный харнесс: по одному мутанту
применяем к src/leeth/contrastive.py, гоняем тесты, считаем kill score.
Тест, не убивший ни одного мутанта, — тавтология (AP4).
"""

import shutil
import subprocess
import sys
from pathlib import Path

TARGET = Path("src/leeth/contrastive.py")
BACKUP = Path(".argent/_contrastive_backup.py")
TESTS = ["tests/test_leeth_contrastive.py", "tests/test_lmtrust_blindspots.py"]

MUTANTS = [
    ("invert shape check", "if pos_scores.shape != neg_scores.shape:", "if pos_scores.shape == neg_scores.shape:"),
    ("sub->add in hinge", "F.relu(margin - (pos_scores - neg_scores))", "F.relu(margin - (pos_scores + neg_scores))"),
    ("margin sign flip", "F.relu(margin - (pos_scores - neg_scores))", "F.relu(margin + (pos_scores - neg_scores))"),
    ("invert empty check", "if pos_scores.numel() == 0:", "if pos_scores.numel() != 0:"),
    ("zeros->ones empty", "return pos_scores.new_zeros(())\n    return F.relu", "return pos_scores.new_ones(())\n    return F.relu"),
    ("temp boundary <0", "if temperature <= 0:", "if temperature < 0:"),
    ("drop temp validation", 'raise ValueError(f"temperature должна быть > 0, получено {temperature}")', "pass  # mutated"),
    ("invert empty batch", "if query_emb.size(0) == 0:", "if query_emb.size(0) != 0:"),
    ("empty batch ones", "return query_emb.new_zeros(())\n    if neg_embs is None", "return query_emb.new_ones(())\n    if neg_embs is None"),
    ("drop neg-empty validation", 'raise ValueError("info_nce_loss требует хотя бы один negative")', "pass  # валидация снята"),
    ("or->and neg stack", "if neg_stack.dim() != 3 or neg_stack.size(1) == 0:", "if neg_stack.dim() != 3 and neg_stack.size(1) == 0:"),
    ("pos_sim /->*", "pos_sim = (query_emb * pos_emb).sum(dim=-1, keepdim=True) / temperature", "pos_sim = (query_emb * pos_emb).sum(dim=-1, keepdim=True) * temperature"),
    ("neg_sim /->*", 'neg_sim = torch.einsum("qd,qnd->qn", query_emb, neg_stack) / temperature', 'neg_sim = torch.einsum("qd,qnd->qn", query_emb, neg_stack) * temperature'),
    ("targets zeros->ones", "targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)", "targets = torch.ones(logits.size(0), dtype=torch.long, device=logits.device)"),
    ("or->and pos targets", "if (pos_targets < 0).any() or (pos_targets >= n_cls).any():", "if (pos_targets < 0).any() and (pos_targets >= n_cls).any():"),
    ("drop pos validation", 'raise ValueError(f"pos_targets вне [0, {n_cls}): {pos_targets.tolist()}")', "pass  # валидация снята"),
    ("or->and neg targets", "if (neg_targets < 0).any() or (neg_targets >= neg_logits.size(1)).any():", "if (neg_targets < 0).any() and (neg_targets >= neg_logits.size(1)).any():"),
    ("drop neg validation", 'raise ValueError(f"neg_targets вне [0, {neg_logits.size(1)}): {neg_targets.tolist()}")', "pass  # валидация снята"),
    ("swap pos/neg scores", "return margin_loss(pos_score, neg_score, margin=margin)", "return margin_loss(neg_score, pos_score, margin=margin)"),
    ("weight boundary <=0", "if contrastive_weight < 0:", "if contrastive_weight <= 0:"),
    ("combined +->-", "return supervised + contrastive_weight * contrastive", "return supervised - contrastive_weight * contrastive"),
    ("or->and neg_embs None", "if neg_embs is None or (isinstance(neg_embs, (list, tuple)) and len(neg_embs) == 0):", "if neg_embs is None and (isinstance(neg_embs, (list, tuple)) and len(neg_embs) == 0):"),
    ("invert isinstance tensor", "if isinstance(neg_embs, torch.Tensor):\n        neg_stack = neg_embs", "if not isinstance(neg_embs, torch.Tensor):\n        neg_stack = neg_embs"),
    ("noul delegates swapped", "return margin_loss(pos_probs, neg_probs, margin=margin)", "return margin_loss(neg_probs, pos_probs, margin=margin)"),
]


def run_tests() -> bool:
    """True = все тесты прошли (мутант выжил)."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "--no-header", "--timeout=120"],
        capture_output=True, text=True, timeout=300,
    )
    return r.returncode == 0


def main() -> None:
    shutil.copy(TARGET, BACKUP)
    killed, survived, errors = [], [], []
    try:
        for name, old, new in MUTANTS:
            src = BACKUP.read_text(encoding="utf-8")
            if src.count(old) != 1:
                errors.append((name, "target not unique/absent"))
                continue
            TARGET.write_text(src.replace(old, new), encoding="utf-8")
            try:
                if run_tests():
                    survived.append(name)
                else:
                    killed.append(name)
            except subprocess.TimeoutExpired:
                errors.append((name, "timeout"))
            finally:
                shutil.copy(BACKUP, TARGET)
    finally:
        shutil.copy(BACKUP, TARGET)
        BACKUP.unlink(missing_ok=True)

    total = len(killed) + len(survived)
    score = len(killed) / total * 100 if total else 0.0
    print(f"Мутантов: {total} | убито: {len(killed)} | выжило: {len(survived)} | ошибок: {len(errors)}")
    print(f"KILL SCORE: {score:.0f}%  ({'ОТЛИЧНО' if score >= 80 else 'ПРИЕМЛЕМО' if score >= 50 else 'СЛАБО'})")
    if survived:
        print("Выжившие (слабые места тестов):")
        for s in survived:
            print("  -", s)
    if errors:
        print("Ошибки применения:")
        for e in errors:
            print("  -", e)
    sys.exit(0 if score >= 50 else 1)


if __name__ == "__main__":
    main()