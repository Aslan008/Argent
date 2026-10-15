"""src/leeth/contrastive.py — Фаза D1: контрастные лоссы для RLCD.

Зачем: обычный supervised-лосс учит модель «угадать верный ответ», но не
«отодвинуть неверный». Контрастные лоссы работают на ПАРАХ (positive, negative)
и требуют, чтобы скор правильного варианта был выше неправильного на заданную
маржу. Это точечно бьёт по слабым местам: похожие инструменты и маскированные
деструктивы, где модель ошибается именно «в пользу похожего».

Все лоссы чисто тензорные — юнит-тестируются без загрузки весов.
"""

from typing import List

import torch
import torch.nn.functional as F


def margin_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Hinge-маржа: score(positive) должен превышать score(negative) на margin.

    Зачем: простейший контрастный сигнал — «правильное должно быть заметно
    лучше неправильного», а не просто «правильное — максимум».
    """
    if pos_scores.shape != neg_scores.shape:
        raise ValueError(f"shape mismatch: {pos_scores.shape} vs {neg_scores.shape}")
    if pos_scores.numel() == 0:
        # mean() по пустому тензору даёт NaN и отравляет весь лосс.
        return pos_scores.new_zeros(())
    return F.relu(margin - (pos_scores - neg_scores)).mean()


def info_nce_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_embs: List[torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE: softmax по [positive, negatives] — вся масса на positive.

    Зачем: стандартный контрастивный критерий для bi-encoder'а. Учит
    ранжировать: верный элемент должен набрать больше сходства, чем все
    дистракторы вместе.

    Контракт: temperature > 0 (иначе inf/NaN или перевёрнутый softmax);
    neg_embs непустой список [bsz, dim] ИЛИ готовый стек [bsz, n_neg, dim]
    (без negatives контрастивного сигнала нет — это ошибка вызова, а не 0).
    """
    if temperature <= 0:
        raise ValueError(f"temperature должна быть > 0, получено {temperature}")
    if query_emb.size(0) == 0:
        # cross_entropy по пустому батчу даёт NaN.
        return query_emb.new_zeros(())
    if neg_embs is None or (isinstance(neg_embs, (list, tuple)) and len(neg_embs) == 0):
        raise ValueError("info_nce_loss требует хотя бы один negative")
    if isinstance(neg_embs, torch.Tensor):
        neg_stack = neg_embs  # уже стек [bsz, n_neg, dim]
    else:
        neg_stack = torch.stack(list(neg_embs), dim=1)  # [bsz, n_neg, dim]
    if neg_stack.dim() != 3 or neg_stack.size(1) == 0:
        raise ValueError(f"neg_embs пуст: форма {tuple(neg_stack.shape)}")
    neg_sim = torch.einsum("qd,qnd->qn", query_emb, neg_stack) / temperature
    pos_sim = (query_emb * pos_emb).sum(dim=-1, keepdim=True) / temperature  # [bsz, 1]
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def noul_contrastive_loss(
    pos_probs: torch.Tensor,
    neg_probs: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Контраст для Noul-головы: p(safe|positive) должна быть выше p(safe|negative).

    Зачем: пара «безопасная команда vs маскированный деструктив» в одинаковой
    обёртке состояния. Лосс требует не просто «0/1», а заметную маржу между
    ними — это и есть калибровка на OOD.
    """
    return margin_loss(pos_probs, neg_probs, margin=margin)


def choice_contrastive_loss(
    pos_logits: torch.Tensor,
    pos_targets: torch.Tensor,
    neg_logits: torch.Tensor,
    neg_targets: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Контраст для Choice-головы: логит верного класса у positive должен
    превышать логит (неверного) класса у negative на margin.

    Зачем: пары «один запрос, верный инструмент vs похожий неверный» учат
    голову раздвигать похожие классы, которые CE-лосс держит слишком близко.
    """
    # Валидация таргетов: out-of-bounds дал бы RuntimeError без контекста,
    # отрицательные индексы — молчаливый gather чужого класса.
    n_cls = pos_logits.size(1)
    if (pos_targets < 0).any() or (pos_targets >= n_cls).any():
        raise ValueError(f"pos_targets вне [0, {n_cls}): {pos_targets.tolist()}")
    if (neg_targets < 0).any() or (neg_targets >= neg_logits.size(1)).any():
        raise ValueError(f"neg_targets вне [0, {neg_logits.size(1)}): {neg_targets.tolist()}")
    pos_score = pos_logits.gather(1, pos_targets.unsqueeze(1)).squeeze(1)
    neg_score = neg_logits.gather(1, neg_targets.unsqueeze(1)).squeeze(1)
    return margin_loss(pos_score, neg_score, margin=margin)


def combined_loss(
    supervised: torch.Tensor,
    contrastive: torch.Tensor,
    contrastive_weight: float = 0.4,
) -> torch.Tensor:
    """Смешанный лосс: supervised + weight * contrastive.

    Зачем: чистый контраст может «съесть» общую точность — смешивание
    сохраняет базовые метрики, добавляя маржу.
    """
    if contrastive_weight < 0:
        raise ValueError(f"contrastive_weight должна быть >= 0, получено {contrastive_weight}")
    return supervised + contrastive_weight * contrastive