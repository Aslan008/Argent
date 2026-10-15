"""src/leeth/ranking.py — Фаза F1: bi-encoder ranking-голова.

Зачем: выбор скилла, плагина или браузерного элемента — это выбор из
ДИНАМИЧЕСКОГО списка (пользователь добавляет файлы когда хочет). Фиксированная
choice-голова так не умеет в принципе. Решение — bi-encoder: кодируем запрос
и описание элемента в общее векторное пространство, скорим косинусом.
Новый скилл = новый текст описания, переобучение НЕ нужно.

Дизайн:
  - Общий backbone LeethModel (разделяемый с decision-головами).
  - Две проекции: rank_query_proj (запрос) и rank_item_proj (элемент).
  - Пулинг: mean-pool по attention mask (стабильнее первого токена).
  - Скор: косинус между L2-нормализованными эмбеддингами.

Обучение (Фаза F3/G2): InfoNCE по батчу (positive + distractors).
До обучения головы route() возвращает None -> делегат большой модели.
"""

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.leeth.model import LeethModel


@dataclass
class RankResult:
    index: int
    score: float
    margin_to_second: float


class LeethRankingModel(nn.Module):
    """Bi-encoder поверх общего backbone LeethModel.

    query и item кодируются ОДНИМ энкодером, но разными проекциями —
    это стандартная asymmetric bi-encoder схема для retrieval.
    """

    def __init__(self, backbone: LeethModel, embed_dim: int = 256, trainable_backbone: bool = False):
        super().__init__()
        # ВНИМАНИЕ: при trainable_backbone=False (по умолчанию) замораживаются
        # параметры ПЕРЕДАННОГО backbone НА МЕСТЕ (requires_grad=False). Если
        # тот же экземпляр backbone используется другой головой — передавайте
        # отдельный экземпляр (build_skill_router так и делает).
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.rank_query_proj = nn.Linear(hidden, embed_dim)
        self.rank_item_proj = nn.Linear(hidden, embed_dim)
        if not trainable_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids, attention_mask=attention_mask)
        emb = self.rank_query_proj(out["pooled_repr"])
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_item(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids, attention_mask=attention_mask)
        emb = self.rank_item_proj(out["pooled_repr"])
        return F.normalize(emb, dim=-1)

    def encode_query_trainable(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Grad-enabled кодирование запроса для ОБУЧЕНИЯ проекций.

        Зачем: encode_query/encode_item под @torch.no_grad() — для инференса.
        Тренировочный цикл обязан идти через эти методы, иначе градиенты
        не текут и обучение превращается в no-op.
        """
        out = self.backbone(input_ids, attention_mask=attention_mask)
        return F.normalize(self.rank_query_proj(out["pooled_repr"]), dim=-1)

    def encode_item_trainable(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids, attention_mask=attention_mask)
        return F.normalize(self.rank_item_proj(out["pooled_repr"]), dim=-1)

    def forward(self, query_ids, query_mask, item_ids, item_mask) -> torch.Tensor:
        """Trainable forward: скоры [bsz] для пар (query, item).

        Контракт: батчи запросов и элементов равны (парный скоринг).
        """
        if query_ids.shape[0] != item_ids.shape[0]:
            raise ValueError(
                f"батчи не парны: query {query_ids.shape[0]} vs item {item_ids.shape[0]}"
            )
        q_out = self.backbone(query_ids, attention_mask=query_mask)
        i_out = self.backbone(item_ids, attention_mask=item_mask)
        q_emb = F.normalize(self.rank_query_proj(q_out["pooled_repr"]), dim=-1)
        i_emb = F.normalize(self.rank_item_proj(i_out["pooled_repr"]), dim=-1)
        return (q_emb * i_emb).sum(dim=-1)

    def rank(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        item_ids: torch.Tensor,
        item_mask: torch.Tensor,
    ) -> List[RankResult]:
        """Скорит N элементов против ОДНОГО запроса (item_ids: [N, seq]).

        Контракт: query_ids — ровно один запрос [1, seq] (или [seq]); батч
        запросов вызовет ValueError, а не молчаливое взятие первого.
        Инференс сам переключает eval-режим: dropout замороженной базы
        не должен шуметь в скорах, даже если вызывающий забыл .eval().
        margin_to_second при единственном элементе = -1.0 (sentinel:
        «второго нет», не реальная маржа).
        """
        if query_ids.dim() == 1:
            query_ids = query_ids.unsqueeze(0)
            query_mask = query_mask.unsqueeze(0)
        if query_ids.shape[0] != 1:
            raise ValueError(
                f"rank() принимает один запрос, получен батч {query_ids.shape[0]}; "
                "используйте цикл или encode_query для батчей"
            )
        was_training = self.training
        if was_training:
            self.eval()  # dropout выключен на время инференса
        try:
            with torch.no_grad():
                q_emb = self.encode_query(query_ids[:1], query_mask[:1])[0]
                i_emb = self.encode_item(item_ids, item_mask)
                scores = (i_emb * q_emb).sum(dim=-1)
        finally:
            if was_training:
                self.train()
        order = torch.argsort(scores, descending=True)
        results = []
        for rank_idx in order.tolist():
            s = float(scores[rank_idx].item())
            second = float(scores[order[1]].item()) if len(order) > 1 else -1.0
            results.append(RankResult(score=s, margin_to_second=s - second, index=int(rank_idx)))
        return results


def load_ranking_head(
    backbone: "LeethModel",
    weights_path: Optional[str] = None,
    embed_dim: int = 256,
) -> Optional[LeethRankingModel]:
    """Загружает обученную ranking-голову; None = головы ещё нет (fallback).

    Зачем: до Фазы F3 весов нет — router обязан честно возвращать None
    (делегат), а не гадать необученной головой.
    """
    if weights_path is None:
        return None
    path = Path(weights_path)
    if not path.exists():
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        # embed_dim читаем из чекпоинта: обучение могло идти с другой размерностью.
        saved_dim = state.get("embed_dim", embed_dim)
        if torch.is_tensor(saved_dim):
            saved_dim = saved_dim.item()
        model = LeethRankingModel(backbone, embed_dim=int(saved_dim))
        model.rank_query_proj.load_state_dict(state["query_proj"])
        model.rank_item_proj.load_state_dict(state["item_proj"])
        return model
    except (KeyError, RuntimeError, EOFError, ValueError, TypeError,
            zipfile.BadZipFile, pickle.UnpicklingError) as load_err:
        # Контракт: None = головы нет (fallback на делегата), а не падение
        # рантайма из-за чужого/битого чекпоинта.
        from logger import get_logger
        get_logger("leeth_ranking").warning("ranking head load failed (%s): %s", path, load_err)
        return None