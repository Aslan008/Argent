"""src/leeth/browser_router.py — Фаза G3/G4/G5: браузерные решения через ranking.

Зачем: браузерное управление (кликнуть кнопку, заполнить поле, выбрать опцию) —
выбор из ДИНАМИЧЕСКОГО множества (DOM меняется на каждой странице). Ranking-
голова скорит каждый видимый элемент против цели; лучший выше порога -> клик,
ниже порога -> делегат большой модели (browser_act).

Безопасность (Фаза G4): элементы с опасными действиями (submit payment,
delete, confirm purchase) ВСЕГДА требуют подтверждения — вероятностный клик
без предохранителя = риск «случайно оплатить».
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from logger import get_logger

logger = get_logger("leeth_browser_router")

RANKING_WEIGHTS = Path("src/leeth/data/ranking_head.pt")
BASE_DIR = Path(".argent/models/mmbert-argent")
DEFAULT_THRESHOLD = 0.20

# Опасные действия в браузере: клик по ним никогда не идёт автоматически.
DANGEROUS_PATTERNS = [
    re.compile(r"\b(?:delete|remove|erase)\b", re.I),
    re.compile(r"\b(?:pay|payment|checkout|purchase|place order|buy now)\b", re.I),
    re.compile(r"\bconfirm\b", re.I),
    re.compile(r"\bauthorize\b|\bauthorise\b", re.I),
    re.compile(r"\b(?:submit payment|final step)\b", re.I),
    re.compile(r"\b(?:удалить|оплатить|подтвердить|купить|оформить заказ)\b", re.I),
    # Опасные команды, спрятанные в тексте элемента («Run rm -rf /»):
    re.compile(r"\brm\s+-[a-z]*[rf]", re.I),
    re.compile(r"\bformat\s+\w:", re.I),
    re.compile(r"\b(?:rd|rmdir)\s+/s\b", re.I),
    re.compile(r"\bdel\s+/[\w]*[fsq]", re.I),
    re.compile(r"\bdrop\s+(?:database|table)\b", re.I),
]


@dataclass
class BrowserRoute:
    index: int
    score: float
    margin_to_second: float
    requires_confirmation: bool
    reason: Optional[str] = None


def requires_confirmation_check(element: Dict) -> bool:
    """Процедурный гейт опасных кнопок — не зависит от модели.

    Ловит и опасные КОМАНДЫ, спрятанные в тексте элемента («Run rm -rf /»):
    deny-паттерны диспетчера применяются к тексту кнопки так же, как к команде.
    """
    text = (
        (element.get("text") or element.get("label") or element.get("name") or "")
        + " "
        + (element.get("role") or element.get("tag") or "")
    )
    if any(p.search(text) for p in DANGEROUS_PATTERNS):
        return True
    try:
        from src.leeth.dispatcher import DENY_PATTERNS  # deferred: avoids import cycle
        raw_txt = element.get("text") or element.get("label") or element.get("name") or ""
        return any(p.search(raw_txt) for p in DENY_PATTERNS)
    except Exception:
        return False


class BrowserRouter:
    """Скорит элементы DOM против цели на обученной bi-encoder ranking-голове."""

    def __init__(
        self,
        ranking: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        threshold: float = DEFAULT_THRESHOLD,
        weights_path: Optional[Path] = None,
    ):
        self.ranking = ranking
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.weights_path = weights_path or RANKING_WEIGHTS

        # Bi-encoder projection tensors (ModernBERT 768 -> 256)
        self.q_w: Optional[torch.Tensor] = None
        self.q_b: Optional[torch.Tensor] = None
        self.i_w: Optional[torch.Tensor] = None
        self.i_b: Optional[torch.Tensor] = None
        self._backbone: Optional[Any] = None

        self._init_engine()

    def _init_engine(self) -> None:
        """Инициализирует ModernBERT bi-encoder если веса доступны."""
        if not self.weights_path.exists():
            logger.info("Ranking weights not found at %s. BrowserRouter in fallback mode.", self.weights_path)
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            if self.tokenizer is None and BASE_DIR.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(BASE_DIR))

            if self._backbone is None and BASE_DIR.exists():
                self._backbone = AutoModel.from_pretrained(str(BASE_DIR)).eval()

            sd = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            if "rank_query_proj.weight" in sd:
                self.q_w = sd["rank_query_proj.weight"]
                self.q_b = sd["rank_query_proj.bias"]
                self.i_w = sd["rank_item_proj.weight"]
                self.i_b = sd["rank_item_proj.bias"]
            elif "query_proj" in sd and "item_proj" in sd:
                self.q_w = sd["query_proj"]["weight"]
                self.q_b = sd["query_proj"].get("bias")
                self.i_w = sd["item_proj"]["weight"]
                self.i_b = sd["item_proj"].get("bias")
            logger.info("BrowserRouter bi-encoder engine successfully initialized from %s.", self.weights_path)
        except Exception as e:
            logger.warning("Failed to initialize bi-encoder ranking engine: %s", e)

    @property
    def is_ready(self) -> bool:
        has_projections = self.q_w is not None and self.i_w is not None and self._backbone is not None
        has_custom_ranking = self.ranking is not None and self.tokenizer is not None
        has_dopt = Path(".argent/models/dopt/dopt_v2.onnx").exists() or Path(".argent/models/mmbert_dynamic_opt_v2.pt").exists()
        return has_projections or has_custom_ranking or has_dopt

    def _item_text(self, element: Dict) -> str:
        """Единый текст элемента для кодирования с поддержкой A11y и DOM: 'role: text (aria/placeholder)'."""
        role = element.get("role") or element.get("tag") or "element"
        text = (element.get("text") or element.get("name") or element.get("label") or element.get("title") or "").strip()
        aria = (element.get("aria_label") or element.get("placeholder") or "").strip()
        extra = f" ({aria})" if aria and aria != text else ""
        return f"{role}: {text}{extra}".strip()

    def route(self, goal: str, elements: List[Dict]) -> Optional[BrowserRoute]:
        """Верный элемент для цели или None (делегат большой модели).

        Порядок решений:
          1. Движок не готов или список elements пуст -> None.
          2. Скор top-1 ниже порога -> None.
          3. Опасный элемент -> маршрут с requires_confirmation=True.
        """
        if not self.is_ready or not elements or not goal:
            return None

        item_texts = [self._item_text(el) for el in elements]

        # Path A: Universal Task Solver (DoPT zero-shot dynamic option selection)
        try:
            from src.leeth.dopt_solver import DoptSolver
            if not hasattr(self, "_dopt_solver") or self._dopt_solver is None:
                self._dopt_solver = DoptSolver()

            candidate_indices = list(range(len(elements)))
            
            # Если элементов много (> 8), используем bi-encoder для быстрого top-8 отбора
            if len(elements) > 8 and self.q_w is not None and self.i_w is not None and self._backbone is not None and self.tokenizer is not None:
                try:
                    with torch.no_grad():
                        q_enc = self.tokenizer(goal, return_tensors="pt", max_length=64, truncation=True)
                        q_out = self._backbone(**q_enc)
                        q_mask = q_enc["attention_mask"].unsqueeze(-1)
                        q_pooled = (q_out.last_hidden_state * q_mask).sum(1) / q_mask.sum(1).clamp(min=1e-9)
                        q_proj = F.normalize(F.linear(q_pooled, self.q_w, self.q_b), dim=-1)

                        i_enc = self.tokenizer(item_texts, padding=True, return_tensors="pt", max_length=64, truncation=True)
                        i_out = self._backbone(**i_enc)
                        i_mask = i_enc["attention_mask"].unsqueeze(-1)
                        i_pooled = (i_out.last_hidden_state * i_mask).sum(1) / i_mask.sum(1).clamp(min=1e-9)
                        i_proj = F.normalize(F.linear(i_pooled, self.i_w, self.i_b), dim=-1)

                        sims = (q_proj * i_proj).sum(dim=-1).squeeze(0)
                        candidate_indices = torch.argsort(sims, descending=True)[:8].tolist()
                except Exception as be_filter_err:
                    logger.debug("Bi-encoder candidate filtering failed: %s", be_filter_err)
                    candidate_indices = list(range(min(len(elements), 8)))

            opts = [item_texts[i] for i in candidate_indices]
            dopt_res = self._dopt_solver.decide(state_text="", question=goal, options=opts)
            if dopt_res and dopt_res.choice in opts:
                c_idx = opts.index(dopt_res.choice)
                best_idx = candidate_indices[c_idx]
                element = elements[best_idx]
                requires = requires_confirmation_check(element)
                probs = list(dopt_res.probabilities.values())
                probs_sorted = sorted(probs, reverse=True)
                margin = (probs_sorted[0] - probs_sorted[1]) if len(probs_sorted) > 1 else 0.0
                if dopt_res.confidence >= self.threshold:
                    return BrowserRoute(
                        index=best_idx,
                        score=float(dopt_res.confidence),
                        margin_to_second=float(margin),
                        requires_confirmation=requires,
                        reason="dangerous_action" if requires else None,
                    )
        except Exception as dopt_err:
            logger.debug("DoPT routing failed (%s), proceeding to fallback bi-encoder.", dopt_err)

        # Path B: Fast Native Bi-Encoder (ModernBERT + 256 projection)
        if self.q_w is not None and self.i_w is not None and self._backbone is not None and self.tokenizer is not None:
            try:
                with torch.no_grad():
                    # 1. Encode query
                    q_enc = self.tokenizer(goal, return_tensors="pt", max_length=64, truncation=True)
                    q_out = self._backbone(**q_enc)
                    q_mask = q_enc["attention_mask"].unsqueeze(-1)
                    q_pooled = (q_out.last_hidden_state * q_mask).sum(1) / q_mask.sum(1).clamp(min=1e-9)
                    q_proj = F.normalize(F.linear(q_pooled, self.q_w, self.q_b), dim=-1)

                    # 2. Encode elements in batch
                    i_enc = self.tokenizer(item_texts, padding=True, return_tensors="pt", max_length=64, truncation=True)
                    i_out = self._backbone(**i_enc)
                    i_mask = i_enc["attention_mask"].unsqueeze(-1)
                    i_pooled = (i_out.last_hidden_state * i_mask).sum(1) / i_mask.sum(1).clamp(min=1e-9)
                    i_proj = F.normalize(F.linear(i_pooled, self.i_w, self.i_b), dim=-1)

                    # 3. Similarities
                    scores = (q_proj * i_proj).sum(dim=-1).squeeze(0)
                    order = torch.argsort(scores, descending=True)
                    best_idx = int(order[0].item())
                    best_score = float(scores[best_idx].item())
                    second_score = float(scores[order[1].item()]) if len(order) > 1 else -1.0
                    margin = best_score - second_score

                if best_score >= self.threshold:
                    element = elements[best_idx]
                    requires = requires_confirmation_check(element)
                    return BrowserRoute(
                        index=best_idx,
                        score=best_score,
                        margin_to_second=margin,
                        requires_confirmation=requires,
                        reason="dangerous_action" if requires else None,
                    )
            except Exception as e:
                logger.warning("Bi-encoder routing failed (%s), trying custom ranking fallback.", e)

        # Path C: LeethRankingModel fallback
        if self.ranking is not None and self.tokenizer is not None:
            try:
                item_ids, item_mask = [], []
                for el in elements:
                    text = self._item_text(el)
                    ids = self.tokenizer.encode(text, max_length=64, truncation=True, add_special_tokens=True)
                    item_ids.append(ids)
                    item_mask.append([1] * len(ids))
                max_len = max(len(x) for x in item_ids)
                pad = self.tokenizer.pad_token_id
                item_ids_t = torch.tensor([x + [pad] * (max_len - len(x)) for x in item_ids], dtype=torch.long)
                item_mask_t = torch.tensor(item_mask, dtype=torch.long)

                q_ids = self.tokenizer.encode(goal, max_length=64, truncation=True, add_special_tokens=True)
                q_ids_t = torch.tensor([q_ids], dtype=torch.long)
                q_mask_t = torch.tensor([[1] * len(q_ids)], dtype=torch.long)

                results = self.ranking.rank(q_ids_t, q_mask_t, item_ids_t, item_mask_t)
                best = results[0]
                if best.score < self.threshold:
                    return None

                element = elements[best.index]
                requires = requires_confirmation_check(element)
                return BrowserRoute(
                    index=best.index,
                    score=best.score,
                    margin_to_second=best.margin_to_second,
                    requires_confirmation=requires,
                    reason="dangerous_action" if requires else None,
                )
            except Exception as e:
                logger.warning("Custom ranking fallback failed: %s", e)

        return None


_BROWSER_ROUTER_SINGLETON: Optional[BrowserRouter] = None


def get_browser_router() -> BrowserRouter:
    """Ленивый синглтон BrowserRouter для быстрых решений в browser_act."""
    global _BROWSER_ROUTER_SINGLETON
    if _BROWSER_ROUTER_SINGLETON is None:
        _BROWSER_ROUTER_SINGLETON = BrowserRouter()
    return _BROWSER_ROUTER_SINGLETON