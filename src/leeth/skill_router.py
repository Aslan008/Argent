"""src/leeth/skill_router.py — Фаза F1/F4: маршрутизация скиллов через ranking.

Зачем: скиллы — динамическое множество (пользователь добавляет SKILL.md
когда хочет). Router индексирует описания скиллов при старте, скорит запрос
против всех описаний и отдаёт top-1 ТОЛЬКО если уверенность выше порога.
Ниже порога или нет обученной головы -> None -> делегат большой модели.

Гибридная схема (F4):
  1. (опционально, после обучения) Noul-фильтр «нужен ли скилл»
  2. ranking top-1 по описаниям
  3. порог уверенности -> делегат
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

from src.leeth.ranking import LeethRankingModel, RankResult, load_ranking_head
from src.leeth.tokenizer import LeethTokenizer
from logger import get_logger

logger = get_logger("leeth_skill_router")

SKILLS_DIR = Path("skills")
RANKING_WEIGHTS = Path("src/leeth/data/ranking_head.pt")
DEFAULT_THRESHOLD = 0.55


@dataclass
class SkillRoute:
    skill: str
    score: float
    margin_to_second: float


@dataclass
class SkillIndexEntry:
    name: str
    description: str
    path: Path


def _extract_description(skill_md: str) -> str:
    """Первый заголовок + первый содержательный абзац SKILL.md."""
    lines = skill_md.strip().splitlines()
    heading = ""
    body = ""
    for line in lines:
        if line.startswith("#") and not heading:
            heading = line.lstrip("#").strip()
            continue
        if heading and line.strip() and not line.startswith("#") and not line.startswith("```"):
            body = line.strip()
            break
    if heading or body:
        return f"{heading}. {body}".strip(". ")
    return ""


class SkillRouter:
    """Индексирует skills/*/SKILL.md и маршрутизирует запросы на ranking-голове."""

    def __init__(
        self,
        ranking: Optional[LeethRankingModel],
        tokenizer: Optional[LeethTokenizer],
        skills_dir: Path = SKILLS_DIR,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.ranking = ranking
        self.tokenizer = tokenizer
        self.skills_dir = skills_dir
        self.threshold = threshold
        self.entries: List[SkillIndexEntry] = []
        self._item_ids: Optional[torch.Tensor] = None
        self._item_mask: Optional[torch.Tensor] = None

    @property
    def is_ready(self) -> bool:
        """Готов ли router к скорингу (нужны и голова, и индекс)."""
        return self.ranking is not None and self.tokenizer is not None and bool(self.entries)

    def rebuild_index(self) -> int:
        """Сканирует skills/*/SKILL.md и кодирует описания.

        Зачем: новый скилл-файл подхватывается простым вызовом rebuild —
        без переобучения модели.
        """
        self.entries = []
        if not self.skills_dir.exists():
            return 0
        for skill_dir in sorted(self.skills_dir.iterdir()):
            md = skill_dir / "SKILL.md"
            if not skill_dir.is_dir() or not md.exists():
                continue
            try:
                text = md.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            desc = _extract_description(text) or skill_dir.name
            self.entries.append(SkillIndexEntry(name=skill_dir.name, description=desc, path=md))
        self._encode_items()
        logger.info("SkillRouter: indexed %d skills", len(self.entries))
        return len(self.entries)

    def _encode_items(self) -> None:
        if not self.is_ready:
            return
        ids, masks = [], []
        for entry in self.entries:
            text = f"Skill {entry.name}: {entry.description}"
            token_ids = self.tokenizer.encode(text, max_length=128, truncation=True, add_special_tokens=True)
            ids.append(token_ids)
            masks.append([1] * len(token_ids))
        max_len = max(len(x) for x in ids)
        pad = self.tokenizer.pad_token_id
        self._item_ids = torch.tensor([x + [pad] * (max_len - len(x)) for x in ids], dtype=torch.long)
        self._item_mask = torch.tensor(masks, dtype=torch.long)

    def route(self, query: str) -> Optional[SkillRoute]:
        """Верный скилл для запроса или None (делегат).

        Порядок решений:
          - головы/индекса нет -> None (честный делегат, не гадание)
          - скор top-1 ниже порога -> None
        """
        if not self.is_ready:
            return None
        q_ids = self.tokenizer.encode(query, max_length=128, truncation=True, add_special_tokens=True)
        q_mask = [1] * len(q_ids)
        q_ids_t = torch.tensor([q_ids], dtype=torch.long)
        q_mask_t = torch.tensor([q_mask], dtype=torch.long)
        results = self.ranking.rank(q_ids_t, q_mask_t, self._item_ids, self._item_mask)
        best: RankResult = results[0]
        if best.score < self.threshold:
            return None
        return SkillRoute(skill=self.entries[best.index].name, score=best.score,
                          margin_to_second=best.margin_to_second)


def build_skill_router(backbone, tokenizer: Optional[LeethTokenizer] = None,
                       threshold: float = DEFAULT_THRESHOLD) -> SkillRouter:
    """Фабрика: грузит ranking-голову если веса есть, иначе fallback-роутер."""
    ranking = load_ranking_head(backbone, weights_path=str(RANKING_WEIGHTS))
    router = SkillRouter(ranking=ranking, tokenizer=tokenizer, threshold=threshold)
    router.rebuild_index()
    return router