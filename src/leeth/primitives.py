"""
src/leeth/primitives.py — Core Typed Decision Primitives for Leeth.

Implements the System One non-autoregressive decision interface compatible with Jev:
- Choice: Categorical selection from predefined options with calibrated distribution.
- Noul: Calibrated boolean judgment (P(True) in [0.0, 1.0]).
- Score: Scalar rubric level rating (0..100).
- Extract: Entity/parameter span extraction without regex heuristics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class BaseQuestion:
    """Base class for all Leeth typed questions."""
    id: str
    question: str


@dataclass(frozen=True)
class ChoiceQuestion(BaseQuestion):
    """Categorical question: select one option from a predefined list of choices."""
    choices: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.choices:
            raise ValueError(f"ChoiceQuestion '{self.id}' must provide at least one choice.")
        if len(self.choices) > 255:
            raise ValueError(f"ChoiceQuestion '{self.id}' exceeds maximum choices limit of 255.")


@dataclass(frozen=True)
class NoulQuestion(BaseQuestion):
    """Boolean judgment question: returns probability that the statement/predicate is True."""
    pass


@dataclass(frozen=True)
class ScoreQuestion(BaseQuestion):
    """Rubric scoring question: rates state on an ordered discrete scale (2 to 10 levels) or 0..100."""
    levels: int = 5
    rubric: Optional[List[str]] = None

    def __post_init__(self):
        if self.levels < 2 or self.levels > 10:
            raise ValueError(f"ScoreQuestion '{self.id}' levels must be between 2 and 10 (got {self.levels}).")


@dataclass(frozen=True)
class ExtractQuestion(BaseQuestion):
    """Span extraction question: identifies entity (path, flag, command, identifier) from the state."""
    target_type: str = "general"


# ---------------------------------------------------------------------------
# Shared question texts (single source of truth for runtime AND dataset
# generator — the model conditions on the exact string, so drift = garbage).
# ---------------------------------------------------------------------------
INTENT_COUNT_QUESTION = (
    "Сколько независимых задач (интентов) содержит этот запрос? "
    "Мета-текст, болтовня и приветствия задачами не считаются."
)
FIRST_INTENT_QUESTION = (
    "Извлеките текст первой задачи (конкретное действие с инструментом, "
    "без мета-текста и болтовни)"
)


# ---------------------------------------------------------------------------
# Decision Results
# ---------------------------------------------------------------------------

@dataclass
class ChoiceResult:
    """Result of a ChoiceQuestion evaluation."""
    choice: str
    confidence: float
    probabilities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


@dataclass
class NoulResult:
    """Result of a NoulQuestion evaluation (calibrated boolean judgment)."""
    value: bool
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "noul",
            "value": self.value,
            "probability": round(self.probability, 4),
        }


@dataclass
class ScoreResult:
    """Result of a ScoreQuestion evaluation (ordinal rubric with per-level distribution)."""
    score: float
    level: int
    label: str
    probabilities: Dict[str, float] = field(default_factory=dict)  # per-level probs, keys "1".."L"
    confidence: float = 0.0  # modal concentration (Jev-analysis formula)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "score",
            "score": round(self.score, 2),
            "level": self.level,
            "label": self.label,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ExtractResult:
    """Result of an ExtractQuestion evaluation (non-autoregressive span)."""
    text: str
    span: Tuple[int, int]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "extract",
            "text": self.text,
            "span": list(self.span),
            "confidence": round(self.confidence, 4),
        }


LeethResult = Union[ChoiceResult, NoulResult, ScoreResult, ExtractResult]


@dataclass
class LeethDecisionBatch:
    """Batch container holding decision results for multiple questions evaluated on a state."""
    state_preview: str
    results: Dict[str, LeethResult] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    device: str = "cpu"
    state_nll: Optional[float] = None  # MLM familiarity proxy (low = familiar)

    def __getitem__(self, key: str) -> LeethResult:
        if key not in self.results:
            raise KeyError(f"Question ID '{key}' not found in LeethDecisionBatch.")
        return self.results[key]

    def get(self, key: str, default: Optional[Any] = None) -> Optional[LeethResult]:
        return self.results.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.results

    def __len__(self) -> int:
        return len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_preview": self.state_preview,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "device": self.device,
            "state_nll": round(self.state_nll, 4) if self.state_nll is not None else None,
            "decisions": {k: v.to_dict() for k, v in self.results.items()},
        }
