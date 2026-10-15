"""Unified record format for Leeth 2 data.

Every dataset adapter produces ``Sample`` objects, whatever the source looked like.
A sample asks ONE question about ONE state:

    choice   pick one of ``options`` (2..255)          target = index into options, or a probability list (soft label, e.g. several equally good moves)
    noul     calibrated yes/no                         target = P(true) in [0, 1] (soft allowed)
    score    ordinal rating on ``levels`` (2..10)      target = level in [1, levels] (float allowed)
    extract  span(s) of ``state`` answering the question  target = [start_char, end_char), a list of such pairs (every answer:
             "all file paths"), or None (no answer)

``state`` is the long free-form context, ``question`` the instruction / criterion.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

PRIMITIVES = ("choice", "noul", "score", "extract")
MAX_OPTIONS = 255
MIN_LEVELS = 2
MAX_LEVELS = 10
MAX_SPANS = 64


def span_list(target: Any) -> List[Tuple[int, int]]:
    """The spans of an extract target: ``None`` -> [], ``[s, e]`` -> [(s, e)], ``[[s, e], ...]`` -> several (as given)."""
    if target is None:
        return []
    if isinstance(target, (list, tuple)) and len(target) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in target):
        return [(int(target[0]), int(target[1]))]
    if isinstance(target, (list, tuple)) and target and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in target):
        return [(int(p[0]), int(p[1])) for p in target]
    raise ValueError("extract target must be [start, end), a list of such pairs, or None")


@dataclass
class Sample:
    task: str  # dataset / generator id, e.g. "imdb", "argent_route"
    family: str  # coarse group used for reports and mixing: sentiment, nli, mcq, qa, safety, ...
    lang: str  # ISO-ish code of the state/question language ("en", "ru", "mix", "code")
    ptype: str  # one of PRIMITIVES
    state: str
    question: str
    options: Optional[List[str]] = None
    target: Any = None
    levels: Optional[int] = None
    weight: float = 1.0
    split: str = "train"
    meta: Optional[Dict[str, Any]] = None

    # -- validation -----------------------------------------------------------------
    def validate(self) -> None:
        if self.ptype not in PRIMITIVES:
            raise ValueError(f"unknown ptype {self.ptype!r}")
        if not isinstance(self.state, str) or not isinstance(self.question, str):
            raise ValueError("state and question must be strings")
        if not self.question.strip():
            raise ValueError("empty question")
        if self.ptype == "choice":
            opts = self.options
            if not opts or not (2 <= len(opts) <= MAX_OPTIONS):
                raise ValueError(f"choice needs 2..{MAX_OPTIONS} options, got {0 if not opts else len(opts)}")
            if any((not isinstance(o, str)) or not o.strip() for o in opts):
                raise ValueError("empty option")
            if len(set(opts)) != len(opts):
                raise ValueError("duplicate options")
            t = self.target
            if isinstance(t, (list, tuple)):
                if len(t) != len(opts) or any((not isinstance(x, (int, float))) or x < 0 for x in t) or abs(sum(t) - 1.0) > 1e-3:
                    raise ValueError("soft choice target must be a probability list over the options")
            elif isinstance(t, bool) or not isinstance(t, int) or not (0 <= t < len(opts)):
                raise ValueError(f"choice target must be an index into options, got {t!r}")
        elif self.ptype == "noul":
            t = self.target
            if isinstance(t, bool):
                t = float(t)
            if not isinstance(t, (int, float)) or not (0.0 <= float(t) <= 1.0) or math.isnan(float(t)):
                raise ValueError(f"noul target must be in [0, 1], got {t!r}")
        elif self.ptype == "score":
            lv = self.levels
            if lv is None or not (MIN_LEVELS <= lv <= MAX_LEVELS):
                raise ValueError(f"score levels must be {MIN_LEVELS}..{MAX_LEVELS}, got {lv!r}")
            t = self.target
            if not isinstance(t, (int, float)) or isinstance(t, bool) or math.isnan(float(t)):
                raise ValueError(f"score target must be a number, got {t!r}")
            if not (1.0 - 1e-6 <= float(t) <= lv + 1e-6):
                raise ValueError(f"score target {t} outside 1..{lv}")
        else:  # extract
            t = self.target
            if t is not None:
                spans = span_list(t)
                if not 1 <= len(spans) <= MAX_SPANS:
                    raise ValueError(f"extract target needs 1..{MAX_SPANS} spans, got {len(spans)}")
                prev_end = 0
                for s, e in spans:
                    if not (0 <= s < e <= len(self.state)):
                        raise ValueError(f"extract span {[s, e]} outside state of {len(self.state)} chars")
                    if s < prev_end:
                        raise ValueError("extract spans must be sorted and must not overlap")
                    prev_end = e

    # -- (de)serialisation ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and not (k == "weight" and v == 1.0) and not (k == "split" and v == "train")} | {
            "ptype": self.ptype
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, line: str) -> "Sample":
        return cls.from_dict(json.loads(line))

    # -- helpers --------------------------------------------------------------------
    def hard_choice(self) -> int:
        """Index of the (most probable) correct option."""
        t = self.target
        if isinstance(t, (list, tuple)):
            return max(range(len(t)), key=lambda i: t[i])
        return int(t)

    def spans(self) -> List[Tuple[int, int]]:
        """Character spans of an extract target (empty = no answer)."""
        return span_list(self.target) if self.ptype == "extract" else []

    def char_len(self) -> int:
        n = len(self.state) + len(self.question)
        if self.options:
            n += sum(len(o) + 1 for o in self.options)
        return n
