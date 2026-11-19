"""Inference runtime with the same typed API as the old ``src.leeth.runtime`` (Choice / Noul / Score / Extract).

    from src.leeth2.runtime import Leeth2Runtime
    rt = Leeth2Runtime.load("path/to/final")
    batch = rt.decide("User request: ...", [ChoiceQuestion("tool", "Which tool?", ["read_file", "run_command"]), NoulQuestion("safe", "Is it safe?")])
    batch["tool"].choice, batch["safe"].probability

Questions, options and the state are plain strings, so the same model serves routing, sorting, NPC decisions,
browser steps and anything else that can be said in words.
"""

from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..leeth.primitives import (  
    BaseQuestion,
    ChoiceQuestion,
    ChoiceResult,
    ExtractQuestion,
    ExtractResult,
    LeethDecisionBatch,
    LeethResult,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
)
from .encoding import Leeth2Encoder
from .evaluate import Prediction, predict
from .window import shrink
from .model import Leeth2Model, compress_for_inference
from .schema import Sample


@dataclass
class GatedChoice:
    """A choice under rules: the best option whose gate (a yes/no question about that option) passed."""

    choice: Optional[str]  # None: every checked candidate was blocked -> the caller should not act on its own
    confidence: float  # probability of ``choice`` among the options that were NOT blocked
    probabilities: Dict[str, float]  # the unconstrained distribution
    gates: Dict[str, float]  # P(gate passes) of every candidate that was checked
    blocked: List[str]  # candidates that were checked and failed, most likely first
    unconstrained: str  # what the plain choice would have been


GateSpec = Union[str, Dict[str, str], Callable[[str], str]]


def _gate_text(gate: GateSpec, option: str) -> str:
    if callable(gate):
        return gate(option)
    if isinstance(gate, dict):
        return gate[option]
    return gate.replace("{option}", option)


class Leeth2Runtime:
    def __init__(self, model: Leeth2Model, encoder: Leeth2Encoder, device: str):
        self.model = model.eval()
        self.encoder = encoder
        self.device = device
        self.amp = device.startswith("cuda")
        # A list longer than ``shortlist_above`` options can be cut to ``shortlist_k`` by an outside retriever first (off unless
        # set_shortlist is called). Why it exists: the encoder gives each option ``cap`` tokens, which is ~48 at 8-32 options and
        # ~15 at 128-255, so the model reads big lists with half-erased descriptions — accuracy 0.57-0.62 against 0.77-0.82 at 8.
        # Measured 2026-11-04 on the frozen exams: with the right option among 96 candidates the same model answers 0.762 at
        # 128-option lists against 0.569 in one pass (an embedding retriever, see reports/leeth3_decision_2026-11-04.md §4b).
        self.retriever: Optional[Callable[[str, str, List[str], int], Sequence[str]]] = None
        self.shortlist_above = 64
        self.shortlist_k = 96

    def set_shortlist(self, retriever: Optional[Callable[[str, str, List[str], int], Sequence[str]]],
                      above: int = 64, k: int = 96) -> None:
        """Install a two-stage chooser: ``retriever(state, question, options, k)`` returns at most ``k`` of the options, and only
        those are asked. It must never invent an option — the answer and the confidences are then relative to what it returned,
        exactly like ``choose_gated``. ``None`` turns the stage off again."""
        self.retriever = retriever
        self.shortlist_above = int(above)
        self.shortlist_k = int(k)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None, max_len: Optional[int] = None, compress: Optional[str] = None) -> "Leeth2Runtime":
        from transformers import AutoTokenizer

        p = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = Leeth2Model.from_pretrained(p, device=device, dtype=torch.float16 if device.startswith("cuda") else torch.float32)
        if not device.startswith("cuda"):  # on a GPU the fp16 model is already small; on CPU the lossless squeeze saves ~400 MB
            model = compress_for_inference(model, compress or os.environ.get("LEETH2_COMPRESS", "emb16"))
        tok = AutoTokenizer.from_pretrained(str(p))
        ml = max_len or int(os.environ.get("LEETH2_MAXLEN", "4096"))
        return cls(model, Leeth2Encoder(tok, model.cfg.vocab_size, max_len=ml, hard_max_len=8192), device)

    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _to_sample(state: str, q: BaseQuestion) -> Sample:
        if isinstance(q, ChoiceQuestion):
            return Sample("rt", "rt", "mix", "choice", state, q.question, list(q.choices), 0)
        if isinstance(q, NoulQuestion):
            return Sample("rt", "rt", "mix", "noul", state, q.question, None, 0.0)
        if isinstance(q, ScoreQuestion):
            q_text = q.question
            if q.rubric:
                q_text += "\n" + "\n".join(f"{i + 1} = {r}" for i, r in enumerate(q.rubric))
            return Sample("rt", "rt", "mix", "score", state, q_text, None, 1.0, levels=q.levels)
        if isinstance(q, ExtractQuestion):
            return Sample("rt", "rt", "mix", "extract", state, q.question, None, None)
        raise TypeError(f"unsupported question type {type(q).__name__}")

    @staticmethod
    def _to_result(q: BaseQuestion, p: Prediction) -> LeethResult:
        if isinstance(q, ChoiceQuestion):
            probs = {c: float(x) for c, x in zip(q.choices, p.probs)}
            best = int(np.argmax(p.probs))
            return ChoiceResult(choice=q.choices[best], confidence=float(p.probs[best]), probabilities=probs)
        if isinstance(q, NoulQuestion):
            return NoulResult(value=p.p_true >= 0.5, probability=float(p.p_true))
        if isinstance(q, ScoreQuestion):
            lv = q.levels
            top = int(np.argmax(p.level_probs))
            label = q.rubric[top] if q.rubric and top < len(q.rubric) else str(top + 1)
            return ScoreResult(
                score=float(p.level / lv * 100.0), level=top + 1, label=label,
                probabilities={str(i + 1): float(x) for i, x in enumerate(p.level_probs)}, confidence=float(np.max(p.level_probs)),
            )
        return ExtractResult(text=p.text or "", span=tuple(p.span) if p.span else (0, 0), confidence=float(p.confidence or 0.0))

    # -- long states -------------------------------------------------------------------------------
    def _state_windows(self, state: str) -> Optional[List[Tuple[int, int]]]:
        """Character ranges of overlapping windows when the state does not fit one pass, None when it does. One pass sees at most
        ``max_len`` tokens; the encoder would keep the head and the tail of a longer state and lose the middle, which is fine for
        a routing decision (the request is at the ends) but not for finding a value somewhere inside a long document."""
        enc = self.encoder
        room = max(64, enc.max_len - enc.q_cap - 8)
        if len(state) <= room:  # a token is at least one character: a state with fewer characters than the window cannot overflow it
            return None
        ids, offs = enc._ids_offsets(state)
        if len(ids) <= room:
            return None
        step = max(1, int(room * 0.75))
        out: List[Tuple[int, int]] = []
        a = 0
        while True:
            b = min(len(ids), a + room)
            out.append((int(offs[a][0]), int(offs[b - 1][1])))
            if b >= len(ids):
                return out
            a += step

    def fit_state(self, state: str, question: str, budget_tokens: int = 0) -> str:
        """What the model gets to read from a long state: the parts that relate to the question.

        The encoder keeps the first 70 % and the last 30 % of a long state (src/leeth2/encoding.py:174), so a fact in the middle is
        never shown to the model — the frozen long exam scored 0.52-0.56 there, the same as guessing, against 0.76 at the edges.
        Measured with this window instead (2026-11-04, 80 items per condition, same weights): the WORST position rose 0.06-0.11
        (0.525→0.625, 0.563→0.625, 0.550→0.663) and the average over nine conditions rose from 0.679 to 0.692, at no cost in memory
        or latency — it is word overlap over the lines of the text.

        LEETH2_WINDOW=0 turns it off (the model then reads head-and-tail as it did before).
        """
        if not state or not state.strip():
            return state
        if os.environ.get("LEETH2_WINDOW", "1").strip() in ("0", "false", "no"):
            return state
        budget = int(budget_tokens or (self.encoder.max_len or 4096) * 0.95)
        try:
            tok = self.encoder.tok
            if len(tok(state)["input_ids"]) <= budget:
                return state
            chars = int(budget * 3.4)            # Cyrillic measures ~3.8 chars per token here; a low estimate keeps the result inside
            out = state
            for _ in range(6):
                out = shrink(state, question, chars)
                n = len(tok(out)["input_ids"])
                if n <= budget:
                    break
                chars = max(400, int(chars * budget / max(1, n) * 0.95))
            return out
        except Exception:                        # never lose an answer because the window could not be computed
            return state

    def decide(self, state: str, questions: Sequence[BaseQuestion]) -> LeethDecisionBatch:
        t0 = time.perf_counter()
        wins = self._state_windows(state) if any(isinstance(q, ExtractQuestion) for q in questions) else None
        samples: List[Sample] = []
        plan: List[List[Tuple[int, int]]] = []  # per question: (index of its sample, character offset of the window it saw)
        for q in questions:
            if wins and isinstance(q, ExtractQuestion):
                base = self._to_sample(state, q)
                rows = []
                for a, b in wins:
                    samples.append(dataclasses.replace(base, state=state[a:b]))
                    rows.append((len(samples) - 1, a))
                plan.append(rows)
            else:
                samples.append(self._to_sample(self.fit_state(state, q.question), q))
                plan.append([(len(samples) - 1, 0)])
        preds = predict(self.model, self.encoder, samples, device=self.device, amp=self.amp, calibrated=True, tokens_per_batch=24000)
        results: Dict[str, LeethResult] = {}
        for q, rows in zip(questions, plan):
            if len(rows) == 1:
                p = preds[rows[0][0]]
                if p.span is not None and rows[0][1]:
                    p = dataclasses.replace(p, span=(p.span[0] + rows[0][1], p.span[1] + rows[0][1]))
            else:  # a long document: the window with the most confident answer wins
                found = [(preds[i], a) for i, a in rows if not preds[i].no_answer]
                if found:
                    p, a = max(found, key=lambda x: x[0].confidence or 0.0)
                    p = dataclasses.replace(p, span=(p.span[0] + a, p.span[1] + a))
                else:
                    p = preds[rows[0][0]]
            results[q.id] = self._to_result(q, p)
        return LeethDecisionBatch(state_preview=state[:120], results=results, elapsed_ms=(time.perf_counter() - t0) * 1000.0, device=self.device)

    def extract_all(self, state: str, question: str, max_items: int = 32) -> List[ExtractResult]:
        """Every answer of a question like "all file paths" in one pass (a long state: one pass per window), in text order."""
        wins = self._state_windows(state) or [(0, len(state))]
        samples = [Sample("rt", "rt", "mix", "extract", state[a:b], question, None, None, meta={"multi": True}) for a, b in wins]
        preds = predict(self.model, self.encoder, samples, device=self.device, amp=self.amp, calibrated=True, tokens_per_batch=24000)
        found: List[ExtractResult] = []
        for (a, _), p in zip(wins, preds):
            for o in p.spans or []:
                found.append(ExtractResult(text=o["text"], span=(o["span"][0] + a, o["span"][1] + a), confidence=float(o["confidence"])))
        kept: List[ExtractResult] = []
        for r in sorted(found, key=lambda r: -r.confidence):  # overlapping windows see the same answer twice: the more confident reading stays
            if all(r.span[1] <= k.span[0] or r.span[0] >= k.span[1] for k in kept):
                kept.append(r)
        kept.sort(key=lambda r: r.span)
        return kept[:max_items]

    def choose_gated(self, state: str, question: str, options: Sequence[str], gate: GateSpec, threshold: float = 0.5, top_k: int = 4) -> GatedChoice:
        """Constrained decoding: the most probable option among those whose gate passes.

        ``gate`` is the yes/no question that must come out true for an option to be allowed ("Is it safe to do {option} here?"; a
        string with ``{option}``, a dict option -> question, or a function). A blocked option is removed from the distribution, so
        ``confidence`` is the probability of the winner among what is left, not of the original softmax. The ``top_k`` most
        likely options are gated in ONE batch with the choice itself when they are known in advance (few options), otherwise
        right after it; options below the top_k are never executed unchecked: if all of the checked ones fail the answer is
        ``None`` and the caller hands the decision to someone else."""
        options = list(options)
        small = len(options) <= top_k
        qs: List[BaseQuestion] = [ChoiceQuestion("c", question, options)]
        if small:
            qs += [NoulQuestion(f"g{i}", _gate_text(gate, o)) for i, o in enumerate(options)]
        batch = self.decide(state, qs)
        first = batch["c"]
        probs = dict(first.probabilities)  # type: ignore[union-attr]
        order = sorted(options, key=lambda o: -probs[o])
        checked = order if small else order[:top_k]
        if small:
            gates = {o: float(batch[f"g{i}"].probability) for i, o in enumerate(options)}  # type: ignore[union-attr]
        else:
            res = self.decide(state, [NoulQuestion(f"g{i}", _gate_text(gate, o)) for i, o in enumerate(checked)])
            gates = {o: float(res[f"g{i}"].probability) for i, o in enumerate(checked)}  # type: ignore[union-attr]
        blocked = [o for o in checked if gates[o] < threshold]
        passed = [o for o in checked if gates[o] >= threshold]
        if not passed:
            return GatedChoice(None, 0.0, probs, gates, blocked, order[0])
        win = passed[0]
        left = 1.0 - sum(probs[o] for o in blocked)
        return GatedChoice(win, float(probs[win] / left) if left > 1e-9 else 0.0, probs, gates, blocked, order[0])

    def _many(self, states: Sequence[str], q: BaseQuestion) -> List[LeethResult]:
        samples = [self._to_sample(s, q) for s in states]
        if not samples:
            return []
        preds = predict(self.model, self.encoder, samples, device=self.device, amp=self.amp, calibrated=True, tokens_per_batch=24000)
        return [self._to_result(q, p) for p in preds]

    def judge_many(self, states: Sequence[str], question: str) -> List[NoulResult]:
        """The same yes/no question about many states in ONE batched pass (a list of posts, bookmarks, file names ...)."""
        return self._many(states, NoulQuestion("q", question))  # type: ignore[return-value]

    def choose_many(self, states: Sequence[str], question: str, options: Sequence[str]) -> List[ChoiceResult]:
        """The same choice (the same options) about many states in ONE batched pass: categorise a list."""
        return self._many(states, ChoiceQuestion("q", question, list(options)))  # type: ignore[return-value]

    def choose(self, state: str, question: str, options: Sequence[str]) -> ChoiceResult:
        opts = list(options)
        if self.retriever is not None and len(opts) > self.shortlist_above:
            cut = [o for o in self.retriever(state, question, opts, self.shortlist_k) if o in set(opts)]
            if cut:
                opts = cut                          # two stages: the retriever cuts, the model answers on full descriptions
        return self.decide(state, [ChoiceQuestion("q", question, opts)])["q"]  # type: ignore[return-value]

    def judge(self, state: str, question: str) -> NoulResult:
        return self.decide(state, [NoulQuestion("q", question)])["q"]  # type: ignore[return-value]
