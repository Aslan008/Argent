"""Sample -> token ids.

Layout of one sequence (all segments are tokenised separately, so spans and option
boundaries are exact):

    <bos> state... <eos> [T_type] ([Lk]) question... [RD] <eos> [OPT] option1... [OPT] option2... <eos>

* ``[...]`` are *virtual* marker tokens: ids ``>= vocab_size`` that the model maps to a small
  trainable table, so the 256k pretrained embedding matrix can stay frozen.
* The question sits next to the options and the read-out token ``[RD]`` — mmBERT uses sliding-window
  attention in 2/3 of its layers, and a long state between question and options would hide them
  from each other in those layers.
* For ``choice`` every option owns the positions of its marker and tokens (``opt_index``), the model
  mean-pools them into one vector per option.
* ``extract`` answers are pointers into the state tokens; "no answer" points at ``[RD]``.
"""

from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import MAX_LEVELS, Sample, span_list

MARKER_NAMES: List[str] = (
    ["OPT", "RD", "T_CHOICE", "T_NOUL", "T_SCORE", "T_EXTRACT"]
    + [f"L{k}" for k in range(2, MAX_LEVELS + 1)]
    + [f"SPARE{i}" for i in range(3)]
)
N_MARKERS = len(MARKER_NAMES)
PTYPE_ID = {"choice": 0, "noul": 1, "score": 2, "extract": 3}
ID_PTYPE = {v: k for k, v in PTYPE_ID.items()}
IGNORE = -100


@dataclass
class Encoded:
    input_ids: List[int]
    opt_index: List[int]  # per position: 0 = none, j = j-th option *as shown*
    rd_pos: int
    state_lo: int
    state_hi: int  # state tokens live in [state_lo, state_hi)
    n_opts: int
    ptype: int
    levels: int
    choice_target: int  # index in shown order
    noul_target: float  # -1 = none
    score_pos: float  # position in [0, levels - 1]; -1 = none
    start_target: int
    end_target: int
    perm: List[int]  # shown index -> original option index
    start_set: Optional[List[int]] = None  # extract with several answers: every start / end position that is right (start_target / end_target = the first)
    end_set: Optional[List[int]] = None
    weight: float = 1.0
    choice_probs: Optional[List[float]] = None  # soft choice label in shown order
    state_offsets: Optional[List[Tuple[int, int]]] = None  # char span of every state token in [state_lo, state_hi)

    def __len__(self) -> int:
        return len(self.input_ids)


class Leeth2Encoder:
    BIG_LIST = 128  # more options than this and the window grows (up to hard_max_len) so that every option keeps BIG_LIST_TOKENS tokens
    BIG_LIST_TOKENS = 16

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        max_len: int = 512,
        q_cap: int = 192,
        opt_cap: int = 48,
        min_state: int = 32,
        hard_max_len: int = 8192,
    ):
        self.tok = tokenizer
        self.V = int(vocab_size)
        self.max_len = int(max_len)
        self.q_cap = int(q_cap)
        self.opt_cap = int(opt_cap)
        self.min_state = int(min_state)
        self.hard_max_len = int(hard_max_len)
        self.bos = int(tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id)
        self.eos = int(tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id)
        self.pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    # -- markers --------------------------------------------------------------------
    def marker(self, name: str) -> int:
        return self.V + MARKER_NAMES.index(name)

    # -- tokenisation ---------------------------------------------------------------
    def _ids(self, texts: Sequence[str]) -> List[List[int]]:
        if not texts:
            return []
        return self.tok(list(texts), add_special_tokens=False)["input_ids"]

    def _ids_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    # -- main entry -----------------------------------------------------------------
    def encode(self, s: Sample, rng: Optional[random.Random] = None, want_offsets: bool = False) -> Encoded:
        """Encode one sample. With ``rng`` options are shuffled and a too-long state is windowed
        around the answer at random (training); without it everything is deterministic."""
        ptype = s.ptype
        options = list(s.options or [])
        perm = list(range(len(options)))
        if ptype == "choice" and rng is not None:
            rng.shuffle(perm)
        shown = [options[i] for i in perm]

        need_offsets = ptype == "extract" or want_offsets
        if need_offsets:
            state_ids, state_off = self._ids_offsets(s.state)
            rest = self._ids([s.question] + shown)
        else:
            batch = self._ids([s.state, s.question] + shown)
            state_ids, rest = batch[0], batch[1:]
            state_off = None
        q_ids = rest[0][: self.q_cap]
        opt_ids_raw = rest[1:]

        K = len(shown)
        # a training row may carry its own window (meta.max_len: long states, 255-option lists); the option cap follows it, so a long list is
        # squeezed in training exactly as much as at run time (where max_len is the runtime's)
        base_len = int((s.meta or {}).get("max_len", self.max_len)) if rng is not None else self.max_len
        if K > self.BIG_LIST:
            # a long list gets the room to keep BIG_LIST_TOKENS tokens of every option (the cap below is half a window per list); without it 255
            # options were cut to 7 tokens each at the production window, the only place where the cap, not the model, decided the answer
            base_len = max(base_len, min(self.hard_max_len, 2 * self.BIG_LIST_TOKENS * K))
        if K:
            cap = int(max(2, min(self.opt_cap, (base_len // 2) // max(1, K) - 1)))
            opt_ids = [o[:cap] if o else [self.eos] for o in opt_ids_raw]
        else:
            opt_ids = []

        is_score = ptype == "score"
        fixed = 6 + (1 if is_score else 0)  # bos, eos, type, (level), RD, eos, final eos
        need = fixed + len(q_ids) + sum(len(o) + 1 for o in opt_ids)
        budget_len = max(base_len, min(self.hard_max_len, need + self.min_state))
        state_budget = max(0, budget_len - need)

        # -- target spans in state tokens (before windowing) --
        tspans: List[Tuple[int, int]] = []
        if ptype == "extract" and s.target is not None and state_off is not None:
            tok_ends = [b for _, b in state_off]
            tok_starts = [a for a, _ in state_off]
            for cs, ce in span_list(s.target):
                i = bisect.bisect_right(tok_ends, cs)  # first token that ends after the span starts
                j = bisect.bisect_left(tok_starts, ce) - 1  # last token that starts before the span ends
                if i < len(state_off) and j >= i:  # else the span fell between tokens (whitespace only): not an answer
                    tspans.append((i, j))
        ts = tspans[0][0] if tspans else None
        te = tspans[-1][1] if tspans else None

        # -- window the state if too long --
        lo = 0
        if len(state_ids) > state_budget:
            n = len(state_ids)
            if ts is not None and te is not None:
                span = te - ts + 1
                if span >= state_budget:
                    lo = ts
                else:
                    lo_min = max(0, te + 1 - state_budget)
                    lo_max = min(ts, n - state_budget)
                    lo = rng.randint(lo_min, lo_max) if (rng is not None and lo_max > lo_min) else (lo_min + lo_max) // 2
                hi = min(n, lo + state_budget)
                kept = state_ids[lo:hi]
                kept_off = state_off[lo:hi] if state_off is not None else None
            else:
                head = int(state_budget * 0.7)
                tail = state_budget - head
                kept = state_ids[:head] + (state_ids[n - tail :] if tail else [])
                if state_off is not None:
                    kept_off = state_off[:head] + (state_off[n - tail :] if tail else [])
                else:
                    kept_off = None
                lo = 0
                # a head+tail cut has no single window offset; targets are not defined here (no span)
        else:
            kept = state_ids
            kept_off = state_off

        # -- assemble --
        ids: List[int] = [self.bos]
        ids.extend(kept)
        state_lo, state_hi = 1, 1 + len(kept)
        ids.append(self.eos)
        ids.append(self.marker("T_" + ptype.upper()))
        if is_score:
            ids.append(self.marker(f"L{int(s.levels)}") if int(s.levels) >= 2 else self.marker("L2"))
        ids.extend(q_ids)
        rd_pos = len(ids)
        ids.append(self.marker("RD"))
        ids.append(self.eos)
        opt_index = [0] * len(ids)
        for j, o in enumerate(opt_ids, start=1):
            ids.append(self.marker("OPT"))
            ids.extend(o)
            opt_index.extend([j] * (1 + len(o)))
        ids.append(self.eos)
        opt_index.append(0)

        # -- targets --
        choice_target = IGNORE
        noul_target = -1.0
        score_pos = -1.0
        start_t = end_t = IGNORE
        start_set = end_set = None
        choice_probs = None
        if ptype == "choice":
            choice_target = perm.index(s.hard_choice())
            if isinstance(s.target, (list, tuple)):
                choice_probs = [float(s.target[i]) for i in perm]
        elif ptype == "noul":
            noul_target = float(s.target)
        elif ptype == "score":
            score_pos = float(s.target) - 1.0
        else:
            inside = [(a - lo, b - lo) for a, b in tspans if a - lo >= 0 and b - lo < len(kept)]
            if not inside:
                start_t = end_t = rd_pos  # no answer, or every answer cut off by truncation: unanswerable in this view
            else:
                start_t, end_t = state_lo + inside[0][0], state_lo + inside[0][1]
                if len(inside) > 1:
                    start_set = [state_lo + a for a, _ in inside]
                    end_set = [state_lo + b for _, b in inside]

        return Encoded(
            input_ids=ids,
            opt_index=opt_index,
            rd_pos=rd_pos,
            state_lo=state_lo,
            state_hi=state_hi,
            n_opts=K,
            ptype=PTYPE_ID[ptype],
            levels=int(s.levels or 0),
            choice_target=choice_target,
            noul_target=noul_target,
            score_pos=score_pos,
            start_target=start_t,
            end_target=end_t,
            perm=perm,
            start_set=start_set,
            end_set=end_set,
            weight=float(s.weight),
            choice_probs=choice_probs,
            state_offsets=list(kept_off) if (kept_off is not None and (ptype == "extract" or want_offsets)) else None,
        )


def collate(batch: Sequence[Encoded], pad_id: int = 0) -> Dict[str, Any]:
    """Pad a list of ``Encoded`` into tensors."""
    import torch

    B = len(batch)
    L = max(len(e) for e in batch)
    Kmax = max(1, max(e.n_opts for e in batch))
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    opt_index = torch.zeros((B, L), dtype=torch.long)
    state_mask = torch.zeros((B, L), dtype=torch.bool)
    choice_probs = torch.zeros((B, Kmax), dtype=torch.float)
    has_soft = torch.zeros(B, dtype=torch.bool)
    M = max([1] + [len(e.start_set) for e in batch if e.start_set])
    start_set = torch.full((B, M), IGNORE, dtype=torch.long)
    end_set = torch.full((B, M), IGNORE, dtype=torch.long)
    for i, e in enumerate(batch):
        start_set[i, 0], end_set[i, 0] = e.start_target, e.end_target
        if e.start_set:
            start_set[i, : len(e.start_set)] = torch.tensor(e.start_set, dtype=torch.long)
            end_set[i, : len(e.end_set)] = torch.tensor(e.end_set, dtype=torch.long)
        n = len(e)
        if e.choice_probs is not None:
            choice_probs[i, : len(e.choice_probs)] = torch.tensor(e.choice_probs, dtype=torch.float)
            has_soft[i] = True
        input_ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        attn[i, :n] = 1
        opt_index[i, :n] = torch.tensor(e.opt_index, dtype=torch.long)
        state_mask[i, e.state_lo : e.state_hi] = True
        state_mask[i, e.rd_pos] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "opt_index": opt_index,
        "n_opts": torch.tensor([e.n_opts for e in batch], dtype=torch.long),
        "k_max": Kmax,
        "rd_pos": torch.tensor([e.rd_pos for e in batch], dtype=torch.long),
        "state_mask": state_mask,
        "ptype": torch.tensor([e.ptype for e in batch], dtype=torch.long),
        "levels": torch.tensor([e.levels for e in batch], dtype=torch.long),
        "choice_target": torch.tensor([e.choice_target for e in batch], dtype=torch.long),
        "choice_probs": choice_probs,
        "has_soft": has_soft,
        "noul_target": torch.tensor([e.noul_target for e in batch], dtype=torch.float),
        "score_pos": torch.tensor([e.score_pos for e in batch], dtype=torch.float),
        "start_target": torch.tensor([e.start_target for e in batch], dtype=torch.long),
        "end_target": torch.tensor([e.end_target for e in batch], dtype=torch.long),
        "start_set": start_set,
        "end_set": end_set,
        "weight": torch.tensor([e.weight for e in batch], dtype=torch.float),
    }
