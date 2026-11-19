"""Cutting a long option list down before the model reads it.

The encoder gives every option ``cap`` tokens (src/leeth2/encoding.py:134): ~48 at 8-32 options, ~15 at 128-255, where the list
fills the window. So at 128-255 options the model reads half-erased descriptions and scores 0.57-0.62 against 0.77-0.82 at 8.

Measured on the frozen exams (2026-11-04, reports/leeth3_decision_2026-11-04.md §4b): with the right option among 96 candidates
the same model answers 0.762 at 128-option lists against 0.569 in one pass; a hybrid of embeddings and word overlap is the best
choice at 255-option lists (0.651 against 0.619). Word overlap alone is poor (recall@16 0.25-0.32) — it is only useful as a
second opinion next to embeddings.

    retriever = ShortlistRetriever(make_mmbert_embedder())
    rt.set_shortlist(retriever, above=64, k=96)

Nothing here loads a model by itself: the embedder is passed in, so the runtime keeps its own dependencies and a test can run
without one.
"""

from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

_WORD = re.compile(r"[a-zа-яё0-9_]{2,}", re.I)
Embed = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def _words(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class ShortlistRetriever:
    """Keep the ``k`` options that most plausibly answer the question: embeddings first, word overlap as the second opinion.

    Vectors of options are cached by their text — a dropdown, a tool catalogue or a category list repeats itself between
    requests, so after the first call the cost is one embedding of the question.
    """

    def __init__(self, embed: Embed, cache_entries: int = 20000, semantic_weight: float = 1.0, lexical_weight: float = 1.0,
                 lexical_floor: float = 0.0, embed_query: Optional[Embed] = None, query_from_state: bool = True,
                 keep_least_relevant: bool = True) -> None:
        self.embed = embed
        self.embed_query = embed_query or embed
        # The measuring probe embedded the STATE only, at 256 tokens: the question is the same for every item of a request, so it
        # adds no discrimination, and appending it to a long state only pushes the state out of the 128-token window. Measured
        # 2026-11-05: question+state at 128 tokens gave 0.597 at 128 options and dropped none-recall to 0.52 against 0.762/0.72.
        self.query_from_state = query_from_state
        self.keep_least_relevant = keep_least_relevant
        self.cache_entries = max(256, cache_entries)
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.lexical_floor = lexical_floor
        self.calls = 0                       # for tests and for a sanity line in logs
        self.cache_hits = 0

    # -- vectors -------------------------------------------------------------------------------------
    def _vectors(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        missing: List[str] = []
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is None:
                missing.append(t)
            else:
                self._cache.move_to_end(t)
                self.cache_hits += 1
                out[i] = v
        if missing:
            fresh = list(self.embed(missing))
            for t, v in zip(missing, fresh):
                v = [float(x) for x in v]
                self._cache[t] = v
                while len(self._cache) > self.cache_entries:
                    self._cache.popitem(last=False)
            by_text = dict(zip(missing, fresh))
            for i, t in enumerate(texts):
                if out[i] is None:
                    out[i] = [float(x) for x in by_text[t]]
        return [v or [] for v in out]

    # -- ranking -------------------------------------------------------------------------------------
    def rank(self, state: str, question: str, options: Sequence[str]) -> List[str]:
        """Best first. Every option appears exactly once, so a bad retriever degrades to the old behaviour rather than losing one."""
        self.calls += 1
        opts = list(options)
        if len(opts) <= 1:
            return opts
        # two different queries on purpose: the embedding sees the state alone (a long state put behind a constant question would be
        # cut at the window edge), while the word overlap needs the question too — it is where the subject usually is.
        sem_query = (state or question).strip() if self.query_from_state else f"{question}\n{state}".strip()
        lex_query = f"{question}\n{state}".strip()
        qv = list(self.embed_query([sem_query]))[0]
        ov = self._vectors(opts)
        semantic = [_cos(qv, v) for v in ov]

        toks = [_words(o) for o in opts]
        df = Counter()
        for t in toks:
            for w in set(t):
                df[w] += 1
        total = max(1, len(toks))
        qt = Counter(_words(lex_query))
        lexical = []
        for t in toks:
            have = Counter(t)
            s = sum(qt[w] * math.log(1 + total / (1 + df[w])) for w in have if w in qt)
            lexical.append(s / math.sqrt(max(1, len(t))))
        top_lex = max(lexical) if lexical else 0.0
        lexical = [(x / top_lex) if top_lex > 0 else 0.0 for x in lexical]

        score = [self.semantic_weight * semantic[i] + self.lexical_weight * lexical[i] for i in range(len(opts))]
        order = sorted(range(len(opts)), key=lambda i: (-score[i], i))
        return [opts[i] for i in order]

    def __call__(self, state: str, question: str, options: Sequence[str], k: int) -> List[str]:
        ranked = self.rank(state, question, options)
        k = max(1, int(k))
        if self.keep_least_relevant and len(ranked) > k >= 2:
            # The shortlist keeps the options most similar to the request, which are exactly its hardest distractors — so the one
            # option that means "none of these fits" is the LEAST similar and would be cut away, and the model then has nothing to
            # decline with. Measured 2026-11-05: without this, "none of these" recall fell 0.722 -> 0.519 while list accuracy rose.
            return ranked[:k - 1] + [ranked[-1]]
        return ranked[:k]


class LazyEmbedder:
    """Build the real embedder on the first call: mmBERT is a 1.2 GB file, and most requests never see a long option list."""

    def __init__(self, factory: Callable[[], Embed]) -> None:
        self.factory = factory
        self._fn: Optional[Embed] = None

    def _inner(self):
        if self._fn is None:
            self._fn = self.factory()
        return self._fn

    def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self._inner()(texts)

    def query(self, texts: Sequence[str]):
        """The request is embedded with a wider window than an option description (see MmbertEmbedder)."""
        inner = self._inner()
        return inner.query(texts) if hasattr(inner, "query") else inner(texts)


class MmbertEmbedder:
    """One loaded model, two windows: a request is embedded with more text than an option description needs (256 against 48, as in
    the probe that measured the 0.762), so both call shapes share the weights instead of loading them twice."""

    def __init__(self, model, tok, device, doc_tokens: int = 48, query_tokens: int = 256, batch: int = 32) -> None:
        self.model, self.tok, self.device = model, tok, device
        self.doc_tokens, self.query_tokens, self.batch = doc_tokens, query_tokens, batch

    def _run(self, texts: Sequence[str], max_tokens: int):
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(list(texts[i:i + self.batch]), padding=True, truncation=True, max_length=max_tokens,
                               return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out) if out else []

    def __call__(self, texts: Sequence[str]):
        return self._run(texts, self.doc_tokens)

    def query(self, texts: Sequence[str]):
        return self._run(texts, self.query_tokens)


def make_mmbert_embedder(model_dir: str = "", device: Optional[str] = None, max_tokens: int = 128, batch: int = 32) -> Embed:
    """Mean-pooled, L2-normalised vectors from the mmBERT base Leeth already ships with. Imported here so that a caller who never
    has a long option list never pays for torch or the weights."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = model_dir or "mmbert-base"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()

    def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                enc = tok(list(texts[i:i + batch]), padding=True, truncation=True, max_length=max_tokens,
                          return_tensors="pt").to(model.device)
                h = model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu())
        return torch.cat(out) if out else []

    return embed


def _mmbert(model_dir: str, device: Optional[str]):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir or "mmbert-base")
    model = AutoModel.from_pretrained(model_dir or "mmbert-base").to(device or ("cuda" if torch.cuda.is_available() else "cpu")).eval()
    return MmbertEmbedder(model, tok, model.device)


def install_shortlist(rt, model_dir: str = "", device: Optional[str] = None, above: int = 64, k: int = 96) -> None:
    """Give a runtime the two-stage chooser. Shared by the product (src/leeth/runtime.py) and the exam, so that what is measured
    is what ships — the Index otherwise scores the model on lists the product never sends it."""
    lazy = LazyEmbedder(lambda: _mmbert(model_dir, device))
    rt.set_shortlist(ShortlistRetriever(lazy, embed_query=lambda texts: lazy.query(texts)), above=above, k=k)
