"""Cross-encoder reranking for research.

Ordering candidate chunks by relevance with a bi-encoder (embed query and chunk
separately, compare cosines — the old _rerank_chunks) is fast but coarse. A
cross-encoder scores each (query, chunk) pair *jointly*, which is the standard
accuracy jump for retrieval reranking (~55% -> ~85% on the project's docs).

The default model (ms-marco-MiniLM, 92MB) runs on CPU and loads lazily, kept out
of Argent's startup path. It degrades gracefully: cross-encoder -> bi-encoder ->
input order, so research never hard-fails on a missing or broken model.

LANGUAGE: the default model is ENGLISH-ONLY, and measurement shows this is not a
mild degradation. On a Russian query against Russian passages it ranked a borscht
recipe (7.06) ABOVE a directly relevant .NET GC document (6.41) — inverted, worse
than random. Worse, in a MIXED pool a relevant Russian passage scores about -0.33
where the same passage in English scores +6.99, so almost any mediocre English
chunk outranks an excellent Russian one and non-English sources are effectively
buried. Setting a multilingual model (see config.get_reranker_model) fixes this;
it is opt-in because the multilingual weights are ~471MB versus 92MB — the
difference is the 250k-token vocabulary, not depth, so inference speed is
comparable.
"""

from logger import get_logger

log = get_logger("research")

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Same MiniLM cross-encoder architecture, trained on mMARCO (14 languages incl.
# Russian) — a drop-in swap, no code path changes.
MULTILINGUAL_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_cross_encoder = None       # cached across a session; the model is expensive to load
_cross_encoder_name = None  # so switching the setting reloads instead of reusing


def active_model_name() -> str:
    """The reranker the session should use: multilingual when configured."""
    try:
        from config import get_reranker_model
        return get_reranker_model() or _DEFAULT_MODEL
    except Exception:
        return _DEFAULT_MODEL


def _get_cross_encoder(model_name: str = None):
    global _cross_encoder, _cross_encoder_name
    model_name = model_name or active_model_name()
    if _cross_encoder is None or _cross_encoder_name != model_name:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
        _cross_encoder_name = model_name
        log.info("reranker loaded: %s", model_name)
    return _cross_encoder


def _bi_encoder_rank(query: str, chunks: list, top_n: int) -> list:
    """Fallback: the old bi-encoder cosine ranking."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q = model.encode([query])
    c = model.encode(chunks)
    sims = np.dot(c, q.T).flatten()
    idx = np.argsort(sims)[-top_n:][::-1]
    return [chunks[i] for i in idx]


def rerank(query: str, chunks: list, top_n: int = 15, _scorer=None) -> list:
    """Return the ``top_n`` chunks most relevant to ``query``.

    ``_scorer`` (list[(query, chunk)] -> list[float]) is injectable for testing;
    in production it's the cross-encoder's predict. On any failure, falls back to
    a bi-encoder, then to input order.
    """
    if not chunks:
        return []
    try:
        if _scorer is None:
            _scorer = _get_cross_encoder().predict
        scores = _scorer([(query, c) for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [c for c, _ in ranked[:top_n]]
    except Exception as e:
        log.warning("cross-encoder rerank failed (%s); falling back to bi-encoder", e)
        try:
            return _bi_encoder_rank(query, chunks, top_n)
        except Exception as e2:
            log.warning("bi-encoder rerank failed (%s); using input order", e2)
            return chunks[:top_n]
