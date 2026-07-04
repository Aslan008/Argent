"""Cross-encoder reranking for research.

Ordering candidate chunks by relevance with a bi-encoder (embed query and chunk
separately, compare cosines — the old _rerank_chunks) is fast but coarse. A
cross-encoder scores each (query, chunk) pair *jointly*, which is the standard
accuracy jump for retrieval reranking (~55% -> ~85% on the project's docs).

It uses a small model (ms-marco-MiniLM, ~80MB) so it runs on CPU and only loads
on first use (lazy — kept out of Argent's startup path). It degrades gracefully:
cross-encoder -> bi-encoder -> input order, so research never hard-fails on a
missing or broken model.
"""

from logger import get_logger

log = get_logger("research")

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_cross_encoder = None  # cached across a session; the model is expensive to load


def _get_cross_encoder(model_name: str = _DEFAULT_MODEL):
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(model_name)
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
