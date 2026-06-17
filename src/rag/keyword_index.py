"""
Cached BM25 keyword index for RAG hybrid search.

The previous keyword retriever called collection.get() — loading the ENTIRE
collection into memory — and substring-counted every document on EVERY query
(O(corpus) per search). On a large corpus (e.g. tens of thousands of Unity doc
pages) that is slow and doesn't scale.

This builds a proper inverted BM25 index once per collection and caches it,
invalidated when the collection's document count changes (re-index). Queries
then touch only the postings for the query terms, not the whole corpus.
"""

import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str):
    return [t.lower() for t in _TOKEN.findall(text) if len(t) > 2]


class BM25Index:
    """Minimal in-memory BM25 over a fixed document set."""

    def __init__(self, ids, docs, metas, k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.docs = docs
        self.metas = metas
        self.k1 = k1
        self.b = b
        self.postings = defaultdict(list)  # term -> [(doc_idx, term_freq)]
        self.doc_len = []
        df = defaultdict(int)
        for idx, doc in enumerate(docs):
            toks = tokenize(doc)
            self.doc_len.append(len(toks))
            for term, freq in Counter(toks).items():
                self.postings[term].append((idx, freq))
                df[term] += 1
        self.N = max(len(docs), 1)
        self.avgdl = (sum(self.doc_len) / self.N) if self.doc_len else 1.0
        self.idf = {
            t: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for t, d in df.items()
        }

    def search(self, query: str, top_k: int):
        """Return up to top_k (id, doc, meta) ranked by BM25."""
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for idx, freq in postings:
                dl = self.doc_len[idx] or 1
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[idx] += idf * (freq * (self.k1 + 1)) / (denom or 1)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.ids[i], self.docs[i], self.metas[i]) for i, _ in ranked]


# Per-collection cache: name -> (doc_count, BM25Index)
_CACHE = {}


def get_index(col) -> BM25Index:
    """Return a cached BM25 index for a Chroma collection, rebuilding it only
    when the collection's document count has changed."""
    name = getattr(col, "name", id(col))
    try:
        count = col.count()
    except Exception:
        count = -1
    cached = _CACHE.get(name)
    if cached and cached[0] == count:
        return cached[1]
    data = col.get(include=["documents", "metadatas"])
    idx = BM25Index(
        data.get("ids", []) or [],
        data.get("documents", []) or [],
        data.get("metadatas", []) or [],
    )
    _CACHE[name] = (count, idx)
    return idx


def clear_cache():
    """Drop all cached indexes (call after re-indexing a collection)."""
    _CACHE.clear()
