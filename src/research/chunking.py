"""Structure-aware chunking for research.

Blind fixed-size character slicing (the old _chunk_text) splits sentences,
words and code mid-token, which degrades both reranking and synthesis. Here we
split on document structure — blank-line paragraphs, or lines when there are no
blank lines — and greedily pack whole units up to a target size, never splitting
a unit (only a single oversized unit is hard-split, as a last resort).
Consecutive chunks overlap by a trailing unit for continuity, unless that unit
alone already fills a chunk.
"""

import re


def _split_units(text: str) -> list:
    """Paragraphs (blank-line separated), falling back to non-empty lines."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return blocks


def _hard_split(unit: str, size: int) -> list:
    return [unit[i:i + size] for i in range(0, len(unit), size)]


def chunk_document(text: str, target_size: int = 1000, overlap_units: int = 1) -> list:
    """Split text into coherent, roughly ``target_size``-sized chunks that
    respect paragraph/line boundaries."""
    if not text or not text.strip():
        return []

    # Explode any single unit larger than the target so packing never emits a
    # giant chunk from one over-long paragraph.
    exploded = []
    for u in _split_units(text):
        exploded.extend(_hard_split(u, target_size) if len(u) > target_size else [u])

    chunks, cur, cur_len = [], [], 0
    for u in exploded:
        if cur and cur_len + len(u) > target_size:
            chunks.append("\n".join(cur))
            # Carry a trailing unit as overlap, but never one that already fills
            # a chunk on its own (that would just duplicate a whole chunk).
            keep = cur[-overlap_units:] if overlap_units else []
            while keep and sum(len(x) for x in keep) >= target_size:
                keep = keep[1:]
            cur, cur_len = list(keep), sum(len(x) for x in keep)
        cur.append(u)
        cur_len += len(u)

    if cur:
        chunks.append("\n".join(cur))
    return chunks
