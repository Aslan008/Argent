"""Pure helpers for the research synthesis stage: parsing LLM query lists and
grounding chunks with numbered source citations.

Kept free of network/LLM/console so they can be unit-tested directly; the
orchestration in deep_research wires them to the live calls.
"""

import json


def parse_query_list(raw: str, limit: int = 5) -> list:
    """Parse an LLM response that should be a JSON array of query strings.

    Tolerant of a ```json fenced block and of trailing prose. Returns [] on
    anything that isn't a non-empty JSON list, so callers can treat "no
    queries" and "bad output" identically.
    """
    if not raw:
        return []
    candidates = [raw]
    stripped = raw.replace("```json", "").replace("```", "").strip()
    if stripped != raw:
        candidates.append(stripped)
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, list) and data:
            return [str(x) for x in data][:limit]
    return []


def number_sources(chunks: list, source_map: dict) -> tuple:
    """Annotate chunks with [n] citation markers and build the source list.

    Each distinct source URL (in order of first appearance) gets a number; every
    chunk is prefixed with its source's marker. Returns
    ``(annotated_text, sources_lines)`` where annotated_text is the chunks joined
    with separators (fed to extraction/synthesis so the model can cite inline)
    and sources_lines is ``["[1] https://...", ...]`` for the SOURCES section.
    """
    url_to_num = {}
    annotated = []
    for c in chunks:
        url = source_map.get(c)
        if url and url not in url_to_num:
            url_to_num[url] = len(url_to_num) + 1
        num = url_to_num.get(url)
        annotated.append(f"[{num}] {c}" if num else c)
    sources = [f"[{num}] {url}" for url, num in url_to_num.items()]
    return "\n---\n".join(annotated), sources
