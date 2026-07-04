"""Docker-free meta-search: our own lightweight federation of keyless sources.

The old search_web hit a single backend (DuckDuckGo HTML scraping). When DDG
rate-limited, everything stalled behind an exponential backoff — the slowness
the user felt. Here we query several independent, keyless sources and merge
them, so one engine failing (or rate-limiting) just means the others carry the
result. No API keys, no Docker, no per-query cost.

Engines (each isolated — a failure yields [] instead of taking down the search):
  * DuckDuckGo  — general web (single fast attempt, no long backoff);
  * Wikipedia   — encyclopedic, stable MediaWiki JSON API;
  * StackOverflow — programming Q&A, keyless StackExchange API.

Each engine has the same shape ``fn(query, limit) -> list[dict]`` with dicts
``{title, url, snippet, source}``, so adding an engine is a one-liner and the
merge logic stays engine-agnostic. Results are interleaved round-robin across
sources for diversity and de-duplicated by normalised URL.
"""

import html as _html
import re
import urllib.parse

from logger import get_logger

log = get_logger("research")

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover - dependency always present in practice
    DDGS = None

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"


def _get_json(url: str, params: dict, timeout: int = 10) -> dict:
    """GET a JSON endpoint. Isolated so engine parsers are testable offline."""
    import requests
    resp = requests.get(url, params=params, headers={"User-Agent": _UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _ddg_search(query: str, limit: int = 5) -> list:
    """General web via DuckDuckGo — a single fast attempt (no long backoff);
    the other engines cover for it when it rate-limits."""
    if DDGS is None:
        return []
    out = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=limit):
            out.append({
                "title": r.get("title", "") or "(no title)",
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "source": "duckduckgo",
            })
    return out


def _wikipedia_search(query: str, limit: int = 3) -> list:
    """Encyclopedic results via the MediaWiki search API (stable JSON)."""
    data = _get_json("https://en.wikipedia.org/w/api.php", {
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": limit,
    })
    out = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = re.sub("<[^>]+>", "", item.get("snippet", "")).strip()
        url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        out.append({"title": title, "url": url, "snippet": snippet, "source": "wikipedia"})
    return out


def _stackoverflow_search(query: str, limit: int = 3) -> list:
    """Programming Q&A via the keyless StackExchange API."""
    data = _get_json("https://api.stackexchange.com/2.3/search/advanced", {
        "order": "desc", "sort": "relevance", "q": query,
        "site": "stackoverflow", "pagesize": limit,
    })
    out = []
    for item in data.get("items", []):
        score = item.get("score", 0)
        answered = "answered" if item.get("is_answered") else "unanswered"
        out.append({
            "title": _html.unescape(item.get("title", "")),
            "url": item.get("link", ""),
            "snippet": f"[{score} votes, {answered}]",
            "source": "stackoverflow",
        })
    return out


DEFAULT_ENGINES = [_ddg_search, _wikipedia_search, _stackoverflow_search]


def _norm_url(u: str) -> str:
    """Normalise a URL for dedup: host lowercased, trailing slash and fragment
    dropped. Different sources often return the same page with cosmetic diffs."""
    try:
        p = urllib.parse.urlsplit(u)
        return f"{p.netloc.lower()}{p.path.rstrip('/')}"
    except Exception:
        return u


def meta_search(query: str, max_results: int = 8, engines=None,
                per_engine: int = None) -> list:
    """Federate the engines and return merged, de-duplicated results.

    Engines run independently; one raising is logged and treated as empty.
    Results are interleaved round-robin so no single source dominates the top.
    ``engines`` is injectable for testing.
    """
    engines = engines if engines is not None else DEFAULT_ENGINES
    if per_engine is None:
        per_engine = max(3, max_results // max(1, len(engines)) + 2)

    buckets = []
    for eng in engines:
        try:
            buckets.append(list(eng(query, per_engine)) or [])
        except Exception as e:
            log.debug("search engine %s failed: %s", getattr(eng, "__name__", eng), e)
            buckets.append([])

    merged, seen = [], set()
    depth = max((len(b) for b in buckets), default=0)
    for i in range(depth):
        for b in buckets:
            if i >= len(b):
                continue
            r = b[i]
            key = _norm_url(r.get("url", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(r)
            if len(merged) >= max_results:
                return merged
    return merged
