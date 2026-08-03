"""Docker-free meta-search: our own lightweight federation of keyless sources.

The old search_web hit a single backend (DuckDuckGo HTML scraping). When DDG
rate-limited, everything stalled behind an exponential backoff — the slowness
the user felt. Here we query several independent, keyless sources and merge
them, so one engine failing (or rate-limiting) just means the others carry the
result. No API keys, no Docker, no per-query cost.

Engines (each isolated — a failure yields [] instead of taking down the search):
  * DuckDuckGo  — general web (single fast attempt, no long backoff);
  * Wikipedia   — encyclopedic, stable MediaWiki JSON API;
  * StackOverflow — programming Q&A, keyless StackExchange API;
  * GitHub      — issues/PRs, where an error string usually appears verbatim
                  next to the maintainer's answer;
  * Brave       — optional, needs an API key: a second INDEPENDENT index
                  (DDG's results are largely Bing's), so it widens recall
                  rather than re-ranking the same pages.

Each engine has the same shape ``fn(query, limit) -> list[dict]`` with dicts
``{title, url, snippet, source}``, so adding an engine is a one-liner and the
merge logic stays engine-agnostic. Results are interleaved round-robin across
sources for diversity and de-duplicated by normalised URL.
"""

import html as _html
import re
import threading
import time
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


def _github_search(query: str, limit: int = 3) -> list:
    """Issues and pull requests across public GitHub, via the keyless search API.

    For a coding agent this often beats a web page: an error string usually
    appears verbatim in an issue, together with the maintainer's answer and the
    version it was fixed in. Sorted by reactions so the thread people actually
    found useful comes first. Unauthenticated search is rate-limited (~10/min),
    which the engine isolation already tolerates — a 403 just yields [].
    """
    data = _get_json("https://api.github.com/search/issues", {
        "q": query, "sort": "reactions", "order": "desc", "per_page": limit,
    })
    out = []
    for item in data.get("items", []):
        state = item.get("state", "")
        comments = item.get("comments", 0)
        body = (item.get("body") or "").strip().replace("\r", "")
        body = re.sub(r"\s+", " ", body)[:180]
        # repository_url looks like https://api.github.com/repos/<owner>/<name>
        repo = (item.get("repository_url") or "").rsplit("/repos/", 1)[-1]
        out.append({
            "title": item.get("title", "") or "(no title)",
            "url": item.get("html_url", ""),
            "snippet": f"[{repo} · {state} · {comments} comments] {body}".strip(),
            "source": "github",
        })
    return out


def _stackoverflow_search(query: str, limit: int = 3) -> list:
    """Programming Q&A via the keyless StackExchange API.

    This API expresses filters as PARAMETERS rather than query operators, so
    `intitle:` is lifted out of the text into the native `title` field instead
    of being searched for literally.
    """
    params = {"order": "desc", "sort": "relevance",
              "site": "stackoverflow", "pagesize": limit}

    title_terms = []

    def _lift_title(m):
        title_terms.append(m.group("val").strip('"'))
        return ""

    remainder = re.sub(r'\bintitle:(?P<val>"[^"]*"|\S+)', _lift_title, query,
                       flags=re.IGNORECASE).strip()
    if title_terms:
        params["title"] = " ".join(title_terms)
    if remainder:
        params["q"] = remainder
    if "title" not in params and "q" not in params:
        return []

    data = _get_json("https://api.stackexchange.com/2.3/search/advanced", params)
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


# Serialises Brave calls so concurrent or back-to-back queries still respect
# the per-second quota.
_BRAVE_LOCK = threading.Lock()
_BRAVE_MIN_INTERVAL = 1.1
_brave_last_call = 0.0


def _brave_pace():
    """Block just long enough to stay under the free tier's rate limit."""
    global _brave_last_call
    with _BRAVE_LOCK:
        wait = _BRAVE_MIN_INTERVAL - (time.monotonic() - _brave_last_call)
        if wait > 0:
            time.sleep(wait)
        _brave_last_call = time.monotonic()


def _brave_search(query: str, limit: int = 5) -> list:
    """General web via the Brave Search API — an INDEPENDENT index.

    This is deliberately additive rather than a replacement for DuckDuckGo.
    DDG's results are largely Bing's, so running Brave alongside it unions two
    genuinely different indexes, and union is what lifts RECALL — the one thing
    the cross-encoder reranker downstream cannot fix (it can only reorder what
    was retrieved). It also removes a fragility: ddgs is a scraper, Brave is a
    contracted API.

    Off unless an API key is configured, so the keyless zero-setup default is
    untouched; when the quota runs out the other engines simply carry the query.
    """
    from config import get_brave_api_key
    key = get_brave_api_key()
    if not key:
        return []

    # The free tier allows about one query per second, and deep research fires
    # several in a row. Without pacing, the burst gets 429s — and because engine
    # failures are isolated, that would SILENTLY drop Brave from the federation
    # exactly during the multi-query runs it helps most. Waiting a fraction of a
    # second is cheaper than losing the results.
    _brave_pace()

    import requests
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": limit},
        headers={"Accept": "application/json", "X-Subscription-Token": key},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    out = []
    for item in (data.get("web", {}) or {}).get("results", []):
        out.append({
            "title": item.get("title", "") or "(no title)",
            "url": item.get("url", ""),
            "snippet": re.sub("<[^>]+>", "", item.get("description", "") or ""),
            "source": "brave",
        })
    # Brave also surfaces forum/discussion threads separately; for debugging
    # questions those are often worth more than an article.
    for item in (data.get("discussions", {}) or {}).get("results", []):
        out.append({
            "title": item.get("title", "") or "(discussion)",
            "url": item.get("url", ""),
            "snippet": re.sub("<[^>]+>", "", item.get("description", "") or ""),
            "source": "brave-discussions",
        })
    return out[:limit]


# Keyless engines, always available — the zero-setup baseline.
DEFAULT_ENGINES = [_ddg_search, _wikipedia_search, _stackoverflow_search, _github_search]


# Display names, kept next to the engines themselves so /doctor and /search
# cannot drift into describing the same federation differently.
ENGINE_LABELS = {
    "_ddg_search": "DuckDuckGo", "_wikipedia_search": "Wikipedia",
    "_stackoverflow_search": "StackOverflow", "_github_search": "GitHub",
    "_brave_search": "Brave",
}


def engine_labels(engines=None) -> list:
    """Readable names of the engines that will run."""
    return [ENGINE_LABELS.get(e.__name__, e.__name__)
            for e in (engines if engines is not None else active_engines())]


def active_engines() -> list:
    """The engine set for this run: the keyless baseline plus any that the user
    has configured (currently Brave). Built per call so enabling a key takes
    effect without a restart."""
    engines = list(DEFAULT_ENGINES)
    try:
        from config import get_brave_api_key
        if get_brave_api_key():
            engines.insert(0, _brave_search)   # independent index goes first
    except Exception:
        pass
    return engines


# Google-style operators the model may write. They are NOT portable: the same
# string is sent to five engines that speak different query languages, and an
# operator the engine does not know is treated as literal text — measured, a
# query carrying `site:` returned 5 results from DuckDuckGo and Brave and ZERO
# from Wikipedia and GitHub, silently removing them from the federation. So each
# engine declares what it understands and the rest is stripped before sending.
#
# Quoted phrases are deliberately absent from this table: every engine here
# supports them, so they always pass through untouched.
_OPERATOR_RE = re.compile(
    r'(?P<op>-?\b(?:site|filetype|ext|inurl|intitle|related|cache)):(?P<val>"[^"]*"|\S+)',
    re.IGNORECASE)
_EXCLUDE_RE = re.compile(r'(?<!\S)-(?!\s)(?!\w+:)[^\s"]+')

_OPERATOR_SUPPORT = {
    # DuckDuckGo passes Google's operator set through almost entirely.
    "_ddg_search": {"site", "filetype", "ext", "inurl", "intitle", "exclude"},
    # Brave supports the common filters; the rarer ones are ignored, not honoured.
    "_brave_search": {"site", "filetype", "ext", "exclude"},
    # MediaWiki has its own language (intitle:, insource:) and no notion of a site.
    "_wikipedia_search": {"intitle", "exclude"},
    # The StackExchange API takes filters as query PARAMETERS; operators inside
    # `q` are matched as text and quietly wreck the search. intitle: is kept
    # because the engine lifts it into the `title` parameter itself.
    "_stackoverflow_search": {"intitle"},
    # GitHub has its own qualifiers (repo:, in:title, language:) — Google's are
    # not valid there, but `-term` exclusion is, and intitle: TRANSLATES (below).
    "_github_search": {"exclude", "intitle"},
}

# Where an operator has a native equivalent, translate instead of stripping.
# The model writes ONE query for the whole federation, so it cannot express a
# single engine's dialect — deriving it here is the only place that can.
# Measured: 'Addressables in:title' returns a correctly-titled top hit where the
# unqualified query returns an unrelated repository.
def _github_intitle(value: str) -> str:
    return f"{value} in:title"


_OPERATOR_TRANSLATION = {
    # engine -> {operator: fn(value) -> replacement text}
    "_github_search": {"intitle": _github_intitle},
    # Wikipedia's intitle: is already native, and StackExchange's engine lifts it
    # into the API's `title` parameter itself (see _stackoverflow_search).
}


def adapt_query(query: str, supported: set, translation: dict = None) -> str:
    """Rewrite a query into what the target engine can actually honour:
    translate an operator when the engine has a native equivalent, keep it when
    the syntax matches, drop it otherwise.

    Dropping beats sending: the engine would search for the operator's literal
    text and return nothing, which reads as "this source had no answer".
    """
    if not query:
        return query
    translation = translation or {}

    def _rewrite(m):
        name = m.group("op").lstrip("-").lower()
        if name in translation:
            return translation[name](m.group("val").strip('"'))
        return m.group(0) if name in supported else ""

    out = _OPERATOR_RE.sub(_rewrite, query)
    if "exclude" not in supported:
        out = _EXCLUDE_RE.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def adapt_query_for_engine(query: str, engine) -> str:
    """adapt_query for a specific engine function. An engine we don't know
    (a test double, a user-added one) gets the query unchanged."""
    name = getattr(engine, "__name__", "")
    if name not in _OPERATOR_SUPPORT:
        return query
    return adapt_query(query, _OPERATOR_SUPPORT[name],
                       _OPERATOR_TRANSLATION.get(name))


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
    engines = engines if engines is not None else active_engines()
    if per_engine is None:
        per_engine = max(3, max_results // max(1, len(engines)) + 2)

    buckets = []
    for eng in engines:
        try:
            # Each engine gets the query in the dialect it actually speaks.
            adapted = adapt_query_for_engine(query, eng)
            if not adapted:
                # The query was nothing BUT operators this engine cannot use.
                buckets.append([])
                continue
            buckets.append(list(eng(adapted, per_engine)) or [])
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
