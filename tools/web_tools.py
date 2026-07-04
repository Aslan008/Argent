from logger import get_logger

log = get_logger("tools")


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web across several keyless sources (DuckDuckGo, Wikipedia,
    StackOverflow) and return merged snippets.

    Federated so one engine rate-limiting no longer stalls the whole search —
    the others still deliver. Only returns short snippets; call read_webpage on
    the most relevant URL to get the actual content.
    """
    from src.research.search import meta_search
    try:
        results = meta_search(query, max_results=max_results)
    except Exception as e:
        log.error(f"Web search failed: {e}")
        return f"Error searching the web: {e}"

    if not results:
        return f"No results found for query: '{query}'"

    lines = [f"Search results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '(no title)')}  [{r.get('source', '?')}]")
        lines.append(f"   URL: {r.get('url', '')}")
        if r.get("snippet"):
            lines.append(f"   Snippet: {r['snippet']}\n")

    lines.append("\n[SYSTEM REMINDER: These are only short snippets. To get the actual answer, you MUST call `read_webpage(url)` on the most relevant URL above! Do not just search again.]")
    return "\n".join(lines)


def read_webpage(url: str, timeout: int = 15, raw_mode: bool = False) -> str:
    """Read and extract meaningful text content from a webpage URL.
    If raw_mode is True, bypasses semantic extraction and returns all visible text.

    Delegates to the research fetch layer (trafilatura main-content extraction,
    PDF support, BeautifulSoup fallback, on-disk cache); this function only
    shapes the tool-facing string and enforces the length cap.
    """
    from src.research.fetch import fetch_page
    res = fetch_page(url, timeout=timeout, raw=raw_mode)
    if not res["ok"]:
        return f"Error reading webpage '{url}': {res.get('error') or 'no content'}"

    text = res["text"]
    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [Content Truncated]"

    return f"Content of {url}:\n\n{text}"
