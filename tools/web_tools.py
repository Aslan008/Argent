import time
from ddgs import DDGS
from logger import get_logger

log = get_logger("tools")


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo with retry logic for rate limits."""
    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return f"No results found for query: '{query}'"

            formatted_results = [f"Search results for: '{query}'\n"]
            for i, res in enumerate(results, 1):
                formatted_results.append(f"{i}. {res.get('title', 'No Title')}")
                formatted_results.append(f"   URL: {res.get('href', 'No URL')}")
                formatted_results.append(f"   Snippet: {res.get('body', 'No Snippet')}\n")

            formatted_results.append("\n[SYSTEM REMINDER: These are only short snippets. To get the actual answer, you MUST call `read_webpage(url)` on the most relevant URL above! Do not just search again.]")
            return "\n".join(formatted_results)

        except Exception as e:
            error_msg = str(e).lower()
            if "ratelimit" in error_msg or "202" in error_msg or attempt < max_retries - 1:
                log.warning(f"DDGS search failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {base_delay}s...")
                time.sleep(base_delay)
                base_delay *= 2  # Exponential backoff
            else:
                log.error(f"Error searching the web after {max_retries} attempts: {e}")
                return f"Error searching the web: {e}"

    return f"Error: Could not fetch results for '{query}' due to rate limits."


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
