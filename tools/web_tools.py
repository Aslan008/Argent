from logger import get_logger

log = get_logger("tools")


def search_web(query: str, max_results: int = 5, engine: str = None) -> str:
    """Search the web across several keyless sources (DuckDuckGo, Wikipedia,
    StackOverflow, GitHub) or an explicitly chosen search engine (Brave, Ollama, DDG, etc.).

    Federated by default so one engine rate-limiting no longer stalls the whole search —
    the others still deliver. Only returns short snippets; call read_webpage on
    the most relevant URL to get the actual content.
    """
    from src.research.search import (
        meta_search, _ddg_search, _wikipedia_search,
        _stackoverflow_search, _github_search, _brave_search, _ollama_search
    )

    engines = None
    if engine and str(engine).strip().lower() not in ("all", "default", "auto", ""):
        eng_norm = str(engine).strip().lower()
        if eng_norm in ("duckduckgo", "ddg"):
            engines = [_ddg_search]
        elif eng_norm == "brave":
            from config import get_brave_api_key
            if not get_brave_api_key():
                return "Error: Brave search is not configured. Set your Brave API key via /search command."
            engines = [_brave_search]
        elif eng_norm == "ollama":
            from config import get_ollama_api_key
            if not get_ollama_api_key():
                return "Error: Ollama web search is not configured. Set your Ollama API key via /search command."
            engines = [_ollama_search]
        elif eng_norm in ("github", "gh"):
            engines = [_github_search]
        elif eng_norm in ("stackoverflow", "stackexchange", "so"):
            engines = [_stackoverflow_search]
        elif eng_norm in ("wikipedia", "wiki"):
            engines = [_wikipedia_search]
        else:
            return (
                f"Error: Unknown search engine '{engine}'. "
                f"Supported engines: 'duckduckgo', 'brave', 'ollama', 'github', 'stackoverflow', 'wikipedia', or 'all'."
            )

    try:
        if engines is not None:
            results = meta_search(query, max_results=max_results, engines=engines)
        else:
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
    If static fetch returns no content (e.g. dynamic JavaScript/React SPA),
    gracefully attempts a headless browser render fallback.
    """
    from src.research.fetch import fetch_page
    res = fetch_page(url, timeout=timeout, raw=raw_mode)
    text = res.get("text", "")

    # If static extraction yields no content or an empty/SPA stub, try headless browser fallback
    if (not res.get("ok") or len(text.strip()) < 100) and not url.lower().endswith(".pdf"):
        try:
            from browser_engine import browser_engine
            session_name = "reader_fallback"
            try:
                browser_engine.run(browser_engine.open_page(url, session=session_name, headed=False, return_state=False))
                browser_engine.run(browser_engine.wait_stable(timeout=5000, session=session_name))
                rendered_md = browser_engine.run(browser_engine.get_markdown(session=session_name))
                if rendered_md and len(rendered_md.strip()) > len(text.strip()):
                    text = rendered_md.strip()
                    res["ok"] = True
            finally:
                browser_engine.run(browser_engine.close_session(session_name))
        except Exception as e:
            log.debug("Headless browser fallback for %s failed: %s", url, e)

    if not res.get("ok") or not text:
        return f"Error reading webpage '{url}': {res.get('error') or 'no content'}"

    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [Content Truncated]"

    return f"Content of {url}:\n\n{text}"
