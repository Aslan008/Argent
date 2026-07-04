"""Robust page fetching for research.

Upgrades the old requests+BeautifulSoup crutch on three axes:
  * main-content extraction via trafilatura (far better boilerplate removal than
    hand-rolled article/main heuristics), with a graceful BeautifulSoup +
    markdownify fallback so nothing breaks if trafilatura is unavailable or a
    page defeats it;
  * PDF text extraction (pypdf), so a .pdf link yields readable text instead of
    binary garbage;
  * an on-disk cache, so a URL is downloaded and parsed at most once per day.

The network call is isolated in ``_http_get`` so the whole pipeline is testable
without touching the network.
"""

import io
import random
import urllib.parse

from logger import get_logger
from src.research.cache import get_cached, set_cached

log = get_logger("research")

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0",
]


def _http_get(url: str, timeout: int) -> dict:
    """Fetch a URL. Returns {status, content_type, text, content}.

    Isolated so tests can monkeypatch it and drive the extraction logic with
    canned responses — no network in unit tests.
    """
    import requests
    headers = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    return {
        "status": resp.status_code,
        "content_type": (resp.headers.get("Content-Type") or "").lower(),
        "text": resp.text,
        "content": resp.content,
    }


def _extract_pdf(content: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _bs4_extract(html: str, url: str, raw: bool) -> str:
    """Legacy extraction path: main-content heuristics (or full body in raw
    mode) via BeautifulSoup, links absolutised and converted to markdown."""
    from bs4 import BeautifulSoup
    try:
        from markdownify import markdownify
    except ImportError:
        markdownify = None

    soup = BeautifulSoup(html, "html.parser")

    def render(element) -> str:
        if not element:
            return ""
        if markdownify:
            for a in element.find_all("a", href=True):
                a["href"] = urllib.parse.urljoin(url, a["href"])
            return markdownify(str(element), heading_style="ATX").strip()
        return element.get_text(separator="\n", strip=True)

    if raw:
        for element in soup(["script", "style", "noscript"]):
            element.extract()
        return render(soup.body or soup)

    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        element.extract()

    main = soup.find("article") or soup.find("main")
    if not main:
        divs = soup.find_all("div", class_=lambda x: x and ("content" in x.lower() or "article" in x.lower()))
        if divs:
            main = max(divs, key=lambda d: len(d.get_text(strip=True)))

    text = render(main) if main else ""
    body = soup.find("body")
    if body:
        body_text = render(body)
        # If the semantic pick was suspiciously small, prefer the fuller body.
        if len(text) < 300 and len(body_text) > len(text) * 2:
            text = body_text
    elif not text:
        text = render(soup)
    return text


def _extract_html(html: str, url: str, raw: bool) -> tuple[str, str]:
    """(text, source). trafilatura for clean main content, else BeautifulSoup."""
    if not raw:
        try:
            import trafilatura
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=True,
                favor_recall=True, url=url,
            )
            if extracted and len(extracted.strip()) > 200:
                return extracted.strip(), "trafilatura"
        except Exception as e:
            log.debug("trafilatura extraction failed for %s: %s", url, e)
    return _bs4_extract(html, url, raw), "bs4"


def _tidy(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines)


def fetch_page(url: str, timeout: int = 15, raw: bool = False,
               use_cache: bool = True) -> dict:
    """Fetch and extract readable text from a URL.

    Returns a dict: {ok, url, text, source, error}. ``source`` is one of
    cache / trafilatura / bs4 / pdf (or "error"/"empty" when ok is False).
    Never raises — failures come back as ok=False with a human error string.
    """
    if use_cache:
        cached = get_cached(url, raw)
        if cached is not None:
            return {"ok": True, "url": url, "text": cached, "source": "cache", "error": ""}

    try:
        resp = _http_get(url, timeout)
    except Exception as e:
        return {"ok": False, "url": url, "text": "", "source": "error", "error": str(e)}

    status = resp.get("status")
    if status != 200:
        return {"ok": False, "url": url, "text": "", "source": "error", "error": f"HTTP {status}"}

    ctype = resp.get("content_type", "")
    path = url.lower().split("?", 1)[0]
    is_pdf = "application/pdf" in ctype or path.endswith(".pdf")

    if is_pdf:
        try:
            text = _extract_pdf(resp.get("content") or b"")
            source = "pdf"
        except Exception as e:
            return {"ok": False, "url": url, "text": "", "source": "error",
                    "error": f"PDF extraction failed: {e}"}
    else:
        text, source = _extract_html(resp.get("text") or "", url, raw)

    text = _tidy(text)
    if not text:
        return {"ok": False, "url": url, "text": "", "source": "empty",
                "error": "no extractable content"}

    if use_cache:
        set_cached(url, raw, text)
    return {"ok": True, "url": url, "text": text, "source": source, "error": ""}
