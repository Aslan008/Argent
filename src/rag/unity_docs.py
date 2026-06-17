"""
Unity documentation cleaning and symbol-aware chunking for RAG ingestion.

Raw Unity HTML is ~60-70% boilerplate (nav, search box, the "Leave feedback"
widget, footer). Naive BeautifulSoup get_text() over the whole page poisons the
embeddings with that noise, so retrieval finds navigation instead of API text.

This module:
- detects Unity doc pages,
- strips the chrome and the recurring feedback widget,
- extracts the fully-qualified API symbol from <h1> (e.g. "Rigidbody.AddForce")
  and prepends it to every chunk so a query like "Rigidbody AddForce" matches,
- splits large pages into bounded chunks.
"""

import re

# Lines from Unity's recurring feedback widget that survive tag removal.
_FEEDBACK_NOISE = (
    "Leave feedback", "Suggest a change", "Success!", "Close",
    "Thank you for helping us improve the quality of Unity Documentation",
    "Although we cannot accept all submissions",
    "Submission failed",
    "For some reason your suggested change could not be submitted",
    "Please <a>try again</a>", "Your name", "Your email",
    "And thank you for taking the time to help us improve",
)

# class/id substrings marking navigation/chrome rather than content.
_NOISE_SELECTORS = (
    "menu", "nav", "sidebar", "search", "breadcrumb", "toc", "feedback",
    "suggest", "footer", "header", "otremove", "cookie", "tooltip",
)


def is_unity_doc(html: str) -> bool:
    """Heuristic: is this an offline Unity documentation HTML page?"""
    head = html[:4000].lower()
    return "docs.unity3d.com" in head or (
        "unity technologies" in head and 'name="author"' in head
    )


def _extract_symbol(soup) -> str:
    h1 = soup.find("h1")
    raw = h1.get_text(" ", strip=True) if h1 else ""
    if not raw:
        title = soup.find("title")
        raw = (title.get_text(strip=True).replace("Unity - Scripting API:", "").strip()
               if title else "")
    if not raw:
        return ""
    collapsed = re.sub(r"\s+", "", raw)
    # API symbols are dotted identifiers (Rigidbody.AddForce) — keep them tight.
    # Manual titles are prose — keep them readable with single spaces.
    if re.fullmatch(r"[A-Za-z_]\w*(\.\w+)+", collapsed):
        return collapsed
    return re.sub(r"\s+", " ", raw)


def clean_unity_html(html: str):
    """Return (symbol, clean_text) for a Unity doc page. Requires bs4."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript", "svg"]):
        tag.decompose()

    # Collect noise elements first, then decompose — decomposing during the scan
    # detaches descendants and breaks the iterator (their .attrs become None).
    symbol = _extract_symbol(soup)  # capture before removal
    to_remove = []
    for el in soup.find_all(True):
        attrs = el.attrs or {}
        cls = " ".join(attrs.get("class") or []).lower()
        eid = str(attrs.get("id") or "").lower()
        if any(n in cls for n in _NOISE_SELECTORS) or any(n in eid for n in _NOISE_SELECTORS):
            to_remove.append(el)
    for el in to_remove:
        try:
            el.decompose()
        except Exception:
            pass
    main = soup.find("div", class_="section") or soup.find("body") or soup
    text = main.get_text(separator="\n", strip=True)

    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or any(s == p or s.startswith(p) for p in _FEEDBACK_NOISE):
            continue
        kept.append(s)
    # Drop a leading line that just repeats the symbol (the H1 echoed in body).
    if kept and re.sub(r"\s+", "", kept[0]) == re.sub(r"\s+", "", symbol):
        kept = kept[1:]
    return symbol, "\n".join(kept)


def chunk_unity_doc(html: str, rel_path: str, max_chars: int = 1800):
    """Return [(chunk_text, metadata)] for a Unity page, each symbol-prefixed."""
    symbol, text = clean_unity_html(html)
    if not text.strip():
        return []
    header = f"Unity API: {symbol}\n" if symbol else f"Unity doc: {rel_path}\n"

    if len(text) <= max_chars:
        parts = [text]
    else:
        parts, cur, cur_len = [], [], 0
        for para in text.split("\n"):
            if cur_len + len(para) > max_chars and cur:
                parts.append("\n".join(cur))
                cur, cur_len = [], 0
            cur.append(para)
            cur_len += len(para) + 1
        if cur:
            parts.append("\n".join(cur))

    chunks = []
    for i, part in enumerate(parts):
        meta = {"file": rel_path, "symbol": symbol, "start_line": i, "context": "Unity doc"}
        chunks.append((header + part, meta))
    return chunks
