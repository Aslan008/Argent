"""Research fetch layer: on-disk cache + robust page extraction.

The network boundary (_http_get) and PDF extractor are monkeypatched, so these
tests never touch the network and are deterministic.
"""

import pytest

from src.research import cache as cache_mod
from src.research import fetch as fetch_mod
from src.research.fetch import fetch_page


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "_DB_PATH", tmp_path / "web_cache.db")
    return cache_mod


class TestCache:
    def test_roundtrip(self, tmp_cache):
        assert tmp_cache.get_cached("http://x", False) is None
        tmp_cache.set_cached("http://x", False, "hello")
        assert tmp_cache.get_cached("http://x", False) == "hello"

    def test_raw_keyed_separately(self, tmp_cache):
        tmp_cache.set_cached("http://x", False, "clean")
        tmp_cache.set_cached("http://x", True, "raw")
        assert tmp_cache.get_cached("http://x", False) == "clean"
        assert tmp_cache.get_cached("http://x", True) == "raw"

    def test_ttl_expiry(self, tmp_cache):
        tmp_cache.set_cached("http://x", False, "old")
        assert tmp_cache.get_cached("http://x", False, ttl=-1) is None

    def test_empty_not_stored(self, tmp_cache):
        tmp_cache.set_cached("http://x", False, "")
        assert tmp_cache.get_cached("http://x", False) is None

    def test_clear(self, tmp_cache):
        tmp_cache.set_cached("http://x", False, "hi")
        tmp_cache.clear_cache()
        assert tmp_cache.get_cached("http://x", False) is None


def _resp(status=200, ctype="text/html; charset=utf-8", text="", content=b""):
    return {"status": status, "content_type": ctype, "text": text, "content": content}


ARTICLE = """<html><head><title>T</title></head><body>
<nav>menu junk</nav>
<article><h1>Widgets</h1>
<p>A widget is a small gadget. Widgets are made of metal and plastic and are used
in many industries worldwide for a wide variety of practical purposes.</p>
<p>The history of widgets goes back centuries. Early widgets were carved from wood
by skilled artisans working in small village workshops across the region.</p>
</article><footer>copyright junk</footer></body></html>"""


class TestFetchPage:
    def test_html_extracted(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "_http_get", lambda url, t: _resp(text=ARTICLE))
        res = fetch_page("http://x/a", use_cache=False)
        assert res["ok"]
        assert "widget is a small gadget" in res["text"]
        assert res["source"] in ("trafilatura", "bs4")

    def test_http_error(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "_http_get", lambda url, t: _resp(status=404))
        res = fetch_page("http://x", use_cache=False)
        assert not res["ok"] and "404" in res["error"]

    def test_network_exception(self, monkeypatch):
        def boom(url, t):
            raise ConnectionError("dns fail")
        monkeypatch.setattr(fetch_mod, "_http_get", boom)
        res = fetch_page("http://x", use_cache=False)
        assert not res["ok"] and "dns fail" in res["error"] and res["source"] == "error"

    def test_empty_content(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "_http_get", lambda url, t: _resp(text="<html><body></body></html>"))
        res = fetch_page("http://x", use_cache=False)
        assert not res["ok"] and res["source"] == "empty"

    def test_pdf_by_content_type(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "_http_get",
                            lambda url, t: _resp(ctype="application/pdf", content=b"%PDF-1.4"))
        monkeypatch.setattr(fetch_mod, "_extract_pdf", lambda c: "Extracted PDF body text here.")
        res = fetch_page("http://x/doc", use_cache=False)
        assert res["ok"] and res["source"] == "pdf" and "PDF body" in res["text"]

    def test_pdf_by_extension(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "_http_get",
                            lambda url, t: _resp(ctype="application/octet-stream", content=b"%PDF"))
        monkeypatch.setattr(fetch_mod, "_extract_pdf", lambda c: "pdf via extension path")
        res = fetch_page("http://x/file.pdf", use_cache=False)
        assert res["ok"] and res["source"] == "pdf"

    def test_pdf_extract_failure(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "_http_get",
                            lambda url, t: _resp(ctype="application/pdf", content=b"broken"))
        def boom(c):
            raise ValueError("corrupt pdf")
        monkeypatch.setattr(fetch_mod, "_extract_pdf", boom)
        res = fetch_page("http://x/doc.pdf", use_cache=False)
        assert not res["ok"] and "PDF extraction failed" in res["error"]

    def test_cache_hit_skips_network(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "get_cached", lambda url, raw: "cached body")

        def boom(url, t):
            raise AssertionError("network must not be called on a cache hit")
        monkeypatch.setattr(fetch_mod, "_http_get", boom)
        res = fetch_page("http://x", use_cache=True)
        assert res["ok"] and res["source"] == "cache" and res["text"] == "cached body"

    def test_cache_write_on_miss(self, monkeypatch):
        monkeypatch.setattr(fetch_mod, "get_cached", lambda url, raw: None)
        stored = {}
        monkeypatch.setattr(fetch_mod, "set_cached",
                            lambda url, raw, text: stored.update({"url": url, "text": text}))
        monkeypatch.setattr(fetch_mod, "_http_get", lambda url, t: _resp(text=ARTICLE))
        res = fetch_page("http://x", use_cache=True)
        assert res["ok"] and stored.get("url") == "http://x" and "widget" in stored.get("text", "")


class TestReadWebpageIntegration:
    def test_success_format(self, monkeypatch):
        import tools.web_tools as wt
        monkeypatch.setattr("src.research.fetch.fetch_page",
                            lambda url, timeout=15, raw=False:
                            {"ok": True, "url": url, "text": "hello world", "source": "trafilatura", "error": ""})
        out = wt.read_webpage("http://x")
        assert out.startswith("Content of http://x:") and "hello world" in out

    def test_error_format(self, monkeypatch):
        import tools.web_tools as wt
        monkeypatch.setattr("src.research.fetch.fetch_page",
                            lambda url, timeout=15, raw=False:
                            {"ok": False, "url": url, "text": "", "source": "error", "error": "HTTP 500"})
        out = wt.read_webpage("http://x")
        assert out.startswith("Error reading webpage") and "HTTP 500" in out

    def test_truncation(self, monkeypatch):
        import tools.web_tools as wt
        big = "a" * 20000
        monkeypatch.setattr("src.research.fetch.fetch_page",
                            lambda url, timeout=15, raw=False:
                            {"ok": True, "url": url, "text": big, "source": "bs4", "error": ""})
        out = wt.read_webpage("http://x")
        assert "[Content Truncated]" in out and len(out) < 20000 + 200
