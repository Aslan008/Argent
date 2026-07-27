"""The browser refuses non-http(s) schemes: file:// would exfiltrate local files."""

import asyncio

import pytest

import browser_engine
from browser_engine import _blocked_url_reason


class TestSchemeGuard:
    @pytest.mark.parametrize("url", [
        "file:///C:/Users/me/secret.txt",
        "file:///etc/passwd",
        "data:text/html,<script>alert(1)</script>",
        "javascript:fetch('http://evil.tld')",
        "view-source:https://example.com",
        "chrome://settings",
        "ftp://files.example.com/x",
    ])
    def test_dangerous_schemes_blocked(self, url):
        reason = _blocked_url_reason(url)
        assert reason is not None and reason.startswith("Error:")

    @pytest.mark.parametrize("url", [
        "https://example.com/page",
        "http://example.com",
        "about:blank",
        "example.com/path",              # bare host — Playwright adds http
        "http://localhost:5173",         # dev server: a first-class use case
        "http://127.0.0.1:8080/preview",
    ])
    def test_allowed(self, url):
        assert _blocked_url_reason(url) is None

    def test_empty_url_rejected(self):
        assert _blocked_url_reason("") is not None


class TestNavigationRefuses:
    def test_open_page_refuses_file_url_without_launching(self):
        engine = browser_engine.BrowserEngine.__new__(browser_engine.BrowserEngine)
        engine._sessions = {}

        async def boom(*a, **k):
            raise AssertionError("must not create a session for a blocked URL")

        engine._create_session = boom
        out = asyncio.run(engine.open_page("file:///C:/secret.txt"))
        assert out.startswith("Error: refusing to open a 'file:' URL")

    def test_navigate_refuses_file_url_without_a_page(self):
        engine = browser_engine.BrowserEngine.__new__(browser_engine.BrowserEngine)

        async def boom(*a, **k):
            raise AssertionError("must not touch the session for a blocked URL")

        engine._get_session = boom
        out = asyncio.run(engine.navigate("file:///etc/passwd"))
        assert "read_file" in out          # points at the right tool instead
