"""Fetcher SSRF guard: only http(s) URLs resolving to public IPs are fetched,
and redirects are re-validated hop by hop."""

import pytest

import src.research.fetch as fetch
from src.research.fetch import UnsafeURLError, _guard_url, _ip_is_public, fetch_page


class TestIpClassification:
    @pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"])
    def test_public(self, ip):
        assert _ip_is_public(ip) is True

    @pytest.mark.parametrize("ip", [
        "127.0.0.1", "::1",           # loopback
        "10.0.0.5", "192.168.1.1", "172.16.0.1",   # private
        "169.254.169.254",            # cloud metadata (link-local)
        "0.0.0.0",                    # unspecified
    ])
    def test_blocked(self, ip):
        assert _ip_is_public(ip) is False


class TestGuardUrl:
    def test_rejects_non_http_scheme(self):
        with pytest.raises(UnsafeURLError):
            _guard_url("file:///etc/passwd")
        with pytest.raises(UnsafeURLError):
            _guard_url("gopher://x/")

    def test_rejects_metadata_ip(self, monkeypatch):
        monkeypatch.setattr(fetch.socket, "getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("169.254.169.254", 80))])
        with pytest.raises(UnsafeURLError):
            _guard_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_localhost(self, monkeypatch):
        monkeypatch.setattr(fetch.socket, "getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 8756))])
        with pytest.raises(UnsafeURLError):
            _guard_url("http://localhost:8756/ws")

    def test_allows_public_host(self, monkeypatch):
        monkeypatch.setattr(fetch.socket, "getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
        _guard_url("https://example.com/page")     # no raise


class TestFetchPageRejectsUnsafe:
    def test_fetch_page_returns_error_for_internal_url(self, monkeypatch):
        monkeypatch.setattr(fetch, "get_cached", lambda *a, **k: None)
        # Real _http_get runs; only DNS is stubbed to a private IP.
        monkeypatch.setattr(fetch.socket, "getaddrinfo",
                            lambda *a, **k: [(2, 1, 6, "", ("192.168.0.10", 80))])
        out = fetch_page("http://192.168.0.10/admin", use_cache=False)
        assert out["ok"] is False
        assert "non-public" in out["error"] or "192.168" in out["error"]


class TestRedirectRevalidation:
    def test_redirect_into_private_space_is_blocked(self, monkeypatch):
        # First hop public, redirects to the metadata endpoint — must be caught.
        hosts = {"evil.example": "93.184.216.34", "169.254.169.254": "169.254.169.254"}
        monkeypatch.setattr(fetch.socket, "getaddrinfo",
                            lambda host, *a, **k: [(2, 1, 6, "", (hosts[host], 80))])

        class _Resp:
            def __init__(self, redirect):
                self.is_redirect = redirect
                self.headers = {"Location": "http://169.254.169.254/latest/"} if redirect else {}
                self.status_code = 302 if redirect else 200
                self.text = "ok"
                self.content = b"ok"

        calls = {"n": 0}

        def fake_get(url, **kw):
            calls["n"] += 1
            return _Resp(redirect=(calls["n"] == 1))

        import requests
        monkeypatch.setattr(requests, "get", fake_get)
        with pytest.raises(UnsafeURLError):
            fetch._http_get("http://evil.example/start", timeout=5)
