"""The local server's real context window is asked for before the first request.

The window used to be learnt only from KoboldCPP's HTTP 400 ("request (65919 tokens) exceeds the available
context size (65536 tokens)") - at the end of a long turn, after the work was done.
"""

import pytest
import requests

import config
from src.agent import context_limit
from src.agent.context_limit import probe_server_context


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def fresh_cache():
    context_limit._PROBE_CACHE.clear()
    yield
    context_limit._PROBE_CACHE.clear()


def _serve(monkeypatch, routes):
    asked = []

    def get(url, timeout=None):
        asked.append(url)
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return _Resp(404, {})
    monkeypatch.setattr(requests, "get", get)
    return asked


def test_koboldcpp_reports_its_window(monkeypatch):
    _serve(monkeypatch, {"/api/extra/true_max_context_length": _Resp(200, {"value": 65536})})
    assert probe_server_context("http://127.0.0.1:5001/v1") == 65536


def test_llama_server_props(monkeypatch):
    _serve(monkeypatch, {"/props": _Resp(200, {"default_generation_settings": {"n_ctx": 32768}})})
    assert probe_server_context("http://127.0.0.1:8080/v1") == 32768


def test_a_server_that_is_down_is_asked_once(monkeypatch):
    asked = _serve(monkeypatch, {"/api/extra/true_max_context_length": requests.ConnectionError("refused")})
    assert probe_server_context("http://127.0.0.1:5001/v1", now=1000.0) is None
    assert probe_server_context("http://127.0.0.1:5001/v1", now=1010.0) is None
    assert len(asked) == 1                        # the rest of the endpoints are skipped, the answer cached


def test_nonsense_is_not_a_window(monkeypatch):
    _serve(monkeypatch, {"/api/extra/true_max_context_length": _Resp(200, {"value": "big"}),
                         "/api/v1/config/max_context_length": _Resp(200, {"value": 12}),
                         "/props": _Resp(500, {})})
    assert probe_server_context("http://h:1/v1") is None


def _agent(provider):
    from agent import ArgentAgent
    a = ArgentAgent.__new__(ArgentAgent)
    a.provider = provider
    a.max_context_tokens = config.get_context_window()
    return a


def test_a_smaller_server_window_lowers_the_configured_one(monkeypatch):
    config.set_context_window(131072)
    monkeypatch.setattr(context_limit, "probe_server_context", lambda url, **k: 65536)
    a = _agent("koboldcpp")
    note = a._sync_server_context()
    assert config.get_context_window() == int(65536 * 0.95)
    assert a.max_context_tokens == int(65536 * 0.95)
    assert note and "65536" in note


def test_a_bigger_server_window_changes_nothing(monkeypatch):
    config.set_context_window(24000)
    monkeypatch.setattr(context_limit, "probe_server_context", lambda url, **k: 65536)
    assert _agent("koboldcpp")._sync_server_context() is None
    assert config.get_context_window() == 24000


def test_cloud_providers_are_not_probed(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not probe a cloud provider")
    monkeypatch.setattr(context_limit, "probe_server_context", boom)
    assert _agent("openrouter")._sync_server_context() is None
