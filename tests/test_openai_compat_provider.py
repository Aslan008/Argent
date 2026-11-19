"""The OpenAI Compatible provider: an address, an optional key, a model name.

KoboldCPP could not be pointed at a server that wants a key — its provider hardcodes
`api_key="dummy_key"` and takes no key at all. LM Studio, vLLM, a bare llama-server and
hosted gateways all speak the OpenAI dialect, so they get their own provider instead of an
unauthenticated KoboldCPP wearing another server's name.

The address is also what decides the model's TIER: on localhost this is a local model with
a local model's prompt and history budget, on any other machine it is a hosted API. Reading
the size out of "llama-3.3-70b-instruct" would hand a hosted model the local treatment.
"""

import pytest

import argent.config as config
import argent.providers as providers
from argent.providers import OpenAICompatProvider, ProviderError
from src.agent import subagents
from src.agent.strategy import CloudStrategy, StandardLocalStrategy, get_model_strategy
from src.cli.commands import config_commands as cc
from src.cli.commands.base import CommandContext


@pytest.fixture(autouse=True)
def _no_keys_from_the_environment(monkeypatch):
    """A developer with OPENAI_API_KEY exported is not a keyless server."""
    for name in ("OPENAI_API_KEY", "OPENAI_COMPAT_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


class _Ask:
    """What questionary returns: an object whose .ask() is the answer."""

    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


# ── the address ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("typed, expected", [
    ("192.168.1.5:8000", "http://192.168.1.5:8000/v1"),
    ("http://localhost:1234", "http://localhost:1234/v1"),
    ("http://pc:8000/v1/", "http://pc:8000/v1"),
    ("https://gateway.example.com/openai", "https://gateway.example.com/openai"),
    # A whole endpoint pasted from a playground: the client appends /chat/completions itself.
    ("https://api.example.com/v1/chat/completions", "https://api.example.com/v1"),
])
def test_an_address_becomes_the_openai_base_url(typed, expected):
    assert config.normalize_openaicompat_url(typed) == expected


@pytest.mark.parametrize("typed", ["", "   ", "ftp://pc:8000", "http://", "http://pc:notaport"])
def test_a_non_address_is_refused(typed):
    with pytest.raises(ValueError):
        config.normalize_openaicompat_url(typed)


# ── the provider ──────────────────────────────────────────────────────────────

def test_the_address_and_key_come_from_the_settings():
    config.set_openaicompat_url("http://localhost:8080/v1")
    config.set_openaicompat_api_key("sk-test")
    provider = providers._build_provider("openaicompat")
    assert isinstance(provider, OpenAICompatProvider)
    assert provider.name == "openaicompat"
    assert provider._base_url == "http://localhost:8080/v1"
    assert provider._api_key == "sk-test"


@pytest.mark.parametrize("typed", ["openaicompat", "openai-compatible", "OpenAI Compatible", "custom"])
def test_the_way_a_person_writes_it_builds_the_same_provider(typed):
    config.set_openaicompat_url("http://pc:8000/v1")
    assert config.normalize_provider_name(typed) == "openaicompat"
    assert isinstance(providers._build_provider(typed), OpenAICompatProvider)


def test_a_keyless_server_is_allowed():
    """The whole point of the provider: a key is optional, not mandatory."""
    config.set_openaicompat_url("http://pc:8000/v1")
    provider = providers._build_provider("openaicompat")
    assert config.get_openaicompat_api_key() is None
    # Some string is still needed: the openai client refuses to build with an empty key.
    assert provider._api_key == "not-needed"


def test_no_address_says_how_to_set_one():
    message = OpenAICompatProvider().validate_config()
    assert message and "/provider openaicompat" in message


def test_a_server_that_wants_a_key_is_not_reported_as_unreachable(monkeypatch):
    provider = OpenAICompatProvider(base_url="http://pc:8000/v1")

    class _Unauthorized(Exception):
        status_code = 401

    def _client():
        raise _Unauthorized("401")

    monkeypatch.setattr(provider, "_get_client", _client)
    message = provider.validate_config()
    assert "ключ" in message and "/key" in message


def test_a_server_that_is_down_is_reported_as_unreachable(monkeypatch):
    provider = OpenAICompatProvider(base_url="http://pc:8000/v1")

    class _Down(Exception):
        status_code = 0

    monkeypatch.setattr(provider, "_get_client", lambda: (_ for _ in ()).throw(_Down("refused")))
    assert "Cannot connect" in provider.validate_config()


def test_the_model_list_is_the_live_one(monkeypatch):
    provider = OpenAICompatProvider(base_url="http://pc:8000/v1")

    class _Model:
        def __init__(self, id):
            self.id = id

    class _List:
        data = [_Model("qwen3-coder-30b"), _Model("mistral-small")]

    class _Models:
        @staticmethod
        def list():
            return _List()

    class _Client:
        models = _Models()

    monkeypatch.setattr(provider, "_get_client", lambda: _Client())
    assert provider.list_models() == ["qwen3-coder-30b", "mistral-small"]


def test_a_server_without_a_catalogue_offers_the_saved_model(monkeypatch):
    """A gateway that hides /models must not leave /model with an empty menu — and the name
    is always typable (argent/ui.py select_model)."""
    config.set_model_for_provider("openaicompat", "gpt-oss-120b")
    provider = OpenAICompatProvider(base_url="http://pc:8000/v1")
    monkeypatch.setattr(provider, "_get_client", lambda: (_ for _ in ()).throw(RuntimeError("404")))
    assert provider.list_models() == ["gpt-oss-120b"]


def test_a_401_at_request_time_points_at_the_key():
    provider = OpenAICompatProvider(base_url="http://pc:8000/v1")
    error = Exception("unauthorized")
    error.status_code = 401
    with pytest.raises(ProviderError) as caught:
        provider._handle_api_status_error(error)
    assert "/key" in str(caught.value)


def test_a_404_at_request_time_points_at_the_address_and_the_model():
    provider = OpenAICompatProvider(base_url="http://pc:8000/v1")
    error = Exception("not found")
    error.status_code = 404
    with pytest.raises(ProviderError) as caught:
        provider._handle_api_status_error(error)
    text = str(caught.value)
    assert "/v1" in text and "/model" in text


# ── the tier ──────────────────────────────────────────────────────────────────

def test_a_hosted_endpoint_is_a_cloud_model():
    config.set_openaicompat_url("http://gpu-farm.example.com:8000/v1")
    strategy = get_model_strategy("llama-3.3-70b-instruct", "openaicompat")
    assert isinstance(strategy, CloudStrategy)


def test_this_machines_endpoint_keeps_the_local_tier():
    config.set_openaicompat_url("http://127.0.0.1:8080/v1")
    strategy = get_model_strategy("llama-3.3-70b-instruct", "openaicompat")
    assert isinstance(strategy, StandardLocalStrategy)


def test_the_tier_can_still_be_set_by_hand():
    """`/model` writes the override; it wins over whatever the address suggests."""
    config.set_openaicompat_url("http://gpu-farm.example.com:8000/v1")
    config.set_model_category_override("large")
    assert isinstance(get_model_strategy("some-small-name", "openaicompat"), StandardLocalStrategy)


def test_a_model_on_this_machine_costs_nothing():
    from argent.model_pricing import estimate_cost
    config.set_openaicompat_url("http://127.0.0.1:8080/v1")
    assert estimate_cost("openaicompat", "gpt-4o", 1000, 1000) is None


def test_a_subagent_may_run_on_it_without_a_key():
    assert subagents.provider_ready("openaicompat") == (True, "")


# ── the commands ──────────────────────────────────────────────────────────────

class _Agent:
    def set_provider(self, name):
        self.provider = name


@pytest.fixture
def offline(monkeypatch):
    """No network and no model menus: the commands are exercised on the config alone."""
    monkeypatch.setattr(cc, "_server_problem", lambda url, timeout=3.0, api_key="": None)
    monkeypatch.setattr(cc, "select_model", lambda current=None: current)
    monkeypatch.setattr("argent.command_handler.switch_model", lambda agent, model: None)
    said, errors = [], []
    monkeypatch.setattr(cc, "print_system", said.append)
    monkeypatch.setattr(cc, "print_error", errors.append)
    return said, errors


def _run(handler, text):
    return handler(CommandContext(agent=_Agent(), user_input=text))


def test_provider_with_an_address_saves_it_and_switches(offline, monkeypatch):
    said, errors = offline
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask(""))
    _run(cc.handle_provider, "/provider openaicompat 192.168.1.50:8000")
    assert config.get_openaicompat_url() == "http://192.168.1.50:8000/v1"
    assert config.get_provider() == "openaicompat"
    assert not errors
    assert any("http://192.168.1.50:8000/v1" in line for line in said)


def test_the_name_alone_asks_for_the_address_then_the_key(offline, monkeypatch):
    said, errors = offline
    monkeypatch.setattr(cc.questionary, "text", lambda *a, **k: _Ask("http://gpu-box:8000"))
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask("sk-secret"))
    _run(cc.handle_provider, "/provider openaicompat")
    assert config.get_openaicompat_url() == "http://gpu-box:8000/v1"
    assert config.get_openaicompat_api_key() == "sk-secret"
    assert config.get_provider() == "openaicompat"


def test_an_empty_key_answer_leaves_the_server_keyless(offline, monkeypatch):
    said, errors = offline
    monkeypatch.setattr(cc.questionary, "text", lambda *a, **k: _Ask("http://pc:8000"))
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask(""))
    _run(cc.handle_provider, "/provider openaicompat")
    assert config.get_openaicompat_api_key() is None


def test_a_bad_address_changes_nothing(offline, monkeypatch):
    said, errors = offline
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask(""))
    provider_before = config.get_provider()
    _run(cc.handle_provider, "/provider openaicompat ftp://pc")
    assert config.get_openaicompat_url() == ""
    assert config.get_provider() == provider_before
    assert errors and "http" in errors[0]


def test_the_probe_carries_the_key_and_a_401_is_not_a_dead_server(offline, monkeypatch):
    """A server that wants a key answers 401 to a bare probe: that is the address WORKING."""
    said, errors = offline
    monkeypatch.setattr(cc, "_server_problem", lambda url, timeout=3.0, api_key="": "HTTP 401")
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask(""))
    _run(cc.handle_provider, "/provider openaicompat http://pc:8000")
    assert config.get_openaicompat_url() == "http://pc:8000/v1"
    assert any("ключ" in line for line in said)
    assert not any("не отвечает" in line for line in said)


def test_the_key_command_saves_and_clears_it(offline, monkeypatch):
    said, errors = offline
    config.set_provider("openaicompat")
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask("sk-from-key-command"))
    _run(cc.handle_key, "/key")
    assert config.get_openaicompat_api_key() == "sk-from-key-command"
    _run(cc.handle_key, "/key off")
    assert config.get_openaicompat_api_key() is None


def test_endpoint_sets_the_address_while_openaicompat_is_active(offline):
    said, errors = offline
    config.set_provider("openaicompat")
    _run(cc.handle_endpoint, "/endpoint http://lan-box:8000")
    assert config.get_openaicompat_url() == "http://lan-box:8000/v1"


def test_a_changed_address_forgets_what_was_learned_about_the_old_server(offline, monkeypatch):
    import argent.vision as vision
    monkeypatch.setattr(cc.questionary, "password", lambda *a, **k: _Ask(""))
    vision._CAPABILITY_CACHE[("openaicompat", "m")] = False
    _run(cc.handle_provider, "/provider openaicompat http://other:8000")
    assert ("openaicompat", "m") not in vision._CAPABILITY_CACHE
    vision._CAPABILITY_CACHE.clear()


# ── over the wire ─────────────────────────────────────────────────────────────

def test_the_whole_path_against_a_real_server():
    """A real HTTP server, a real Bearer check, a real streamed answer.

    The tests above prove the wiring; this one proves it over the wire: the key reaches the
    server, a 401 reads as a key problem rather than a dead server, and a token stream from
    an OpenAI-compatible endpoint comes back.
    """
    import http.server
    import json
    import socket
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self):
            return self.headers.get("Authorization") == "Bearer sk-live"

        def do_GET(self):
            if self.path.rstrip("/") != "/v1/models":
                self._json(404, {"error": {"message": "not found"}})
                return
            if not self._authorized():
                self._json(401, {"error": {"message": "missing key"}})
                return
            self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            if not self._authorized():
                self._json(401, {"error": {"message": "missing key"}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunk = {"id": "c1", "object": "chat.completion.chunk", "created": 0,
                     "model": "fake-model",
                     "choices": [{"index": 0, "delta": {"content": "привет"},
                                   "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config.set_openaicompat_url(f"http://127.0.0.1:{port}/v1")

        # No key: the server answers 401, and that must read as "wants a key", not "unreachable".
        keyless = providers._build_provider("openaicompat")
        message = keyless.validate_config() or ""
        assert "ключ" in message and "/key" in message

        config.set_openaicompat_api_key("sk-live")
        keyed = providers._build_provider("openaicompat")
        assert keyed.validate_config() is None
        assert keyed.list_models() == ["fake-model"]

        answer = "".join(
            part.get("content", "")
            for part in keyed.stream_chat(
                model="fake-model", messages=[{"role": "user", "content": "привет"}])
        )
        assert answer == "привет"
    finally:
        config.set_openaicompat_api_key("")
        server.shutdown()
        server.server_close()
