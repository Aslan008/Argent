"""The KoboldCPP address can be changed from inside Argent.

`set_koboldcpp_url` existed but nothing called it: the only way to point Argent at another KoboldCPP was to
edit the JSON config by hand. Now `/provider koboldcpp <address>` and `/endpoint <address>` do it, the address
is normalised to the OpenAI-compatible base URL, and a server that is not up yet does not stop it from being
saved.
"""

import pytest

import config
import vision
from config import normalize_koboldcpp_url
from src.cli.commands import config_commands as cc
from src.cli.commands.base import CommandContext


# ── what a person types -> the base URL ───────────────────────────────────────

@pytest.mark.parametrize("typed, expected", [
    ("192.168.1.5", "http://192.168.1.5/v1"),
    ("192.168.1.5:5001", "http://192.168.1.5:5001/v1"),
    ("http://pc:5001", "http://pc:5001/v1"),
    ("http://pc:5001/", "http://pc:5001/v1"),
    ("http://pc:5001/v1", "http://pc:5001/v1"),
    ("http://pc:5001/v1/", "http://pc:5001/v1"),
    ("  'https://kobold.example.com'  ", "https://kobold.example.com/v1"),
    ("http://localhost:5002/api", "http://localhost:5002/api"),     # an explicit path is the person's own
    ("http://[::1]:5001", "http://[::1]:5001/v1"),
])
def test_an_address_becomes_the_openai_base_url(typed, expected):
    assert normalize_koboldcpp_url(typed) == expected


@pytest.mark.parametrize("typed", ["", "   ", "ftp://pc:5001", "http://", "http://pc:notaport", "http://pc:99999"])
def test_a_non_address_is_refused(typed):
    with pytest.raises(ValueError):
        normalize_koboldcpp_url(typed)


# ── the commands ──────────────────────────────────────────────────────────────

class _Agent:
    def set_provider(self, name):
        self.provider = name


@pytest.fixture
def offline(monkeypatch):
    """No network and no model menus: the commands are exercised on the config alone."""
    monkeypatch.setattr(cc, "_server_problem", lambda url, timeout=3.0: None)
    monkeypatch.setattr(cc, "select_model", lambda current=None: current)
    monkeypatch.setattr("command_handler.switch_model", lambda agent, model: None)
    said, errors = [], []
    monkeypatch.setattr(cc, "print_system", said.append)
    monkeypatch.setattr(cc, "print_error", errors.append)
    return said, errors


def _run(handler, text):
    return handler(CommandContext(agent=_Agent(), user_input=text))


def test_provider_with_an_address_saves_it_and_switches(offline):
    said, errors = offline
    _run(cc.handle_provider, "/provider koboldcpp 192.168.1.50:5001")
    assert config.get_koboldcpp_url() == "http://192.168.1.50:5001/v1"
    assert config.get_provider() == "koboldcpp"
    assert not errors
    assert any("http://192.168.1.50:5001/v1" in line for line in said)


def test_a_bad_address_changes_nothing(offline):
    said, errors = offline
    before, provider = config.get_koboldcpp_url(), config.get_provider()
    _run(cc.handle_provider, "/provider koboldcpp ftp://pc")
    assert config.get_koboldcpp_url() == before
    assert config.get_provider() == provider
    assert errors and "http" in errors[0]


def test_an_address_for_another_provider_is_refused(offline):
    said, errors = offline
    provider = config.get_provider()
    _run(cc.handle_provider, "/provider zai http://x")
    assert config.get_provider() == provider
    assert errors and "koboldcpp" in errors[0]


def test_a_server_that_is_down_does_not_stop_the_address_from_being_saved(offline, monkeypatch):
    said, errors = offline
    monkeypatch.setattr(cc, "_server_problem", lambda url, timeout=3.0: "ConnectionError")
    _run(cc.handle_provider, "/provider koboldcpp http://pc:5001")
    assert config.get_koboldcpp_url() == "http://pc:5001/v1"
    assert any("не отвечает" in line for line in said)


def test_endpoint_sets_the_address_while_koboldcpp_is_active(offline):
    said, errors = offline
    config.set_provider("koboldcpp")
    _run(cc.handle_endpoint, "/endpoint http://gpu-box:5001")
    assert config.get_koboldcpp_url() == "http://gpu-box:5001/v1"


def test_endpoint_still_belongs_to_zai_for_other_providers(offline, monkeypatch):
    said, errors = offline
    config.set_provider("zai")
    before = config.get_koboldcpp_url()

    class _Select:
        def ask(self):
            return "General API (api.z.ai/api/paas/v4) - standard pay-per-token"
    monkeypatch.setattr(cc.questionary, "select", lambda *a, **k: _Select())
    _run(cc.handle_endpoint, "/endpoint")
    assert config.get_zai_endpoint() == config.ZAI_ENDPOINT_GENERAL
    assert config.get_koboldcpp_url() == before


def test_a_changed_address_forgets_what_was_learned_about_the_old_server(offline):
    vision._CAPABILITY_CACHE[("koboldcpp", "m")] = False
    vision._CAPABILITY_CACHE[("ollama", "m")] = True
    _run(cc.handle_provider, "/provider koboldcpp http://other:5001")
    assert ("koboldcpp", "m") not in vision._CAPABILITY_CACHE
    assert vision._CAPABILITY_CACHE[("ollama", "m")] is True
    vision._CAPABILITY_CACHE.clear()


def test_the_provider_is_built_on_the_new_address(offline):
    import providers
    _run(cc.handle_provider, "/provider koboldcpp http://lan-box:5001")
    provider = providers._build_provider("koboldcpp")
    assert provider._base_url == "http://lan-box:5001/v1"
