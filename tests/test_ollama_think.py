"""/think reaches Ollama.

Thinking mode and depth were stored in the config and shown in the badge ("OFF", "ON/high") but the
Ollama provider never put them into the request, so switching thinking off left a thinking model
thinking. They are sent now as the `think` field, only when the person chose something and only to a
model that lists the "thinking" capability (sent to any other it is an HTTP 400 for the whole turn).
"""

import pytest

import config
from providers import OllamaProvider


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    store = {}
    monkeypatch.setattr(config, "load_config", lambda: store)
    monkeypatch.setattr(config, "save_config", lambda c: store.update(c))
    OllamaProvider._think_capable.clear()
    yield store
    OllamaProvider._think_capable.clear()


class _Show:
    def __init__(self, capabilities):
        self.capabilities = capabilities


class _FakeOllama:
    def __init__(self, capabilities=("completion", "thinking"), show_error=None):
        self._caps = list(capabilities)
        self._show_error = show_error
        self.show_calls = 0
        self.chat_kwargs = None

    def show(self, model):
        self.show_calls += 1
        if self._show_error:
            raise self._show_error
        return _Show(self._caps)

    def chat(self, **kwargs):
        self.chat_kwargs = kwargs
        return iter([{"message": {"content": "ok"}, "done": True}])


def _sent(model="qwen3:8b", **fake):
    p = OllamaProvider.__new__(OllamaProvider)
    p._ollama = _FakeOllama(**fake)
    list(p.stream_chat(model=model, messages=[{"role": "user", "content": "hi"}]))
    return p._ollama


def test_the_default_sends_no_think_and_asks_nothing():
    fake = _sent()
    assert "think" not in fake.chat_kwargs
    assert fake.show_calls == 0


def test_off_is_sent_as_false():
    config.set_thinking_mode("off")
    assert _sent().chat_kwargs["think"] is False


def test_on_is_sent_as_true():
    config.set_thinking_mode("on")
    assert _sent().chat_kwargs["think"] is True


def test_a_level_without_a_depth_aware_model_is_a_plain_true():
    config.set_thinking_level("high")
    assert _sent(model="qwen3:8b").chat_kwargs["think"] is True


@pytest.mark.parametrize("level, expected", [("low", "low"), ("medium", "medium"), ("high", "high"), ("xhigh", "high")])
def test_a_depth_aware_model_gets_the_level(level, expected):
    config.set_thinking_level(level)
    assert _sent(model="gpt-oss:20b").chat_kwargs["think"] == expected


def test_a_model_that_cannot_think_is_left_alone():
    config.set_thinking_mode("off")
    fake = _sent(model="llama3.2:3b", capabilities=("completion",))
    assert "think" not in fake.chat_kwargs


def test_an_unreadable_capability_list_means_nothing_is_sent():
    config.set_thinking_mode("on")
    fake = _sent(show_error=ConnectionError("down"))
    assert "think" not in fake.chat_kwargs


def test_the_capability_is_asked_once_per_model():
    config.set_thinking_mode("off")
    p = OllamaProvider.__new__(OllamaProvider)
    p._ollama = _FakeOllama()
    for _ in range(3):
        list(p.stream_chat(model="qwen3:8b", messages=[]))
    assert p._ollama.show_calls == 1
