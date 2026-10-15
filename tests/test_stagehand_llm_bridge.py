"""Stagehand's fallback model is Argent's own provider, and it has to work.

With no OpenAI / Anthropic key the semantic browser tools (act, extract, observe)
borrow Argent's configured provider through a callback. That callback called
`stream_chat(model_name=...)` (the parameter is `model`) so every call raised and
was reported as an empty completion; it kept only `str` chunks while providers
yield dicts; it ran the blocking stream on the browser's event loop; and
`_get_llm_config` handed Stagehand a bare function instead of a ClientLLM.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

sht = pytest.importorskip("tools.stagehand_tools")
if not sht.STAGEHAND_AVAILABLE:
    pytest.skip("stagehand is not installed", allow_module_level=True)

import providers


class _Provider:
    """Records how it was called; yields what real providers yield: dicts."""

    def __init__(self, chunks, delay=0.0):
        self.calls = []
        self._chunks = chunks
        self._delay = delay

    def stream_chat(self, model, messages, tools=None, context_window=None, temperature=None, **kw):
        self.calls.append({"model": model, "messages": messages, "tools": tools,
                           "temperature": temperature})
        if self._delay:
            time.sleep(self._delay)           # blocking, like a real HTTP stream
        yield from self._chunks


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("config.get_current_model", lambda *a, **k: "test-model")

    def install(provider):
        monkeypatch.setattr(providers, "create_provider", lambda *a, **k: provider)
        return sht._get_llm_config()["model"].generate

    return install


def _params(text="what is on the page?"):
    return SimpleNamespace(messages=[SimpleNamespace(role="user", content=text)])


class TestConfig:
    def test_the_fallback_is_a_client_llm(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert isinstance(sht._get_llm_config()["model"], sht.ClientLLM)

    def test_a_key_in_the_environment_still_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert sht._get_llm_config()["model"] == "openai/gpt-4o"


class TestTheCallback:
    def test_dict_chunks_become_the_completion(self, bridge):
        provider = _Provider([{"content": "Two "}, {"content": "buttons."}, {"thinking": "x"}])
        generate = bridge(provider)
        result = asyncio.run(generate(_params()))
        assert result.content[0].root.text == "Two buttons."

    def test_it_calls_the_provider_with_the_arguments_it_really_has(self, bridge):
        provider = _Provider([{"content": "ok"}])
        generate = bridge(provider)
        asyncio.run(generate(_params("find the login button")))
        call = provider.calls[0]
        assert call["model"] == "test-model"
        assert "find the login button" in call["messages"][0]["content"]
        assert call["tools"] is None
        assert call["temperature"] == 0.0

    def test_a_provider_failure_is_reported_not_raised(self, bridge):
        class Broken:
            def stream_chat(self, *a, **k):
                raise RuntimeError("HTTP 502")

        generate = bridge(Broken())
        result = asyncio.run(generate(_params()))
        assert "Error generating completion" in result.content[0].root.text
        assert "HTTP 502" in result.content[0].root.text

    def test_the_blocking_stream_does_not_freeze_the_event_loop(self, bridge):
        """The loop that runs the callback is the one driving the browser."""
        generate = bridge(_Provider([{"content": "done"}], delay=0.4))
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.02)

        async def main():
            task = asyncio.create_task(ticker())
            result = await generate(_params())
            task.cancel()
            return result

        result = asyncio.run(main())
        assert result.content[0].root.text == "done"
        assert len(ticks) >= 8, f"the loop only got {len(ticks)} turns during a 0.4 s completion"
