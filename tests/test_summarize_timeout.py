"""summarize_messages bounds a hung provider via a per-call sync_chat timeout,
and returns None (not a wedged thread) when the call overruns."""

import time

import providers
import src.agent.service_llm as service_llm
import src.agent.trimmer as trimmer

SUMMARY = "## Done\nСделано: добавлена проверка контекста и описаны изменения в файле config.py."


class _FakeProvider:
    name = "fake"

    def __init__(self, seen, hang=False):
        self._seen = seen
        self._hang = hang

    def validate_config(self):
        return None

    def sync_chat(self, model, messages, timeout=None, **kw):
        self._seen["timeout"] = timeout
        if self._hang:
            # Simulate a network call that respects the deadline: sleep the
            # given timeout then raise, exactly like a timing-out HTTP client.
            time.sleep(min(timeout or 0.2, 0.2))
            raise TimeoutError("provider timed out")
        return SUMMARY


def _msgs():
    return [{"role": "user", "content": "did X"}, {"role": "assistant", "content": "then Y"}]


def test_timeout_is_passed_to_sync_chat(monkeypatch):
    seen = {}
    monkeypatch.setattr(providers, "create_service_provider",
                        lambda: (_FakeProvider(seen), "svc-model"))
    out = trimmer.summarize_messages(_msgs(), "main-model", timeout=30.0)
    assert out == SUMMARY
    # The caller's timeout bounds each request, so a hung provider cannot hold a turn.
    assert seen["timeout"] == 30.0


def test_hung_provider_returns_none_not_a_wedge(monkeypatch):
    seen = {}
    service_llm.clear_cooldowns()
    monkeypatch.setattr(service_llm, "_pause", lambda seconds, started: False)
    monkeypatch.setattr(providers, "create_service_provider",
                        lambda: (_FakeProvider(seen, hang=True), "svc-model"))
    # A short timeout; the fake raises within it on every try, so summarize returns None.
    out = trimmer.summarize_messages(_msgs(), "main-model", timeout=1.0)
    assert out is None
