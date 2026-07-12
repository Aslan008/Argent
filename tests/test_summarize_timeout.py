"""summarize_messages bounds a hung provider via a per-call sync_chat timeout,
and returns None (not a wedged thread) when the call overruns."""

import time

import providers
import src.agent.trimmer as trimmer


class _FakeProvider:
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
        return "a concise summary"


def _msgs():
    return [{"role": "user", "content": "did X"}, {"role": "assistant", "content": "then Y"}]


def test_timeout_is_passed_to_sync_chat(monkeypatch):
    seen = {}
    monkeypatch.setattr(providers, "create_service_provider",
                        lambda: (_FakeProvider(seen), "svc-model"))
    out = trimmer.summarize_messages(_msgs(), "main-model", timeout=30.0)
    assert out == "a concise summary"
    # Slightly under the join timeout so the network gives up first.
    assert seen["timeout"] == 29.0


def test_hung_provider_returns_none_not_a_wedge(monkeypatch):
    seen = {}
    monkeypatch.setattr(providers, "create_service_provider",
                        lambda: (_FakeProvider(seen, hang=True), "svc-model"))
    # A short join timeout; the fake raises within it, so summarize returns None.
    out = trimmer.summarize_messages(_msgs(), "main-model", timeout=1.0)
    assert out is None
