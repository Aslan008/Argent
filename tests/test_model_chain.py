"""The model chain: when the chosen model refuses, another one answers.

What matters here is the decision, not the HTTP call — which failures are worth
switching for, which model comes next, and that the person's own setting is never
silently rewritten. The refusal classification is not re-implemented: these tests
go through the same `service_llm.classify` the compression path uses.
"""

import pytest

from argent import model_chain
from argent.providers import ProviderError
from src.agent import service_llm

FREE_A, FREE_B, FREE_C = "free-a:free", "free-b:free", "free-c:free"


class FakeAgent:
    """Just enough agent for the chain: a model, a provider name, and set_model."""

    def __init__(self, model=FREE_A, provider="openrouter"):
        self.model_name = model
        self.provider = provider
        self.switches = []

    def set_model(self, name):
        self.switches.append(name)
        self.model_name = name


def _key(agent) -> str:
    """The cooldown key the chain itself uses — not a second guess at it."""
    return model_chain._name(model_chain._provider(agent), agent)


@pytest.fixture(autouse=True)
def chain_config(monkeypatch):
    service_llm.clear_cooldowns()
    monkeypatch.setattr("argent.config.get_model_fallback_enabled", lambda: True)
    monkeypatch.setattr("argent.config.get_model_fallback_switch_limit", lambda: 5)
    monkeypatch.setattr("argent.config.get_model_fallback_chain",
                        lambda: [FREE_B, FREE_C])
    monkeypatch.setattr("argent.config.get_current_model", lambda: FREE_A)
    yield
    service_llm.clear_cooldowns()


def _quota_error() -> ProviderError:
    return ProviderError("OpenRouter: Rate limited (HTTP 429, free-models-per-day)")


def _retired_error() -> ProviderError:
    return ProviderError("OpenRouter: model not found (HTTP 404)")


def test_a_spent_free_quota_moves_to_the_next_model():
    agent = FakeAgent()
    rotated = model_chain.next_model(agent, _quota_error(), set())

    assert rotated is not None, "a per-day quota is exactly what the chain exists for"
    model, notice = rotated
    assert model == FREE_B
    assert FREE_B in notice


def test_the_failed_model_is_parked_for_its_own_period():
    agent = FakeAgent()
    model_chain.next_model(agent, _quota_error(), set())

    assert service_llm.cooldown_left(_key(agent), FREE_A) > 0, (
        "without the park the chain would hand the same exhausted model back on the next try")


def test_a_retired_model_is_also_switched_away_from():
    rotated = model_chain.next_model(FakeAgent(), _retired_error(), set())
    assert rotated is not None and rotated[0] == FREE_B


def test_a_bad_api_key_does_not_switch_models():
    """A wrong key belongs to the provider: swapping models would hide the real problem."""
    assert model_chain.next_model(FakeAgent(), ProviderError("invalid api key (HTTP 401)"), set()) is None


def test_a_cooling_model_is_skipped():
    agent = FakeAgent()
    service_llm.mark_unavailable(_key(agent), FREE_B, 600, "test")

    rotated = model_chain.next_model(agent, _quota_error(), set())
    assert rotated is not None and rotated[0] == FREE_C


def test_the_budget_stops_an_endless_chain():
    agent = FakeAgent()
    spent = {FREE_A, FREE_B, FREE_C, "d:free", "e:free"}
    assert model_chain.next_model(agent, _quota_error(), spent) is None


def test_an_empty_chain_falls_back_to_the_providers_live_free_list(monkeypatch):
    monkeypatch.setattr("argent.config.get_model_fallback_chain", lambda: [])
    monkeypatch.setattr(model_chain, "live_free_models", lambda provider: ["listed:free"])

    rotated = model_chain.next_model(FakeAgent(), _quota_error(), set())
    assert rotated is not None and rotated[0] == "listed:free"


def test_disabled_means_no_switching(monkeypatch):
    monkeypatch.setattr("argent.config.get_model_fallback_enabled", lambda: False)
    assert model_chain.next_model(FakeAgent(), _quota_error(), set()) is None


def test_restore_goes_back_to_the_model_the_person_chose():
    agent = FakeAgent(model=FREE_B)
    assert model_chain.restore(agent) == FREE_A
    assert agent.model_name == FREE_A

def test_restore_waits_while_the_chosen_model_is_still_cooling():
    agent = FakeAgent(model=FREE_B)
    service_llm.mark_unavailable(_key(agent), FREE_A, 600, "test")

    assert model_chain.restore(agent) is None
    assert agent.model_name == FREE_B


def test_the_persons_setting_is_never_written_to_config(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("config must keep the model the person chose")

    monkeypatch.setattr("argent.config.set_current_model", boom)
    agent = FakeAgent(model=FREE_B)
    model_chain.restore(agent)
    assert agent.model_name == FREE_A


def test_the_agents_own_switch_method_counts_its_switches():
    """`ArgentAgent._rotate_model` bound to a fake: the bookkeeping the loop relies on."""
    from argent.agent import ArgentAgent

    agent = FakeAgent()
    tried = set()
    notice = ArgentAgent._rotate_model(agent, _quota_error(), tried)

    assert notice and FREE_B in notice
    assert agent.model_name == FREE_B
    assert tried == {FREE_A, FREE_B}, "both the abandoned and the new model must be remembered"


def test_a_pinned_bot_model_wins_over_the_global_one():
    """The bot has a model of its own: `restore()` brings THIS agent back to it."""
    agent = FakeAgent(model=FREE_C)
    agent.preferred_model = FREE_B

    assert model_chain.restore(agent) == FREE_B
    assert agent.model_name == FREE_B


def test_a_pinned_model_also_waits_out_its_cooldown():
    agent = FakeAgent(model=FREE_C)
    agent.preferred_model = FREE_B
    service_llm.mark_unavailable(_key(agent), FREE_B, 600, "test")

    assert model_chain.restore(agent) is None
    assert agent.model_name == FREE_C
