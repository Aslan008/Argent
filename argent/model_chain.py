"""Switch to another model when this one refuses to answer.

A free OpenRouter model that has spent its daily counter answers 429 until midnight
UTC; a retired id answers 404 for good. Neither gets better by waiting, and both used
to surface as a red error while other free models sat idle.

This module asks the refusal engine the compression path already trusts
(``src/agent/service_llm``) what kind of failure it was, parks that model for exactly
as long as the failure says, and picks the next one — so the durations are one table,
not two that drift apart.

Deliberately NOT written to config: the person's chosen model stays chosen. A switch
lives in ``agent.model_name`` for the turn and the turns after it, and ``restore()``
puts the configured model back as soon as its cooldown is over.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from argent.logger import get_logger
from src.agent.service_llm import (
    Kind,
    classify,
    cooldown_for,
    cooldown_left,
    live_free_models,
    mark_unavailable,
)

log = get_logger("model_chain")

#: Refusals a different model may well answer. AUTH is absent on purpose: a wrong key
#: belongs to the provider, not to the model, and swapping models would hide it.
SWITCH_KINDS = frozenset({Kind.EXHAUSTED, Kind.RATE_LIMITED, Kind.UNFIT})


def _provider(agent):
    """The provider object behind the agent's provider name, or None."""
    try:
        from argent.providers import create_provider
        return create_provider(getattr(agent, "provider", None))
    except Exception as e:
        log.debug("fallback chain: no provider (%s)", e)
        return None


def _name(provider, agent) -> str:
    """The cooldown key — the same string the compression path uses."""
    return str(getattr(provider, "name", "") or getattr(agent, "provider", "") or "")


def switchable(exc: BaseException) -> bool:
    """Could another model answer this? Asked before waiting out a failure."""
    try:
        return classify(exc).kind in SWITCH_KINDS
    except Exception:
        return False


def candidates(agent, tried: Iterable[str] = ()) -> list:
    """Models to try next: the configured chain, else the provider's live free list.

    Only models that are not cooling down are returned, best first, without repeats.
    """
    from argent.config import get_model_fallback_chain, get_model_fallback_enabled

    if not get_model_fallback_enabled():
        return []
    provider = _provider(agent)
    if provider is None:
        return []

    skipped = set(tried or ())
    skipped.add(str(getattr(agent, "model_name", "") or ""))
    source = get_model_fallback_chain() or live_free_models(provider)
    name = _name(provider, agent)

    fresh = []
    for model in source:
        model = str(model).strip()
        if not model or model in skipped or model in fresh:
            continue
        if cooldown_left(name, model) > 0:
            continue
        fresh.append(model)
    return fresh


def next_model(agent, exc: BaseException, tried: Iterable[str] = ()) -> Optional[Tuple[str, str]]:
    """Park the model that just failed and return ``(model, notice)``, or None.

    None means "do not switch": the failure is not one a model can fix, the feature is
    off, the switch budget is spent, or the whole chain is cooling down.
    """
    from argent.config import get_model_fallback_switch_limit

    info = classify(exc)
    if info.kind not in SWITCH_KINDS:
        return None

    seen = set(tried or ())
    if len(seen) >= get_model_fallback_switch_limit():
        return None

    provider = _provider(agent)
    if provider is None:
        return None

    failed = str(getattr(agent, "model_name", "") or "")
    if failed:
        mark_unavailable(_name(provider, agent), failed, cooldown_for(info), info.detail)
        seen.add(failed)

    fresh = candidates(agent, seen)
    if not fresh:
        return None
    chosen = fresh[0]
    return chosen, _notice(failed, chosen, info)


def restore(agent) -> Optional[str]:
    """Go back to the model the person chose, once it is no longer cooling down."""
    try:
        from argent.config import get_current_model
        # An agent may be pinned to a model of its own (the bot's). Without reading it,
        # every turn would drag that agent back to the terminal's model.
        preferred = str(getattr(agent, "preferred_model", "") or "")
        configured = preferred or str(get_current_model() or "")
    except Exception:
        return None
    if not configured or configured == str(getattr(agent, "model_name", "") or ""):
        return None

    provider = _provider(agent)
    if provider is None:
        return None
    if cooldown_left(_name(provider, agent), configured) > 0:
        return None
    try:
        agent.set_model(configured)
    except Exception as e:
        log.debug("could not restore the configured model: %s", e)
        return None
    return configured


def _notice(failed: str, chosen: str, info) -> str:
    """One line for the chat: what happened, what answers now, and that it comes back."""
    status = f" (HTTP {info.status})" if getattr(info, "status", None) else ""
    return (f"[Argent: {failed or 'модель'} {info.why()}{status} — переключаюсь на {chosen}. "
            f"К прежней вернусь, когда она оживёт.]")
