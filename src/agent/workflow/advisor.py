"""The advisor: a stronger, separate model consulted at the decision points of a workflow.

Why a separate one. The workers and the critic are the same small local model as the main agent — that is the
point of running locally. Such a model is most likely to be wrong exactly where being wrong is expensive: whether
the plan is sound, why a step keeps failing, whether the result is really finished. A second opinion from a
stronger model, spent only at those three moments, is the cheapest quality there is.

It is configurable on purpose and by hand — a cloud model or another local one, exactly as the critic is:

    /advisor model <provider>:<model>      /advisor clear      /advisor consent on|off

When it is a cloud model and the main model is local, the plan and the patches leave the PC. That needs the
person's yes (once; `consent`), and everything sent goes through the secret redactor first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from logger import get_logger
from src.agent.workflow import roles

log = get_logger("advisor")

MAX_BODY_CHARS = 24000


class AdvisorError(RuntimeError):
    pass


@dataclass
class Target:
    provider: str
    model: str
    configured: bool           # False: nothing was set, the main model stands in
    cloud: bool                # runs on somebody else's machine
    leaves_pc: bool            # cloud while the main model is local: the person's code is sent out

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class Advice:
    kind: str
    verdict: str
    text: str
    target: str
    fallback: bool = False     # the configured advisor failed and the main model answered


def resolve(main_provider: str, main_model: str) -> Target:
    import config
    from src.agent.local_backend import classify

    model = (config.get_advisor_model() or "").strip()
    provider = (config.get_advisor_provider() or "").strip()
    configured = bool(model)
    if not configured:
        provider, model = main_provider, main_model
    elif not provider:
        provider = main_provider
    cloud = not classify(provider, model).local
    main_local = classify(main_provider, main_model).local
    return Target(provider=provider, model=model, configured=configured, cloud=cloud,
                  leaves_pc=cloud and main_local)


def _ask(target: Target, messages: list, timeout: float) -> str:
    from providers import create_provider
    from src.agent import governor
    try:
        with governor.priority(governor.P_ADVISOR):
            return create_provider(target.provider).sync_chat(
                target.model, messages, temperature=0.2, timeout=timeout) or ""
    except Exception as e:
        raise AdvisorError(f"{target.label}: {e}") from e


def consult(kind: str, goal: str, body: str, target: Target, fallback: Optional[Target] = None,
            timeout: float = 240.0) -> Advice:
    """Ask the advisor. If the configured one fails and a fallback is given, that one answers (and says so)."""
    from src.agent.secret_redactor import redact

    clean_goal = redact(goal)
    clean = redact(body)
    if len(clean) > MAX_BODY_CHARS:
        half = MAX_BODY_CHARS // 2
        clean = clean[:half] + "\n[…]\n" + clean[-half:]
    messages = roles.advisor_messages(kind, clean_goal, clean)
    try:
        text = _ask(target, messages, timeout)
        used, was_fallback = target, False
    except AdvisorError as first:
        if fallback is None or fallback.label == target.label:
            raise
        log.warning("advisor %s failed (%s); asking %s instead", target.label, first, fallback.label)
        text = _ask(fallback, messages, timeout)
        used, was_fallback = fallback, True
    verdict = roles.parse_advisor_verdict(text)
    if kind == "stuck":
        shown = roles.advisor_guidance(text)
    else:
        shown = text.strip()
    return Advice(kind=kind, verdict=verdict, text=shown, target=used.label, fallback=was_fallback)
