"""What a sub-agent runs on: model, provider and thinking level, with defaults.

The default matters more than the feature: a helper must never cost more than the
agent that called it unless the model asked for something else on purpose.
"""

import json
import sys
from pathlib import Path

import pytest

import argent.config as config
from src.agent import subagents


@pytest.fixture
def defaults(monkeypatch):
    """No sub-agent defaults configured: everything inherits the main agent."""
    monkeypatch.setattr(config, "get_subagent_model", lambda: "")
    monkeypatch.setattr(config, "get_subagent_provider", lambda: "")
    monkeypatch.setattr(config, "get_subagent_thinking", lambda: "")
    return monkeypatch


def _child_ok(report="done"):
    """A stand-in worker: prints its verdict and finishes."""
    payload = json.dumps({"status": "ok", "report": report, "denied": []})
    return [sys.executable, "-c",
            f"print({subagents.RESULT_MARKER!r} + {payload!r}, flush=True)"]


def recording(captured):
    """An argv builder that reads the spec before the process even exists."""
    def builder(spec_path):
        captured.update(json.loads(Path(spec_path).read_text(encoding="utf-8")))
        return _child_ok()
    return builder


# ── resolving a target ────────────────────────────────────────────────────────

def test_nothing_asked_means_the_main_agent(defaults):
    assert subagents.resolve_target() == ("", "", "")


def test_a_model_alone_keeps_the_main_provider(defaults):
    provider = config.get_provider()
    model, resolved_provider, thinking = subagents.resolve_target("some-model")
    assert model == "some-model"
    assert resolved_provider == provider
    assert thinking == ""


def test_a_provider_alone_uses_that_providers_saved_model(defaults):
    saved = config.get_current_model("ollama")
    model, provider, _ = subagents.resolve_target(provider="ollama")
    assert provider == "ollama"
    assert model == (saved or "")


def test_the_persons_defaults_apply_when_nothing_is_asked(defaults):
    defaults.setattr(config, "get_subagent_model", lambda: "cheap-model")
    defaults.setattr(config, "get_subagent_thinking", lambda: "low")
    model, _provider, thinking = subagents.resolve_target()
    assert model == "cheap-model"
    assert thinking == "low"


def test_an_explicit_choice_beats_the_configured_default(defaults):
    defaults.setattr(config, "get_subagent_model", lambda: "cheap-model")
    model, _provider, _thinking = subagents.resolve_target("better-model")
    assert model == "better-model"


def test_an_unknown_provider_is_refused_before_anything_starts(defaults):
    with pytest.raises(ValueError) as err:
        subagents.resolve_target(provider="not-a-provider")
    assert "unknown provider" in str(err.value)


def test_an_unconfigured_provider_is_refused_when_the_caller_picked_it(defaults, monkeypatch):
    monkeypatch.setattr(subagents, "provider_ready", lambda name: (False, f"{name} has no API key configured"))
    with pytest.raises(ValueError) as err:
        subagents.resolve_target("m", provider="openrouter")
    assert "API key" in str(err.value)


def test_a_configured_provider_is_not_refused_when_it_came_from_config(defaults, monkeypatch):
    # The person's own critic configuration must keep its existing fallback.
    monkeypatch.setattr(subagents, "provider_ready", lambda name: (False, "no key"))
    monkeypatch.setattr(config, "get_subagent_provider", lambda: "openrouter")
    monkeypatch.setattr(config, "get_subagent_model", lambda: "m")
    model, provider, _ = subagents.resolve_target(strict=False)
    assert (model, provider) == ("m", "openrouter")


def test_a_bad_thinking_level_is_refused(defaults):
    with pytest.raises(ValueError):
        subagents.resolve_target(thinking="deep")


def test_the_list_of_targets_marks_what_runs_here(defaults):
    text = subagents.available_targets()
    assert "Main agent:" in text
    assert "thinking:" in text
    assert "ollama:" in text                       # a local server needs no key


# ── the run carries the target ────────────────────────────────────────────────

def test_the_spec_and_the_run_name_the_model_and_the_level(tmp_path):
    spec = {}
    mgr = subagents.SubagentManager(argv_builder=recording(spec))

    run = mgr.spawn("researcher", "read things", isolation="none", cwd=tmp_path,
                    model="cheap-1", provider="ollama", thinking="low")
    assert mgr.wait(run, timeout=60)

    assert spec["model"] == "cheap-1"
    assert spec["provider"] == "ollama"
    assert spec["thinking"] == "low"
    assert (run.model, run.provider, run.thinking) == ("cheap-1", "ollama", "low")


def test_a_bad_level_never_starts_a_process(tmp_path):
    mgr = subagents.SubagentManager(argv_builder=lambda spec: [sys.executable, "-c", "pass"])
    with pytest.raises(ValueError):
        mgr.spawn("coder", "x", cwd=tmp_path, thinking="louder")


def test_the_report_says_what_the_worker_ran_on(tmp_path):
    mgr = subagents.SubagentManager(argv_builder=recording({}))
    run = mgr.spawn("coder", "fix it", isolation="none", cwd=tmp_path,
                    model="cheap-1", provider="ollama", thinking="low")
    mgr.wait(run, timeout=60)

    text = subagents.describe(run)
    assert "ollama:cheap-1" in text
    assert "thinking low" in text


def test_an_unadorned_run_does_not_claim_a_model(tmp_path):
    mgr = subagents.SubagentManager(argv_builder=recording({}))
    run = mgr.spawn("coder", "fix it", isolation="none", cwd=tmp_path)
    mgr.wait(run, timeout=60)

    assert " on " not in subagents.describe(run).splitlines()[0]


# ── what is available ─────────────────────────────────────────────────────────

def test_a_local_provider_is_always_ready():
    assert subagents.provider_ready("ollama")[0] is True
    assert subagents.provider_ready("koboldcpp")[0] is True


def test_an_unknown_provider_says_what_the_known_ones_are():
    ok, why = subagents.provider_ready("gpt5")
    assert ok is False
    assert "ollama" in why and "openrouter" in why
