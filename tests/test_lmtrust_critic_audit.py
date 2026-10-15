"""LMTrust Blind-Spot & Contract Audit for the Critic System in Argent.

Covers layers L1-L9 and directions D1-D10:
- L1 x D7 (Contract / Interface Violations & Verdict Parsing)
- L2 x D1 (Boundary / Huge Diffs & Sanitization)
- L4 x D5 (Token Duplication Prevention & Resource Efficiency)
- L4b x D4 (Fallback Handling & User Notification)
- L5 x D2 (Pre-Commit Gating & Confirmation Flow)
- L6 x D7 (Tool Schema & Subagent Role Alignment)
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.critic import (
    CRITIC_SYSTEM, build_critique_task, parse_critic_model,
    parse_critic_verdict, CriticVerdict, sanitize_and_truncate_diff,
)
from agent import ArgentAgent, ArgentSubAgent, run_plan_critique
from tools.misc_tools import run_subagent
import config


class TestCriticContractAndLogicAudit:
    """Mode B & C: Logic Audit, Contract Hardening & Invariant Verification for Critic."""

    def test_l4_d5_subagent_task_no_token_duplication(self):
        """L4 x D5: ArgentSubAgent must NOT duplicate the entire task/target payload
        in both messages[0] (system prompt) and messages[1] (user prompt).
        """
        payload = "UNIQUE_CRITIQUE_TARGET_ABC_12345"
        sub = ArgentSubAgent("Critic", payload)
        
        # Advance generator by one step to enter _run_turn where messages are appended
        turn_gen = sub.process_user_input(f"Start task: {sub.task}")
        try:
            next(turn_gen)
        except Exception:
            pass
        
        m0_content = sub.messages[0]["content"]
        m1_content = sub.messages[1]["content"]
        
        in_m0 = payload in m0_content
        in_m1 = payload in m1_content
        
        # In the fixed architecture, task payload is NOT duplicated in system prompt:
        assert not in_m0, "System prompt should not duplicate the specific task payload"
        assert in_m1, "User prompt must carry the task instructions"

    def test_l1_d7_critic_verdict_parser(self):
        """L1 x D7: Critic prompt specifies 'VERDICT: SOLID | FIXABLE | RETHINK'.
        parse_critic_verdict must reliably extract structured enum.
        """
        assert parse_critic_verdict("Everything looks good.\nVERDICT: SOLID") == CriticVerdict.SOLID
        assert parse_critic_verdict("Fix minor issues.\n**VERDICT: FIXABLE**") == CriticVerdict.FIXABLE
        assert parse_critic_verdict("Fatal flaws.\nVerdict: Rethink.") == CriticVerdict.RETHINK
        assert parse_critic_verdict("No verdict given.") == CriticVerdict.UNKNOWN
        assert parse_critic_verdict("") == CriticVerdict.UNKNOWN

    def test_l2_d1_diff_sanitization_and_truncation(self):
        """L2 x D1 / D5: Large git diffs (>24k chars) or noisy lockfiles must be
        filtered/truncated before feeding to the Critic subagent.
        """
        # Test noise filtering
        noisy_diff = (
            "diff --git a/package-lock.json b/package-lock.json\n"
            "+huge lockfile content\n"
            "diff --git a/app.py b/app.py\n"
            "+def important(): pass\n"
        )
        clean = sanitize_and_truncate_diff(noisy_diff)
        assert "package-lock.json" not in clean
        assert "def important(): pass" in clean

        # Test truncation of giant diffs
        huge_diff = "diff --git a/main.py b/main.py\n" + ("+line\n" * 10_000)
        bounded = sanitize_and_truncate_diff(huge_diff, max_chars=1000)
        assert len(bounded) <= 1200
        assert "Diff truncated" in bounded

    def test_l2_d1_build_critique_task_bounded(self):
        """L2 x D1: build_critique_task bounds enormous input targets."""
        giant_target = "A" * 100_000
        task = build_critique_task(giant_target, max_chars=5000)
        assert len(task) < 6000
        assert "Content truncated" in task

    def test_l4b_d4_fallback_notifies_user(self, monkeypatch):
        """L4b x D4: When custom critic model fails, fallback to current model
        notifies the user in the console, avoiding a silent degradation.
        """
        monkeypatch.setattr(config, "get_critic_model", lambda: "anthropic/claude-3.5-sonnet")
        monkeypatch.setattr(config, "get_critic_provider", lambda: "openrouter")
        
        printed = []
        import ui
        monkeypatch.setattr(ui.console, "print", lambda msg: printed.append(str(msg)))
        
        calls = []
        def fake_init(self, role, task, tools_override=None, model_override=None, provider_override=None):
            calls.append((role, model_override, provider_override))
            if model_override:
                raise RuntimeError("OpenRouter connection timeout")
            self.execute = lambda: "Fallback critic output"

        with patch.object(ArgentSubAgent, "__init__", fake_init):
            res = run_plan_critique("some plan")
            assert res == "Fallback critic output"
            assert len(calls) == 2
            # Verify user was alerted on console
            assert any("failed" in p.lower() or "falling back" in p.lower() for p in printed)

    def test_l6_d7_tools_schema_has_critic_role(self):
        """L6 x D7: tools/schemas.py run_subagent description lists 'Critic'
        allowing the model to invoke Critic sub-agents.
        """
        from tools.schemas import TOOL_SCHEMAS
        subagent_tool = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "run_subagent")
        role_desc = subagent_tool["function"]["parameters"]["properties"]["role"]["description"]
        assert "Critic" in role_desc

    def test_l6_d7_run_subagent_critic_config_propagation(self, monkeypatch):
        """L6 x D7: Calling run_subagent with role='Critic' properly picks up
        critic_model and critic_provider overrides and defaults tools to [].
        """
        monkeypatch.setattr(config, "get_critic_model", lambda: "custom-critic-model")
        monkeypatch.setattr(config, "get_critic_provider", lambda: "openrouter")

        sub_args = {}
        def fake_init(self, role, task, tools_override=None, model_override=None, provider_override=None):
            sub_args["role"] = role
            sub_args["tools"] = tools_override
            sub_args["model"] = model_override
            sub_args["provider"] = provider_override
            self.execute = lambda: "Critic review completed"

        with patch.object(ArgentSubAgent, "__init__", fake_init):
            res = run_subagent(role="Critic", task="Audit this implementation plan")
            assert res == "Critic review completed"
            assert sub_args["role"] == "Critic"
            assert sub_args["model"] == "custom-critic-model"
            assert sub_args["provider"] == "openrouter"
            assert sub_args["tools"] == []

    def test_l1_d7_cli_prompt_description(self):
        """L1 x D7: /critic on in COMMAND_HELP accurately describes pre-commit checking."""
        from src.cli.cli_prompt import COMMAND_HELP
        critic_on_help = COMMAND_HELP.get("/critic on", "")
        assert "commit" in critic_on_help.lower()
