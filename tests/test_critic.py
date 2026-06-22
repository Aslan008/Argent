"""The Critic prompt + task builder (the /critic command's core)."""

from src.agent.critic import CRITIC_SYSTEM, build_critique_task, parse_critic_model


class TestCriticPrompt:
    def test_is_adversarial(self):
        assert "GUILTY UNTIL PROVEN INNOCENT" in CRITIC_SYSTEM
        assert "did NOT write" in CRITIC_SYSTEM

    def test_is_bounded(self):
        # caps the number of findings so it can't dump a wall of nitpicks
        assert "AT MOST 5" in CRITIC_SYSTEM

    def test_allows_no_issues(self):
        # the escape hatch that prevents a timid agent / invented problems
        assert "No significant issues found." in CRITIC_SYSTEM

    def test_demands_a_verdict(self):
        assert "VERDICT" in CRITIC_SYSTEM and "RETHINK" in CRITIC_SYSTEM


class TestCritiqueTask:
    def test_includes_goal_and_target(self):
        t = build_critique_task("Step 1: delete prod DB", goal="ship a website", what="plan")
        assert "ship a website" in t
        assert "Step 1: delete prod DB" in t
        assert "did not write it" in t

    def test_defaults_goal_to_na(self):
        assert "n/a" in build_critique_task("some idea")

    def test_strips_and_wraps_target(self):
        t = build_critique_task("  padded idea  ")
        assert "--- BEGIN ---\npadded idea\n--- END ---" in t

    def test_handles_empty_target(self):
        # shouldn't crash on None/empty
        assert isinstance(build_critique_task("", goal=None), str)
        assert isinstance(build_critique_task(None), str)


class TestParseCriticModel:
    def test_provider_prefix_splits(self):
        assert parse_critic_model("openrouter:anthropic/claude-sonnet-4-6") == (
            "openrouter", "anthropic/claude-sonnet-4-6")

    def test_ollama_tag_is_not_a_provider(self):
        # 'qwen2.5' is not a provider — keep the whole tag as the model
        assert parse_critic_model("qwen2.5:3b") == ("", "qwen2.5:3b")

    def test_plain_model(self):
        assert parse_critic_model("gpt-4") == ("", "gpt-4")

    def test_unknown_prefix_kept_whole(self):
        assert parse_critic_model("foo:bar") == ("", "foo:bar")

    def test_empty(self):
        assert parse_critic_model("") == ("", "")
        assert parse_critic_model(None) == ("", "")
