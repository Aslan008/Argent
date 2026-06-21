"""The Critic prompt + task builder (the /critic command's core)."""

from src.agent.critic import CRITIC_SYSTEM, build_critique_task


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
