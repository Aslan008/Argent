"""feature-workflow skill: interview -> spec/prototype -> deviation journal ->
review doc + comprehension quiz, plus the core prompt hook."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from skill_manager import SkillManager

_SKILL = Path(__file__).resolve().parents[1] / "skills" / "feature-workflow.md"


class TestSkillContent:
    def test_frontmatter_valid(self):
        meta, body = SkillManager._parse_frontmatter(_SKILL.read_text(encoding="utf-8"))
        assert meta["name"] == "feature-workflow"
        assert "interview" in meta["description"].lower()
        assert body

    def test_three_phases_present(self):
        body = _SKILL.read_text(encoding="utf-8")
        assert "Phase 1" in body and "Phase 2" in body and "Phase 3" in body

    def test_interview_is_one_question_at_a_time(self):
        body = _SKILL.read_text(encoding="utf-8")
        assert "ONE question at a time" in body
        assert "overrides" in body and "blind-spot-pass" in body   # explicit precedence
        assert "ARCHITECTURE" in body                              # priority of questions

    def test_deviation_journal_shape(self):
        body = _SKILL.read_text(encoding="utf-8")
        for field in ("Planned:", "Found:", "Did instead:", "Why:"):
            assert field in body, f"journal entry field missing: {field}"
        assert "append_to_file" in body
        assert "STOP" in body            # approved decisions are not silently overridden

    def test_review_and_quiz(self):
        body = _SKILL.read_text(encoding="utf-8")
        assert "_review.md" in body
        assert "ask_user_questions" in body
        assert "3–4" in body or "3-4" in body          # quiz size bounded
        assert "request_user_approval" in body          # approval gate before build

    def test_uses_existing_tools_only(self):
        # The workflow must not invent tools that don't exist in Argent.
        body = _SKILL.read_text(encoding="utf-8")
        from tools import get_available_tools
        available = set(get_available_tools())
        import re
        for name in re.findall(r"`(\w+)\(", body):
            assert name in available, f"skill references unknown tool: {name}"


@pytest.fixture
def make_agent(monkeypatch):
    mem = MagicMock()
    mem.data = {}
    monkeypatch.setattr(agent_module, "memory", mem)
    monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
    monkeypatch.setattr(config, "get_model_category_override", lambda: None)

    def _make(category):
        monkeypatch.setattr(agent_module, "get_model_size_category", lambda name: category)
        return ArgentAgent()
    return _make


class TestCorePromptHook:
    def test_capable_models_get_the_workflow(self, make_agent):
        prompt = make_agent("cloud").build_system_prompt()
        assert "feature-workflow" in prompt
        assert "one-question-at-a-time" in prompt

    def test_tiny_models_are_not_burdened(self, make_agent):
        assert "feature-workflow" not in make_agent("tiny").build_system_prompt()
