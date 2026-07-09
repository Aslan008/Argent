"""Blind Spot Pass: the general skill, its Unity integration, and the core
system-prompt hook that points models at the method."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import config
from agent import ArgentAgent
from skill_manager import SkillManager

_ROOT = Path(__file__).resolve().parents[1]
_GENERAL = _ROOT / "skills" / "blind-spot-pass.md"
_UNITY = _ROOT / "skills" / "unity-dev" / "SKILL.md"


class TestGeneralSkill:
    def test_frontmatter_valid(self):
        meta, body = SkillManager._parse_frontmatter(_GENERAL.read_text(encoding="utf-8"))
        assert meta["name"] == "blind-spot-pass"
        assert "unknowns" in meta["description"].lower()
        assert body  # non-empty instructions

    def test_covers_all_four_quadrants(self):
        body = _GENERAL.read_text(encoding="utf-8")
        for quadrant in ("Known knowns", "Known unknowns", "Unknown knowns", "Unknown unknowns"):
            assert quadrant in body, f"missing quadrant: {quadrant}"

    def test_actionable_wiring(self):
        body = _GENERAL.read_text(encoding="utf-8")
        assert "ask_user_questions" in body      # structured questions tool
        assert "set_goal" in body                 # pin the refined contract
        assert "Max 3" in body or "max 3" in body # economy rule
        assert "ASSUMPTIONS" in body              # report shape
        assert "FACTS" in body and "RISKS" in body and "QUESTIONS" in body


class TestUnityIntegration:
    def test_unity_skill_references_method(self):
        body = _UNITY.read_text(encoding="utf-8")
        assert "blind-spot-pass" in body
        assert "Unknown unknowns" in body
        assert "unity_context.py" in body         # unknown-unknowns opener
        assert "SerializeField]` tunables" in body  # game-feel move


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
    def test_capable_models_get_blind_spot_pass(self, make_agent):
        prompt = make_agent("cloud").build_system_prompt()
        assert "Blind Spot Pass" in prompt
        assert "blind-spot-pass" in prompt        # the skill pointer

    def test_tiny_models_are_not_burdened(self, make_agent):
        prompt = make_agent("tiny").build_system_prompt()
        assert "Blind Spot Pass" not in prompt
