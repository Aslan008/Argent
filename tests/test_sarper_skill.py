from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "sarper-web-architect" / "SKILL.md"


def test_sarper_skill_is_discoverable_and_names_the_mcp_contract():
    text = SKILL.read_text(encoding="utf-8")
    assert "name: sarper-web-architect" in text
    assert "get_status" in text
    assert "start_task" in text
    assert "wait_task" in text
    assert "call_mcp_tool" in text
    assert "fresh" in text and "close_tab" in text


def test_the_skill_links_to_its_exception_reference():
    text = SKILL.read_text(encoding="utf-8")
    assert "references/edge-cases.md" in text
    assert "## Что если" in text
    ref = SKILL.parent / "references" / "edge-cases.md"
    assert ref.is_file()
    body = ref.read_text(encoding="utf-8")
    for trigger in ("бесплатный лимит", "reasoning_level", "Неясно, какую модель", "Бюджет обращений"):
        assert trigger in body


def test_the_skill_keeps_the_local_verifier_in_charge():
    text = SKILL.read_text(encoding="utf-8")
    assert "локальный исполнитель" in text
    assert "не доверяй ответу веб-ии" in text.lower()
