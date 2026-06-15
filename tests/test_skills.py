import pytest

import skill_manager as sm_module


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(sm_module, "get_skills_dir", lambda: str(tmp_path))
    return sm_module.SkillManager()


class TestCreateAndRead:
    def test_create_and_read_body(self, manager):
        manager.create_skill("greet", "Say hello, then ask a question.", "Greeting flow")
        assert manager.read_skill("greet") == "Say hello, then ask a question."

    def test_description_round_trips(self, manager):
        manager.create_skill("s", "Body.", "Simple description")
        meta = next(s for s in manager.list_skills() if s["name"] == "s")
        assert meta["description"] == "Simple description"

    def test_description_with_quotes_and_colon_preserved(self, manager):
        # Regression: a hand-built YAML f-string corrupted these, losing the
        # description on read ("No description provided.").
        tricky = 'Skill for: parsing "tricky" input'
        manager.create_skill("t", "Body.", tricky)
        meta = next(s for s in manager.list_skills() if s["name"] == "t")
        assert meta["description"] == tricky

    def test_unicode_description(self, manager):
        manager.create_skill("u", "Тело.", "Навык: работа с проектом")
        meta = next(s for s in manager.list_skills() if s["name"] == "u")
        assert meta["description"] == "Навык: работа с проектом"

    def test_read_missing_returns_none(self, manager):
        assert manager.read_skill("nope") is None


class TestListAndDelete:
    def test_list_multiple(self, manager):
        manager.create_skill("a", "x", "first")
        manager.create_skill("b", "y", "second")
        names = {s["name"] for s in manager.list_skills()}
        assert {"a", "b"} <= names

    def test_delete(self, manager):
        manager.create_skill("temp", "x", "d")
        assert "deleted" in manager.delete_skill("temp")
        assert manager.read_skill("temp") is None

    def test_delete_missing(self, manager):
        assert "not found" in manager.delete_skill("ghost")
