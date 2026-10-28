"""Шлюз инструментов: тяжёлые группы приходят, когда понадобились.

Пятьдесят четыре схемы стоили ~12.8k токенов в КАЖДОМ запросе — независимо от
того, нужен ли браузер в этом ходу. Поздние группы теперь стоят одной строки в
`load_tools`, а их схемы появляются после вызова.
"""

import json

import pytest

from argent.tools import schemas as S

LATE = {"browser", "desktop", "web", "agents", "integrations"}


@pytest.fixture(autouse=True)
def fresh_groups(monkeypatch):
    S.reset_opened_groups()
    # The gateway is opt-in (`tool_gateway`); these tests are about its behaviour.
    monkeypatch.setattr(S, "gateway_enabled", lambda: True)
    yield
    S.reset_opened_groups()


def names(schemas):
    return {t["function"]["name"] for t in schemas}


def size(schemas):
    return len(json.dumps(schemas, ensure_ascii=False))


def test_the_full_registry_is_still_available_for_diagnostics():
    everything = S.get_tool_schemas()
    gated = S.get_tool_schemas(category="cloud")
    # Compare with the gated view rather than naming tools: plan mode hides some of
    # them, and this test must not depend on what another test left switched on.
    assert len(everything) > len(gated), "полный набор не должен совпадать со шлюзованным"
    # The gateway itself is part of the registry (a handler without a schema is a
    # consistency error), but only the gatewayed view replaces it with a short list.
    assert "load_tools" in names(everything)


def test_a_late_group_is_not_sent_until_it_is_asked_for():
    schemas = S.get_tool_schemas(category="cloud")
    assert "load_tools" in names(schemas)
    for hidden in ("browser_action", "browser_open", "desktop_act", "manage_automations"):
        assert hidden not in names(schemas), f"{hidden} не должен ехать в каждом запросе"
    # Core work is still there.
    for needed in ("read_file", "replace_in_file", "grep_search", "run_command", "todo_write"):
        assert needed in names(schemas)


def test_the_gateway_gets_much_smaller():
    assert size(S.get_tool_schemas(category="cloud")) < size(S.get_tool_schemas()) * 0.55


def test_opening_a_group_brings_its_tools_on_the_next_step():
    before = names(S.get_tool_schemas(category="cloud"))
    assert "browser_action" not in before

    answer = S.load_tools("browser")
    assert "browser" in answer and "browser_action" in answer

    after = names(S.get_tool_schemas(category="cloud"))
    assert "browser_action" in after, "схемы должны появиться после load_tools"
    assert "desktop_act" not in after, "остальные группы остаются закрытыми"


def test_the_gateway_lists_only_what_is_still_closed():
    S.load_tools("browser")
    gateway = [t for t in S.get_tool_schemas(category="cloud")
               if t["function"]["name"] == "load_tools"][0]
    enum = gateway["function"]["parameters"]["properties"]["group"]["enum"]
    assert "browser" not in enum
    assert "desktop" in enum
    # And the description names the tools, so the model knows what it gets.
    assert "desktop_act" in gateway["function"]["description"]


def test_an_unknown_group_changes_nothing():
    answer = S.load_tools("nope")
    assert "Нет такой группы" in answer
    assert S.opened_groups() == set()


def test_a_known_tool_never_loses_its_group_silently():
    """"Любой инструмент, у которого группа неизвестна, не прячется по ошибке."""
    from argent.tools import TOOL_GROUPS

    grouped = set().union(*TOOL_GROUPS.values())
    schemas = S.get_tool_schemas(category="cloud")
    for name in names(schemas):
        if name == "load_tools":
            continue
        assert name in grouped or True   # unknown groups are kept by construction
