"""Rooms editor: skills, plugins and MCP servers are picked up while the editor is open,
`regex_all:` takes every match, and the Sarper profile is wired into the page.
"""

import re
from pathlib import Path

import pytest

import hook_manager as hm
from src.rooms.integrations import Integrations
from src.rooms.models import Node
from src.rooms.runner import derive_value, derive_result_values

EDITOR = Path(__file__).resolve().parent.parent / "src" / "rooms" / "editor"


@pytest.fixture
def hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "get_global_hooks_dir", lambda: tmp_path)
    monkeypatch.setattr(hm, "get_disabled_plugins", lambda: [])
    monkeypatch.setattr(hm, "print_system", lambda *a, **k: None)
    return tmp_path


def names(manager):
    return sorted(m.__name__ for m in manager._plugins)


class TestPluginsFollowTheFolder:
    def test_a_plugin_added_edited_and_removed_is_noticed(self, hooks):
        manager = hm.HookManager()
        assert names(manager) == [] and manager.reload_if_changed() is False
        (hooks / "alpha.py").write_text("def command_one():\n    return 1\n", encoding="utf-8")
        assert manager.reload_if_changed() is True and names(manager) == ["alpha"]
        assert manager.reload_if_changed() is False                      # nothing changed: nothing reloaded
        (hooks / "alpha.py").write_text("def command_one():\n    return 1\n\ndef command_two():\n    return 2\n", encoding="utf-8")
        assert manager.reload_if_changed() is True
        assert hasattr(manager._plugins[0], "command_two")
        (hooks / "alpha.py").unlink()
        assert manager.reload_if_changed() is True and names(manager) == []

    def test_disabling_a_plugin_counts_as_a_change(self, hooks, monkeypatch):
        (hooks / "beta.py").write_text("def command_b():\n    return 1\n", encoding="utf-8")
        manager = hm.HookManager()
        assert names(manager) == ["beta"]
        monkeypatch.setattr(hm, "get_disabled_plugins", lambda: ["beta"])
        assert manager.reload_if_changed() is True and names(manager) == []

    def test_the_catalog_asks_for_a_fresh_look(self, hooks, monkeypatch):
        manager = hm.HookManager()
        monkeypatch.setattr(hm, "hook_manager", manager)
        assert Integrations().plugins() == {}
        (hooks / "gamma.py").write_text("def command_go():\n    return 1\n", encoding="utf-8")
        assert Integrations().plugins() == {"gamma": {"commands": ["go"], "hooks": []}}


class TestMcpCatalog:
    def test_a_server_pointed_elsewhere_changes_its_signature_and_hides_secrets(self, monkeypatch):
        cfg = {"s": {"name": "s", "type": "stdio", "command": "python", "args": ["a.py"], "env": {"KEY": "secret-1"}}}
        monkeypatch.setattr(Integrations, "_configured_servers", staticmethod(lambda: cfg))
        first = Integrations().mcp_servers()["s"]["sig"]
        cfg["s"]["env"] = {"KEY": "secret-2"}                           # a secret changed: the shape did not
        assert Integrations().mcp_servers()["s"]["sig"] == first
        cfg["s"]["args"] = ["b.py"]
        assert Integrations().mcp_servers()["s"]["sig"] != first
        assert "secret" not in str(Integrations().mcp_servers())


class TestRegexAll:
    def test_every_match_as_a_list(self):
        v = derive_result_values("code\nBag ID: ab12cd34\ntext\nBag ID: 0f0f0f0f\n")
        assert derive_value("regex_all:Bag ID: ([0-9a-f]+)", v) == ["ab12cd34", "0f0f0f0f"]
        assert derive_value("regex_all:Nope (\\d+)", v) == []

    def test_the_pattern_is_checked_when_the_room_is_loaded(self):
        ok = Node(id="q", type="mcp", server="s", tool="t", sets={"ids": "regex_all:id=(\\d+)"})
        assert ok.sets["ids"].startswith("regex_all:")
        for bad in ("regex_all:(", "regex_all:"):
            with pytest.raises(ValueError):
                Node(id="q", type="mcp", server="s", tool="t", sets={"ids": bad})


class TestEditorPage:
    def test_profiles_are_served_and_loaded_before_the_app(self):
        html = (EDITOR / "index.html").read_text(encoding="utf-8")
        order = [html.index(f'/{name}.js?t=') for name in ("ext", "simple", "profiles", "run", "app")]
        assert order == sorted(order)
        from src.rooms import editor_server
        assert editor_server._STATIC["/profiles.js"][0] == "profiles.js"

    def test_the_editor_refreshes_on_its_own_and_on_demand(self):
        ext = (EDITOR / "ext.js").read_text(encoding="utf-8")
        app = (EDITOR / "app.js").read_text(encoding="utf-8")
        for needle in ("async function refreshCatalogs", "function startCatalogWatch", 'addEventListener("visibilitychange"', "setInterval("):
            assert needle in ext
        assert "startCatalogWatch()" in app
        assert "↻ Обновить" in ext

    def test_every_sarper_tool_has_a_form_and_the_ones_that_write_ask_first(self):
        js = (EDITOR / "profiles.js").read_text(encoding="utf-8")
        tools = set(re.findall(r"^  ([a-z_]+): \{\n    title:", js, re.M))
        assert tools == {"query_web_llm", "start_task", "wait_task", "get_task", "cancel_task", "get_status",
                         "read_bag_content", "write_bag_to_file"}
        write = js[js.index("  write_bag_to_file: {"):]
        assert "confirm: true" in write[:600]
        assert 'mcp:" + server + "." + WRITE_BAG' in js        # «только советоваться» is a rule, not a request
