"""Blind-spot tests for hook_manager.py — edge cases around plugin discovery,
event dispatch, modifier chaining, custom commands, reloading, disabled
plugins, and empty-state behaviour.

The module creates a singleton ``hook_manager = HookManager()`` at import time,
which calls ``config.get_hooks_dir`` / ``config.get_disabled_plugins``.  We
monkeypatch those *before* importing the module so the singleton points at a
throwaway tmp_path, and then build fresh ``HookManager`` instances per test for
isolation.
"""

import sys
import textwrap
import importlib

import pytest


# ─── Helpers: write plugin files into a hooks dir ────────────────────────────

def _write_plugin(hooks_dir, name, body):
    """Write a .py plugin file (name without extension) into hooks_dir."""
    path = hooks_dir / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def hooks_dir(tmp_path):
    """A fresh empty hooks directory inside tmp_path."""
    d = tmp_path / "hooks"
    d.mkdir()
    return d


@pytest.fixture
def patched_config(monkeypatch, hooks_dir):
    """Monkeypatch config.get_hooks_dir and config.get_disabled_plugins so the
    hook_manager module (and any HookManager instance) uses our tmp_path."""
    import config

    monkeypatch.setattr(config, "get_hooks_dir", lambda: str(hooks_dir))
    monkeypatch.setattr(config, "get_disabled_plugins", lambda: [])

    # Make sure hook_manager is (re)imported with the patched config so the
    # singleton does not blow up pointing at the real ~/.argent/hooks dir.
    if "hook_manager" in sys.modules:
        del sys.modules["hook_manager"]
    import hook_manager  # noqa: F401
    return hook_manager


@pytest.fixture
def hm(patched_config):
    """A HookManager class bound to the patched config (not the singleton)."""
    return patched_config.HookManager


# ─── 1. Loads .py but skips __init__.py (underscore prefix) ──────────────────

class TestPluginDiscovery:
    def test_loads_py_files_but_skips_underscore_prefix(self, hm, hooks_dir):
        """A regular plugin is loaded; __init__.py and any _-prefixed file are
        silently skipped so they don't pollute the plugin list."""
        _write_plugin(hooks_dir, "real_plugin", """
            def on_start():
                return "real"
        """)
        _write_plugin(hooks_dir, "__init__", """
            # should be skipped
            raise RuntimeError("init should never load")
        """)
        _write_plugin(hooks_dir, "_private", """
            # underscore-prefixed, should be skipped
            raise RuntimeError("private should never load")
        """)

        mgr = hm()
        names = [p.__name__ for p in mgr._plugins]
        assert "real_plugin" in names
        assert "__init__" not in names
        assert "_private" not in names
        assert len(mgr._plugins) == 1


# ─── 2 & 3 & 10. call_hook results ───────────────────────────────────────────

class TestCallHook:
    def test_returns_list_of_results_from_implementing_plugins(self, hm, hooks_dir):
        """call_hook collects one result per plugin that implements the event."""
        _write_plugin(hooks_dir, "plug_a", """
            def on_event():
                return "a"
        """)
        _write_plugin(hooks_dir, "plug_b", """
            def on_event():
                return "b"
        """)

        mgr = hm()
        results = mgr.call_hook("on_event")
        assert sorted(results) == ["a", "b"]
        assert isinstance(results, list)

    def test_returns_empty_list_when_no_plugin_implements_event(self, hm, hooks_dir):
        """If no plugin has the event, call_hook returns an empty list."""
        _write_plugin(hooks_dir, "plug_a", """
            def on_other():
                return 1
        """)

        mgr = hm()
        assert mgr.call_hook("nonexistent_event") == []

    def test_returns_empty_list_when_no_plugins_loaded(self, hm, hooks_dir):
        """call_hook with zero loaded plugins returns an empty list."""
        mgr = hm()
        assert mgr._plugins == []
        assert mgr.call_hook("anything") == []

    def test_passes_args_and_kwargs_to_plugin(self, hm, hooks_dir):
        """Positional and keyword arguments are forwarded to the event handler."""
        _write_plugin(hooks_dir, "plug", """
            def on_event(value, multiplier=1):
                return value * multiplier
        """)

        mgr = hm()
        assert mgr.call_hook("on_event", 5, multiplier=3) == [15]


# ─── 4. call_hook catches exceptions and continues ───────────────────────────

class TestCallHookExceptions:
    def test_exception_in_one_plugin_does_not_stop_others(self, hm, hooks_dir):
        """A plugin that raises is skipped (its error is printed) but the next
        plugin's result still appears in the output list."""
        _write_plugin(hooks_dir, "boom", """
            def on_event():
                raise ValueError("boom")
        """)
        _write_plugin(hooks_dir, "ok", """
            def on_event():
                return "ok"
        """)

        mgr = hm()
        results = mgr.call_hook("on_event")
        # The failing plugin contributed nothing; the good one still ran.
        assert results == ["ok"]


# ─── 5 & 6. call_modifier_hook ───────────────────────────────────────────────

class TestCallModifierHook:
    def test_passes_value_through_all_plugins_sequentially(self, hm, hooks_dir):
        """The initial value flows through every plugin in order, each one
        transforming the output of the previous."""
        _write_plugin(hooks_dir, "upper", """
            def modify(value):
                return value.upper()
        """)
        _write_plugin(hooks_dir, "exclaim", """
            def modify(value):
                return value + "!"
        """)

        mgr = hm()
        result = mgr.call_modifier_hook("modify", "hello")
        # upper runs first -> "HELLO", then exclaim -> "HELLO!"
        assert result == "HELLO!"

    def test_skips_plugins_returning_none(self, hm, hooks_dir):
        """A plugin that returns None is skipped — the current value is passed
        unchanged to the next plugin."""
        _write_plugin(hooks_dir, "noop", """
            def modify(value):
                return None
        """)
        _write_plugin(hooks_dir, "exclaim", """
            def modify(value):
                return value + "!"
        """)

        mgr = hm()
        result = mgr.call_modifier_hook("modify", "hi")
        # noop returned None so value stayed "hi", then exclaim appended "!"
        assert result == "hi!"

    def test_returns_initial_value_when_no_plugin_implements_event(self, hm, hooks_dir):
        """With no implementing plugins, the initial value is returned unchanged."""
        _write_plugin(hooks_dir, "plug", """
            def on_other():
                return "x"
        """)

        mgr = hm()
        assert mgr.call_modifier_hook("modify", "untouched") == "untouched"

    def test_modifier_exception_does_not_stop_chain(self, hm, hooks_dir):
        """A plugin that raises during modification is skipped, the chain
        continues with the current value."""
        _write_plugin(hooks_dir, "boom", """
            def modify(value):
                raise RuntimeError("boom")
        """)
        _write_plugin(hooks_dir, "exclaim", """
            def modify(value):
                return value + "!"
        """)

        mgr = hm()
        assert mgr.call_modifier_hook("modify", "hi") == "hi!"


# ─── 7. get_custom_commands ─────────────────────────────────────────────────

class TestGetCustomCommands:
    def test_finds_command_prefixed_functions(self, hm, hooks_dir):
        """Functions named command_<name> are discovered and mapped by name."""
        _write_plugin(hooks_dir, "cmds", """
            def command_deploy():
                return "deploying"
            def command_status():
                return "status-ok"
            def not_a_command():
                return "ignored"
        """)

        mgr = hm()
        commands = mgr.get_custom_commands()
        assert set(commands.keys()) == {"deploy", "status"}
        assert commands["deploy"]() == "deploying"
        assert commands["status"]() == "status-ok"

    def test_returns_empty_dict_when_no_commands(self, hm, hooks_dir):
        """No command_ functions → empty dict."""
        _write_plugin(hooks_dir, "plain", """
            def on_event():
                return 1
        """)

        mgr = hm()
        assert mgr.get_custom_commands() == {}

    def test_returns_empty_dict_when_no_plugins(self, hm, hooks_dir):
        """No plugins loaded at all → empty command dict."""
        mgr = hm()
        assert mgr.get_custom_commands() == {}


# ─── 8. reload_plugins loads from a new dir ─────────────────────────────────

class TestReloadPlugins:
    def test_reload_from_new_dir(self, hm, hooks_dir, tmp_path):
        """reload_plugins(new_dir) switches to a different directory and loads
        the plugins found there, dropping the old ones."""
        # Start with one plugin in the original dir.
        _write_plugin(hooks_dir, "old_plugin", """
            def on_event():
                return "old"
        """)
        mgr = hm()
        assert len(mgr._plugins) == 1
        assert mgr._plugins[0].__name__ == "old_plugin"

        # New directory with a different plugin.
        new_dir = tmp_path / "new_hooks"
        new_dir.mkdir()
        _write_plugin(new_dir, "new_plugin", """
            def on_event():
                return "new"
        """)

        mgr.reload_plugins(str(new_dir))
        names = [p.__name__ for p in mgr._plugins]
        assert "new_plugin" in names
        assert "old_plugin" not in names
        assert str(mgr.hooks_dir) == str(new_dir.resolve())

    def test_reload_without_new_dir_reloads_same_dir(self, hm, hooks_dir):
        """reload_plugins() with no argument reloads from the current dir,
        picking up files added after the initial load."""
        mgr = hm()
        assert mgr._plugins == []

        _write_plugin(hooks_dir, "added_later", """
            def on_event():
                return "late"
        """)
        mgr.reload_plugins()
        names = [p.__name__ for p in mgr._plugins]
        assert "added_later" in names


# ─── 9. disabled plugins are skipped ────────────────────────────────────────

class TestDisabledPlugins:
    def test_disabled_plugin_is_not_loaded(self, monkeypatch, tmp_path, hooks_dir):
        """A plugin whose stem name appears in get_disabled_plugins() is skipped
        during discovery."""
        import config

        _write_plugin(hooks_dir, "enabled_plug", """
            def on_event():
                return "ok"
        """)
        _write_plugin(hooks_dir, "disabled_plug", """
            # this should never load
            raise RuntimeError("disabled plugin loaded!")
        """)

        monkeypatch.setattr(config, "get_hooks_dir", lambda: str(hooks_dir))
        monkeypatch.setattr(config, "get_disabled_plugins",
                            lambda: ["disabled_plug"])

        if "hook_manager" in sys.modules:
            del sys.modules["hook_manager"]
        import hook_manager as hm_mod

        mgr = hm_mod.HookManager()
        names = [p.__name__ for p in mgr._plugins]
        assert "enabled_plug" in names
        assert "disabled_plug" not in names
        assert len(mgr._plugins) == 1