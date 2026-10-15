"""/verbose, /results, /debug and /max_tokens.

They were dispatched by the old command_handler chain, lost in the move to the
command registry, and left behind in the README, the Tab menu and config's
setters — typing one answered "unknown command". /max_tokens was the only way to
change max_generation_tokens, which the agent and the providers both read.
"""

from types import SimpleNamespace

import pytest

import config
from src.cli.commands import CommandContext, dispatch_command


@pytest.fixture(autouse=True)
def _own_config(tmp_path, monkeypatch):
    """Whatever a command writes lands in a throw-away file."""
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "_CONFIG_CACHE", None)
    monkeypatch.setattr(config, "_CONFIG_MTIME", None)


def _run(text: str, window: int = 0):
    ctx = CommandContext(agent=SimpleNamespace(max_context_tokens=window), user_input=text)
    return dispatch_command(text, ctx)


SWITCHES = [
    ("/verbose", config.get_verbose_status, True),          # default ON
    ("/results", config.get_show_tool_results, False),      # default OFF
    ("/debug", config.get_debug_mode, False),               # default OFF
]


class TestSwitches:
    @pytest.mark.parametrize("command, getter, default", SWITCHES)
    def test_it_is_handled(self, command, getter, default):
        assert _run(command).handled

    @pytest.mark.parametrize("command, getter, default", SWITCHES)
    def test_it_flips_each_time(self, command, getter, default):
        _run(command)
        assert getter() is (not default)
        _run(command)
        assert getter() is default

    @pytest.mark.parametrize("command, getter, default", SWITCHES)
    def test_on_and_off_are_explicit(self, command, getter, default):
        _run(f"{command} on")
        assert getter() is True
        _run(f"{command} on")
        assert getter() is True            # saying it twice must not flip it back
        _run(f"{command} off")
        assert getter() is False

    @pytest.mark.parametrize("command, getter, default", SWITCHES)
    def test_garbage_changes_nothing_and_says_how_to_use_it(self, command, getter, default, capsys):
        before = getter()
        _run(f"{command} perhaps")
        assert getter() is before
        assert "Использование" in capsys.readouterr().out

    def test_results_off_reminds_that_errors_still_print_in_full(self, capsys):
        _run("/results on")
        capsys.readouterr()
        _run("/results off")
        assert "ошибки показываются целиком" in capsys.readouterr().out


class TestMaxTokens:
    def test_without_an_argument_it_reports_and_changes_nothing(self, capsys):
        config.set_max_generation_tokens(12345)
        _run("/max_tokens")
        assert config.get_max_generation_tokens() == 12345
        assert "12345" in capsys.readouterr().out

    @pytest.mark.parametrize("typed, expected", [
        ("16384", 16384),
        ("32k", 32 * 1024),
        ("32K", 32 * 1024),
        ("1m", 1024 * 1024),
        ("16_384", 16384),
        ("8,192", 8192),
    ])
    def test_it_sets_the_limit(self, typed, expected):
        _run(f"/max_tokens {typed}")
        assert config.get_max_generation_tokens() == expected

    def test_default_restores_8192(self):
        config.set_max_generation_tokens(32768)
        _run("/max_tokens default")
        assert config.get_max_generation_tokens() == 8192

    def test_the_alias_works(self):
        _run("/maxtokens 4096")
        assert config.get_max_generation_tokens() == 4096

    @pytest.mark.parametrize("typed", ["lots", "12.5", "-5", "0", "255", "2000000", "k"])
    def test_nonsense_is_refused_and_the_old_value_kept(self, typed):
        config.set_max_generation_tokens(9000)
        _run(f"/max_tokens {typed}")
        assert config.get_max_generation_tokens() == 9000

    def test_a_limit_as_large_as_the_context_window_is_flagged(self, capsys):
        _run("/max_tokens 16384", window=8192)
        out = capsys.readouterr().out
        assert config.get_max_generation_tokens() == 16384      # allowed, but not silently
        assert "окна контекста" in out
