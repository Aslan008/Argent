"""Unit tests for Turn Snapshots, Prompt Cache accounting, and /snapshot command."""

import pytest
from unittest.mock import MagicMock

import config
from providers import _parse_openai_usage
from usage_tracker import SessionUsage
from src.cli.cli_ui import _print_turn_snapshot
from src.cli.commands.base import CommandContext
from src.cli.commands.config_commands import handle_snapshot


# ---------------------------------------------------------------------------
# 1. Tests for _parse_openai_usage with prompt cache
# ---------------------------------------------------------------------------

class TestParseOpenAIUsageCache:
    def test_parse_none(self):
        assert _parse_openai_usage(None) == {}

    def test_parse_standard_openai_with_cached_tokens(self):
        class Details:
            cached_tokens = 1500

        class UsageObj:
            prompt_tokens = 2000
            completion_tokens = 300
            total_tokens = 2300
            prompt_tokens_details = Details()

        res = _parse_openai_usage(UsageObj())
        assert res["prompt"] == 2000
        assert res["completion"] == 300
        assert res["total"] == 2300
        assert res["cached"] == 1500

    def test_parse_openai_dict_format(self):
        raw = {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {
                "cached_tokens": 800,
            },
            "cost": 0.005,
        }
        res = _parse_openai_usage(raw)
        assert res["prompt"] == 1000
        assert res["completion"] == 200
        assert res["total"] == 1200
        assert res["cached"] == 800
        assert res["cost"] == 0.005

    def test_parse_deepseek_prompt_cache_hit(self):
        raw = {
            "prompt_tokens": 4000,
            "completion_tokens": 500,
            "total_tokens": 4500,
            "prompt_cache_hit_tokens": 3200,
            "prompt_cache_miss_tokens": 800,
        }
        res = _parse_openai_usage(raw)
        assert res["prompt"] == 4000
        assert res["cached"] == 3200

    def test_parse_openrouter_anthropic_model_extra(self):
        class UsageObj:
            prompt_tokens = 5000
            completion_tokens = 400
            total_tokens = 5400
            model_extra = {
                "cost": 0.0125,
                "cache_read_input_tokens": 3500,
            }

        res = _parse_openai_usage(UsageObj())
        assert res["prompt"] == 5000
        assert res["cost"] == 0.0125
        assert res["cached"] == 3500

    def test_parse_no_cache_info(self):
        raw = {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600}
        res = _parse_openai_usage(raw)
        assert res["prompt"] == 500
        assert res["completion"] == 100
        assert "cached" not in res


# ---------------------------------------------------------------------------
# 2. Tests for SessionUsage with cached tokens
# ---------------------------------------------------------------------------

class TestSessionUsageCached:
    def test_accumulates_cached(self):
        su = SessionUsage()
        su.add({"prompt": 1000, "completion": 200, "cached": 600})
        su.add({"prompt": 2000, "completion": 300, "cached": 1400})
        assert su.prompt == 3000
        assert su.completion == 500
        assert su.cached == 2000
        assert su.requests == 2

    def test_reset_clears_cached(self):
        su = SessionUsage()
        su.add({"prompt": 1000, "completion": 200, "cached": 800})
        su.reset()
        assert su.cached == 0
        assert su.prompt == 0

    def test_format_last_includes_cached(self):
        res = SessionUsage.format_last({"prompt": 2000, "completion": 500, "cached": 1500})
        assert "2k->500 tok" in res
        assert "cached 1.5k 75%" in res

    def test_format_last_without_cached(self):
        res = SessionUsage.format_last({"prompt": 2000, "completion": 500})
        assert "2k->500 tok" in res
        assert "cached" not in res


# ---------------------------------------------------------------------------
# 3. Tests for Turn Snapshot Config & Commands
# ---------------------------------------------------------------------------

class TestSnapshotConfig:
    def test_default_is_compact(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        assert config.get_turn_snapshot_mode() == "compact"

    def test_set_mode_valid(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        config.set_turn_snapshot_mode("verbose")
        assert config.get_turn_snapshot_mode() == "verbose"
        config.set_turn_snapshot_mode("off")
        assert config.get_turn_snapshot_mode() == "off"
        config.set_turn_snapshot_mode("compact")
        assert config.get_turn_snapshot_mode() == "compact"

    def test_set_mode_invalid_raises(self):
        with pytest.raises(ValueError):
            config.set_turn_snapshot_mode("super_detailed")

    def test_handle_snapshot_command_query(self, monkeypatch):
        printed = []
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: printed.append(m))
        ctx = CommandContext(agent=MagicMock(), user_input="/snapshot")
        res = handle_snapshot(ctx)
        assert res.handled
        assert any("Текущий режим снимков хода" in p for p in printed)

    def test_handle_snapshot_command_change(self, monkeypatch):
        test_conf = {}
        monkeypatch.setattr(config, "load_config", lambda: test_conf)
        monkeypatch.setattr(config, "save_config", lambda c: test_conf.update(c))
        printed = []
        monkeypatch.setattr("src.cli.commands.config_commands.print_system", lambda m: printed.append(m))
        ctx = CommandContext(agent=MagicMock(), user_input="/snapshot verbose")
        handle_snapshot(ctx)
        assert config.get_turn_snapshot_mode() == "verbose"
        assert any("✓ Режим снимков хода изменён на: [bold cyan]verbose[/bold cyan]" in p for p in printed)

    def test_handle_snapshot_command_invalid(self, monkeypatch):
        errors = []
        monkeypatch.setattr("src.cli.commands.config_commands.print_error", lambda m: errors.append(m))
        ctx = CommandContext(agent=MagicMock(), user_input="/snapshot unknown_mode")
        handle_snapshot(ctx)
        assert any("Неизвестный режим" in e for e in errors)


# ---------------------------------------------------------------------------
# 4. Tests for _print_turn_snapshot rendering
# ---------------------------------------------------------------------------

class TestPrintTurnSnapshot:
    def test_compact_rendering(self, monkeypatch):
        monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "compact")
        printed = []
        monkeypatch.setattr("src.cli.cli_ui.safe_print", lambda m: printed.append(m))

        agent = MagicMock()
        agent.get_context_usage.return_value = {
            "tokens": 12400,
            "max": 32768,
            "percent": 38.0,
        }

        usage = {"prompt": 2400, "completion": 1100, "cached": 1800, "cost": 0.0021}
        _print_turn_snapshot(
            agent=agent,
            elapsed=4.2,
            timing={"first": 1.1, "tools": 0.3},
            usage=usage,
            tool_calls=3,
        )

        output_text = "\n".join(printed)
        assert "4.2s" in output_text
        assert "2.4k→1.1k tok" in output_text
        assert "cached 1.8k (75%)" in output_text
        assert "3 tools" in output_text
        assert "ctx 12.4k/32.8k (38%)" in output_text or "ctx 12.4k/32k (38%)" in output_text
        assert "$0.0021" in output_text

    def test_verbose_rendering(self, monkeypatch):
        monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "verbose")
        printed = []
        monkeypatch.setattr("src.cli.cli_ui.safe_print", lambda m: printed.append(m))

        agent = MagicMock()
        agent.get_context_usage.return_value = {
            "tokens": 12400,
            "max": 32768,
            "percent": 38.0,
        }
        agent.get_context_breakdown.return_value = {
            "system": 3200,
            "tools": 2100,
            "history": 7100,
            "total": 12400,
            "max": 32768,
            "percent": 38.0,
        }

        usage = {"prompt": 2400, "completion": 1100, "cached": 1800, "cost": 0.0021}
        _print_turn_snapshot(
            agent=agent,
            elapsed=4.2,
            timing={"first": 1.1, "tools": 0.3},
            usage=usage,
            tool_calls=3,
        )

        output_text = "\n".join(printed)
        # Should have multi-line breakdown
        assert "ожидание:" in output_text
        assert "cached: 1.8k (75%)" in output_text
        assert "sys 3.2k" in output_text
        assert "tools 2.1k" in output_text
        assert "history 7.1k" in output_text
        assert "3 tools" in output_text

    def test_off_rendering(self, monkeypatch):
        monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "off")
        printed = []
        monkeypatch.setattr("src.cli.cli_ui.safe_print", lambda m: printed.append(m))

        _print_turn_snapshot(
            agent=None,
            elapsed=4.2,
            timing={},
            usage={"prompt": 2400, "completion": 1100},
            tool_calls=0,
        )

        output_text = "\n".join(printed)
        assert "(4.2s" in output_text

    def test_context_overflow_warning_above_85(self, monkeypatch):
        monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "compact")
        printed = []
        monkeypatch.setattr("src.cli.cli_ui.safe_print", lambda m: printed.append(m))

        agent = MagicMock()
        agent.get_context_usage.return_value = {
            "tokens": 28000,
            "max": 32768,
            "percent": 88.0,
        }

        _print_turn_snapshot(
            agent=agent,
            elapsed=3.0,
            timing={},
            usage={},
            tool_calls=0,
        )

        output_text = "\n".join(printed)
        assert "▲ Контекст заполнен на 88%" in output_text
        assert "/clear или /compact" in output_text

    def test_no_overflow_warning_below_85(self, monkeypatch):
        monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "compact")
        printed = []
        monkeypatch.setattr("src.cli.cli_ui.safe_print", lambda m: printed.append(m))

        agent = MagicMock()
        agent.get_context_usage.return_value = {
            "tokens": 15000,
            "max": 32768,
            "percent": 45.0,
        }

        _print_turn_snapshot(
            agent=agent,
            elapsed=3.0,
            timing={},
            usage={},
            tool_calls=0,
        )

        output_text = "\n".join(printed)
        assert "▲ Контекст заполнен" not in output_text

    def test_graceful_when_agent_none(self, monkeypatch):
        monkeypatch.setattr(config, "get_turn_snapshot_mode", lambda: "compact")
        printed = []
        monkeypatch.setattr("src.cli.cli_ui.safe_print", lambda m: printed.append(m))

        # Should not throw any exception
        _print_turn_snapshot(
            agent=None,
            elapsed=2.5,
            timing=None,
            usage=None,
            tool_calls=0,
        )
        assert any("2.5s" in p for p in printed)
