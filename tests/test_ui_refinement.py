"""
Тесты для премиального Human-First интерфейса Argent:
- Двухстрочный вывод инструментов (Вариант Б: │ header \n │ ↳ summary)
- Детализированные заголовки инструментов (read_file, run_command, grep_search, browser_*)
- Человечные сводки результатов (pytest, git, правки файлов, браузер)
- Отсутствие дешевых эмодзи в reasoning, rethink, approval
- Расшифровка сетевых ошибок (Wormsoft 404, Ollama 502/DNS)
"""
import pytest
from unittest.mock import patch
from io import StringIO

from ui import _format_tool_header, _format_tool_header_markup, _format_tool_summary, print_reasoning, print_rethink, print_tool_end
from approval import _ALWAYS_PREFIX
from error_help import explain_error


class TestToolHeaderFormatting:
    def test_read_file_header_with_path(self):
        hdr = _format_tool_header("read_file", {"path": "src/cli/cli_ui.py"})
        assert hdr == "read_file: src/cli/cli_ui.py"

    def test_read_file_header_with_offset_and_limit(self):
        hdr = _format_tool_header("read_file", {"path": "agent.py", "offset": 100, "limit": 50})
        assert hdr == "read_file: agent.py:100-150"

    def test_read_file_header_with_string_limit_and_no_offset(self):
        # Reproduces the exact bug where model emits limit="60" as string without offset
        hdr = _format_tool_header("read_file", {"path": "scripts/_e5.py", "limit": "60"})
        assert hdr == "read_file: scripts/_e5.py:1-61"

    def test_read_file_header_with_mixed_string_int_bounds(self):
        hdr1 = _format_tool_header("read_file", {"path": "a.py", "offset": "10", "limit": 20})
        assert hdr1 == "read_file: a.py:10-30"
        hdr2 = _format_tool_header("read_file", {"path": "b.py", "offset": "5", "limit": "15"})
        assert hdr2 == "read_file: b.py:5-20"
        hdr3 = _format_tool_header("read_file", {"path": "c.py", "offset": 1, "limit": "invalid"})
        assert "c.py:1" in hdr3

    def test_write_file_header(self):
        hdr = _format_tool_header("write_file", {"path": "tests/test_sample.py"})
        assert hdr == "write_file: tests/test_sample.py"

    def test_replace_in_file_header(self):
        hdr = _format_tool_header("replace_in_file", {"path": "ui.py"})
        assert hdr == "replace_in_file: ui.py"

    def test_run_command_header(self):
        hdr = _format_tool_header("run_command", {"command": "pytest tests/test_approval.py -v"})
        assert hdr == "run_command: pytest tests/test_approval.py -v"

    def test_grep_search_header(self):
        hdr = _format_tool_header("grep_search", {"query": "print_tool_end", "path": "src"})
        assert hdr == 'grep_search: "print_tool_end" in src'

    def test_file_search_header(self):
        hdr = _format_tool_header("file_search", {"pattern": "*.py"})
        assert hdr == 'file_search: "*.py"'

    def test_browser_headers(self):
        assert _format_tool_header("browser_navigate", {"url": "https://yandex.ru"}) == "browser_navigate: https://yandex.ru"
        assert _format_tool_header("browser_click", {"selector": "button.submit"}) == "browser_click: button.submit"
        assert _format_tool_header("browser_type", {"selector": "#query", "text": "hello"}) == "browser_type: #query"

    def test_web_url_headers_prioritize_and_preserve_full_link(self):
        # Reproduces exact user issue: read_webpage with timeout and long URL
        hf_url = "https://huggingface.co/convaiinnovations/laya/blob/main/README.md"
        hdr = _format_tool_header("read_webpage", {"timeout": "20", "url": hf_url})
        assert hdr == f"read_webpage: {hf_url}"
        assert "timeout" not in hdr

        # Alternative URL key aliases
        assert _format_tool_header("fetch_url", {"target_url": "https://docs.anthropic.com/api"}) == "fetch_url: https://docs.anthropic.com/api"
        assert _format_tool_header("api_get", {"uri": "https://api.github.com/repos"}) == "api_get: https://api.github.com/repos"

    def test_web_url_truncation_only_when_exceeding_250_chars(self):
        long_url = "https://example.com/api/v1/search?q=" + "a" * 250
        hdr = _format_tool_header("read_webpage", {"url": long_url})
        assert len(hdr.removeprefix("read_webpage: ")) <= 250
        assert hdr.endswith("…")

    def test_fallback_header(self):
        assert _format_tool_header("custom_tool", None) == "custom_tool"
        assert _format_tool_header("custom_tool", {"name": "test"}) == "custom_tool: name='test'"

    def test_markup_header_coloring(self):
        markup = _format_tool_header_markup("grep_search", {"query": "sample", "path": "src"})
        assert markup == '[tool_name]grep_search[/tool_name][dim]:[/dim] [bold white]"sample" in src[/bold white]'

        markup_plain = _format_tool_header_markup("custom_tool", None)
        assert markup_plain == "[tool_name]custom_tool[/tool_name]"


class TestToolSummaryFormatting:
    def test_pytest_summary_extracted(self):
        raw = "...\n======= 10 passed, 1 failed in 2.34s =======\n"
        summary = _format_tool_summary("run_command", raw)
        assert "10 passed, 1 failed in 2.34s" in summary

    def test_git_stat_summary_extracted(self):
        raw = " 3 files changed, 25 insertions(+), 4 deletions(-)\n"
        summary = _format_tool_summary("run_command", raw)
        assert "3 files changed" in summary

    def test_browser_null_result_beautified(self):
        summary = _format_tool_summary("browser_click", "JS result: null")
        assert summary == "Действие выполнено"

    def test_file_write_summary(self):
        summary = _format_tool_summary("write_file", "Файл успешно записан (512 байт)")
        assert "Создан файл" in summary or "записан" in summary

    def test_file_replace_summary(self):
        summary = _format_tool_summary("replace_in_file", "Заменено 2 вхождения в ui.py")
        assert "Заменено 2 вхождения" in summary


class TestTwoLineToolEndDisplay:
    def test_print_tool_end_two_line_layout(self):
        out = StringIO()
        with patch("ui.console.print", side_effect=lambda *args, **kw: out.write(str(args[0]) + "\n")):
            print_tool_end("read_file", "line 1\nline 2\nline 3", duration=0.045, args={"path": "agent.py"})
        output = out.getvalue()
        # Проверяем наличие границы '│' и стрелки '↳'
        assert "│" in output
        assert "↳" in output
        assert "[tool_name]read_file[/tool_name]" in output
        assert "agent.py" in output
        assert "45ms" in output


class TestNoCheapEmojis:
    def test_reasoning_uses_glyph_not_emoji(self):
        out = StringIO()
        with patch("ui.console.print", side_effect=lambda *args, **kw: out.write(str(args[0]) + "\n")):
            print_reasoning("Some thought process", duration=4.2)
        val = out.getvalue()
        assert "◇" in val
        assert "Анализ" in val
        assert "🧠" not in val
        assert "🤔" not in val

    def test_rethink_uses_glyph_not_emoji(self):
        out = StringIO()
        with patch("ui.console.print", side_effect=lambda *args, **kw: out.write(str(args[0]) + "\n")):
            print_rethink("Let me double check", duration=1.5)
        val = out.getvalue()
        assert "◈" in val
        assert "Перепроверка" in val
        assert "🤔" not in val

    def test_approval_uses_glyphs(self):
        assert "✓" in _ALWAYS_PREFIX
        assert "✅" not in _ALWAYS_PREFIX
        assert "❌" not in _ALWAYS_PREFIX


class TestNetworkErrorExplanations:
    def test_wormsoft_404_chat_completions(self):
        msg = "Error code: 404 - {'message': 'Cannot POST /api/v1/chat/completions', 'error': 'Not Found'}"
        explanation = explain_error(msg)
        assert explanation is not None
        assert "/v1/chat/completions" in explanation or "api_base" in explanation

    def test_ollama_dns_failure(self):
        msg = "failed to connect: dial tcp: lookup ollama.lan: no such host"
        explanation = explain_error(msg)
        assert explanation is not None
        assert "Ollama" in explanation or "DNS" in explanation

    def test_server_502_bad_gateway(self):
        msg = "HTTP 502 Bad Gateway from upstream model proxy"
        explanation = explain_error(msg)
        assert explanation is not None
        assert "502" in explanation or "Шлюз" in explanation or "прокси" in explanation
