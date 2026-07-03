import os
from itertools import islice
from unittest.mock import MagicMock

import agent as agent_module
from agent import ArgentAgent, MAX_SALVAGE_CONTINUES
from src.agent.salvage import (
    extract_partial_write, trim_to_last_line, last_lines, SALVAGEABLE_TOOLS,
    verify_salvaged_file,
)


class TestExtractPartialWrite:
    def test_truncated_json_string(self):
        # Realistic: arguments cut off mid-content (no closing quote/brace).
        args = '{"file_path": "index.html", "content": "<!DOCTYPE html>\\n<html>\\n<body>\\n'
        fp, content = extract_partial_write(args)
        assert fp == "index.html"
        assert content == "<!DOCTYPE html>\n<html>\n<body>\n"

    def test_decodes_escapes(self):
        args = '{"file_path": "a.py", "content": "def f():\\n    return \\"x\\"\\n'
        fp, content = extract_partial_write(args)
        assert content == 'def f():\n    return "x"\n'

    def test_drops_dangling_backslash(self):
        args = '{"file_path": "a.txt", "content": "line1\\nhalf\\'
        fp, content = extract_partial_write(args)
        assert content == "line1\nhalf"

    def test_already_parsed_dict(self):
        assert extract_partial_write({"file_path": "x", "content": "abc"}) == ("x", "abc")

    def test_alternative_path_keys(self):
        fp, _ = extract_partial_write('{"path": "x.md", "content": "hi')
        assert fp == "x.md"

    def test_no_content_returns_none(self):
        assert extract_partial_write('{"file_path": "x", "mode": "r"}') is None

    def test_no_path_returns_none(self):
        assert extract_partial_write('{"content": "orphan text') is None

    def test_empty_path_returns_none(self):
        assert extract_partial_write('{"file_path": "", "content": "x') is None


class TestTrimToLastLine:
    def test_drops_partial_last_line(self):
        assert trim_to_last_line("a\nb\nhalf-li") == "a\nb\n"

    def test_complete_trailing_newline_kept(self):
        assert trim_to_last_line("a\nb\n") == "a\nb\n"

    def test_single_line_kept(self):
        assert trim_to_last_line("only one line") == "only one line"


class TestLastLines:
    def test_returns_final_nonempty_lines(self):
        assert last_lines("a\nb\n\nc\n", n=2) == "b\nc"

    def test_fewer_lines_than_requested(self):
        assert last_lines("solo", n=3) == "solo"


class TestSalvageableTools:
    def test_includes_write_and_append(self):
        assert "write_file" in SALVAGEABLE_TOOLS
        assert "append_to_file" in SALVAGEABLE_TOOLS


class TestVerifySalvagedFile:
    def test_broken_python_flagged(self, tmp_path):
        p = tmp_path / "a.py"
        p.write_text("def foo():\n", encoding="utf-8")  # header, no body
        ok, detail = verify_salvaged_file(str(p))
        assert not ok
        assert "syntax error" in detail.lower()

    def test_valid_python_ok(self, tmp_path):
        p = tmp_path / "a.py"
        p.write_text("def foo():\n    return 1\n", encoding="utf-8")
        assert verify_salvaged_file(str(p)) == (True, "")

    def test_broken_json_flagged(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text('{"a": 1, "b":', encoding="utf-8")  # cut off
        ok, detail = verify_salvaged_file(str(p))
        assert not ok
        assert "json" in detail.lower()

    def test_valid_json_ok(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        assert verify_salvaged_file(str(p)) == (True, "")

    def test_unknown_extension_not_flagged(self, tmp_path):
        # A half-written .md/.txt has no cheap, reliable check — never cry wolf.
        p = tmp_path / "a.md"
        p.write_text("# Title\nhalf sen", encoding="utf-8")
        assert verify_salvaged_file(str(p)) == (True, "")

    def test_missing_file_flagged(self, tmp_path):
        ok, detail = verify_salvaged_file(str(tmp_path / "nope.py"))
        assert not ok
        assert "read back" in detail


class FakeProvider:
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


class TestSalvageIntegration:
    def _agent(self, monkeypatch, provider):
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        return ArgentAgent()

    def test_truncated_write_is_persisted_and_continued(self, tmp_path, monkeypatch):
        target = str(tmp_path / "big.html")
        json_path = target.replace("\\", "\\\\")
        truncated = [{
            "tool_call_deltas": [{
                "index": 0, "id": "x",
                "function_name_delta": "write_file",
                "function_arguments_delta": f'{{"file_path": "{json_path}", "content": "<html>\\n<body>\\n<p>hi</p>\\n',
            }],
            "truncated": True,
        }]
        provider = FakeProvider([truncated, [{"content": "Done."}]])
        agent = self._agent(monkeypatch, provider)

        gen = agent.process_user_input("создай большой html")
        list(islice(gen, 500))

        # The partial content was actually written to disk.
        assert os.path.exists(target)
        text = open(target, encoding="utf-8").read()
        assert "<body>" in text and "<p>hi</p>" in text
        # The model was asked to continue via append, not to start over.
        user_msgs = [m["content"] for m in agent.messages if m.get("role") == "user"]
        assert any("append_to_file" in m and "INCOMPLETE" in m for m in user_msgs)

    def test_cap_reached_reports_broken_python(self, tmp_path, monkeypatch):
        # A .py write that never completes: after the continuation cap, the file
        # on disk is a syntactically broken stub. The stop message must say so
        # rather than reporting a clean "saved".
        target = str(tmp_path / "broken.py")
        json_path = target.replace("\\", "\\\\")
        # trim_to_last_line keeps "def foo():\n" (header, no body) → won't parse.
        truncated = [{
            "tool_call_deltas": [{
                "index": 0, "id": "x",
                "function_name_delta": "write_file",
                "function_arguments_delta": f'{{"file_path": "{json_path}", "content": "def foo():\\n    x = ',
            }],
            "truncated": True,
        }]
        provider = FakeProvider([truncated])  # always truncates → hits the cap
        agent = self._agent(monkeypatch, provider)

        chunks = list(islice(agent.process_user_input("создай питон файл"), 500))
        errors = [c["content"] for c in chunks if c.get("type") == "error"]
        stop = [e for e in errors if "kept hitting the length limit" in e]
        assert stop, "expected a cap-reached stop message"
        assert "INCOMPLETE" in stop[-1]
        assert "syntax error" in stop[-1].lower()
        # Sanity: the salvage loop was actually bounded by the cap.
        assert agent._salvage_continues > MAX_SALVAGE_CONTINUES
