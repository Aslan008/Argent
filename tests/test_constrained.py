import json

from src.agent.constrained import build_step_schema, build_tool_catalog, StepStreamExtractor


def feed_in_chunks(extractor, text, size):
    out = ""
    for i in range(0, len(text), size):
        out += extractor.feed(text[i:i + size])
    return out


class TestSchema:
    def test_schema_has_two_branches(self):
        schema = build_step_schema()
        assert len(schema["anyOf"]) == 2
        assert schema["anyOf"][0]["required"] == ["reply"]
        assert schema["anyOf"][1]["required"] == ["tool"]

    def test_schema_is_json_serializable(self):
        json.dumps(build_step_schema())


class TestReplyStreaming:
    def test_reply_text_streams_incrementally(self):
        ex = StepStreamExtractor()
        raw = '{"reply": "Hello world"}'
        emitted = feed_in_chunks(ex, raw, 3)
        assert emitted == "Hello world"
        assert ex.mode == "reply"

    def test_single_chunk(self):
        ex = StepStreamExtractor()
        assert ex.feed('{"reply": "ok"}') == "ok"

    def test_escapes_decoded(self):
        ex = StepStreamExtractor()
        raw = '{"reply": "line1\\nline2 \\"quoted\\" \\\\ end"}'
        emitted = feed_in_chunks(ex, raw, 1)
        assert emitted == 'line1\nline2 "quoted" \\ end'

    def test_unicode_escape_split_across_chunks(self):
        ex = StepStreamExtractor()
        raw = '{"reply": "\\u041f\\u0440\\u0438\\u0432\\u0435\\u0442"}'
        emitted = feed_in_chunks(ex, raw, 2)
        assert emitted == "Привет"

    def test_text_after_closing_quote_not_emitted(self):
        ex = StepStreamExtractor()
        emitted = feed_in_chunks(ex, '{"reply": "done"}  ', 4)
        assert emitted == "done"

    def test_finalize_reply(self):
        ex = StepStreamExtractor()
        feed_in_chunks(ex, '{"reply": "answer"}', 5)
        assert ex.finalize() == {"reply": "answer"}


class TestToolStreaming:
    def test_tool_mode_emits_nothing(self):
        ex = StepStreamExtractor()
        raw = '{"tool": {"name": "read_file", "arguments": {"file_path": "a.py"}}}'
        assert feed_in_chunks(ex, raw, 4) == ""
        assert ex.mode == "tool"

    def test_finalize_tool(self):
        ex = StepStreamExtractor()
        raw = '{"tool": {"name": "run_command", "arguments": {"command": "dir"}}}'
        feed_in_chunks(ex, raw, 7)
        step = ex.finalize()
        assert step == {"tool": {"name": "run_command", "arguments": {"command": "dir"}}}

    def test_finalize_flattened_tool_shape(self):
        # Some models flatten the branch: {"name": ..., "arguments": ...}
        ex = StepStreamExtractor()
        ex.feed('{"name": "read_file", "arguments": {"file_path": "x"}}')
        step = ex.finalize()
        assert step["tool"]["name"] == "read_file"

    def test_finalize_tool_with_non_dict_arguments(self):
        ex = StepStreamExtractor()
        ex.feed('{"tool": {"name": "t", "arguments": "oops"}}')
        assert ex.finalize() == {"tool": {"name": "t", "arguments": {}}}

    def test_finalize_garbage_returns_none(self):
        ex = StepStreamExtractor()
        ex.feed("complete nonsense")
        assert ex.finalize() is None

    def test_finalize_empty_returns_none(self):
        assert StepStreamExtractor().finalize() is None


class TestToolCatalog:
    SCHEMAS = [
        {"function": {
            "name": "read_file",
            "description": "Read a file. Second sentence dropped.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["file_path"],
            },
        }},
        {"function": {"name": "noop", "description": "", "parameters": {}}},
    ]

    def test_catalog_lines(self):
        catalog = build_tool_catalog(self.SCHEMAS)
        assert "- read_file(file_path, limit?) — Read a file" in catalog
        assert "Second sentence" not in catalog
        assert "- noop()" in catalog

    def test_catalog_header_mentions_exact_names(self):
        assert "EXACT names" in build_tool_catalog(self.SCHEMAS)
