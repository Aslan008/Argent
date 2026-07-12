"""Verbatim ```write_file``` block — file content with zero JSON escaping."""

from src.agent.parser import parse_fenced_write, parse_raw_tool_call


def test_verbatim_roundtrip_with_backslashes_and_quotes():
    body = 'using UnityEngine;\npublic class Foo {\n    string p = "C:\\Users\\x";\n}'
    text = f"```write_file C:/proj/Foo.cs\n{body}\n```"
    r = parse_fenced_write(text)
    assert r["parsed"]["name"] == "write_file"
    assert r["parsed"]["arguments"]["file_path"] == "C:/proj/Foo.cs"
    assert r["parsed"]["arguments"]["content"] == body   # taken verbatim, no unescaping


def test_windows_backslash_path_preserved():
    r = parse_fenced_write("```write_file C:\\Users\\a\\b.cs\nx\n```")
    assert r["parsed"]["arguments"]["file_path"] == "C:\\Users\\a\\b.cs"


def test_routes_through_raw_parser():
    r = parse_raw_tool_call("```write_file a.txt\nhello\nworld\n```")
    assert r["parsed"]["name"] == "write_file"
    assert r["parsed"]["arguments"]["content"] == "hello\nworld"


def test_does_not_misfire_on_ordinary_code_block():
    assert parse_fenced_write("intro\n```python\nprint('hi')\n```") is None
    # …and the raw parser doesn't treat it as a write.
    r = parse_raw_tool_call("```python\nprint('hi')\n```")
    assert r is None or r["parsed"]["name"] != "write_file"


def test_markdown_file_with_inner_code_fence_is_captured_whole():
    # The truncation bug: content that itself contains a ``` block used to be
    # cut at the first inner fence. It must now survive to the last fence.
    body = "# Notes\n\n```python\nprint(1)\n```\n\nMore text after the block."
    text = f"```write_file notes.md\n{body}\n```"
    r = parse_fenced_write(text)
    assert r is not None
    assert r["parsed"]["arguments"]["file_path"] == "notes.md"
    assert r["parsed"]["arguments"]["content"] == body


def test_trailing_prose_after_block_is_not_swallowed():
    text = "```write_file a.py\nx = 1\n```\nDone — I wrote the file."
    r = parse_fenced_write(text)
    assert r["parsed"]["arguments"]["content"] == "x = 1"


def test_missing_path_returns_none():
    assert parse_fenced_write("```write_file \ncontent\n```") is None


def test_absent_marker_returns_none():
    assert parse_fenced_write('{"name": "write_file", "arguments": {}}') is None
