import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.parser import (
    extract_balanced_json,
    repair_json_strings,
    try_parse_json_tool,
    parse_raw_tool_call,
    decode_json_escapes,
    try_recover_malformed_tool,
)

class TestAgentParser(unittest.TestCase):
    def test_extract_balanced_json(self):
        text = 'some prefix {"name": "test", "arguments": {"a": 1}} suffix'
        start = text.find('{')
        extracted = extract_balanced_json(text, start)
        self.assertEqual(extracted, '{"name": "test", "arguments": {"a": 1}}')

        # Test nested braces
        nested = '{"outer": {"inner": {}}}'
        self.assertEqual(extract_balanced_json(nested, 0), nested)

        # Test incomplete braces
        incomplete = '{"incomplete": {'
        self.assertIsNone(extract_balanced_json(incomplete, 0))

    def test_repair_json_strings(self):
        # Escapes raw newlines in string values
        json_with_newline = '{"content": "line 1\nline 2"}'
        repaired = repair_json_strings(json_with_newline)
        self.assertEqual(repaired, '{"content": "line 1\\nline 2"}')

    def test_try_parse_json_tool(self):
        # Test standard format
        valid_json = '{"name": "read_file", "arguments": {"file_path": "test.txt"}}'
        parsed = try_parse_json_tool(valid_json)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["parsed"]["name"], "read_file")
        self.assertEqual(parsed["parsed"]["arguments"]["file_path"], "test.txt")

    def test_parse_raw_tool_call(self):
        # Markdown JSON code block format
        md_block = """Here is the tool call:
```json
{
  "name": "write_file",
  "arguments": {
    "file_path": "hello.py",
    "content": "print('hello')"
  }
}
```
"""
        parsed = parse_raw_tool_call(md_block)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["parsed"]["name"], "write_file")
        self.assertEqual(parsed["parsed"]["arguments"]["file_path"], "hello.py")

    def test_decode_escapes_preserves_cyrillic(self):
        # Regression: codecs 'unicode_escape' mojibaked non-ASCII content,
        # corrupting recovered files with Russian/Unicode text.
        self.assertEqual(decode_json_escapes("Привет\\nмир"), "Привет\nмир")
        self.assertEqual(decode_json_escapes("café\\ttab"), "café\ttab")
        self.assertNotIn("Ð", decode_json_escapes("Документ"))

    def test_decode_escapes_unicode_and_backslash(self):
        self.assertEqual(decode_json_escapes("\\u0041\\u0042"), "AB")
        self.assertEqual(decode_json_escapes("a\\\\b"), "a\\b")  # \\ -> one backslash

    def test_shorthand_tool_format(self):
        parsed = try_parse_json_tool('{"read_file": {"file_path": "a.txt"}}')
        self.assertEqual(parsed["parsed"]["name"], "read_file")
        self.assertEqual(parsed["parsed"]["arguments"]["file_path"], "a.txt")

    def test_param_alias_normalization(self):
        parsed = try_parse_json_tool('{"name":"read_file","arguments":{"filename":"a.txt"}}')
        self.assertEqual(parsed["parsed"]["arguments"]["file_path"], "a.txt")
        self.assertNotIn("filename", parsed["parsed"]["arguments"])

    def test_trailing_comma_recovered(self):
        parsed = try_parse_json_tool('{"name":"read_file","arguments":{"file_path":"a.txt",}}')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["parsed"]["arguments"]["file_path"], "a.txt")

    def test_json_embedded_in_prose(self):
        text = 'Sure! {"name": "read_file", "arguments": {"file_path": "x.py"}} done.'
        parsed = parse_raw_tool_call(text)
        self.assertEqual(parsed["parsed"]["name"], "read_file")

    def test_recover_malformed_write_keeps_cyrillic(self):
        raw = '{"name": "write_file", "arguments": {"file_path": "z.txt", "content": "Привет\\nмир"}}'
        result = try_recover_malformed_tool(raw, "write_file")
        self.assertIsNotNone(result)
        content = result["parsed"]["arguments"]["content"]
        self.assertIn("Привет", content)
        self.assertNotIn("Ð", content)  # no mojibake

if __name__ == "__main__":
    unittest.main()
