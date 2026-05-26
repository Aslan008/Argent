import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent.parser import (
    extract_balanced_json,
    repair_json_strings,
    try_parse_json_tool,
    parse_raw_tool_call
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

if __name__ == "__main__":
    unittest.main()
