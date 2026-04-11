import json
import re
import sys
import os

# Create a mock ArgentAgent with the UPDATED logic
class MockAgent:
    def _normalize_tool_params(self, parsed):
        return parsed

    def _try_parse_json_tool(self, json_str):
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        
        if not isinstance(parsed, dict):
            return None
        
        # Updated logic
        tool_id = parsed.get("name") or parsed.get("function")
        if tool_id and isinstance(tool_id, str) and "arguments" in parsed:
            parsed["name"] = tool_id # Canonicalize
            parsed = self._normalize_tool_params(parsed)
            return {"parsed": parsed}
        
        return None

    def _try_recover_malformed_tool(self, content, tool_name):
        CONTENT_TOOLS = ["write_file", "write_obsidian_note", "replace_python_function", "replace_in_file"]
        if tool_name not in CONTENT_TOOLS:
            return None
        
        fp_match = re.search(
            r'"(?:file_path|filename|filepath|path|file|note_path)"\s*:\s*"([^"]+)"', content
        )
        ct_match = re.search(r'"content"\s*:\s*"([\s\S]*)', content)
        if not fp_match or not ct_match: return None
        
        recovered = ct_match.group(1)
        recovered = re.sub(r'"\s*\}\s*\}?\s*$', '', recovered)
        recovered = recovered.rstrip().rstrip('"')
        recovered = recovered.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
        
        path_param = "file_path"
        if tool_name == "write_obsidian_note":
            path_param = "note_path"
            
        return {"parsed": {"name": tool_name, "arguments": {path_param: fp_match.group(1), "content": recovered}}}

# Test cases
agent = MockAgent()

# 1. Non-standard "function" key
test_json_1 = '{"function": "write_obsidian_note", "arguments": {"note_path": "test.md", "content": "hello"}}'
res1 = agent._try_parse_json_tool(test_json_1)
print(f"Test 1 (function key): {'PASS' if res1 and res1['parsed']['name'] == 'write_obsidian_note' else 'FAIL'}")

# 2. Malformed JSON with literal newlines for write_obsidian_note
test_malformed_obsidian = '''{
    "name": "write_obsidian_note",
    "arguments": {
        "note_path": "templates/tech.md",
        "content": "Line 1
Line 2
Line 3"
    }
}'''
res2 = agent._try_recover_malformed_tool(test_malformed_obsidian, 'write_obsidian_note')
print(f"Test 2 (malformed obsidian): {'PASS' if res2 and res2['parsed']['name'] == 'write_obsidian_note' and 'Line 2' in res2['parsed']['arguments']['content'] else 'FAIL'}")
