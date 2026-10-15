"""Tests for multi_replace_in_file hardening:
1. Duplicate key protection (non-empty replacement_text preserved over empty string).
2. Atomic multi-replacement on a single file (bottom-to-top execution, no line drift).
3. Overlap detection (abort without modifying file).
4. Atomicity (if one target fails, no changes are written to the file).
5. Alias mapping (target_content, replacement_content, path).
6. Missing replacement_text protection (refuses to delete code if key was omitted).
"""
import json
import pytest
from tools.file_ops import multi_replace_in_file

class TestMultiReplaceHardening:
    def test_duplicate_key_preserves_non_empty_replacement(self, tmp_path):
        """Streaming / model duplicate key: 'replacement_text': 'code' then 'replacement_text': ''
        must NEVER erase the code."""
        fp = tmp_path / "dataset.py"
        fp.write_text("for d in list_dirs:\n    loop_body()\n", encoding="utf-8")

        raw_json = json.dumps([
            {
                "file_path": str(fp),
                "target_text": "for d in list_dirs:\n    loop_body()",
                "replacement_text": "for d in list_dirs:\n    loop_body()\n    contrast_block()",
            }
        ])
        # Inject duplicate key into the raw JSON string
        duplicate_json = raw_json.replace(
            '"replacement_text": "for d in list_dirs:\\n    loop_body()\\n    contrast_block()"',
            '"replacement_text": "for d in list_dirs:\\n    loop_body()\\n    contrast_block()", "replacement_text": ""'
        )

        res = multi_replace_in_file(duplicate_json)
        assert "Multi-replace execution finished" in res
        content = fp.read_text(encoding="utf-8")
        assert "contrast_block()" in content
        assert "loop_body()" in content

    def test_atomic_multi_replace_bottom_to_top(self, tmp_path):
        """Multiple edits in same file are applied bottom-to-top without line shift corruption."""
        fp = tmp_path / "code.py"
        original = (
            "def fn():\n"
            "    # BLOCK 1\n"
            "    val1 = 1\n"
            "    # BLOCK 2\n"
            "    val2 = 2\n"
            "    return val1 + val2\n"
        )
        fp.write_text(original, encoding="utf-8")

        # Edit 1 expands block 1 with 10 lines
        # Edit 2 replaces block 2
        # If applied in naive forward order, line offsets of block 2 would drift!
        changes = json.dumps([
            {
                "file_path": str(fp),
                "target_text": "    # BLOCK 1\n    val1 = 1",
                "replacement_text": "    # EXPANDED BLOCK 1\n" + "".join(f"    v1_{i} = {i}\n" for i in range(5)) + "    val1 = 100",
            },
            {
                "file_path": str(fp),
                "target_text": "    # BLOCK 2\n    val2 = 2",
                "replacement_text": "    # MODIFIED BLOCK 2\n    val2 = 200",
            }
        ])

        res = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in res
        assert "Successfully applied 2 non-overlapping replacements" in res
        content = fp.read_text(encoding="utf-8")
        assert "EXPANDED BLOCK 1" in content
        assert "MODIFIED BLOCK 2" in content
        assert "val1 = 100" in content
        assert "val2 = 200" in content
        assert "return val1 + val2" in content

    def test_atomicity_if_one_target_fails_file_untouched(self, tmp_path):
        """If chunk 1 succeeds but chunk 2 target does not exist, the file must remain 100% untouched."""
        fp = tmp_path / "safe.py"
        original = "a = 1\nb = 2\nc = 3\n"
        fp.write_text(original, encoding="utf-8")

        changes = json.dumps([
            {"file_path": str(fp), "target_text": "a = 1", "replacement_text": "a = 100"},
            {"file_path": str(fp), "target_text": "NONEXISTENT_TEXT", "replacement_text": "z = 999"},
        ])

        res = multi_replace_in_file(changes)
        assert "target text in change #2 was not found" in res
        # File MUST NOT be partially modified
        assert fp.read_text(encoding="utf-8") == original

    def test_overlapping_edits_rejected(self, tmp_path):
        """Overlapping edits in the same file must be detected and rejected."""
        fp = tmp_path / "overlap.py"
        original = "first_line\nmiddle_line\nlast_line\n"
        fp.write_text(original, encoding="utf-8")

        changes = json.dumps([
            {"file_path": str(fp), "target_text": "first_line\nmiddle_line", "replacement_text": "A"},
            {"file_path": str(fp), "target_text": "middle_line\nlast_line", "replacement_text": "B"},
        ])

        res = multi_replace_in_file(changes)
        assert "overlap" in res.lower()
        assert fp.read_text(encoding="utf-8") == original

    def test_alias_parameter_support(self, tmp_path):
        """Supports path, target_content, and replacement_content aliases."""
        fp = tmp_path / "alias.py"
        fp.write_text("old_function()\n", encoding="utf-8")

        changes = json.dumps([
            {
                "path": str(fp),
                "target_content": "old_function()",
                "replacement_content": "new_function()",
            }
        ])

        res = multi_replace_in_file(changes)
        assert "Multi-replace execution finished" in res
        assert fp.read_text(encoding="utf-8") == "new_function()\n"

    def test_missing_replacement_text_not_defaulted_to_empty(self, tmp_path):
        """If replacement_text is omitted from change, it must NOT delete code."""
        fp = tmp_path / "nodelete.py"
        original = "important_function()\n"
        fp.write_text(original, encoding="utf-8")

        changes = json.dumps([
            {"file_path": str(fp), "target_text": "important_function()"}
        ])

        res = multi_replace_in_file(changes)
        assert "replacement_text" in res and "missing" in res.lower()
        assert fp.read_text(encoding="utf-8") == original
