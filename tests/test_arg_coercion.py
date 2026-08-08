"""Giving each tool the argument types its schema promised.

Models emit JSON slightly wrong all the time, and Python does not complain
until the failure is already dishonest. Every case below was measured against
the real tools before it was fixed:

* write_file(overwrite="false") destroyed a 400-line file that
  overwrite=False protects — a non-empty string is truthy, so `not overwrite`
  turned the guard off;
* grep_search(max_results="3") reported "No matches found" for a pattern with
  three matches;
* read_file(start_line="5") leaked "'>' not supported between instances of
  'str' and 'int'" at a model that cannot act on it.
"""

import logging

import pytest

from src.agent import arg_coercion
from src.agent.arg_coercion import coerce_and_log, coerce_arguments


def _c(tool, **args):
    return coerce_arguments(tool, args)[0]


class TestTheDangerousOne:
    def test_a_string_false_disarms_a_guard(self):
        """The whole reason this module exists."""
        assert _c("write_file", file_path="x", content="y",
                  overwrite="false")["overwrite"] is False

    @pytest.mark.parametrize("text,expected", [
        ("true", True), ("True", True), ("TRUE", True), ("yes", True), ("1", True),
        ("false", False), ("False", False), ("no", False), ("0", False),
        ("null", False), ("none", False),
    ])
    def test_the_spellings_that_show_up(self, text, expected):
        assert _c("write_file", file_path="x", content="y",
                  overwrite=text)["overwrite"] is expected

    def test_an_ambiguous_string_is_left_for_the_tool_to_reject(self):
        """Guessing here would be worse than the tool's own error message."""
        assert _c("write_file", file_path="x", content="y",
                  overwrite="maybe")["overwrite"] == "maybe"

    def test_a_real_boolean_is_untouched(self):
        assert _c("write_file", file_path="x", content="y",
                  overwrite=True)["overwrite"] is True


class TestNumbers:
    def test_an_integer_sent_as_text(self):
        assert _c("grep_search", directory=".", pattern="p",
                  max_results="3")["max_results"] == 3

    def test_a_whole_float_is_an_integer(self):
        assert _c("read_file", file_path="x", start_line="10.0")["start_line"] == 10

    def test_a_fractional_value_is_not_silently_truncated(self):
        """Rounding somebody's 2.5 to 2 invents an intention they did not have."""
        assert _c("read_file", file_path="x", start_line="2.5")["start_line"] == "2.5"

    def test_nonsense_is_passed_through(self):
        assert _c("grep_search", directory=".", pattern="p",
                  max_results="many")["max_results"] == "many"

    def test_a_boolean_is_not_a_number(self):
        """True is not 1 here — a boolean in an integer field is a real mistake,
        and quietly turning it into 1 hides it."""
        assert _c("grep_search", directory=".", pattern="p",
                  max_results=True)["max_results"] is True


class TestArrays:
    def test_a_json_string(self):
        assert _c("filter_new_items", items='["a", "b"]')["items"] == ["a", "b"]

    def test_one_item_where_a_list_was_declared(self):
        assert _c("filter_new_items", items="http://a")["items"] == ["http://a"]

    def test_newline_separated(self):
        assert _c("filter_new_items", items="a\nb")["items"] == ["a", "b"]

    def test_a_real_list_is_untouched(self):
        value = ["a", "b"]
        assert _c("filter_new_items", items=value)["items"] is value

    def test_broken_json_becomes_a_single_item_not_a_crash(self):
        assert _c("filter_new_items", items='["a", ')["items"] == ['["a", ']


class TestJsonInAString:
    """A parameter named *_json declares a string that CONTAINS json, which
    invites the model to send the object itself."""

    def test_a_list_is_serialised_back(self):
        changes = [{"file_path": "a.py", "changes": []}]
        assert _c("multi_replace_in_file", changes_json=changes) == {
            "changes_json": '[{"file_path": "a.py", "changes": []}]'}

    def test_a_dict_is_serialised_back(self):
        assert _c("call_mcp_tool", server_name="unity", tool_name="t",
                  arguments_json={"type": "Scene"})["arguments_json"] == '{"type": "Scene"}'

    def test_an_ordinary_string_parameter_does_not_swallow_objects(self):
        """Only *_json invites this; turning any list into text elsewhere would
        invent an intention the model did not have."""
        value = ["line one", "line two"]
        assert _c("write_file", file_path="x", content=value)["content"] is value

    def test_a_correct_json_string_is_untouched(self):
        text = '[{"file_path": "a.py"}]'
        assert _c("multi_replace_in_file", changes_json=text)["changes_json"] is text

    def test_the_tool_then_works(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from tools.file_ops import multi_replace_in_file

        (tmp_path / "t.txt").write_text("привет мир", encoding="utf-8")
        payload = [{"file_path": "t.txt", "target_text": "мир",
                    "replacement_text": "друг"}]
        args = coerce_and_log("multi_replace_in_file", {"changes_json": payload})
        multi_replace_in_file(**args)
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "привет друг"

    def test_the_wrong_shape_is_diagnosed_honestly(self, tmp_path, monkeypatch):
        """A nested {"changes": [...]} entry leaves target_text empty. That used
        to be reported as "the target appears 11 times", with advice to make the
        target MORE distinctive — the opposite of the real problem."""
        monkeypatch.chdir(tmp_path)
        from tools.file_ops import multi_replace_in_file

        (tmp_path / "t.txt").write_text("привет мир", encoding="utf-8")
        payload = [{"file_path": "t.txt",
                    "changes": [{"target_text": "мир", "replacement_text": "друг"}]}]
        out = multi_replace_in_file(**coerce_and_log(
            "multi_replace_in_file", {"changes_json": payload}))
        assert "'target_text' is empty" in out
        assert "flat" in out                       # says what the shape must be
        assert (tmp_path / "t.txt").read_text(encoding="utf-8") == "привет мир"


class TestUntouched:
    def test_a_tool_with_no_schema(self):
        args = {"anything": "5"}
        assert coerce_arguments("not_a_tool", args) == (args, [])

    def test_a_parameter_the_schema_does_not_declare(self):
        assert _c("write_file", file_path="x", content="y",
                  mystery="7")["mystery"] == "7"

    def test_strings_stay_strings(self):
        assert _c("write_file", file_path="x", content="y")["content"] == "y"

    def test_a_number_in_a_string_field_becomes_text(self):
        assert _c("write_file", file_path=5, content="y")["file_path"] == "5"

    def test_empty_and_non_dict_input(self):
        assert coerce_arguments("write_file", {}) == ({}, [])
        assert coerce_arguments("write_file", None) == (None, [])


class TestRecord:
    def test_a_repair_is_written_down(self, caplog):
        """A tool whose arguments are always repaired is a tool whose
        description needs rewriting — invisible without the record."""
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            coerce_and_log("write_file", {"file_path": "x", "content": "y",
                                          "overwrite": "false"})
        assert "coerced 1 argument(s) for write_file" in caplog.text
        assert "overwrite='false'->False" in caplog.text

    def test_a_clean_call_is_silent(self, caplog):
        with caplog.at_level(logging.INFO, logger=arg_coercion.log.name):
            coerce_and_log("write_file", {"file_path": "x", "content": "y",
                                          "overwrite": False})
        assert "coerced" not in caplog.text


class TestAgainstTheRealTools:
    """End-to-end: the coerced arguments must actually restore the behaviour."""

    def test_the_overwrite_guard_holds_again(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from tools.file_ops import write_file

        target = tmp_path / "notes.md"
        target.write_text("\n".join(f"line {i}" for i in range(400)), encoding="utf-8")

        args = coerce_and_log("write_file", {"file_path": "notes.md",
                                             "content": "WIPED",
                                             "overwrite": "false"})
        result = write_file(**args)
        assert "already exists" in result
        assert len(target.read_text(encoding="utf-8").splitlines()) == 400

    def test_grep_finds_what_is_there(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from tools.search_ops import grep_search

        (tmp_path / "a.txt").write_text("needle\nneedle\nneedle\n", encoding="utf-8")
        args = coerce_and_log("grep_search", {"directory": str(tmp_path),
                                              "pattern": "needle",
                                              "max_results": "3"})
        assert "No matches found" not in grep_search(**args)

    def test_read_file_does_not_leak_a_typeerror(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from tools.file_ops import read_file

        (tmp_path / "a.txt").write_text("\n".join(str(i) for i in range(20)),
                                        encoding="utf-8")
        args = coerce_and_log("read_file", {"file_path": "a.txt",
                                            "start_line": "5", "end_line": "8"})
        out = read_file(**args)
        assert "not supported between instances" not in out
