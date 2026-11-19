"""Every file tool must be callable with exactly the arguments it advertises.

The mechanics around the path are covered elsewhere: the read cache has its own suite,
cache invalidation and rehydration too, and the permission rules are exercised through
`tests/test_permissions.py`. The gap was one level up — nothing checked that a call built
**only from the schema** actually reaches the implementation and does the work.

That is the shape a model produces, and it is the shape a rename breaks quietly: the
schema says `path`, the function says `file_path`, the call arrives, the dispatcher
drops an unknown argument, and the tool answers something plausible and wrong. Signature
parity (in `test_tool_contracts`) catches the names but not the wiring — coercion,
alias repair, dispatch — which is where the path actually travels.

These tests run the real pipeline in a temporary directory: args -> `coerce_and_log`
(the layer a turn uses) -> the tool itself.
"""

import json

import pytest

import argent.tools.command_ops as command_ops
from argent.tools.misc_tools import memory
from src.agent.arg_coercion import coerce_and_log

#: The tools that take a file or directory, with the smallest arguments the schema asks
#: for. `view_image` is absent on purpose: it needs a real image, and its own suite
#: covers the decode path.
CASES = {
    "read_file": lambda p: {"path": p},
    "write_file": lambda p: {"path": p, "content": "written\n"},
    "replace_in_file": lambda p: {"path": p, "target_text": "alpha", "replacement_text": "beta"},
    "get_file_outline": lambda p: {"path": p},
    "create_directory": lambda p: {"path": p},
    "list_directory": lambda p: {"path": p},
    "search_files": lambda p: {"path": p, "pattern": "*.py"},
    "delete_file": lambda p: {"path": p},
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # `delete_file` imports the gate inside the call, so the module itself is patched.
    monkeypatch.setattr("argent.approval.request_approval", lambda *a, **k: True)
    monkeypatch.setattr(memory, "add_fact", lambda *a, **k: None)
    return tmp_path


def _call(tool: str, arguments: dict) -> str:
    """Exactly what a turn does: repair/coerce the arguments, then call the tool."""
    from argent.tools import AVAILABLE_TOOLS

    args = coerce_and_log(tool, dict(arguments))
    return AVAILABLE_TOOLS[tool](**args)


class TestEveryFileToolIsCallableAsAdvertised:
    def test_read_file(self, workspace):
        target = workspace / "a.py"
        target.write_text("alpha\n", encoding="utf-8")
        out = _call("read_file", {"path": str(target)})
        assert "alpha" in out and not out.startswith("Error")

    def test_write_file(self, workspace):
        target = workspace / "new.py"
        out = _call("write_file", {"path": str(target), "content": "alpha\n"})
        assert not out.startswith("Error")
        assert target.read_text(encoding="utf-8") == "alpha\n"

    def test_replace_in_file(self, workspace):
        target = workspace / "r.py"
        target.write_text("alpha\n", encoding="utf-8")
        out = _call("replace_in_file",
                    {"path": str(target), "target_text": "alpha", "replacement_text": "beta"})
        assert not out.startswith("Error")
        assert target.read_text(encoding="utf-8") == "beta\n"

    def test_get_file_outline(self, workspace):
        target = workspace / "o.py"
        target.write_text("class A:\n    def m(self):\n        pass\n", encoding="utf-8")
        out = _call("get_file_outline", {"path": str(target)})
        assert "A" in out and "m" in out

    def test_create_directory(self, workspace):
        target = workspace / "made"
        out = _call("create_directory", {"path": str(target)})
        assert not out.startswith("Error")
        assert target.is_dir()

    def test_list_directory(self, workspace):
        (workspace / "listed.py").write_text("x\n", encoding="utf-8")
        out = _call("list_directory", {"path": str(workspace)})
        assert "listed.py" in out

    def test_search_files(self, workspace):
        (workspace / "found.py").write_text("x\n", encoding="utf-8")
        out = _call("search_files", {"path": str(workspace), "pattern": "*.py"})
        assert "found.py" in out

    def test_delete_file(self, workspace):
        target = workspace / "gone.txt"
        target.write_text("x\n", encoding="utf-8")
        out = _call("delete_file", {"path": str(target)})
        assert not out.startswith("Error")
        assert not target.exists()

    def test_a_missing_file_is_reported_not_crashed(self, workspace):
        out = _call("read_file", {"path": str(workspace / "nope.py")})
        assert out.startswith("Error")


class TestPermissionRulesSeeThePath:
    """`subject()` in shell_hooks is what turns a tool call into the file a rule is
    about. If it stopped recognising `path`, deny rules would simply stop matching —
    everything would pass, and no test would notice."""

    def test_a_deny_rule_matches_the_folder_it_names(self, tmp_path):
        from src.agent.permissions import matches, parse_rule

        rule = parse_rule("write_file(src/**)", "deny")
        args = {"path": str(tmp_path / "src" / "app.py"), "content": "x"}
        assert matches(rule, "write_file", args, tmp_path)

    def test_a_deny_rule_does_not_match_another_folder(self, tmp_path):
        from src.agent.permissions import matches, parse_rule

        rule = parse_rule("write_file(src/**)", "deny")
        args = {"path": str(tmp_path / "docs" / "readme.md"), "content": "x"}
        assert not matches(rule, "write_file", args, tmp_path)

    def test_a_rule_without_a_pattern_covers_every_call(self, tmp_path):
        from src.agent.permissions import matches, parse_rule

        rule = parse_rule("delete_file", "ask")
        assert matches(rule, "delete_file", {"path": str(tmp_path / "a.txt")}, tmp_path)

    def test_command_rules_still_match_the_command(self, tmp_path):
        from src.agent.permissions import matches, parse_rule

        rule = parse_rule("run_command(git push*)", "ask")
        assert matches(rule, "run_command", {"command": "git push origin main"}, tmp_path)

    def test_a_multi_replace_rule_sees_the_path_inside_changes(self, tmp_path):
        """`replace_in_file` can name its files only inside `changes`. Looking at the
        top level alone let a `deny` rule on a folder be bypassed by using the batch
        form — the call simply did not match any file and went through."""
        from src.agent.permissions import matches, parse_rule

        rule = parse_rule("replace_in_file(src/**)", "deny")
        args = {
            "changes_json": json.dumps([{"path": str(tmp_path / "src" / "x.py"),
                                         "target_text": "a", "replacement_text": "b"}]),
        }
        assert matches(rule, "replace_in_file", args, tmp_path)

    def test_an_allow_rule_must_cover_every_file_in_the_batch(self, tmp_path):
        """An allow is a permission: one entry outside the allowed folder must stop it
        from approving the whole call."""
        from src.agent.permissions import matches, parse_rule

        rule = parse_rule("replace_in_file(src/**)", "allow")
        inside = {"path": str(tmp_path / "src" / "a.py"),
                  "target_text": "a", "replacement_text": "b"}
        assert matches(rule, "replace_in_file", inside, tmp_path)

        mixed = {"changes_json": json.dumps([
            {"path": str(tmp_path / "src" / "a.py"), "target_text": "a", "replacement_text": "b"},
            {"path": str(tmp_path / "docs" / "b.md"), "target_text": "a", "replacement_text": "b"},
        ])}
        assert not matches(rule, "replace_in_file", mixed, tmp_path)
