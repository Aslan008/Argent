"""Rooms: the `git` node (a real repository in a temp folder, a local bare repository as the "server")
and `save_as` (keeping a step's result under a name), plus an agent's answer as its output.
"""

import subprocess

import pytest

from src.rooms import gitnode
from src.rooms.engine import replayable
from src.rooms.integrations import Integrations
from src.rooms.models import Node, Room, State
from src.rooms.runner import NodeRunner
from src.rooms.rules import Rules
from src.rooms.validator import room_warnings, validate_room


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git("init", "-q", "-b", "main", cwd=tmp_path)
    git("config", "user.email", "t@example.com", cwd=tmp_path)
    git("config", "user.name", "tester", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    git("add", "-A", cwd=tmp_path)
    git("commit", "-q", "-m", "first", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def step(mode, message=None, **args):
    return Node(id="g", type="git", mode=mode, message=message, args=args)


def run_step(node, state=None, **kw):
    state = state or State()
    NodeRunner(integrations=Integrations(), **kw).run_node(node, state)
    return state


class TestModel:
    def test_modes_and_required_fields(self):
        assert Node(id="g", type="git").mode is None                   # means "status"
        with pytest.raises(ValueError):
            Node(id="g", type="git", mode="commit")                     # a commit has a message
        with pytest.raises(ValueError):
            Node(id="g", type="git", mode="checkout")                   # and a checkout a branch
        with pytest.raises(ValueError):
            Node(id="g", type="git", mode="explode")

    def test_there_is_no_force(self):
        with pytest.raises(ValueError, match="never forces"):
            Node(id="g", type="git", mode="push", args={"force": True})

    def test_confirm_and_sets_are_allowed_and_looking_is_repeatable(self):
        n = Node(id="g", type="git", mode="push", confirm=True, sets={"ok_push": "ok"})
        assert n.confirm and not n.idempotent                           # changing: never replayed blindly
        assert Node(id="g", type="git", mode="status").idempotent
        assert replayable(Node(id="g", type="git", mode="log"))
        assert not replayable(Node(id="g", type="git", mode="commit", message="m"))

    def test_save_as_is_a_plain_name_on_a_step_with_a_result(self):
        assert Node(id="t", type="tool", tool="run_command", save_as="out").save_as == "out"
        with pytest.raises(ValueError):
            Node(id="t", type="tool", tool="run_command", save_as="my out")
        with pytest.raises(ValueError):
            Node(id="s", type="set", values={"a": 1}, save_as="x")


class TestGitModes:
    def test_status_clean_and_dirty(self, repo):
        st = run_step(step("status"))
        assert st.data["g.clean"] is True and st.data["g.branch"] == "main" and st.data["g.changed"] == 0
        (repo / "b.txt").write_text("new\n", encoding="utf-8")
        (repo / "a.txt").write_text("two\n", encoding="utf-8")
        st = run_step(step("status"))
        assert st.data["g.clean"] is False and st.data["g.changed"] == 2
        assert sorted(st.data["g.files"]) == ["a.txt", "b.txt"] and st.data["g.untracked"] == ["b.txt"]

    def test_diff_counts_lines(self, repo):
        (repo / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        st = run_step(step("diff"))
        assert st.data["g.files"] == ["a.txt"] and st.data["g.added"] == 2 and st.data["g.changed"] is True
        assert "a.txt" in st.data["g.output"]
        git("checkout", "--", "a.txt", cwd=repo)
        assert run_step(step("diff")).data["g.changed"] is False

    def test_commit_stages_everything_by_default_and_reports_it(self, repo):
        (repo / "b.txt").write_text("new\n", encoding="utf-8")
        st = run_step(step("commit", "add b"))
        assert st.data["g.committed"] is True and st.data["g.commit"] and st.data["g.branch"] == "main"
        assert git("log", "-1", "--format=%s", cwd=repo).strip() == "add b"
        again = run_step(step("commit", "nothing"))
        assert again.data["g.committed"] is False and "Nothing to commit" in again.data["g.output"]

    def test_commit_only_staged(self, repo):
        (repo / "b.txt").write_text("b\n", encoding="utf-8")
        (repo / "c.txt").write_text("c\n", encoding="utf-8")
        git("add", "b.txt", cwd=repo)
        run_step(step("commit", "only b", only_staged=True))
        assert git("show", "--name-only", "--format=", "HEAD", cwd=repo).split() == ["b.txt"]

    def test_a_message_is_never_a_command(self, repo):
        (repo / "b.txt").write_text("b\n", encoding="utf-8")
        run_step(step("commit", "x'; touch pwned; echo '"))
        assert not (repo / "pwned").exists()
        assert git("log", "-1", "--format=%s", cwd=repo).strip() == "x'; touch pwned; echo '"

    def test_log_branch_checkout_stash(self, repo):
        st = run_step(step("log", count=5))
        assert st.data["g.count"] == 1 and st.data["g.last"] == "first"
        st = run_step(step("branch", name="feature"))
        assert st.data["g.branch"] == "feature" and "feature" in st.data["g.branches"]
        assert run_step(step("checkout", branch="main")).data["g.branch"] == "main"
        (repo / "a.txt").write_text("changed\n", encoding="utf-8")
        assert run_step(step("stash")).data["g.stashed"] is True
        assert run_step(step("status")).data["g.clean"] is True
        run_step(step("stash", action="pop"))
        assert run_step(step("status")).data["g.clean"] is False

    def test_push_and_pull_with_a_local_server(self, repo, tmp_path_factory):
        server = tmp_path_factory.mktemp("server")
        git("init", "-q", "--bare", "-b", "main", cwd=server)
        git("remote", "add", "origin", str(server), cwd=repo)
        st = run_step(step("push"))
        assert st.data["g.pushed"] is True
        clone = tmp_path_factory.mktemp("clone")
        git("clone", "-q", str(server), str(clone), cwd=clone.parent)
        git("config", "user.email", "t@example.com", cwd=clone)
        git("config", "user.name", "t", cwd=clone)
        (clone / "z.txt").write_text("z\n", encoding="utf-8")
        git("add", "-A", cwd=clone)
        git("commit", "-q", "-m", "from clone", cwd=clone)
        git("push", "-q", "origin", "main", cwd=clone)
        st = run_step(step("pull"))
        assert st.data["g.updated"] is True and (repo / "z.txt").exists()

    def test_outside_a_repository_is_an_error_not_a_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        st = run_step(step("status"))
        assert st.data["g.output"].startswith("Error:") and st.data["g.clean"] is False

    def test_a_ref_that_looks_like_an_option_is_refused(self, repo):
        st = run_step(step("diff", against="--output=/tmp/x"))
        assert st.data["g.output"].startswith("Error:")
        assert run_step(step("branch", name="-D main")).data["g.output"].startswith("Error:")


class TestRunnerWiring:
    def test_confirm_declined_means_it_does_not_run(self, repo):
        (repo / "b.txt").write_text("b\n", encoding="utf-8")
        node = Node(id="g", type="git", mode="commit", message="m", confirm=True)
        st = run_step(node, confirm=lambda n, s: False)
        assert git("log", "--format=%s", cwd=repo).split() == ["first"]
        assert st.data.get("g.declined") is True

    def test_derived_values_work_on_the_report(self, repo):
        node = Node(id="g", type="git", mode="status", sets={"is_ok": "ok", "n": "count"})
        assert run_step(node).data["is_ok"] is True

    def test_save_as_keeps_the_report_and_the_rules_apply(self, repo):
        node = Node(id="g", type="git", mode="log", save_as="history")
        assert "first" in run_step(node).data["history"]
        rules = Rules({"read_only": True})
        assert rules.node_block(Node(id="p", type="git", mode="push")) is not None
        assert rules.node_block(node) is None
        assert Rules({"deny_tools": ["git"]}).node_block(node) is not None

    def test_a_disabled_step_saves_nothing(self, repo):
        node = Node(id="g", type="git", mode="log", save_as="history", disabled=True)
        assert run_step(node).data["history"] == ""

    def test_tool_result_is_kept_under_the_name(self):
        class Tool:
            def __call__(self, node, state):
                return "hello"
        node = Node(id="t", type="tool", tool="run_command", args={"command": "x"}, save_as="greeting")
        st = State()
        NodeRunner(tool_executor=Tool()).run_node(node, st)
        assert st.data["greeting"] == "hello"

    def test_an_agent_with_no_writes_just_answers(self):
        class Agent:
            last_text = ""
            last_tokens = 0
            accepts_brief = False

            def __call__(self, node, context):
                self.last_text = "The bug is in parse()."
                return {}
        node = Node(id="think", type="agent", role="Researcher", budget={"max_iterations": 2, "max_tokens": 100},
                    instruction="Find it", save_as="answer")
        st = State()
        NodeRunner(agent_executor=Agent()).run_node(node, st)
        assert st.data["think.output"] == "The bug is in parse()." and st.data["answer"] == "The bug is in parse()."


class TestValidation:
    def room(self, *nodes):
        edges = [{"from": nodes[-1].id, "to": "EXIT:done"}]
        for a, b in zip(nodes, nodes[1:]):
            edges.append({"from": a.id, "to": b.id})
        return Room(room="t", budget={"max_iterations": 20, "max_tokens": 1000}, nodes=list(nodes), edges=edges)

    def test_push_without_asking_is_noted(self):
        room = self.room(Node(id="p", type="git", mode="push"))
        assert validate_room(room, {"run_command"}) == []
        assert any("without asking" in w for w in room_warnings(room))
        asked = self.room(Node(id="p", type="git", mode="push", confirm=True))
        assert not any("without asking" in w for w in room_warnings(asked))

    def test_two_steps_saving_under_one_name_are_noted(self):
        room = self.room(Node(id="a", type="git", mode="log", save_as="x"), Node(id="b", type="git", mode="status", save_as="x"))
        assert any("overwrites" in w for w in room_warnings(room))


class TestConditionPaths:
    """`state.<step>.<field>` reads: how an arrow looks at what a step left (git's `clean`, an answer's `output`)."""

    def test_a_step_field_is_a_plain_lookup(self):
        from src.rooms import condition as C
        data = {"g.clean": True, "g.changed": 3, "agent.output": "yes please", "cfg": {"mode": "fast", "list": [4, 5]}}
        assert C.evaluate("state.g.clean", data) is True
        assert C.evaluate("state.g.changed >= 3 and not state.g.missing", data) is True
        assert C.evaluate("state.agent.output == 'yes please'", data) is True
        assert C.evaluate("state.cfg.mode == 'fast'", data) is True          # inside a dict the state holds

    def test_still_nothing_that_runs_or_reaches_out(self):
        from src.rooms import condition as C
        for bad in ("state.g.clean()", "state.g.output.upper() == 'X'", "g.clean", "os.system", "state['g']", "__import__('os')"):
            with pytest.raises(C.ConditionError):
                C.validate(bad)
        # dunder names are just keys: nothing is looked up on the object itself
        assert C.evaluate("state.__class__.__mro__", {}) is False

    def test_a_room_branches_on_what_git_left(self, repo):
        from src.rooms.engine import Engine
        room = Room(room="t", budget={"max_iterations": 10, "max_tokens": 1000},
                    nodes=[Node(id="look", type="git", mode="status")],
                    edges=[{"from": "look", "if": "state.look.clean", "to": "EXIT:clean"},
                           {"from": "look", "else": True, "to": "EXIT:dirty"}])
        assert validate_room(room, set()) == []
        runner = NodeRunner(integrations=Integrations())
        assert Engine().run(room, State(), runner.run_node).exit == "clean"
        (repo / "x.txt").write_text("x\n", encoding="utf-8")
        assert Engine().run(room, State(), runner.run_node).exit == "dirty"
