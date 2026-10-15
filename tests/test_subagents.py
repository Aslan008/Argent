"""Sub-agents that run beside the supervisor: worktrees, the process manager, the tools, the worker."""

import json
import subprocess
import sys
import time
from itertools import islice
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import agent as agent_module
import approval
from src.agent import permissions, subagents, wait_conditions, worktrees
from src.agent.permissions import Permissions
from src.agent.plan_mode import PLAN, violation
from src.agent.subagents import MAX_RUNNING, SubagentManager, describe, inherited_authority, table, tools_for


def git(repo, *args):
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@argent.local")
    git(root, "config", "user.name", "T")
    (root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "util.py").write_text("X = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(worktrees.tempfile, "tempdir", str(tmp_path / "tmp"))      # worktrees and spec folders land here
    monkeypatch.chdir(root)
    return root


# ── worktrees ──────────────────────────────────────────────────────────────────

class TestWorktrees:
    def test_the_copy_starts_from_the_tree_as_it_is_now_including_new_files(self, repo):
        (repo / "app.py").write_text("def add(a, b):\n    return a + b + 0\n", encoding="utf-8")
        (repo / "fresh.py").write_text("NEW = True\n", encoding="utf-8")
        wt = worktrees.create("t", repo)
        try:
            assert (wt.path / "app.py").read_text(encoding="utf-8").endswith("a + b + 0\n")
            assert (wt.path / "fresh.py").read_text(encoding="utf-8") == "NEW = True\n"
            assert wt.path != repo and wt.exists()
            assert worktrees.changes(wt) == "" and worktrees.changed_files(wt) == []   # nothing done yet
        finally:
            worktrees.remove(wt)

    def test_the_users_tree_and_stash_are_untouched(self, repo):
        (repo / "app.py").write_text("dirty\n", encoding="utf-8")
        before = git(repo, "status", "--porcelain")
        wt = worktrees.create("t", repo)
        (wt.path / "app.py").write_text("changed in the copy\n", encoding="utf-8")
        try:
            assert (repo / "app.py").read_text(encoding="utf-8") == "dirty\n"
            assert git(repo, "status", "--porcelain") == before
            assert git(repo, "stash", "list") == ""
        finally:
            worktrees.remove(wt)

    def test_the_patch_holds_only_what_the_subagent_did_and_applies_back(self, repo):
        (repo / "fresh.py").write_text("NEW = True\n", encoding="utf-8")
        wt = worktrees.create("t", repo)
        (wt.path / "util.py").write_text("X = 2\n", encoding="utf-8")
        (wt.path / "added.py").write_text("ADDED = 1\n", encoding="utf-8")
        patch = worktrees.changes(wt)
        assert sorted(worktrees.changed_files(wt)) == ["added.py", "util.py"]
        assert "fresh.py" not in patch                 # copied in as the starting point, not "work"
        patch_file = repo / "p.patch"
        patch_file.write_text(patch, encoding="utf-8", newline="\n")
        worktrees.remove(wt)
        assert worktrees.apply_patch(repo, patch_file) == ""
        assert (repo / "util.py").read_text(encoding="utf-8") == "X = 2\n"
        assert (repo / "added.py").read_text(encoding="utf-8") == "ADDED = 1\n"

    def test_a_patch_that_does_not_apply_changes_nothing(self, repo):
        wt = worktrees.create("t", repo)
        (wt.path / "util.py").write_text("X = 2\n", encoding="utf-8")
        (wt.path / "app.py").write_text("changed\n", encoding="utf-8")
        patch_file = repo / "p.patch"
        patch_file.write_text(worktrees.changes(wt), encoding="utf-8", newline="\n")
        worktrees.remove(wt)
        (repo / "util.py").write_text("X = 99\n", encoding="utf-8")          # the user moved on: conflict
        problem = worktrees.apply_patch(repo, patch_file)
        assert problem
        assert (repo / "app.py").read_text(encoding="utf-8").startswith("def add")   # all-or-nothing

    def test_remove_leaves_no_folder_and_no_git_record(self, repo):
        wt = worktrees.create("t", repo)
        worktrees.remove(wt)
        assert not wt.path.exists()
        assert str(wt.path).replace("\\", "/") not in git(repo, "worktree", "list").replace("\\", "/")

    def test_it_needs_a_repository_with_a_commit(self, tmp_path, monkeypatch):
        plain = tmp_path / "plain"
        plain.mkdir()
        with pytest.raises(worktrees.WorktreeError, match="not one"):
            worktrees.create("t", plain)
        empty = tmp_path / "empty"
        empty.mkdir()
        git(empty, "init", "-q")
        with pytest.raises(worktrees.WorktreeError, match="at least one commit"):
            worktrees.create("t", empty)


# ── the manager, with a stand-in child process ─────────────────────────────────

CHILD = r'''
import json, os, sys, time
spec = json.load(open(sys.argv[1], encoding="utf-8"))
task = spec["task"]
print("child started in", os.getcwd())
if task.startswith("sleep:"):
    time.sleep(float(task.split(":")[1]))
if task.startswith("write:"):
    _, name, text = task.split(":", 2)
    open(name, "w", encoding="utf-8").write(text)
if task == "crash":
    sys.exit(3)
if task == "lane":
    print("ARGENT_SUBAGENT_RESULT " + json.dumps({"status": "ok", "report": os.environ.get("ARGENT_LANE_PRIORITY", ""), "denied": []}))
    sys.exit(0)
if task == "dump":
    print("ARGENT_SUBAGENT_RESULT " + json.dumps({"status": "ok", "report": json.dumps(spec), "denied": []}))
    sys.exit(0)
denied = [{"action": "run: rm x", "destructive": True}] if task == "denied" else []
print("ARGENT_SUBAGENT_RESULT " + json.dumps({"status": "ok", "report": "report of " + task, "denied": denied}))
'''


@pytest.fixture
def manager(tmp_path, monkeypatch):
    script = tmp_path / "fake_child.py"
    script.write_text(CHILD, encoding="utf-8")
    mgr = SubagentManager(argv_builder=lambda spec_path: [sys.executable, str(script), str(spec_path)])
    monkeypatch.setattr(subagents, "MANAGER", mgr)
    monkeypatch.setattr(permissions, "PERMISSIONS", Permissions(project_dir=tmp_path / "proj", home=tmp_path / "home"))
    yield mgr
    mgr.stop_all()


def finish(mgr, run, timeout=30):
    assert mgr.wait(run, timeout), "the sub-agent never finished"
    return run


class TestManager:
    def test_a_child_queues_for_the_model_behind_the_supervisor(self, manager):
        run = finish(manager, manager.spawn("Researcher", "lane", cwd=Path.cwd()))
        assert run.report == "2", "a worker process must carry the worker priority of the model's queue"

    def test_a_run_ends_with_the_childs_report(self, manager):
        run = finish(manager, manager.spawn("Researcher", "look", cwd=Path.cwd()))
        assert run.status == "done" and run.report == "report of look" and not run.running
        assert "report of look" in describe(run) and "finished" in describe(run)

    def test_spawn_returns_at_once_and_the_run_is_visible_while_it_works(self, manager):
        started = time.time()
        run = manager.spawn("Researcher", "sleep:3", cwd=Path.cwd())
        assert time.time() - started < 2.5 and run.running
        assert not manager.is_finished(run.id) and "running" in table(manager.all())
        manager.stop(run)

    def test_several_run_at_the_same_time_but_not_more_than_the_limit(self, manager):
        runs = [manager.spawn("Researcher", "sleep:5", cwd=Path.cwd()) for _ in range(MAX_RUNNING)]
        assert len(manager.running()) == MAX_RUNNING
        with pytest.raises(ValueError, match="already running"):
            manager.spawn("Researcher", "one too many", cwd=Path.cwd())
        manager.stop_all()
        assert all(finish(manager, r).status == "stopped" for r in runs)

    def test_a_child_that_dies_without_a_report_is_a_failure_with_its_last_output(self, manager):
        run = finish(manager, manager.spawn("Researcher", "crash", cwd=Path.cwd()))
        assert run.status == "failed" and "without a report" in run.error and "child started" in run.error

    def test_stop_kills_it(self, manager):
        run = manager.spawn("Researcher", "sleep:60", cwd=Path.cwd())
        assert manager.stop(run) and not manager.stop(run)
        finish(manager, run)
        assert run.status == "stopped" and "stopped" in describe(run)

    def test_refused_actions_are_reported(self, manager):
        run = finish(manager, manager.spawn("Coder", "denied", cwd=Path.cwd()))
        assert "REFUSED" in describe(run) and "rm x" in describe(run)

    def test_an_unknown_id_counts_as_finished_so_a_wait_cannot_hang(self, manager):
        assert manager.is_finished(999) and manager.is_finished("nope")

    def test_the_wait_condition_sees_the_run_end(self, manager):
        run = manager.spawn("Researcher", "sleep:1", cwd=Path.cwd())
        cond = f'subagent_finished("{run.id}")'
        assert wait_conditions.evaluate(cond) is False
        finish(manager, run)
        assert wait_conditions.evaluate(cond) is True
        assert wait_conditions.wait_for(cond, timeout=5, poll_seconds=0.1)["met"]

    def test_isolation_none_works_in_the_real_tree(self, manager, repo):
        run = finish(manager, manager.spawn("Coder", "write:direct.txt:hi", cwd=repo))
        assert (repo / "direct.txt").exists() and run.patch is None and run.worktree is None

    def test_a_worktree_run_leaves_a_patch_and_the_real_tree_alone(self, manager, repo):
        run = finish(manager, manager.spawn("Coder", "write:made.txt:hello", isolation="worktree", cwd=repo))
        assert run.status == "done" and run.files == ["made.txt"]
        assert not (repo / "made.txt").exists()                       # the user's tree is untouched
        assert run.patch.is_file() and "made.txt" in run.patch.read_text(encoding="utf-8")
        assert run.worktree is not None and run.worktree.exists()     # kept until applied or dropped
        assert "isolated copy" in describe(run) and 'action="apply"' in describe(run)

    def test_applying_brings_the_work_over_and_removes_the_copy(self, manager, repo):
        run = finish(manager, manager.spawn("Coder", "write:made.txt:hello", isolation="worktree", cwd=repo))
        path = run.worktree.path
        assert manager.apply(run) == ""
        assert (repo / "made.txt").read_text(encoding="utf-8") == "hello"
        assert run.disposition == "applied" and not path.exists() and run.worktree is None
        assert manager.apply(run) == "already applied"

    def test_discarding_drops_the_copy_and_the_patch(self, manager, repo):
        run = finish(manager, manager.spawn("Coder", "write:made.txt:hello", isolation="worktree", cwd=repo))
        patch, path = run.patch, run.worktree.path
        manager.discard(run)
        assert not patch.exists() and not path.exists() and not (repo / "made.txt").exists()
        assert "discarded" in describe(run)

    def test_a_worktree_run_that_changed_nothing_leaves_no_copy(self, manager, repo):
        run = finish(manager, manager.spawn("Reviewer", "just look", isolation="worktree", cwd=repo))
        assert run.worktree is None and run.patch is None and "no changes" in describe(run)
        assert str(Path(run.cwd)).replace("\\", "/") not in git(repo, "worktree", "list").replace("\\", "/")

    def test_what_the_child_is_told(self, manager, repo, monkeypatch):
        PLAN.enter()
        approval._session_grants.add("git")
        Permissions(project_dir=repo, home=repo.parent / "home").add("allow", "run_command(pytest*)", "global")
        Permissions(project_dir=repo, home=repo.parent / "home").add("deny", "write_file(.env*)", "global")
        monkeypatch.setattr(permissions, "PERMISSIONS", Permissions(project_dir=repo, home=repo.parent / "home"))
        try:
            run = finish(manager, manager.spawn("Coder", "dump", tools=["read_file", "run_subagent", "ask_user_questions"],
                                                cwd=repo))
        finally:
            approval.clear_session_grants()
        spec = json.loads(run.report)
        assert spec["plan_mode"] is True
        assert spec["tools"] == ["read_file"]                         # forbidden tools are stripped
        assert "pytest*" in spec["allowed_commands"] and "git *" in spec["allowed_commands"]
        assert spec["deny"] == ["write_file(.env*)"]


class TestToolsAndAuthority:
    def test_roles_get_their_own_tools_and_never_the_forbidden_ones(self):
        assert "write_file" not in tools_for("Researcher") and "write_file" in tools_for("Coder")
        assert tools_for("Critic") == [] and "run_command" not in tools_for("Reviewer")
        assert tools_for("SomethingNew") == tools_for("Researcher")
        for role in ("Coder", "Researcher", "Reviewer", "DocWriter", "BrowserWorker"):
            assert not (set(tools_for(role)) & subagents.FORBIDDEN_TOOLS)
        assert tools_for("Coder", ["run_subagent", "read_file", "wait_heartbeat"]) == ["read_file"]

    def test_authority_is_only_what_the_person_already_gave(self, repo, monkeypatch):
        monkeypatch.setattr(permissions, "PERMISSIONS", Permissions(project_dir=repo, home=repo.parent / "home"))
        assert inherited_authority() == {"allowed_commands": [], "deny": [], "ask": []}
        permissions.PERMISSIONS.add("allow", "run_command(npm test*)")
        permissions.PERMISSIONS.add("allow", "write_file(docs/*)")        # file edits are not gated; not a command allow
        permissions.PERMISSIONS.add("ask", "run_command(git push*)")
        assert inherited_authority()["allowed_commands"] == ["npm test*"]
        assert inherited_authority()["ask"] == ["run_command(git push*)"]


class TestSubagentTools:
    def test_a_background_run_tells_the_model_how_to_get_the_result(self, manager, repo):
        from tools.misc_tools import run_subagent, subagents as subagents_tool
        out = run_subagent("Researcher", "look around", background=True)
        assert "started in the background" in out and "subagent_finished" in out
        run = manager.all()[0]
        finish(manager, run)
        assert "report of look around" in subagents_tool(action="result", id=run.id)

    def test_in_the_tui_a_watch_wakes_the_model_when_it_ends(self, manager, repo):
        from src.agent.watches import WATCHES
        from tools.misc_tools import run_subagent
        woken = []
        WATCHES.set_handler(lambda message, watch: woken.append(message))
        out = run_subagent("Researcher", "sleep:1", background=True)
        assert "woken" in out and len(WATCHES.pending()) == 1
        finish(manager, manager.all()[0])
        for _ in range(60):
            if WATCHES.check_once() or woken:
                break
            time.sleep(0.1)
        assert woken and "subagent_finished" in woken[0] and "Условие выполнено" in woken[0]

    def test_a_foreground_isolated_run_blocks_and_returns_the_report_and_patch(self, manager, repo):
        from tools.misc_tools import run_subagent, subagents as subagents_tool
        out = run_subagent("Coder", "write:made.txt:hello", isolation="worktree")
        assert "report of write" in out and "made.txt" in out
        assert not (repo / "made.txt").exists()
        assert "Applied" in subagents_tool(action="apply", id=1) and (repo / "made.txt").exists()

    def test_wait_true_blocks_until_the_run_ends(self, manager, repo):
        from tools.misc_tools import run_subagent, subagents as subagents_tool
        run_subagent("Researcher", "sleep:1", background=True)
        started = time.time()
        out = subagents_tool(action="result", id=1, wait=True, timeout_seconds=30)
        assert "finished" in out and time.time() - started >= 0.5

    def test_bad_requests_are_answers_not_exceptions(self, manager, repo):
        from tools.misc_tools import run_subagent, subagents as subagents_tool
        assert run_subagent("Coder", "x", isolation="sideways", background=True).startswith("Error")
        assert "no sub-agent #9" in subagents_tool(action="result", id=9)
        assert subagents_tool(action="list") == "No sub-agents in this session."
        assert subagents_tool(action="explode", id=1).startswith("Error")

    def test_isolation_without_a_repository_is_a_clear_error(self, manager, tmp_path, monkeypatch):
        from tools.misc_tools import run_subagent
        plain = tmp_path / "plain"
        plain.mkdir()
        monkeypatch.chdir(plain)
        assert "not one" in run_subagent("Coder", "x", isolation="worktree")


class TestPlanMode:
    def test_only_readers_may_run_while_planning(self):
        PLAN.enter()
        assert violation("run_subagent", {"role": "Coder", "task": "x"})
        assert violation("run_subagent", {"role": "Researcher", "task": "x", "isolation": "worktree"})
        assert violation("run_subagent", {"role": "Researcher", "task": "x"}) is None
        assert violation("subagents", {"action": "apply", "id": 1})
        assert violation("subagents", {"action": "list"}) is None


# ── the worker itself ──────────────────────────────────────────────────────────

class ScriptedProvider:
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def validate_config(self):
        return None

    def supports_constrained_decoding(self):
        return False

    def stream_chat(self, **kwargs):
        if kwargs.get("tools"):
            from providers import ProviderError
            raise ProviderError("HTTP 404: No endpoints found that support tool use.")
        script = self.scripts[min(self.calls, len(self.scripts) - 1)]
        self.calls += 1
        yield from script

    def format_tool_result(self, content, tool_call_id=None):
        return {"role": "tool", "content": content}


@pytest.fixture
def worker_env(monkeypatch, repo):
    def setup(provider):
        mem = MagicMock()
        mem.data = {}
        monkeypatch.setattr(agent_module, "memory", mem)
        monkeypatch.setattr(agent_module, "get_mcp_servers", lambda: [])
        monkeypatch.setattr(agent_module, "create_provider", lambda *a, **k: provider)
        monkeypatch.setattr(agent_module, "estimate_tokens", lambda t, m, p: len(t) // 4)
        monkeypatch.setattr(permissions, "PERMISSIONS", permissions.PERMISSIONS)   # restored afterwards
    yield setup
    approval.reset_approval_backend()


def call(name, **args):
    return [{"content": json.dumps({"name": name, "arguments": args})}]


def test_the_worker_runs_an_agent_under_the_unattended_gate(worker_env, repo):
    from src.agent import subagent_worker
    worker_env(ScriptedProvider([
        call("run_command", command="echo not-allowed"),
        call("run_command", command="echo allowed-one"),
        [{"content": "All done."}],
    ]))
    result = subagent_worker.run({
        "role": "Coder", "task": "do things", "cwd": str(repo), "tools": ["run_command", "read_file"],
        "allowed_commands": ["echo allowed-one"], "deny": [], "ask": [],
    })
    assert result["status"] == "ok" and result["report"].strip().endswith("All done.")
    assert [d["action"] for d in result["denied"]] == [f"{approval.RUN_COMMAND_ACTION}echo not-allowed"]
    assert result["ran"] == ["echo allowed-one"]


def test_the_worker_applies_the_parents_deny_rules_and_plan_mode(worker_env, repo):
    from src.agent import subagent_worker
    worker_env(ScriptedProvider([
        call("write_file", file_path=".env", content="x"),
        call("write_file", file_path="other.txt", content="y"),
        [{"content": "ok"}],
    ]))
    result = subagent_worker.run({
        "role": "Coder", "task": "t", "cwd": str(repo), "tools": ["write_file"],
        "allowed_commands": [], "deny": ["write_file(.env*)"], "ask": [], "plan_mode": False,
    })
    assert not (repo / ".env").exists() and (repo / "other.txt").exists()
    PLAN.leave()

    worker_env(ScriptedProvider([call("write_file", file_path="third.txt", content="z"), [{"content": "ok"}]]))
    subagent_worker.run({"role": "Coder", "task": "t", "cwd": str(repo), "tools": ["write_file"],
                         "allowed_commands": [], "deny": [], "ask": [], "plan_mode": True})
    assert not (repo / "third.txt").exists()
    assert PLAN.active                                     # the child inherited plan mode; the test cleans up


def test_a_subagent_never_goes_through_the_fast_path():
    from agent import ArgentSubAgent
    assert ArgentSubAgent.fast_path_allowed is False


def test_the_in_process_subagent_keeps_its_own_checklist(worker_env, repo):
    from src.agent import todos
    from agent import ArgentSubAgent
    worker_env(ScriptedProvider([call("todo_write", todos=[{"content": "child step", "status": "in_progress"}]),
                                 [{"content": "ok"}]]))
    todos.write(["supervisor step"])
    ArgentSubAgent("Coder", "t", tools_override=["todo_write"]).execute()
    assert [t.content for t in todos.TODOS.get()] == ["supervisor step"]


def test_the_subagent_entry_point_reports_a_broken_spec(tmp_path):
    done = subprocess.run([sys.executable, str(Path(subagents.__file__).with_name("subagent_worker.py")),
                           str(tmp_path / "missing.json")], capture_output=True, text=True)
    assert done.returncode == 2 and "cannot read the spec" in done.stderr


def test_a_broad_grant_does_not_override_an_ask_rule_in_the_child(worker_env, repo):
    """The person allowed `echo *` (a session grant) but asked to be consulted about `echo guarded*`."""
    from src.agent import subagent_worker
    worker_env(ScriptedProvider([
        call("run_command", command="echo guarded one"),
        call("run_command", command="echo free one"),
        [{"content": "ok"}],
    ]))
    result = subagent_worker.run({
        "role": "Coder", "task": "t", "cwd": str(repo), "tools": ["run_command"],
        "allowed_commands": ["echo *"], "deny": [], "ask": ["run_command(echo guarded*)"],
    })
    assert [d["action"] for d in result["denied"]] == [f"{approval.RUN_COMMAND_ACTION}echo guarded one"]
    assert result["ran"] == ["echo free one"]


def test_fetching_the_report_cancels_the_wake_up_for_that_run(manager, repo):
    from src.agent.watches import WATCHES
    from tools.misc_tools import run_subagent, subagents as subagents_tool
    WATCHES.set_handler(lambda message, watch: None)
    run_subagent("Researcher", "sleep:1", background=True)
    assert len(WATCHES.pending()) == 1
    subagents_tool(action="result", id=1, wait=True, timeout_seconds=30)
    assert WATCHES.pending() == []
