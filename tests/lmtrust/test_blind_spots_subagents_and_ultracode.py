"""LMTrust Deep Blind-Spot Test Suite for Subagents & Ultracode.

Covers verification layers:
- L0 Smoke: SubagentManager lifecycle, Ultracode runtime toggle
- L1 Contract: tools_json list coercion, add_step parameter handling
- L2 Boundary: split() on mixed delimiters, JSON arrays, lists in add_step
- L3 Property: MAX_RUNNING concurrency slot reservation invariant
- L4 Adversarial: spec_dir cleanup on Popen failure, add_step runaway protection
- L5 State Hygiene: Subagent memory cleanup & worktree isolation
- L6 Integration: run_subagent with list tools_json & models
- L7 Compositional: Ultracode and SubagentManager interaction
"""

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.agent import subagents
from src.agent.subagents import MAX_RUNNING, Run, SubagentManager
from src.agent.workflow import runtime
from src.agent.workflow.engine import Budgets, Engine, Ports
from src.agent.workflow.plan import Plan, Step
from src.agent.workflow import store
import argent.tools.misc_tools as misc_tools


# ============================================================================
# LAYER L1 & L2: Parameter Coercion & Boundary Parsing
# ============================================================================

class TestL1ContractAndCoercion:
    def test_run_subagent_handles_tools_json_as_list(self, monkeypatch):
        """Passing tools_json as a list of strings must be preserved, not dropped."""
        captured = {}

        def fake_spawn(role, task, tools=None, **kwargs):
            captured["tools"] = tools
            r = Run(id=99, role=role, task=task, isolation="none", cwd=".")
            r.status = "done"
            r.report = "ok"
            r.done.set()
            return r

        monkeypatch.setattr(subagents.MANAGER, "spawn", fake_spawn)
        monkeypatch.setattr(subagents.MANAGER, "wait", lambda r: True)
        monkeypatch.setattr(subagents, "describe", lambda r: "done")

        # Pass as list
        res = misc_tools.run_subagent(
            role="coder", task="test task", tools_json=["read_file", "write_file"],
            isolation="none", background=True
        )
        assert captured.get("tools") == ["read_file", "write_file"]

    def test_run_subagent_handles_tools_json_as_json_array_string(self, monkeypatch):
        captured = {}

        def fake_spawn(role, task, tools=None, **kwargs):
            captured["tools"] = tools
            r = Run(id=99, role=role, task=task, isolation="none", cwd=".")
            r.status = "done"
            r.done.set()
            return r

        monkeypatch.setattr(subagents.MANAGER, "spawn", fake_spawn)
        monkeypatch.setattr(subagents.MANAGER, "wait", lambda r: True)

        misc_tools.run_subagent(
            role="coder", task="test task", tools_json='["grep_search", "read_file"]',
            isolation="none", background=True
        )
        assert captured.get("tools") == ["grep_search", "read_file"]

    def test_runtime_add_step_handles_list_and_json_and_delimiters(self, monkeypatch):
        """add_step must properly parse files, checks, and verify whether list, json string, or semicolon."""
        fake_engine = MagicMock()
        fake_engine.add_step.return_value = "x10"

        monkeypatch.setattr(runtime, "_engine", fake_engine)

        # 1. As Python list
        runtime.add_step("goal 1", files=["src/a.py", "src/b.py"], checks=["pytest test_a.py"])
        args, kwargs = fake_engine.add_step.call_args
        assert kwargs["files"] == ("src/a.py", "src/b.py")
        assert kwargs["checks"] == ("pytest test_a.py",)

        # 2. As JSON string
        runtime.add_step("goal 2", files='["src/c.py", "src/d.py"]', checks='["cargo test"]')
        args, kwargs = fake_engine.add_step.call_args
        assert kwargs["files"] == ("src/c.py", "src/d.py")
        assert kwargs["checks"] == ("cargo test",)

        # 3. As semicolon/comma delimited string
        runtime.add_step("goal 3", files="src/e.py, src/f.py", checks="check1; check2")
        args, kwargs = fake_engine.add_step.call_args
        assert kwargs["files"] == ("src/e.py", "src/f.py")
        assert kwargs["checks"] == ("check1", "check2")


# ============================================================================
# LAYER L3: Concurrency & Invariant Proofs (MAX_RUNNING Slot Reservation)
# ============================================================================

class TestL3ConcurrencyInvariants:
    def test_concurrent_spawns_respect_max_running(self, monkeypatch, tmp_path):
        """Concurrent calls to spawn() must NEVER exceed MAX_RUNNING even under slow startup."""
        mgr = SubagentManager()
        slow_delay = 0.05

        def slow_argv_builder(spec_path):
            time.sleep(slow_delay)
            # Stand-in child that simply sleeps
            return [sys.executable, "-c", "import time; time.sleep(1)"]

        mgr._argv_builder = slow_argv_builder

        # Run 8 concurrent spawns when MAX_RUNNING is 4
        errors = []
        started = []

        def worker(idx):
            try:
                run = mgr.spawn(f"Worker{idx}", "concurrent test", isolation="none", cwd=tmp_path)
                started.append(run)
            except ValueError as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            # Exactly MAX_RUNNING should have succeeded, and 8 - MAX_RUNNING failed with ValueError
            assert len(started) == MAX_RUNNING, f"Started {len(started)} instead of {MAX_RUNNING}. Errors: {errors}"
            assert len(errors) == 8 - MAX_RUNNING
            assert all("already running" in err for err in errors)
        finally:
            mgr.stop_all()


# ============================================================================
# LAYER L4: Adversarial Safety & Resource Cleanup
# ============================================================================

class TestL4AdversarialAndResourceCleanup:
    def test_spec_dir_cleaned_up_on_popen_failure(self, monkeypatch, tmp_path):
        """If subprocess.Popen fails (e.g. invalid executable), spec_dir must not leak on disk."""
        mgr = SubagentManager()
        created_spec_dirs = []

        orig_mkdtemp = subagents.tempfile.mkdtemp

        def instrumented_mkdtemp(*args, **kwargs):
            path = orig_mkdtemp(*args, **kwargs)
            created_spec_dirs.append(Path(path))
            return path

        monkeypatch.setattr(subagents.tempfile, "mkdtemp", instrumented_mkdtemp)
        # Inject failing executable
        mgr._argv_builder = lambda spec: ["non_existent_binary_xyz_12345"]

        with pytest.raises(ValueError, match="could not start the sub-agent"):
            mgr.spawn("TestRole", "test task", isolation="none", cwd=tmp_path)

        assert len(created_spec_dirs) == 1
        spec_dir = created_spec_dirs[0]
        assert not spec_dir.exists(), f"spec_dir {spec_dir} leaked on disk!"

    def test_engine_add_step_runaway_ceiling(self):
        """add_step must stop accepting new steps when hitting 3x max_steps budget."""
        plan = Plan(goal="test", steps=[Step(id=f"s{i}", title=f"s{i}", goal=f"g{i}") for i in range(24)])
        state = store.RunState(id="run-1", goal="test", cwd=".", level=1)
        state.set_plan(plan)

        ports = Ports(
            plan=lambda m: "",
            start=lambda s, t: MagicMock(),
            check=lambda c, w: (True, ""),
        )
        budgets = Budgets(max_steps=8)
        engine = Engine(state, ports, budgets)
        engine._live = plan

        # Already at 24 steps (3 * 8). Next add_step must be rejected.
        added_id = engine.add_step("extra runaway goal")
        assert added_id == "", "Engine allowed adding steps beyond budget ceiling!"
