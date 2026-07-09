"""Concurrent read-only tool batches: parallel when provably safe, otherwise
None so the sequential dispatch stays authoritative."""

import time

from src.agent.parallel_tools import precompute_readonly_parallel


def _call(name, **args):
    return {"function": {"name": name, "arguments": args}}


def _slow_read(file_path):
    time.sleep(0.08)
    return f"content of {file_path}"


def _tools(**extra):
    base = {
        "read_file": _slow_read,
        "grep_search": lambda directory, pattern: f"matches in {directory} for {pattern}",
        "list_directory": lambda dir_path: f"listing {dir_path}",
    }
    base.update(extra)
    return base


def test_batch_runs_concurrently_and_keeps_indexes():
    calls = [_call("read_file", file_path=f"f{i}.py") for i in range(3)]
    start = time.time()
    results = precompute_readonly_parallel(calls, _tools())
    wall = time.time() - start
    assert results == {0: "content of f0.py", 1: "content of f1.py", 2: "content of f2.py"}
    assert wall < 0.2                      # 3 x 0.08s sequential would be ~0.24s


def test_mixed_batch_with_write_falls_back():
    calls = [_call("read_file", file_path="a.py"),
             _call("write_file", file_path="b.py", content="x")]
    assert precompute_readonly_parallel(calls, _tools(write_file=lambda **k: "ok")) is None


def test_single_call_not_parallelized():
    assert precompute_readonly_parallel([_call("read_file", file_path="a.py")], _tools()) is None


def test_missing_required_arg_falls_back():
    calls = [_call("read_file", file_path="a.py"), _call("grep_search", pattern="x")]
    assert precompute_readonly_parallel(calls, _tools()) is None   # grep lacks directory


def test_unknown_tool_falls_back():
    calls = [_call("read_file", file_path="a.py"), _call("nonexistent", x=1)]
    assert precompute_readonly_parallel(calls, _tools()) is None


def test_dedup_note_short_circuits_disk():
    executed = []

    def tracked_read(file_path):
        executed.append(file_path)
        return "real read"

    calls = [_call("read_file", file_path="cached.py"),
             _call("read_file", file_path="fresh.py")]
    results = precompute_readonly_parallel(
        calls, _tools(read_file=tracked_read),
        dedup=lambda args: "[read_file] unchanged" if args["file_path"] == "cached.py" else None,
    )
    assert results[0] == "[read_file] unchanged"
    assert results[1] == "real read"
    assert executed == ["fresh.py"]        # cached one never touched the disk


def test_exception_becomes_error_result():
    def boom(file_path):
        raise RuntimeError("disk on fire")

    calls = [_call("read_file", file_path="a.py"), _call("list_directory", dir_path="d")]
    results = precompute_readonly_parallel(calls, _tools(read_file=boom))
    assert results[0].startswith("Error executing tool read_file")
    assert "disk on fire" in results[0]
    assert results[1] == "listing d"        # one failure doesn't sink the batch
