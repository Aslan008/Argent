"""Tests for LMTrust Agent Bridge — Local zero-API testing protocol."""

import json
from pathlib import Path
import pytest

from src.bridge.lmtrust_bridge import (
    create_test_request,
    list_pending_requests,
    submit_test_response,
    get_test_response,
    apply_test_response,
)
from tools.misc_tools import request_blind_spot_tests


@pytest.mark.lmtrust(layer="L1", direction="D7")
def test_create_test_request_valid(tmp_path):
    src_file = tmp_path / "sample.py"
    src_file.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    req = create_test_request(src_file, focus_layers="L1,L4", note="Test add", base_dir=tmp_path)
    assert req["id"].startswith("req_")
    assert req["source_stem"] == "sample"
    assert req["focus_layers"] == ["L1", "L4"]
    assert "def add(a, b):" in req["source_code"]

    inbox_file = tmp_path / ".argent" / "lmtrust_bridge" / "inbox" / f"{req['id']}.json"
    assert inbox_file.exists()


@pytest.mark.lmtrust(layer="L4b", direction="D4")
def test_create_test_request_nonexistent_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_test_request(tmp_path / "ghost.py", base_dir=tmp_path)


@pytest.mark.lmtrust(layer="L5", direction="D2")
def test_submit_and_get_response(tmp_path):
    src_file = tmp_path / "logic.py"
    src_file.write_text("def compute(): pass", encoding="utf-8")
    req = create_test_request(src_file, base_dir=tmp_path)

    assert len(list_pending_requests(base_dir=tmp_path)) == 1

    submit_test_response(
        req["id"],
        test_code="def test_compute(): assert True",
        analysis_markdown="# Report",
        base_dir=tmp_path,
    )

    # Now should not be pending
    assert len(list_pending_requests(base_dir=tmp_path)) == 0

    resp = get_test_response(req["id"], base_dir=tmp_path)
    assert resp is not None
    assert "def test_compute(): assert True" in resp["test_code"]
    assert resp["analysis_markdown"] == "# Report"


@pytest.mark.lmtrust(layer="L1", direction="D7")
def test_apply_test_response(tmp_path):
    src_file = tmp_path / "math_ops.py"
    src_file.write_text("def sub(a, b): return a - b", encoding="utf-8")
    req = create_test_request(src_file, base_dir=tmp_path)

    submit_test_response(
        req["id"],
        test_code="def test_sub(): assert sub(5, 3) == 2",
        base_dir=tmp_path,
    )

    msg = apply_test_response(req["id"], base_dir=tmp_path)
    assert "Applied LMTrust tests to:" in msg

    applied_file = tmp_path / "tests" / "test_lmtrust_math_ops.py"
    assert applied_file.exists()
    assert "def test_sub(): assert sub(5, 3) == 2" in applied_file.read_text(encoding="utf-8")


@pytest.mark.lmtrust(layer="L7", direction="D8")
def test_tool_request_blind_spot_tests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src_file = tmp_path / "target.py"
    src_file.write_text("def target_fn(): return 42", encoding="utf-8")

    result = request_blind_spot_tests("target.py", focus_layers="L1,L2,L9")
    assert "LMTrust testing request created successfully" in result
    assert "target.py" in result
    assert "L1, L2, L9" in result
