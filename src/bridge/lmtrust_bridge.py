"""LMTrust Local Agent Bridge — Zero-API Testing Oracle Protocol.

Enables local AI agents (Argent, Claude Code, OpenCode, scripts) to submit
blind-spot testing requests and receive full LMTrust test suites from the
active Antigravity chat session without external cloud APIs.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _get_bridge_dir(base_dir: str | Path | None = None) -> Path:
    """Get or create the .argent/lmtrust_bridge directory."""
    if base_dir is None:
        base_dir = Path.cwd()
    bridge_dir = Path(base_dir) / ".argent" / "lmtrust_bridge"
    (bridge_dir / "inbox").mkdir(parents=True, exist_ok=True)
    (bridge_dir / "outbox").mkdir(parents=True, exist_ok=True)
    return bridge_dir


def create_test_request(
    source_file: str | Path,
    focus_layers: list[str] | str | None = None,
    note: str = "",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a new blind-spot testing request in the inbox."""
    bridge_dir = _get_bridge_dir(base_dir)
    src_path = Path(source_file)
    if not src_path.is_absolute():
        src_path = Path(base_dir or Path.cwd()) / src_path

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    source_code = src_path.read_text(encoding="utf-8", errors="replace")

    # Look for existing tests
    test_candidates = [
        Path(base_dir or Path.cwd()) / "tests" / f"test_{src_path.name}",
        Path(base_dir or Path.cwd()) / "tests" / f"test_lmtrust_{src_path.name}",
        Path(base_dir or Path.cwd()) / "tests" / f"test_{src_path.stem}.py",
        Path(base_dir or Path.cwd()) / "tests" / f"test_lmtrust_{src_path.stem}.py",
    ]
    existing_tests = ""
    for cand in test_candidates:
        if cand.exists():
            existing_tests += f"=== {cand.name} ===\n" + cand.read_text(encoding="utf-8", errors="replace") + "\n\n"

    if isinstance(focus_layers, str):
        layers = [l.strip() for l in focus_layers.split(",") if l.strip()]
    elif isinstance(focus_layers, list):
        layers = focus_layers
    else:
        layers = ["L1", "L2", "L3", "L4", "L4b", "L7", "L9"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_stem = re.sub(r"[^\w\-]", "_", src_path.stem)
    req_id = f"req_{timestamp}_{clean_stem}"

    payload = {
        "id": req_id,
        "created_at": datetime.now().isoformat(),
        "source_file": str(src_path),
        "source_stem": src_path.stem,
        "source_code": source_code,
        "existing_tests": existing_tests.strip(),
        "focus_layers": layers,
        "note": note,
        "status": "pending",
    }

    req_file = bridge_dir / "inbox" / f"{req_id}.json"
    req_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def list_pending_requests(base_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List all requests in the inbox that do not yet have an outbox response."""
    bridge_dir = _get_bridge_dir(base_dir)
    inbox = bridge_dir / "inbox"
    outbox = bridge_dir / "outbox"

    results = []
    for req_file in inbox.glob("*.json"):
        try:
            data = json.loads(req_file.read_text(encoding="utf-8"))
            resp_file = outbox / f"{data['id']}_response.json"
            if not resp_file.exists():
                results.append(data)
        except Exception:
            continue
    return sorted(results, key=lambda x: x.get("created_at", ""), reverse=True)


def submit_test_response(
    request_id: str,
    test_code: str,
    analysis_markdown: str = "",
    coverage_summary: str = "",
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Submit generated tests and analysis report to the outbox."""
    bridge_dir = _get_bridge_dir(base_dir)
    resp_id = f"{request_id}_response"

    payload = {
        "request_id": request_id,
        "completed_at": datetime.now().isoformat(),
        "test_code": test_code,
        "analysis_markdown": analysis_markdown,
        "coverage_summary": coverage_summary,
        "status": "completed",
    }

    resp_file = bridge_dir / "outbox" / f"{resp_id}.json"
    resp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def get_test_response(request_id: str, base_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Retrieve completed response for a request ID if available."""
    bridge_dir = _get_bridge_dir(base_dir)
    resp_file = bridge_dir / "outbox" / f"{request_id}_response.json"
    if not resp_file.exists():
        return None
    try:
        return json.loads(resp_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def apply_test_response(
    request_id: str,
    target_file: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> str:
    """Apply generated test code to the project's tests directory."""
    resp = get_test_response(request_id, base_dir=base_dir)
    if not resp:
        return f"Response for request '{request_id}' not found in outbox."

    test_code = resp.get("test_code", "")
    if not test_code.strip():
        return f"Response for request '{request_id}' contains empty test code."

    bridge_dir = _get_bridge_dir(base_dir)
    req_file = bridge_dir / "inbox" / f"{request_id}.json"
    source_stem = "module"
    if req_file.exists():
        try:
            req_data = json.loads(req_file.read_text(encoding="utf-8"))
            source_stem = req_data.get("source_stem", "module")
        except Exception:
            pass

    if target_file is None:
        target_path = Path(base_dir or Path.cwd()) / "tests" / f"test_lmtrust_{source_stem}.py"
    else:
        target_path = Path(target_file)
        if not target_path.is_absolute():
            target_path = Path(base_dir or Path.cwd()) / target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(test_code, encoding="utf-8")
    return f"Applied LMTrust tests to: {target_path}"
