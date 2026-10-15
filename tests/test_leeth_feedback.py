"""
tests/test_leeth_feedback.py — Active-Learning Feedback Loop Tests for Leeth.

Covers: decision logging, verdict submission, LeethSample conversion,
review-response parsing, dispatcher decision-logging integration, and an
end-to-end fine-tune smoke run on an isolated copy of the data directory.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

from src.leeth import feedback


@pytest.fixture()
def fb(tmp_path, monkeypatch, _leeth_feedback_off):
    """Isolated feedback store, forced ON (bypasses the global test disable)."""
    monkeypatch.setenv("LEETH_FEEDBACK_PATH", str(tmp_path / "fb.jsonl"))
    monkeypatch.delenv("LEETH_FEEDBACK_DISABLED", raising=False)
    monkeypatch.setenv("LEETH_FEEDBACK_FORCE", "1")
    yield feedback


ROUTINE_STATE = 'Incoming user request in Argent session:\n"прочитай config.py"\nWorking directory: C:/Users/user/Desktop/Argent'


@pytest.fixture(autouse=True)
def _leeth_cpu():
    """Force the Leeth runtime onto CPU/ONNX: the GPU may be busy with the
    main Argent process, and CUDA OOM must never fail a logic test.

    Reloads at most ONCE per pytest session: each reload of the 400M
    checkpoint peaks at ~3 GB RAM (checkpoint copy + model), and 16
    per-test reloads exhausted laptop memory (native crash in
    nn.Linear.__init__). With the INT8 ONNX present the first reload is
    cheap and every later test reuses the same session.

    Also pins the PRODUCTION router (mmBERT): the dispatcher tests below
    assert per-query dispatch outcomes calibrated against it. Without the
    pin, the router mode leaks here from legacy-pinned modules that run
    earlier alphabetically (test_leeth_fast_dispatch.py).
    """
    from src.leeth.runtime import leeth
    old_router = os.environ.get("LEETH_ROUTER")
    os.environ["LEETH_ROUTER"] = "mmbert"
    if not (leeth.device == "onnx-cpu" and leeth.ort_session is not None and leeth.mmbert_active):
        leeth.reload(force_device="onnx")
        leeth._test_reload_done = True
    assert leeth.mmbert_active, "feedback dispatcher tests need the mmBERT router"
    yield
    if old_router is None:
        os.environ.pop("LEETH_ROUTER", None)
    else:
        os.environ["LEETH_ROUTER"] = old_router


def _routine_proposal():
    return {
        "need_tool": {"value": True, "probability": 1.0},
        "tool": {"choice": "read_file", "confidence": 0.986},
        "param": "config.py",
    }


def _conversational_proposal():
    return {
        "need_tool": {"value": False, "probability": 0.1},
        "tool": {"choice": "NO_TOOL", "confidence": 0.97},
        "param": "",
    }


def test_decision_roundtrip_and_verdict(fb):
    did = fb.log_decision("прочитай config.py", ROUTINE_STATE, _routine_proposal())
    assert did
    rec = fb.get_record(did)
    assert rec["verdict"] is None
    assert rec["query"] == "прочитай config.py"
    assert fb.pending_count() == 1

    assert fb.submit_verdict(did, "accept", source="agent", reason="ok")
    rec = fb.get_record(did)
    assert rec["verdict"] == "accept"
    assert rec["verdict_source"] == "agent"
    assert fb.pending_count() == 0
    assert len(fb.labeled_records()) == 1


def test_record_to_samples_accept(fb):
    did = fb.log_decision(
        "прочитай config.py", ROUTINE_STATE, _routine_proposal(),
        dispatch={"tool_name": "read_file", "arguments": {"file_path": "config.py"}},
    )
    fb.submit_verdict(did, "accept", source="agent")
    samples = fb.record_to_samples(fb.get_record(did))
    kinds = [(s.question_type, s.target_noul, s.target_choice, s.target_span_text) for s in samples]
    assert ("noul", 1.0, None, None) in kinds
    assert ("choice", None, "read_file", None) in kinds
    assert ("extract", None, None, "config.py") in kinds


def test_record_to_samples_partial_correction(fb):
    did = fb.log_decision(
        "какие файлы в scripts", ROUTINE_STATE, _routine_proposal(),
        dispatch={"tool_name": "read_file", "arguments": {"file_path": "scripts"}},
    )
    fb.submit_verdict(did, "partial", correction_tool="list_directory", correction_param="scripts", source="agent")
    samples = fb.record_to_samples(fb.get_record(did))
    kinds = [(s.question_type, s.target_choice, s.target_span_text) for s in samples]
    assert ("choice", "list_directory", None) in kinds
    assert ("extract", None, "scripts") in kinds


def test_record_to_samples_reject_conversational(fb):
    did = fb.log_decision("как дела?", ROUTINE_STATE, _conversational_proposal())
    fb.submit_verdict(did, "reject", source="agent", reason="conversational")
    samples = fb.record_to_samples(fb.get_record(did))
    kinds = [(s.question_type, s.target_noul, s.target_choice) for s in samples]
    assert ("noul", 0.0, None) in kinds
    assert ("choice", None, "NO_TOOL") in kinds
    assert not any(s.question_type == "extract" for s in samples)


def test_record_to_samples_reject_dangerous_with_safety(fb):
    safety_state = "User shell execution request:\nCommand: rm -rf ./x\nCaller: Argent Agent"
    did = fb.log_decision(
        "запусти rm -rf ./x", ROUTINE_STATE, {
            "need_tool": {"value": True, "probability": 1.0},
            "tool": {"choice": "run_command", "confidence": 0.9},
            "param": "rm -rf ./x",
        },
        dispatch={"tool_name": "run_command", "arguments": {"command": "rm -rf ./x"}},
        safety={"state": safety_state, "is_safe_prob": 0.99, "action": "execute_immediately"},
    )
    fb.submit_verdict(did, "reject", source="agent", reason="destructive")
    samples = fb.record_to_samples(fb.get_record(did))
    safety_samples = [s for s in samples if s.state == safety_state]
    assert any(s.question_type == "noul" and s.target_noul == 0.0 for s in safety_samples)
    assert any(s.question_type == "choice" and s.target_choice == "block_command" for s in safety_samples)


def test_parse_review_response(fb):
    assert fb.parse_review_response('{"verdict": "accept", "reason": "ok"}')["verdict"] == "accept"
    fenced = (
        "```json\n{\"verdict\": \"partial\", \"correction_tool\": \"list_directory\", "
        "\"correction_param\": \"src\", \"reason\": \"wrong tool\"}\n```"
    )
    v = fb.parse_review_response(fenced)
    assert v["verdict"] == "partial" and v["correction_tool"] == "list_directory"
    assert fb.parse_review_response("мусор без json") is None
    assert fb.parse_review_response('{"verdict": "maybe"}') is None


def test_dispatcher_logs_decision(fb):
    from src.leeth.dispatcher import dispatcher
    res = dispatcher.try_fast_dispatch("прочитай config.py")
    assert res is not None and res.decision_id
    rec = fb.get_record(res.decision_id)
    assert rec["proposal"]["tool"]["choice"] == "read_file"
    assert rec["verdict"] is None  # pending until reviewed


def test_dispatcher_logs_skipped_conversational(fb):
    from src.leeth.dispatcher import dispatcher
    res = dispatcher.try_fast_dispatch("Привет, как твои дела?")
    assert res is None
    assert any(r.get("skip_reason") == "need_tool_false" for r in fb.load_records())


def test_build_training_samples(fb):
    did1 = fb.log_decision(
        "прочитай config.py", ROUTINE_STATE, _routine_proposal(),
        dispatch={"tool_name": "read_file", "arguments": {"file_path": "config.py"}},
    )
    fb.submit_verdict(did1, "accept", source="agent")
    did2 = fb.log_decision("как дела?", ROUTINE_STATE, _conversational_proposal())
    fb.submit_verdict(did2, "reject", source="agent")
    train, val, info = fb.build_training_samples()
    assert info["samples"] >= 4
    assert (train or val)


def test_finetune_smoke(fb, tmp_path, monkeypatch):
    """End-to-end proof of real learning: gradients flow, weights change,
    gate evaluates, checkpoint saves, ONNX re-exports — all on isolated copies.

    Builds a tiny model with the REAL vocab/classes instead of copying the
    363M-param production checkpoint: the CPU Adam step on the full model
    natively OOMs on dev machines (access violation, no Python traceback).
    The full-size training path is exercised by train.py in production.
    """
    import torch
    from dataclasses import asdict

    from src.leeth import finetune
    from src.leeth.model import LeethConfig, LeethModel

    real_data = Path("src/leeth/data")
    tmp_data = tmp_path / "data"
    tmp_data.mkdir()
    shutil.copy(real_data / "leeth_vocab.json", tmp_data / "leeth_vocab.json")
    shutil.copy(real_data / "choice_classes.json", tmp_data / "choice_classes.json")
    _copy_lines(real_data / "leeth_val.jsonl", tmp_data / "leeth_val.jsonl", 120)
    _copy_lines(real_data / "leeth_train.jsonl", tmp_data / "leeth_train.jsonl", 200)

    with open(real_data / "choice_classes.json", "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    with open(real_data / "leeth_vocab.json", "r", encoding="utf-8") as f:
        vocab_size = max(json.load(f).values()) + 8
    cfg = LeethConfig(vocab_size=vocab_size, hidden_size=64, num_layers=2,
                      num_heads=4, intermediate_size=128,
                      num_choices=len(class_to_idx))
    model = LeethModel(cfg)
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx, "val_loss": 9.9},
               tmp_data / "leeth_best.pt")

    monkeypatch.setattr(finetune, "DATA_DIR", tmp_data)
    monkeypatch.setattr(finetune, "BEST_PATH", tmp_data / "leeth_best.pt")
    monkeypatch.setattr(finetune, "CLASSES_PATH", tmp_data / "choice_classes.json")
    monkeypatch.setattr(finetune, "SYNTH_VAL_PATH", tmp_data / "leeth_val.jsonl")
    monkeypatch.setattr(finetune, "SYNTH_TRAIN_PATH", tmp_data / "leeth_train.jsonl")

    did = fb.log_decision(
        "прочитай config.py", ROUTINE_STATE, _routine_proposal(),
        dispatch={"tool_name": "read_file", "arguments": {"file_path": "config.py"}},
    )
    fb.submit_verdict(did, "accept", source="agent")

    summary = finetune.finetune_from_feedback(epochs=1, min_samples=1, replay_max=10, batch_size=2, device="cpu")
    assert summary["weights_changed"] is True
    assert summary["status"] in ("accepted", "rejected")
    assert summary["onnx_reexported"] is True
    # A versioned artifact must exist either way.
    assert any(tmp_data.glob("leeth_best_v*.pt")) or any(tmp_data.glob("leeth_candidate_v*.pt"))


def _copy_lines(src: Path, dst: Path, count: int) -> None:
    with open(src, "r", encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= count:
                break
            f_out.write(line)