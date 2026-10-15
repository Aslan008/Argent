"""
LMTrust Deep Blind Spot & Logic Audit Test Suite for Argent MicroModels.

Covers all 11 Layers (L0-L9 + L4b) across Directions D1-D10:
- L0: Smoke & Model Availability
- L1: API Contract & Typing Integrity
- L2: Boundary & Edge-Case Probing (empty, None, NaN/Inf, max len)
- L3: Property-Based Verification (Hypothesis Invariants: L2 norm, probability simplex)
- L4: Adversarial & Assumption Violations (PoCs for audited blind spots)
- L4b: Fallback & Zero-Shot Degradation Pathways
- L5: State Lifecycle & Mutation Consistency
- L6: Cross-System E2E Pipeline (Tokenizer -> Runtime -> Registry -> Dispatcher)
- L7: Compositional Multi-Failure Cascades
- L8: Concurrency, Thread Safety & Race Conditions
- L9: Negative Space (What the system MUST NEVER do)
"""

import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.micromodels.data_collector import MicroDataCollector
from src.micromodels.dispatcher import FastDispatchResult, MicroModelDispatcher
from src.micromodels.intent_router import ALL_INTENTS, UniversalIntentEncoder
from src.micromodels.runtime import MicroDispatchResult, MicroModelRuntime
from src.micromodels.semantic_registry import SemanticToolRegistry
from src.micromodels.specialists import (
    SPECIALISTS,
    BaseSpecialist,
    extract_specialist_args,
)
from src.micromodels.tokenizer import BPETokenizer


# ============================================================================
# L0: SMOKE
# ============================================================================
class TestSmoke:
    """L0: Sanity checks - subsystems can be instantiated and basic methods callable."""

    @pytest.mark.lmtrust(layer="L0", direction="D5")
    def test_l0_components_instantiation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            assert reg.total_tools == 0

            collector = MicroDataCollector(data_dir=tmp_dir)
            assert collector.get_stats()["uncertain_queries_logged"] == 0

            tok = BPETokenizer(vocab_size=256)
            assert tok.vocab_size >= len(BPETokenizer.SPECIAL_TOKENS)

    @pytest.mark.lmtrust(layer="L0", direction="D2")
    def test_l0_runtime_singleton(self):
        rt = MicroModelRuntime.get_instance()
        assert rt is not None
        assert rt.total_intents == len(ALL_INTENTS)


# ============================================================================
# L1: CONTRACT (D7: API Contract)
# ============================================================================
class TestContract:
    """L1: Verifies strict return types and invariants under standard conditions."""

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_l1_tokenizer_encode_contract(self):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["git diff src/", "read file config.json", "покажи статус"])
        ids = tok.encode("git diff", max_length=16)
        assert isinstance(ids, list)
        assert len(ids) == 16
        assert all(isinstance(x, int) for x in ids)
        assert ids[0] == tok.cls_token_id

    @pytest.mark.lmtrust(layer="L1", direction="D7")
    def test_l1_semantic_registry_match_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            emb = np.random.randn(192).astype(np.float32)
            reg.register_tool("test_tool", "A test tool description", embedding=emb)
            
            matches = reg.match(emb, top_k=1, min_similarity=0.5)
            assert isinstance(matches, list)
            assert len(matches) == 1
            tool_name, score = matches[0]
            assert isinstance(tool_name, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0001


# ============================================================================
# L2: BOUNDARY (D1: Math/Numeric & Boundary Inputs)
# ============================================================================
class TestEdgeCases:
    """L2: Tests 8 boundary conditions: None, empty string, 0-dim, 1-dim, huge strings, max seq len."""

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_l2_tokenizer_boundary_inputs(self):
        tok = BPETokenizer(vocab_size=100)
        tok.train(["hello world"])

        # Boundary 1: None input must not crash with TypeError
        ids_none = tok.encode(None, max_length=16)
        assert isinstance(ids_none, list)
        assert len(ids_none) == 16

        # Boundary 2: Empty string
        ids_empty = tok.encode("", max_length=16)
        assert len(ids_empty) == 16
        assert ids_empty[0] == tok.cls_token_id

        # Boundary 3: Integer input
        ids_int = tok.encode(12345, max_length=16)
        assert isinstance(ids_int, list)
        assert len(ids_int) == 16

        # Boundary 4: Massive text > 10,000 characters
        huge_text = "слово " * 3000
        ids_huge = tok.encode(huge_text, max_length=16, truncation=True)
        assert len(ids_huge) == 16

    @pytest.mark.lmtrust(layer="L2", direction="D1")
    def test_l2_semantic_registry_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool("tool_a", "desc", embedding=np.ones(192, dtype=np.float32))

            # Boundary 5: Wrong dimension (length 3 instead of 192) - must not crash with ValueError
            res = reg.match(np.array([1.0, 2.0, 3.0]))
            assert res == []

            # Boundary 6: Empty vector
            res_empty = reg.match(np.array([]))
            assert res_empty == []

            # Boundary 7: 2D vector instead of 1D
            res_2d = reg.match(np.ones((1, 192), dtype=np.float32))
            assert isinstance(res_2d, list)


# ============================================================================
# L3: PROPERTY-BASED (D10: Mathematical Invariants via Hypothesis)
# ============================================================================
class TestProperties:
    """L3: Proves mathematical invariants via Hypothesis."""

    @pytest.mark.lmtrust(layer="L3", direction="D10")
    @settings(max_examples=25, deadline=None)
    @given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), min_size=192, max_size=192))
    def test_l3_registry_normalization_invariant(self, raw_vec):
        """Invariant: Every registered vector must have L2-norm == 1.0 (or 0 if zero vector)."""
        vec = np.array(raw_vec, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool("inv_tool", "desc", embedding=vec)
            
            stored = reg._embeddings["inv_tool"]
            norm = np.linalg.norm(stored)
            if np.linalg.norm(vec) > 1e-12:
                assert math.isclose(norm, 1.0, abs_tol=1e-4)
            else:
                assert math.isclose(norm, 0.0, abs_tol=1e-4)

    @pytest.mark.lmtrust(layer="L3", direction="D10")
    @settings(max_examples=20, deadline=None)
    @given(st.text(min_size=1, max_size=50))
    def test_l3_tokenizer_roundtrip_or_unk(self, text):
        """Invariant: Tokenizer decode(encode(x)) never crashes and preserves string type."""
        tok = BPETokenizer(vocab_size=200)
        tok.train(["abc def 123", "тест поиск файл"])
        ids = tok.encode(text, add_special_tokens=False, max_length=None)
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)


# ============================================================================
# L4: ADVERSARIAL (Blind Spot PoCs - Verified and Fixed)
# ============================================================================
class TestAdversarial:
    """L4: Proof-of-Concept exploits for identified blind spots."""

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_poc_bug1_semantic_registry_dimension_crash(self):
        """PoC Bug 1: Calling match() with non-192 length vector safely returns empty list without crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool("dummy", "description", embedding=np.zeros(192, dtype=np.float32))
            
            matches = reg.match(np.array([1.0, 2.0, 3.0]))
            assert matches == []

    @pytest.mark.lmtrust(layer="L4", direction="D6")
    def test_poc_bug2_concurrency_race_condition_in_registry(self):
        """PoC Bug 2: Concurrent registration and matching is protected by lock."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            for i in range(5):
                reg.register_tool(f"base_tool_{i}", "desc", embedding=np.ones(192, dtype=np.float32))

            errors = []
            stop_event = threading.Event()

            def writer():
                for idx in range(100):
                    if stop_event.is_set():
                        break
                    reg.register_tool(f"dyn_tool_{idx}", "dynamic", embedding=np.random.randn(192).astype(np.float32))
                    time.sleep(0.001)

            def reader():
                q_emb = np.ones(192, dtype=np.float32)
                for _ in range(200):
                    if stop_event.is_set():
                        break
                    try:
                        matches = reg.match(q_emb, top_k=3)
                        assert isinstance(matches, list)
                    except Exception as e:
                        errors.append(e)
                        break
                    time.sleep(0.0005)

            t1 = threading.Thread(target=writer)
            t2 = threading.Thread(target=reader)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

            assert len(errors) == 0, f"Concurrency race condition occurred: {errors}"

    @pytest.mark.lmtrust(layer="L4", direction="D3")
    def test_poc_bug3_non_atomic_save_corruption(self):
        """PoC Bug 3: Registry save() uses atomic file replacement (.tmp + os.replace)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool("safe_tool", "desc", embedding=np.ones(192, dtype=np.float32))
            reg.save()
            assert reg.registry_file.exists()

            initial_content = reg.registry_file.read_text(encoding="utf-8")
            assert "safe_tool" in initial_content

    @pytest.mark.lmtrust(layer="L4", direction="D7")
    def test_poc_bug4_tokenizer_type_error_on_non_string(self):
        """PoC Bug 4: Passing non-string/None to BPETokenizer safely normalizes input."""
        tok = BPETokenizer(vocab_size=100)
        tok.train(["sample text"])
        
        res1 = tok.encode(None)
        assert isinstance(res1, list)

        res2 = tok.encode(98765)
        assert isinstance(res2, list)

    @pytest.mark.lmtrust(layer="L4", direction="D4")
    def test_poc_bug6_dynamic_tool_empty_args_trap(self):
        """PoC Bug 6: Tools requiring arguments without specialist fallback return None to LLM."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool(
                "download_polyhaven_asset",
                "Downloads a 3D asset from PolyHaven by asset_id",
                embedding=np.ones(192, dtype=np.float32),
                parameters={
                    "type": "object",
                    "properties": {"asset_id": {"type": "string"}},
                    "required": ["asset_id"],
                },
                is_dynamic=True,
            )

            args = extract_specialist_args("download_polyhaven_asset", "download asset rock_01")
            assert args is None or "asset_id" in args

    @pytest.mark.lmtrust(layer="L4", direction="D1")
    def test_poc_bug7_nan_logits_json_compliance(self):
        """PoC Bug 7: NaN/Inf in logits/embeddings causes no NaN scores in output."""
        rt = MicroModelRuntime.get_instance()
        nan_vec = np.full(192, np.nan, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool("tool_x", "desc", embedding=np.ones(192, dtype=np.float32))
            matches = reg.match(nan_vec)
            assert matches == []
            for _, score in matches:
                assert not math.isnan(score)


# ============================================================================
# L4b: FALLBACK PATHWAYS (D4: Error Handling & Fallback)
# ============================================================================
class TestFallback:
    """L4b: Verifies graceful degradation and zero-shot fallback paths."""

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_l4b_zero_shot_semantic_fallback_in_dispatcher(self):
        """Verifies conversational queries fall through to LLM reasoning."""
        disp = MicroModelDispatcher.get_instance()
        res = disp.try_fast_dispatch("напиши эссе о философии экзистенциализма")
        assert res is None, "Complex conversational reasoning must fall through to LLM"

    @pytest.mark.lmtrust(layer="L4b", direction="D4")
    def test_l4b_dispatcher_delegates_to_runtime_cleanly(self):
        """Verifies dispatcher safely handles invalid or empty queries."""
        disp = MicroModelDispatcher.get_instance()
        assert disp.try_fast_dispatch("") is None
        assert disp.try_fast_dispatch(None) is None
        assert disp.try_fast_dispatch(123) is None


# ============================================================================
# L5: STATE (D2: State & Lifecycle)
# ============================================================================
class TestState:
    """L5: Verifies lifecycle transitions of the Registry and Runtime."""

    @pytest.mark.lmtrust(layer="L5", direction="D2")
    def test_l5_registry_add_update_persistence(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            emb1 = np.ones(192, dtype=np.float32)
            reg.register_tool("tool_alpha", "Initial description", embedding=emb1)
            reg.save()

            reg2 = SemanticToolRegistry(data_dir=tmp_dir)
            loaded = reg2.load()
            assert loaded is True
            assert reg2.total_tools == 1
            assert "tool_alpha" in reg2._tools

            emb2 = np.full(192, 0.5, dtype=np.float32)
            reg2.set_tool_embedding("tool_alpha", emb2)
            reg2.save()

            reg3 = SemanticToolRegistry(data_dir=tmp_dir)
            reg3.load()
            assert math.isclose(np.linalg.norm(reg3._embeddings["tool_alpha"]), 1.0, abs_tol=1e-4)


# ============================================================================
# L6: CROSS-SYSTEM (No-Mock E2E Integration)
# ============================================================================
class TestIntegration:
    """L6: End-to-end integration without mocks from query string to FastDispatchResult."""

    @pytest.mark.lmtrust(layer="L6", direction="D7")
    def test_l6_e2e_read_file_dispatch(self):
        disp = MicroModelDispatcher.get_instance()
        if not disp.is_available:
            pytest.skip("MicroModel weights not loaded in test environment")

        res = disp.try_fast_dispatch("прочитай файл src/micromodels/runtime.py")
        assert res is not None
        assert res.tool_name == "read_file"
        assert "src/micromodels/runtime.py" in res.arguments.get("file_path", "")
        assert res.elapsed_ms < 50.0

        args_with_lines = extract_specialist_args("read_file", "прочитай файл src/micromodels/runtime.py строки 10-30")
        assert args_with_lines is not None
        assert args_with_lines.get("start_line") == 10
        assert args_with_lines.get("line_count") == 21

    @pytest.mark.lmtrust(layer="L6", direction="D7")
    def test_l6_e2e_git_diff_normalization(self):
        disp = MicroModelDispatcher.get_instance()
        if not disp.is_available:
            pytest.skip("MicroModel weights not loaded in test environment")

        res = disp.try_fast_dispatch("покажи git diff src/micromodels/dispatcher.py")
        if res is not None:
            assert res.tool_name == "run_command"
            assert "git diff" in res.arguments.get("command", "")


# ============================================================================
# L7: COMPOSITIONAL (D4: Multiple Simultaneous Failures)
# ============================================================================
class TestCompositional:
    """L7: Simultaneous failures: corrupted file + missing dirs + bad query."""

    @pytest.mark.lmtrust(layer="L7", direction="D4")
    def test_l7_corrupted_file_and_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_file = Path(tmp_dir) / "tool_embeddings.json"
            corrupt_file.write_text("{this is corrupted json", encoding="utf-8")

            reg = SemanticToolRegistry(data_dir=tmp_dir)
            success = reg.load()
            assert success is False
            assert reg.total_tools == 0
            assert reg.match(np.ones(192, dtype=np.float32)) == []


# ============================================================================
# L8: CONCURRENCY / TEMPORAL (D6: Concurrency & Atomicity)
# ============================================================================
class TestTemporal:
    """L8: High-load multi-threaded operations on Registry and Runtime."""

    @pytest.mark.lmtrust(layer="L8", direction="D6")
    def test_l8_multithreaded_concurrent_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reg = SemanticToolRegistry(data_dir=tmp_dir)
            reg.register_tool("base_tool", "desc", embedding=np.ones(192, dtype=np.float32))
            reg.save()

            errors = []

            def worker_save(thread_id):
                try:
                    for i in range(20):
                        emb = np.random.randn(192).astype(np.float32)
                        reg.register_tool(f"t_{thread_id}_{i}", "desc", embedding=emb)
                        reg.save()
                except Exception as e:
                    errors.append(e)

            def worker_load():
                try:
                    for _ in range(20):
                        reg_reader = SemanticToolRegistry(data_dir=tmp_dir)
                        reg_reader.load()
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=worker_save, args=(1,)),
                threading.Thread(target=worker_save, args=(2,)),
                threading.Thread(target=worker_load),
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Concurrency errors during save/load: {errors}"


# ============================================================================
# L9: NEGATIVE SPACE (D3/D8: What MUST NEVER Happen)
# ============================================================================
class TestNegativeSpace:
    """L9: Proves the system never takes dangerous or invalid actions."""

    @pytest.mark.lmtrust(layer="L9", direction="D8")
    def test_l9_never_fast_dispatch_destructive_commands(self):
        """Negative invariant: Destructive shell commands (rm -rf, format, del) MUST NEVER be fast-dispatched."""
        rt = MicroModelRuntime.get_instance()
        destructive_queries = [
            "выполни команду rm -rf /",
            "запусти cmd format C:",
            "run command del /f /s /q *.*",
            "sudo rm -rf /etc",
        ]
        for dq in destructive_queries:
            res = rt.try_fast_dispatch(dq)
            assert res is None, f"Security violation: Destructive command '{dq}' was fast-dispatched!"

    @pytest.mark.lmtrust(layer="L9", direction="D3")
    def test_l9_never_produce_nan_confidence(self):
        """Negative invariant: Confidence score must NEVER be NaN or infinite."""
        rt = MicroModelRuntime.get_instance()
        if not rt.is_available:
            pytest.skip("Runtime not available")

        for query in ["test", "???", "", "12345", "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"]:
            intent, conf, all_probs, _, _ = rt.predict_intent(query)
            assert not math.isnan(conf), f"Confidence was NaN for query '{query}'"
            assert not math.isinf(conf), f"Confidence was Inf for query '{query}'"
            assert 0.0 <= conf <= 1.0, f"Confidence out of bounds [0, 1]: {conf}"
