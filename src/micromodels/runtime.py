"""
MicroModelRuntime: High-performance in-process inference engine for Argent.

Executes UniversalIntentEncoder via ONNX Runtime (< 1.8ms on CPU) or PyTorch fallback,
and extracts tool arguments via dedicated specialists or semantic zero-shot matching.
"""

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.micromodels.schema_dataset_generator import (
    ALL_INTENTS,
    ID2INTENT,
    INTENT2ID,
)
from src.micromodels.semantic_registry import SemanticToolRegistry
from src.micromodels.specialists import extract_specialist_args
from src.micromodels.tokenizer import BPETokenizer
from logger import get_logger

logger = get_logger("micromodel_runtime")

try:
    import onnxruntime as ort

    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


@dataclass
class MicroDispatchResult:
    """Result of a micro-model fast dispatch evaluation."""

    tool_name: str
    arguments: Dict[str, Any]
    confidence: float
    elapsed_ms: float
    raw_query: str
    intent_probabilities: Dict[str, float]
    is_semantic_match: bool = False


class MicroModelRuntime:
    """Unified runtime for UniversalIntentEncoder, SemanticToolRegistry, and micro-specialists."""

    _instance: Optional["MicroModelRuntime"] = None

    def __init__(
        self,
        model_dir: Optional[str] = None,
        confidence_threshold: float = 0.85,
        semantic_threshold: float = 0.72,
        max_seq_len: int = 48,
    ):
        self.model_dir = Path(
            model_dir
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        )
        self.confidence_threshold = confidence_threshold
        self.semantic_threshold = semantic_threshold
        self.max_seq_len = max_seq_len

        self._init_lock = threading.Lock()
        self._tokenizer: Optional[BPETokenizer] = None
        self._onnx_session: Optional[Any] = None
        self._torch_model: Optional[Any] = None
        self._backend: str = "none"
        self._initialized: bool = False
        self._init_error: Optional[str] = None
        self._intents: List[str] = ALL_INTENTS
        self._registry = SemanticToolRegistry.get_instance()

    @classmethod
    def get_instance(cls) -> "MicroModelRuntime":
        if cls._instance is None:
            cls._instance = MicroModelRuntime()
        return cls._instance

    @property
    def is_available(self) -> bool:
        int8_file = self.model_dir / "intent_router_int8.onnx"
        onnx_file = self.model_dir / "intent_router.onnx"
        pt_file = self.model_dir / "intent_router.pt"
        tok_file = self.model_dir / "tokenizer.json"
        return tok_file.exists() and (int8_file.exists() or onnx_file.exists() or pt_file.exists())

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def total_intents(self) -> int:
        return len(self._intents)

    def initialize(self) -> bool:
        """Loads the tokenizer, semantic registry, and model (ONNX or PyTorch) thread-safely."""
        if self._initialized:
            return True

        with self._init_lock:
            if self._initialized:
                return True

            tok_file = self.model_dir / "tokenizer.json"
            if not tok_file.exists():
                self._init_error = f"Tokenizer file not found: {tok_file}"
                return False

            try:
                self._tokenizer = BPETokenizer.load(tok_file)
            except Exception as e:
                self._init_error = f"Failed loading tokenizer: {e}"
                return False

            # Load intent catalog if available
            catalog_file = self.model_dir / "universal_intent_catalog.json"
            if catalog_file.exists():
                try:
                    with open(catalog_file, "r", encoding="utf-8") as f:
                        cat = json.load(f)
                        self._intents = cat.get("intents", ALL_INTENTS)
                except Exception:
                    pass

            # Load semantic registry
            self._registry.load()

            # Attempt 1: ONNX Runtime (prefer INT8 quantized model for sub-2ms latency)
            int8_file = self.model_dir / "intent_router_int8.onnx"
            fp32_file = self.model_dir / "intent_router.onnx"
            onnx_file = int8_file if int8_file.exists() else fp32_file

            if ORT_AVAILABLE and onnx_file.exists():
                try:
                    opts = ort.SessionOptions()
                    threads = min(4, os.cpu_count() or 4)
                    opts.intra_op_num_threads = threads
                    opts.inter_op_num_threads = 1
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self._onnx_session = ort.InferenceSession(str(onnx_file), opts)
                    self._backend = "onnx_int8" if onnx_file == int8_file else "onnx"
                    self._initialized = True
                    logger.info(f"MicroModelRuntime initialized with {self._backend} backend ({onnx_file.name}).")
                    return True
                except Exception as e:
                    logger.warning(f"ONNX session init failed, falling back to PyTorch: {e}")

            # Attempt 2: PyTorch
            pt_file = self.model_dir / "intent_router.pt"
            if pt_file.exists():
                try:
                    import torch
                    from src.micromodels.intent_router import UniversalIntentEncoder

                    checkpoint = torch.load(pt_file, map_location="cpu", weights_only=True)
                    model = UniversalIntentEncoder(
                        vocab_size=self._tokenizer.vocab_size,
                        hidden_dim=192,
                        num_layers=4,
                        num_heads=4,
                        ff_dim=512,
                        max_seq_len=self.max_seq_len,
                        num_classes=len(self._intents),
                        dropout=0.0,
                    )
                    model.load_state_dict(checkpoint["model_state"])
                    model.eval()
                    self._torch_model = model
                    self._backend = "pytorch"
                    self._initialized = True
                    logger.info("MicroModelRuntime initialized with PyTorch CPU backend.")
                    return True
                except Exception as e:
                    self._init_error = f"PyTorch load failed: {e}"
                    logger.error(self._init_error)
                    return False

            self._init_error = "Neither ONNX nor PyTorch model weights found."
            return False

    def predict_intent(
        self, query: str
    ) -> Tuple[str, float, Dict[str, float], float, Optional[np.ndarray]]:
        """Runs inference on a user query string.

        Returns:
            (best_intent, confidence, all_probs, elapsed_ms, embedding_vector)
        """
        if not self.initialize() or self._tokenizer is None:
            return "NO_TOOL", 0.0, {}, 0.0, None

        if query is None:
            query = ""
        elif not isinstance(query, str):
            query = str(query)

        t0 = time.perf_counter()

        # Tokenize
        token_ids = self._tokenizer.encode(
            query,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
        )

        embedding = None
        if self._backend.startswith("onnx") and self._onnx_session is not None:
            inp = np.array([token_ids], dtype=np.int64)
            ort_outs = self._onnx_session.run(None, {"input_ids": inp})
            logits = ort_outs[0][0]
            if len(ort_outs) > 1:
                embedding = ort_outs[1][0]
            # Softmax with NaN / Inf protection
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            max_l = np.max(logits)
            exp_logits = np.exp(logits - max_l)
            sum_exp = np.sum(exp_logits)
            if sum_exp <= 1e-12 or np.isnan(sum_exp):
                probs = np.ones_like(logits) / len(logits)
            else:
                probs = exp_logits / sum_exp
        elif self._backend == "pytorch" and self._torch_model is not None:
            import torch
            import torch.nn.functional as F

            with torch.no_grad():
                inp = torch.tensor([token_ids], dtype=torch.long)
                logits_t, emb_t = self._torch_model(inp)
                probs = F.softmax(logits_t, dim=-1)[0].numpy()
                embedding = emb_t[0].numpy()
        else:
            return "NO_TOOL", 0.0, {}, 0.0, None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        best_idx = int(np.argmax(probs))
        confidence = float(np.clip(probs[best_idx], 0.0, 1.0))
        if np.isnan(confidence):
            confidence = 0.0
        best_intent = self._intents[best_idx] if best_idx < len(self._intents) else "NO_TOOL"

        all_probs = {
            self._intents[i]: round(float(probs[i]), 4)
            for i in range(min(len(self._intents), len(probs)))
        }

        return best_intent, confidence, all_probs, elapsed_ms, embedding

    def register_dynamic_tool(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Registers a dynamic or MCP tool by computing its semantic embedding on the fly."""
        if not self.initialize() or self._tokenizer is None:
            return False

        sig = f"{name.replace('_', ' ')}: {description}"
        tids = self._tokenizer.encode(sig, max_length=self.max_seq_len, padding="max_length", truncation=True)
        inp = np.array([tids], dtype=np.int64)

        if self._backend.startswith("onnx") and self._onnx_session is not None:
            outs = self._onnx_session.run(None, {"input_ids": inp})
            if len(outs) > 1:
                emb = outs[1][0]
            else:
                emb = outs[0][0][:192]
        elif self._torch_model is not None:
            import torch
            with torch.no_grad():
                emb = self._torch_model.get_embedding(torch.tensor(inp, dtype=torch.long))[0].numpy()
        else:
            return False

        self._registry.register_tool(
            name=name,
            description=description,
            embedding=emb,
            parameters=parameters,
            is_dynamic=True,
        )
        self._registry.save()
        logger.info(f"Registered dynamic tool '{name}' into SemanticToolRegistry.")
        return True

    def try_fast_dispatch(
        self, query: str, threshold: Optional[float] = None
    ) -> Optional[MicroDispatchResult]:
        """Evaluates whether query should be dispatched immediately via micromodels.

        Returns MicroDispatchResult if recognized with high confidence and valid arguments.
        Returns None otherwise (passes query to LLM).
        """
        if not query or not isinstance(query, str):
            return None

        q_clean = query.strip()
        if len(q_clean) < 2 or q_clean.startswith("/") or q_clean.startswith("!"):
            return None

        # Negative Space Guard: Destructive / administrative commands must never be fast dispatched
        DANGEROUS_PATTERNS = (
            r"\brm\s+-[rf]{1,2}\b",
            r"\bdel\s+/[fsqa]{1,4}\b",
            r"\bformat\s+[a-z]:",
            r"\bmkfs\b",
            r"\bdrop\s+database\b",
            r"\bshutdown\b",
            r"\bsudo\s+rm\b",
        )
        for pat in DANGEROUS_PATTERNS:
            if re.search(pat, q_clean, re.IGNORECASE):
                logger.warning(f"Query contains dangerous pattern, rejecting fast dispatch: {q_clean}")
                return None

        thresh = threshold if threshold is not None else self.confidence_threshold

        intent, confidence, all_probs, elapsed_ms, embedding = self.predict_intent(q_clean)

        is_semantic = False

        # 1. Check direct classifier
        if confidence >= thresh and intent != "NO_TOOL":
            target_tool = intent
        else:
            # 2. Semantic Zero-Shot Fallback: check if query matches a dynamic tool in SemanticToolRegistry
            target_tool = None
            if embedding is not None:
                matches = self._registry.match(
                    embedding,
                    top_k=1,
                    min_similarity=self.semantic_threshold,
                    only_dynamic=True,
                )
                if matches:
                    top_name, top_sim = matches[0]
                    target_tool = top_name
                    confidence = top_sim
                    is_semantic = True
                    logger.debug(f"Zero-shot semantic match: '{q_clean[:30]}' -> '{top_name}' (sim: {top_sim:.2f})")

        if not target_tool or target_tool == "NO_TOOL":
            return None

        # 3. Extract arguments using specialist
        t_spec_0 = time.perf_counter()
        args = extract_specialist_args(target_tool, q_clean)
        elapsed_ms += (time.perf_counter() - t_spec_0) * 1000.0

        if args is None:
            return None

        # Dynamic tool parameter validation: if tool declares required parameters, arguments must satisfy them
        if is_semantic:
            tool_meta = self._registry._tools.get(target_tool, {})
            req_params = tool_meta.get("parameters", {}).get("required", [])
            if req_params and not all(p in args for p in req_params):
                logger.debug(f"Dynamic tool '{target_tool}' requires parameters {req_params}, falling back to LLM")
                return None

        # Safety check for run_command
        if target_tool == "run_command" and not args.get("is_safe", False):
            logger.debug(f"Command '{args.get('command')}' flagged unsafe for immediate fast-path")
            return None

        return MicroDispatchResult(
            tool_name=target_tool,
            arguments=args,
            confidence=confidence,
            elapsed_ms=elapsed_ms,
            raw_query=query,
            intent_probabilities=all_probs,
            is_semantic_match=is_semantic,
        )


runtime = MicroModelRuntime.get_instance()
