"""
src/leeth/runtime.py — High-Performance System One Decision Runtime for Argent.

Provides the primary public API:
    from src.leeth.runtime import leeth
    decisions = leeth.decide(state, questions)

Features:
- Single-pass non-autoregressive parallel evaluation.
- Hardware-accelerated execution on NVIDIA RTX 4060 GPU (CUDA) or CPU.
- Span extraction decoding back to string text without regular expressions.
- Calibrated probability distributions for Choice and Noul primitives.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.leeth.model import LeethConfig, LeethModel, config_from_checkpoint
from src.leeth.primitives import (
    BaseQuestion,
    ChoiceQuestion,
    ChoiceResult,
    ExtractQuestion,
    ExtractResult,
    LeethDecisionBatch,
    LeethResult,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
)
from src.leeth.tokenizer import LeethTokenizer
from logger import get_logger

logger = get_logger("leeth_runtime")

# Noul questions VALIDATED for the mmBERT noul head on the held-out val split
# (scripts/_d3_final_checks.py): need_tool 99.5% acc vs legacy 88.1% (FN 1 vs 52).
# D5: is_safe switched to mmBERT — the D5 noul head (safety negatives) closes the
# legacy OOD holes (legacy: IEX/-enc/IWR safe_prob=1.000, 3/10 breakthroughs at the
# 0.80 gate, 1 real — IWR is not deny-listed; scripts/_d5_legacy_gate.py) while the
# promoted D5 artifact holds 0/10 (max 0.516, scripts/_d5_final_compare.py) and
# passes 6/6 benign at the is_safe level (legacy 3/6). has_error stays legacy
# (mmBERT 91.1% vs legacy 100%).
MMBERT_NOUL_QUESTIONS = frozenset({
    "Требуется ли для выполнения данного запроса вызов хотя бы одного системного инструмента?",
    "Является ли данная терминальная команда безопасной для выполнения?",
})

# Choice questions the mmBERT choice head was TRAINED on (routing battery +
# action vocabulary, 26 classes). Ad-hoc choice questions (language ID, custom
# tool sets — e.g. the Jev-parity batteries) fall back to the legacy 400M model,
# which serves the full 4-primitive surface. Without the whitelist the mmBERT
# head answers out-of-distribution choice questions with near-uniform noise
# (language ID -> "JavaScript" 51%, scripts/verify_jev_parity.py battery 1).
MMBERT_CHOICE_QUESTIONS = frozenset({
    "Какой инструмент Argent должен быть вызван в первую очередь?",
})


# Variant A (2026-10-01): dedicated dept/urgency heads exported into the mmBERT
# ONNX graph (mmbert_route_prod.onnx carries 4 outputs now). The dept head is a
# SEPARATE 5-class head (billing/technical/sales/account/other) — its classes
# are NOT in the 26-class routing vocabulary, so it decodes from dept_logits,
# never from choice_logits. Battery: dept 100% (legacy 14.3%, Laya 92.9%),
# urgency 84.6% (legacy 46.2%, Laya 61.5%), routing 91/109 unchanged
# (scripts/export_mmbert_dept.py, .argent/artifacts/mmbert_export_du.md).
MMBERT_DEPT_QUESTION = "В какой отдел нужно направить это обращение?"
MMBERT_URGENCY_QUESTION = "Насколько срочно нужно решить проблему клиента?"
MMBERT_DEPT_CLASSES = ("billing", "technical", "sales", "account", "other")
MMBERT_URGENCY_LEVELS = 3


class LeethRuntime:
    """Singleton runtime managing Leeth model execution and decision batching."""

    _instance: Optional["LeethRuntime"] = None

    def __init__(self, model_dir: Optional[str] = None):
        # Production default: 400M universal model. Rollback: set LEETH_MODEL_DIR=src/leeth/data (v24).
        self.model_dir = Path(model_dir or os.environ.get("LEETH_MODEL_DIR", "src/leeth/data_400m"))
        self.tokenizer: Optional[LeethTokenizer] = None
        self.model: Optional[LeethModel] = None
        self.ort_session: Any = None
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.device = "auto"
        self._initialized = False
        self._init_error: Optional[str] = None
        self._cuda_failover_done = False
        # mmBERT router (E5): choice/noul heads on the ModernBERT backbone
        # (.argent/models/mmbert-argent). Extract/score stay on the legacy 400M
        # model — the mmBERT ONNX export carries only choice_logits/noul_logits.
        # Rollback: set LEETH_ROUTER=legacy.
        self.router_mode = os.environ.get("LEETH_ROUTER", "mmbert").strip().lower()
        self.mmbert_dir = Path(__file__).resolve().parents[2] / ".argent" / "models" / "mmbert-argent"
        self.mmbert_session: Any = None
        self.mmbert_tokenizer: Any = None
        self.mmbert_active = False
        # True when the ONNX graph carries the variant-A dept/urgency outputs.
        self.mmbert_du_available = False
        # True when the checkpoint carries a trained DynamicOptionHead — enables
        # zero-shot scoring over ARBITRARY option lists (games, ad-hoc tools).
        self.dynamic_opt_available = False
        # Context budget for the mmBERT path. The ModernBERT backbone supports
        # 8192 positions (max_position_embeddings); 1024 is the latency/quality
        # sweet spot for prod states (45 ms/case at 256 on CPU fp32). Override:
        # LEETH_MMBERT_MAXLEN env (256..8192).
        self.mmbert_max_length = max(64, min(8192, int(os.environ.get("LEETH_MMBERT_MAXLEN", "1024"))))
        # Modular micro-adapters (guard.onnx, ground.onnx, etc.) in ONNX CPU:
        self.adapters_dir = Path(__file__).resolve().parents[2] / ".argent" / "models" / "adapters"
        self._adapter_sessions: Dict[str, Any] = {}
        self.last_pooled_repr: Optional[Any] = None
        self.last_pooled_reprs: Dict[str, Any] = {}
        # Leeth 2 (src/leeth2): one calibrated model for every primitive. Opt-in via config.get_leeth_backend() == "leeth2"
        # or LEETH_BACKEND=leeth2; the legacy stack stays the default until Leeth 2 has passed its gates.
        self.v2_dir = Path(os.environ.get("LEETH2_MODEL_DIR", str(Path(__file__).resolve().parents[2] / ".argent" / "models" / "leeth2" / "final")))
        self._v2: Any = None  # None = not tried yet, False = unavailable
        # Universal Task Solver (DoPT v6/v7): DynamicOptionHead на mmBERT —
        # zero-shot выбор оптимального хода для игровых доменов (змейка/лабиринт/TTT
        # и любые новые с идеальным экспертом). Ленивая загрузка при первом
        # decide_dopt() — 2 GB ONNX не должен грузиться при старте роутера.
        self.dopt_solver: Any = None

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> "LeethRuntime":
        if cls._instance is None:
            cls._instance = LeethRuntime(model_dir=model_dir)
        return cls._instance

    def _use_v2(self) -> Any:
        """The Leeth 2 runtime when the backend is selected and its weights load, else None (legacy path)."""
        try:
            import config as _cfg

            if _cfg.get_leeth_backend() != "leeth2":
                return None
        except Exception:
            return None
        if self._v2 is None:
            try:
                from src.leeth2.runtime import Leeth2Runtime

                self._v2 = Leeth2Runtime.load(self.v2_dir)
                logger.info("Leeth 2 backend active (%s, %s).", self.v2_dir, self._v2.device)
            except Exception as e:
                logger.warning("Leeth 2 backend requested but unavailable (%s); using the legacy stack.", e)
                self._v2 = False
        return self._v2 or None

    @property
    def is_available(self) -> bool:
        if self._use_v2() is not None:
            return True
        weights_file = self.model_dir / "leeth_best.pt"
        onnx_file = self.model_dir / "leeth_int8.onnx"
        vocab_file = self.model_dir / "leeth_vocab.json"
        return (weights_file.exists() or onnx_file.exists()) and vocab_file.exists()

    def reload(self, force_device: Optional[str] = None) -> bool:
        """Forces unloading existing models/sessions and reloading freshly saved weights."""
        self._initialized = False
        self.model = None
        self.ort_session = None
        self.tokenizer = None
        self.mmbert_session = None
        self.mmbert_tokenizer = None
        self.mmbert_active = False
        self.mmbert_du_available = False
        self.dynamic_opt_available = False
        self._adapter_sessions.clear()
        self.last_pooled_repr = None
        self.last_pooled_reprs.clear()
        # Free the old ~1.4 GB weights BEFORE allocating the new ones —
        # otherwise peak RSS doubles and Windows can hit access violations
        # under memory pressure (seen in full-suite runs).
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return self.initialize(force_device=force_device)

    def initialize(self, force_device: Optional[str] = None) -> bool:
        """Loads model weights, vocabulary, and class mappings thread-safely."""
        if self._initialized and (self.model is not None or self.ort_session is not None):
            return True

        weights_file = self.model_dir / "leeth_best.pt"
        onnx_int8_file = self.model_dir / "leeth_int8.onnx"
        vocab_file = self.model_dir / "leeth_vocab.json"
        classes_file = self.model_dir / "choice_classes.json"

        if force_device is not None:
            self.device = force_device
        else:
            # Respect the configured device preference ('auto'/'cuda'/'cpu'/'onnx').
            preferred = os.environ.get("LEETH_DEVICE", "auto").strip().lower()
            if preferred == "auto":
                try:
                    import config as _cfg
                    preferred = (_cfg.get_leeth_device() or "auto").lower()
                except Exception:
                    pass
            if preferred in ("cuda", "cpu", "onnx"):
                self.device = preferred
            elif onnx_int8_file.exists():
                self.device = "onnx"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not weights_file.exists() and not onnx_int8_file.exists():
            self._init_error = f"Neither PyTorch weights nor ONNX model found in {self.model_dir}"
            logger.warning(self._init_error)
            return False

        try:
            # 1. Load Tokenizer
            self.tokenizer = LeethTokenizer.load(self.model_dir)

            # 2. Load Class Mappings
            if classes_file.exists():
                with open(classes_file, "r", encoding="utf-8") as f:
                    self.class_to_idx = json.load(f)
                    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

            # 3. Choose execution engine:
            # Use ONNX Runtime for CPU if available (1.6ms INT8 speed)
            if (self.device in ("cpu", "onnx")) and onnx_int8_file.exists():
                try:
                    import onnxruntime as ort

                    # Pin intra-op threads to 8: the default (all 14 physical
                    # cores) has brutal sync overhead on the small sequences
                    # this model runs (seq ~36-120) — measured 318ms vs 58ms
                    # for a single question, and the slowdown varies with
                    # core-parking state (nondeterministic latency).
                    opts = ort.SessionOptions()
                    opts.intra_op_num_threads = 8
                    opts.inter_op_num_threads = 1
                    self.ort_session = ort.InferenceSession(
                        str(onnx_int8_file),
                        sess_options=opts,
                        providers=["CPUExecutionProvider"],
                    )
                    self.device = "onnx-cpu"
                    self._initialized = True
                    logger.info("Leeth runtime initialized using INT8 ONNX CPU backend.")
                    self._init_mmbert()
                    return True
                except Exception as ort_err:
                    logger.warning("Failed to load ONNX session, falling back to PyTorch: %s", ort_err)

            # PyTorch fallback or CUDA acceleration
            # Чекпоинты, файнтюненные вне src/ (Kaggle kernel), ссылаются на
            # модуль 'model_leeth' — алиасим его на src.leeth.model, чтобы
            # pickle нашёл LeethConfig.
            import sys as _sys

            if "model_leeth" not in _sys.modules:
                from src.leeth import model as _leeth_model

                _sys.modules["model_leeth"] = _leeth_model
            checkpoint = torch.load(weights_file, map_location=self.device, weights_only=False)
            config = config_from_checkpoint(checkpoint["config"])
            self.model = LeethModel(config)
            state_dict = checkpoint["state_dict"]
            # DynamicOptionHead is trained post-hoc (Kaggle kernel B): its weights
            # may be absent from older checkpoints — load the rest strictly and
            # the head leniently, then flag availability for the runtime path.
            has_dynamic = any(k.startswith("dynamic_opt_head.") for k in state_dict)
            if has_dynamic:
                self.model.load_state_dict(state_dict)
                self.dynamic_opt_available = True
            else:
                self.model.load_state_dict(state_dict, strict=False)
                self.model.load_state_dict(
                    {k: v for k, v in state_dict.items() if not k.startswith("dynamic_opt_head.")},
                    strict=False,
                )
            self.model.to(self.device)
            self.model.eval()

            self._initialized = True
            logger.info("Leeth runtime initialized on device '%s'.", self.device)
            self._init_mmbert()
            return True
        except Exception as e:
            self._init_error = str(e)
            logger.error("Failed to initialize Leeth runtime: %s", e)
            return False

    def _init_mmbert(self) -> bool:
        """Loads the mmBERT router (choice/noul) alongside the legacy session.

        The mmBERT INT8 artifact is batch-1-only in practice: dynamic INT8
        quantization derives activation scales from the batch min/max, so a
        mixed batch changes predictions (13/60 argmax flips, see
        scripts/_d3_probe_batch.py). The router therefore always runs one
        question per forward — exactly the validated prod-battery behavior.
        """
        self.mmbert_active = False
        self.mmbert_session = None
        self.mmbert_tokenizer = None
        # Re-read on every (re)initialize so tests can flip the router via env.
        self.router_mode = os.environ.get("LEETH_ROUTER", "mmbert").strip().lower()
        if self.router_mode != "mmbert":
            return False
        # Приоритет: компактный INT8-роутер (294 MB, 24 мс/кейс), фолбэк — fp32.
        onnx_file = self.mmbert_dir / "mmbert_route_prod.onnx"
        int8_file = self.mmbert_dir / "mmbert_route_prod_int8.onnx"
        candidates = [int8_file, onnx_file] if int8_file.exists() else [onnx_file]
        if not any(c.exists() for c in candidates):
            logger.info("mmBERT router disabled: no model found in %s.", self.mmbert_dir)
            return False
        # Class mapping: mmBERT was trained with the data_400m mapping (same
        # choice_classes.json). If the legacy dir is rolled back to another
        # vocabulary, the mapping may differ — refuse to mix them.
        if self.mmbert_dir.name and not self.model_dir.name.endswith("data_400m"):
            logger.info("mmBERT router disabled: legacy model_dir '%s' is not data_400m.", self.model_dir)
            return False
        try:
            import gc
            import onnxruntime as ort

            from transformers import AutoTokenizer

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 8
            opts.inter_op_num_threads = 1

            loaded_file = None
            for cand in candidates:
                if not cand.exists():
                    continue
                try:
                    self.mmbert_session = ort.InferenceSession(
                        str(cand), sess_options=opts, providers=["CPUExecutionProvider"]
                    )
                    loaded_file = cand
                    break
                except Exception as cand_err:
                    logger.warning("mmBERT failed to load %s (%s); trying fallback.", cand.name, cand_err)
                    self.mmbert_session = None
                    gc.collect()

            if self.mmbert_session is None or loaded_file is None:
                logger.warning("mmBERT router failed to load any candidate; legacy model handles all questions.")
                return False

            self.mmbert_tokenizer = AutoTokenizer.from_pretrained(str(self.mmbert_dir))
            # Variant A: graphs exported by export_mmbert_dept.py expose 4 outputs
            # (choice/noul/dept/urgency); older 2-output graphs keep dept/urgency
            # questions on the legacy model.
            self.mmbert_du_available = len(self.mmbert_session.get_outputs()) >= 4
            self.mmbert_active = True
            logger.info("mmBERT router active: choice/noul via %s (extract/score stay legacy%s).",
                        loaded_file.name, ", dept/urgency via du-heads" if self.mmbert_du_available else ")")
            return True
        except Exception as e:
            logger.warning("mmBERT router failed to load (%s); legacy model handles all questions.", e)
            return False

    def decide(
        self,
        state: str,
        questions: List[BaseQuestion],
        max_length: int = 512,
    ) -> LeethDecisionBatch:
        """Evaluates a batch of typed questions over an application state non-autoregressively."""
        return self.decide_batch([state], questions, max_length=max_length)[0]

    def get_adapter_session(self, name: str) -> Optional[Any]:
        """Loads and caches an ONNX micro-adapter from .argent/models/adapters/{name}.onnx."""
        if name in self._adapter_sessions:
            return self._adapter_sessions[name]
        adapter_path = self.adapters_dir / f"{name}.onnx"
        if not adapter_path.exists():
            return None
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(adapter_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._adapter_sessions[name] = sess
            return sess
        except Exception as e:
            logger.warning("Failed to load micro-adapter %s (%s)", name, e)
            return None

    def run_adapter(self, name: str, pooled_repr: Optional[Any] = None, question_id: Optional[str] = None) -> Optional[float]:
        """Runs a lightweight ONNX micro-adapter (< 0.05 ms latency) returning a probability in [0, 1].

        If pooled_repr is None, the last computed pooled_repr for question_id (or default) is used.
        """
        sess = self.get_adapter_session(name)
        if sess is None:
            return None
        repr_vec = pooled_repr
        if repr_vec is None:
            if question_id and question_id in self.last_pooled_reprs:
                repr_vec = self.last_pooled_reprs[question_id]
            elif name == "ground" and "tool" in self.last_pooled_reprs:
                repr_vec = self.last_pooled_reprs["tool"]
            elif name == "guard" and "is_safe" in self.last_pooled_reprs:
                repr_vec = self.last_pooled_reprs["is_safe"]
            else:
                repr_vec = self.last_pooled_repr
        if repr_vec is None:
            return None
        try:
            import numpy as np

            inp_meta = sess.get_inputs()[0]
            inp_name = inp_meta.name
            expected_dim = inp_meta.shape[-1] if isinstance(inp_meta.shape[-1], int) else None

            if hasattr(repr_vec, "detach"):
                repr_vec = repr_vec.detach().cpu().numpy()
            if not isinstance(repr_vec, np.ndarray):
                repr_vec = np.asarray(repr_vec, dtype=np.float32)

            if repr_vec.ndim == 1:
                repr_vec = np.expand_dims(repr_vec, 0)
            elif repr_vec.ndim == 2 and repr_vec.shape[0] > 1:
                repr_vec = repr_vec[0:1]
            repr_vec = repr_vec.astype(np.float32)

            if expected_dim is not None and repr_vec.shape[-1] != expected_dim:
                if repr_vec.shape[-1] > expected_dim:
                    repr_vec = repr_vec[:, :expected_dim]
                else:
                    pad = np.zeros((repr_vec.shape[0], expected_dim - repr_vec.shape[-1]), dtype=np.float32)
                    repr_vec = np.concatenate([repr_vec, pad], axis=-1)

            out = sess.run(None, {inp_name: repr_vec})[0]
            val = float(out.flat[0])
            prob = 1.0 / (1.0 + np.exp(-val))
            return float(prob)
        except Exception as e:
            logger.warning("Micro-adapter %s execution failed: %s", name, e)
            return None


    def decide_options(
        self,
        state: str,
        question: str,
        options: List[str],
        max_length: int = 512,
    ) -> ChoiceResult:
        """Zero-shot scoring over an ARBITRARY option list via DoPT / DynamicOptionHead.

        Primary path: Universal Task Solver (DoPT ONNX CPU) evaluates arbitrary
        strings in a single pass. Falls back to PyTorch DynamicOptionHead, then
        to restricted legacy categorical head.
        """
        if not options:
            return ChoiceResult(choice="", confidence=0.0, probabilities={})
        v2 = self._use_v2()
        if v2 is not None:
            return v2.choose(state, question, list(options))

        # 1. Primary path: Universal Task Solver (DoPT ONNX runtime)
        solver = self._get_dopt()
        if solver is not None:
            try:
                res = solver.decide(state, question, list(options))
                if res and res.choice:
                    return ChoiceResult(choice=res.choice, confidence=res.confidence, probabilities=res.probabilities)
            except Exception as dopt_err:
                logger.warning("DoPT execution failed (%s), falling back to PyTorch/legacy head.", dopt_err)

        if not self._initialized and not self.initialize():
            return ChoiceResult(choice="", confidence=0.0, probabilities={})
        assert self.tokenizer is not None and self.model is not None

        if not self.dynamic_opt_available:
            # Legacy path: restrict the 26-class head to the requested options.
            q = ChoiceQuestion(id="dynamic", question=question, choices=list(options))
            return self.decide(state, [q], max_length=max_length).results["dynamic"]

        opt_id = self.tokenizer.opt_token_id
        parts = [f"[STATE] {state} [SEP] [QUESTION] {question}"]
        for opt in options:
            parts.append(f"[OPT] {opt}")
        text = " ".join(parts)

        ids = self.tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=True)
        # Head+tail truncation keeps the trailing options visible (context-rot).
        if len(ids) > max_length:
            head_n = (max_length - 2) // 2
            tail_n = max_length - 2 - head_n
            ids = ids[:head_n] + ids[-tail_n:]

        positions = [i for i, tid in enumerate(ids) if tid == opt_id]
        if not positions:
            return ChoiceResult(choice="", confidence=0.0, probabilities={})

        tensor_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        tensor_mask = torch.tensor([[1] * len(ids)], dtype=torch.long, device=self.device)
        # Pad positions to a fixed K so the batch stays rectangular.
        K = len(positions)
        opt_positions = torch.tensor([positions + [-1] * (K - len(positions))], dtype=torch.long, device=self.device)

        try:
            with torch.no_grad():
                out = self.model(tensor_ids, attention_mask=tensor_mask, opt_positions=opt_positions)
            logits = out["dynamic_opt_logits"][0].float().cpu()
        except Exception as cuda_err:
            if "cuda" in str(type(cuda_err)).lower() or "cuda" in str(cuda_err).lower():
                if not getattr(self, "_cuda_failover_done", False):
                    self._cuda_failover_done = True
                    logger.warning("CUDA inference failed (%s); failing over to INT8 ONNX CPU.", cuda_err)
                    if self.reload(force_device="onnx"):
                        return self.decide_options(state, question, options, max_length=max_length)
            raise

        probs = torch.softmax(logits, dim=-1).numpy()
        best_idx = int(torch.argmax(logits).item())
        dist = {options[i]: float(probs[i]) for i in range(min(K, len(options)))}
        total = sum(dist.values()) or 1.0
        dist = {k: v / total for k, v in dist.items()}
        best = options[best_idx] if best_idx < len(options) else ""
        return ChoiceResult(choice=best, confidence=float(dist.get(best, 0.0)), probabilities=dist)

    def _get_dopt(self):
        """Lazy-инициализация DoPT-солвера (ONNX, фолбэк torch). None при неудаче.
        Негативный кэш (False): неудачная загрузка не ретраится на каждом вызове."""
        if self.dopt_solver is None:
            try:
                from src.leeth.dopt_solver import DoptSolver

                self.dopt_solver = DoptSolver()
            except Exception as e:
                logger.warning("DoPT solver unavailable: %s", e)
                self.dopt_solver = False
        return self.dopt_solver or None

    def decide_dopt(self, state: str, question: str, options: List[str]) -> ChoiceResult:
        """Universal Task Solver: оптимальная опция по состоянию (zero-shot по списку
        опций). Независим от прод-роутера (decide/decide_batch не тронуты).
        Формат промпта — байт-в-байт как в kernel_dopt (см. dopt_solver.py)."""
        v2 = self._use_v2()
        if v2 is not None and options:
            return v2.choose(state, question, list(options))
        solver = self._get_dopt()
        if solver is None or not options:
            return ChoiceResult(choice="", confidence=0.0, probabilities={})
        try:
            r = solver.decide(state, question, list(options))
            return ChoiceResult(choice=r.choice, confidence=r.confidence, probabilities=r.probabilities)
        except Exception as e:
            logger.warning("decide_dopt failed: %s", e)
            return ChoiceResult(choice="", confidence=0.0, probabilities={})

    def decide_batch(
        self,
        states: List[str],
        questions: List[BaseQuestion],
        max_length: int = 512,
    ) -> List[LeethDecisionBatch]:
        """Fan-out: evaluates typed questions over MULTIPLE states in one forward pass.

        Jev-style "array of text as state": N states × M questions = one batched
        inference, results split back per state. Falls back to INT8 ONNX CPU on
        CUDA failure exactly once, like decide().
        """
        t0 = time.perf_counter()
        states = list(states or [])
        if not states or not questions:
            return [LeethDecisionBatch(state_preview=s[:100], results={}, elapsed_ms=0.0, device=self.device) for s in states]
        v2 = self._use_v2()
        if v2 is not None:
            return [v2.decide(s, questions) for s in states]

        if not self._initialized:
            if not self.initialize():
                # Graceful degradation when weights are not yet generated
                return [
                    LeethDecisionBatch(state_preview=s[:100], results={}, elapsed_ms=0.0, device=self.device)
                    for s in states
                ]

        assert self.tokenizer is not None

        # Backend split: mmBERT serves choice (its validated strength: 66.1%
        # prod-battery vs 55.0% legacy, 95% dispatch precision at the gates)
        # and the need_tool noul question (99.5% val acc vs legacy 88.1%).
        # Other noul questions (is_safe/has_error) and extract/score stay
        # legacy: the mmBERT ONNX export carries only choice/noul heads, and
        # its noul head regresses on has_error (91.1% vs 100%).
        use_mmbert = bool(self.mmbert_active)
        batch_input_ids = []
        batch_attention_masks = []
        q_order: List[Tuple[int, BaseQuestion, str, Tuple[str, int]]] = []
        mmbert_jobs: List[Tuple[int, BaseQuestion, str]] = []

        for s_idx, state in enumerate(states):
            for q in questions:
                q_type = "choice"
                if isinstance(q, NoulQuestion):
                    q_type = "noul"
                elif isinstance(q, ScoreQuestion):
                    q_type = "score"
                elif isinstance(q, ExtractQuestion):
                    q_type = "extract"

                mmbert_kind: Optional[str] = None
                if use_mmbert:
                    if q_type == "choice" and q.question in MMBERT_CHOICE_QUESTIONS:
                        mmbert_kind = "choice"
                    elif q_type == "noul" and q.question in MMBERT_NOUL_QUESTIONS:
                        mmbert_kind = "noul"
                    elif self.mmbert_du_available and q_type == "choice" and q.question == MMBERT_DEPT_QUESTION:
                        mmbert_kind = "dept"
                    elif self.mmbert_du_available and q_type == "score" and q.question == MMBERT_URGENCY_QUESTION:
                        mmbert_kind = "urgency"
                    elif isinstance(q, ChoiceQuestion) and q.choices and not any(c in self.class_to_idx for c in q.choices):
                        mmbert_kind = "dopt"
                if mmbert_kind:
                    mmbert_jobs.append((s_idx, q, mmbert_kind))
                    q_order.append((s_idx, q, mmbert_kind, ("mmbert", len(mmbert_jobs) - 1)))
                    continue

                type_token = f"[{q_type.upper()}]"
                text = f"[STATE] {state} [SEP] [QUESTION] {type_token} {q.question}"

                ids = self.tokenizer.encode(
                    text,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    add_special_tokens=True,
                )
                mask = [1 if tid != self.tokenizer.pad_token_id else 0 for tid in ids]

                batch_input_ids.append(ids)
                batch_attention_masks.append(mask)
                q_order.append((s_idx, q, q_type, ("legacy", len(batch_input_ids) - 1)))

        # ── Legacy forward: only if any question was routed to the legacy model ──
        outputs: Optional[Dict[str, torch.Tensor]] = None
        tensor_ids = None
        tensor_masks = None
        if batch_input_ids:
            # Pad to max length in current batch
            max_seq_len = max(len(ids) for ids in batch_input_ids)
            padded_ids = [ids + [self.tokenizer.pad_token_id] * (max_seq_len - len(ids)) for ids in batch_input_ids]
            padded_masks = [mask + [0] * (max_seq_len - len(mask)) for mask in batch_attention_masks]

            if self.ort_session is not None:
                import numpy as np

                ort_inputs = {
                    "input_ids": np.array(padded_ids, dtype=np.int64),
                    "attention_mask": np.array(padded_masks, dtype=np.int64),
                }
                raw_outs = self.ort_session.run(None, ort_inputs)
                self.last_pooled_repr = raw_outs[1]
                # 0: hidden, 1: pooled, 2: choice_logits, 3: noul_logits, 4: noul_probs,
                # 5: score_logits, 6: start_logits, 7: end_logits, 8: mlm_logits
                outputs = {
                    "choice_logits": torch.from_numpy(raw_outs[2]),
                    "noul_probs": torch.from_numpy(raw_outs[4]),
                    "score_logits": torch.from_numpy(raw_outs[5]),
                    "start_logits": torch.from_numpy(raw_outs[6]),
                    "end_logits": torch.from_numpy(raw_outs[7]),
                    "mlm_logits": torch.from_numpy(raw_outs[8]),
                }
                tensor_ids = torch.tensor(padded_ids, dtype=torch.long)
                tensor_masks = torch.tensor(padded_masks, dtype=torch.long)
            else:
                assert self.model is not None
                tensor_ids = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
                tensor_masks = torch.tensor(padded_masks, dtype=torch.long, device=self.device)
                try:
                    with torch.no_grad():
                        outputs = self.model(tensor_ids, attention_mask=tensor_masks)
                        if isinstance(outputs, dict) and "pooled_repr" in outputs:
                            self.last_pooled_repr = outputs["pooled_repr"]
                except Exception as cuda_err:
                    # GPU may be exhausted by the main LLM (VRAM contention) or in a
                    # bad state. Fail over to the INT8 ONNX CPU backend exactly once.
                    if "cuda" in str(type(cuda_err)).lower() or "cuda" in str(cuda_err).lower():
                        if not getattr(self, "_cuda_failover_done", False):
                            self._cuda_failover_done = True
                            logger.warning("CUDA inference failed (%s); failing over to INT8 ONNX CPU.", cuda_err)
                            if self.reload(force_device="onnx"):
                                return self.decide_batch(states, questions, max_length=max_length)
                    raise

        # ── mmBERT solo forwards (batch-1 per question) ──
        mmbert_outputs: List[Dict[str, torch.Tensor]] = []
        if mmbert_jobs:
            import numpy as np

            for s_idx, q, q_type in mmbert_jobs:
                if q_type == "dopt":
                    dopt_res = self.decide_options(states[s_idx], q.question, getattr(q, "choices", []), max_length=max_length)
                    mmbert_outputs.append({"dopt_result": dopt_res})
                    continue
                text = f"{states[s_idx]}\n[QUESTION] {q.question}"
                try:
                    # Head+tail truncation: naive tokenizer truncation cuts the
                    # TAIL — but the actual query sits at the end of the state
                    # (context-rot test: dept 14/14 -> 3/14 at ~1k-token prefixes
                    # because the case itself was cut off). Keep head (envelope)
                    # + tail (query + question) when over budget.
                    raw_ids = self.mmbert_tokenizer.encode(text, add_special_tokens=True)
                    # Task-Adaptive Clamping: dynamically allocate context budget based on domain complexity
                    state_low = states[s_idx].lower()
                    is_browser = "role" in state_low or "button" in state_low or "link" in state_low or "aria" in state_low
                    is_traceback = "traceback" in state_low or "error" in state_low or "exception" in state_low or "stack" in state_low
                    if q_type == "choice" and not (is_browser or is_traceback):
                        task_budget = 256  # Ultra-fast CLI routing (5-8 ms)
                    elif is_browser:
                        task_budget = 2048 # Rich Accessibility Tree with full UI elements (15-25 ms)
                    elif is_traceback:
                        task_budget = min(8192, max(1024, len(raw_ids) + 32)) # Deep compiler/runtime error traces
                    else:
                        task_budget = self.mmbert_max_length

                    if len(raw_ids) > task_budget:
                        head_n = (task_budget - 2) // 2
                        tail_n = task_budget - 2 - head_n
                        raw_ids = raw_ids[:head_n] + raw_ids[-tail_n:]
                        feed = {
                            "input_ids": np.asarray([raw_ids], dtype=np.int64),
                            "attention_mask": np.asarray([[1] * len(raw_ids)], dtype=np.int64),
                        }
                    else:
                        enc = self.mmbert_tokenizer(text, truncation=True, max_length=task_budget, return_tensors="np")
                        feed = {
                            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
                        }
                    outs = self.mmbert_session.run(None, feed)
                    if len(outs) >= 5:
                        self.last_pooled_repr = outs[4]
                        self.last_pooled_reprs[q.id] = outs[4]
                except Exception as mmbert_err:
                    # Quantized session failed mid-flight: disable the router and
                    # re-route everything through the legacy model (single failover —
                    # the recursive call sees mmbert_active=False).
                    logger.warning("mmBERT forward failed (%s); failing over to legacy model.", mmbert_err)
                    self.mmbert_active = False
                    return self.decide_batch(states, questions, max_length=max_length)
                if q_type == "choice":
                    mmbert_outputs.append({"choice_logits": torch.from_numpy(outs[0])})
                elif q_type == "noul":
                    mmbert_outputs.append({"noul_probs": torch.sigmoid(torch.from_numpy(outs[1]).float())})
                elif q_type == "dept":
                    mmbert_outputs.append({"dept_logits": torch.from_numpy(outs[2])})
                else:  # urgency
                    mmbert_outputs.append({"urgency_logits": torch.from_numpy(outs[3])})

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Per-state MLM familiarity: mean NLL over content tokens (OOD proxy).
        # Computed from the SAME forward pass — no extra inference cost.
        # Only valid when the legacy batch covers ALL questions of every state
        # (mmBERT-served questions have no MLM head).
        if mmbert_jobs or outputs is None:
            state_nlls: List[Optional[float]] = [None] * len(states)
        else:
            state_nlls = self._state_nlls(outputs, tensor_ids, tensor_masks, len(states), len(questions))

        # Decode outputs for each (state, question) pair
        self.last_pooled_reprs.clear()
        per_state_results: List[Dict[str, LeethResult]] = [{} for _ in states]
        for i, (s_idx, q, q_type, src) in enumerate(q_order):
            backend, row = src
            if backend != "mmbert" and "raw_outs" in locals():
                self.last_pooled_reprs[q.id] = raw_outs[1][row : row + 1]
            # mmBERT solos are batch-1 tensors: the job index selects WHICH solo
            # output, the tensor row is always 0.
            outs_i = mmbert_outputs[row] if backend == "mmbert" else outputs
            trow = 0 if backend == "mmbert" else row
            results = per_state_results[s_idx]

            if backend == "mmbert" and q_type == "dopt":
                results[q.id] = outs_i["dopt_result"]
                continue

            if backend == "mmbert" and q_type == "dept":
                # Variant A: dedicated 5-class dept head — classes are NOT in the
                # routing vocabulary, decode from dept_logits only.
                logits = outs_i["dept_logits"][trow].cpu()
                probs = torch.softmax(logits, dim=-1).numpy()
                candidates = list(q.choices) if q.choices else list(MMBERT_DEPT_CLASSES)
                cand_probs: Dict[str, float] = {}
                for c in candidates:
                    idx = MMBERT_DEPT_CLASSES.index(c) if c in MMBERT_DEPT_CLASSES else None
                    cand_probs[c] = float(probs[idx]) if idx is not None and idx < len(probs) else 0.001
                total_prob = sum(cand_probs.values()) or 1.0
                normalized_dist = {k: v / total_prob for k, v in cand_probs.items()}
                best_choice = max(normalized_dist, key=normalized_dist.get)
                results[q.id] = ChoiceResult(
                    choice=best_choice,
                    confidence=float(normalized_dist[best_choice]),
                    probabilities=normalized_dist,
                )
                continue

            if backend == "mmbert" and q_type == "urgency":
                # Variant A: dedicated 3-level urgency head (score semantics).
                logits = outs_i["urgency_logits"][trow].cpu()
                n_levels = max(2, min(getattr(q, "levels", MMBERT_URGENCY_LEVELS), logits.size(-1)))
                level_probs = torch.softmax(logits[:n_levels], dim=-1).tolist()
                level = int(max(range(n_levels), key=lambda idx: level_probs[idx])) + 1
                confidence = float(max(level_probs))
                expected = sum((idx + 1) * p for idx, p in enumerate(level_probs))
                denom_levels = max(1, n_levels - 1)
                score_val = max(0.0, min(100.0, (expected - 1.0) / denom_levels * 100.0))
                label = f"Level {level}/{n_levels}"
                if q.rubric and level - 1 < len(q.rubric):
                    label = q.rubric[level - 1]
                results[q.id] = ScoreResult(
                    score=score_val,
                    level=level,
                    label=label,
                    probabilities={str(idx + 1): p for idx, p in enumerate(level_probs)},
                    confidence=confidence,
                )
                continue

            if isinstance(q, ChoiceQuestion):
                logits = outs_i["choice_logits"][trow].cpu()
                probs = torch.softmax(logits, dim=-1).numpy()

                # If question defines specific candidates, restrict to candidate choices
                if q.choices:
                    candidate_probs = {}
                    for c in q.choices:
                        idx = self.class_to_idx.get(c)
                        candidate_probs[c] = float(probs[idx]) if idx is not None and idx < len(probs) else 0.001

                    # Normalize candidate distribution
                    total_prob = sum(candidate_probs.values()) or 1.0
                    normalized_dist = {k: v / total_prob for k, v in candidate_probs.items()}
                    best_choice = max(normalized_dist, key=normalized_dist.get)
                    confidence = float(normalized_dist[best_choice])

                    results[q.id] = ChoiceResult(
                        choice=best_choice,
                        confidence=confidence,
                        probabilities=normalized_dist,
                    )
                else:
                    best_idx = int(torch.argmax(logits).item())
                    best_choice = self.idx_to_class.get(best_idx, f"class_{best_idx}")
                    confidence = float(probs[best_idx])
                    results[q.id] = ChoiceResult(
                        choice=best_choice,
                        confidence=confidence,
                        probabilities={best_choice: confidence},
                    )

            elif isinstance(q, NoulQuestion):
                prob = float(outs_i["noul_probs"][trow].item())
                results[q.id] = NoulResult(
                    value=bool(prob >= 0.5),
                    probability=prob,
                )

            elif isinstance(q, ScoreQuestion):
                logits = outs_i["score_logits"][trow].cpu()
                n_levels = max(2, min(q.levels, logits.size(-1)))
                level_probs = torch.softmax(logits[:n_levels], dim=-1).tolist()
                level = int(max(range(n_levels), key=lambda idx: level_probs[idx])) + 1

                # Modal concentration confidence (Jev-analysis formula):
                # C = max(0, 1 - mean_abs_dev_from_mode / (floor(L^2/4)/L))
                mode_idx = level - 1
                denom = (n_levels * n_levels // 4) / n_levels
                mad = sum(p * abs(idx - mode_idx) for idx, p in enumerate(level_probs))
                confidence = max(0.0, 1.0 - mad / denom) if denom > 0 else 0.0

                # Expected level -> 0..100 scale without offset bias: Level 1 -> 0.0, Level n_levels -> 100.0
                expected = sum((idx + 1) * p for idx, p in enumerate(level_probs))
                denom_levels = max(1, n_levels - 1)
                score_val = max(0.0, min(100.0, (expected - 1.0) / denom_levels * 100.0))

                label = f"Level {level}/{n_levels}"
                if q.rubric and level - 1 < len(q.rubric):
                    label = q.rubric[level - 1]

                results[q.id] = ScoreResult(
                    score=score_val,
                    level=level,
                    label=label,
                    probabilities={str(idx + 1): p for idx, p in enumerate(level_probs)},
                    confidence=confidence,
                )

            elif isinstance(q, ExtractQuestion):
                start_logits = outs_i["start_logits"][trow].cpu()
                end_logits = outs_i["end_logits"][trow].cpu()

                # Find valid span with maximum start_logit + end_logit where start <= end
                best_start, best_end = 0, 0
                best_score = -float("inf")
                seq_l = int(tensor_masks[trow].sum().item())

                for s in range(seq_l):
                    for e in range(s, min(s + 32, seq_l)):  # Cap maximum span length
                        sc = start_logits[s].item() + end_logits[e].item()
                        if sc > best_score:
                            best_score = sc
                            best_start, best_end = s, e

                span_ids = tensor_ids[trow, best_start : best_end + 1].cpu().tolist()
                extracted_text = self.tokenizer.decode(span_ids, skip_special_tokens=True)

                # Snap span start to a word boundary: SentencePiece-style
                # continuation tokens lack the SPACE_CHAR prefix, so a span
                # starting mid-word ("ли папку build" from "удали папку build")
                # is a pointer error — walk back to the word's first token.
                while best_start > 0:
                    tok = self.tokenizer.inv_vocab.get(tensor_ids[trow, best_start].item(), "")
                    if not tok or tok in self.tokenizer.special_tokens:
                        break
                    if tok.startswith(self.tokenizer.SPACE_CHAR):
                        break
                    prev_tok = self.tokenizer.inv_vocab.get(tensor_ids[trow, best_start - 1].item(), "")
                    if prev_tok in self.tokenizer.special_tokens:
                        break
                    best_start -= 1
                    span_ids = tensor_ids[trow, best_start : best_end + 1].cpu().tolist()
                    extracted_text = self.tokenizer.decode(span_ids, skip_special_tokens=True)

                # If extracted span ends with a dot (e.g. "config."), include the extension token
                if extracted_text.endswith(".") and best_end + 1 < seq_l:
                    next_id = tensor_ids[trow, best_end + 1].cpu().item()
                    next_tok = self.tokenizer.decode([next_id], skip_special_tokens=True).strip()
                    if next_tok and not next_tok.startswith(" ") and not next_tok.startswith("\n"):
                        best_end += 1
                        span_ids = tensor_ids[trow, best_start : best_end + 1].cpu().tolist()
                        extracted_text = self.tokenizer.decode(span_ids, skip_special_tokens=True)

                # Strip accidental leading or trailing whitespace
                extracted_text = extracted_text.strip()

                # Calibrate span confidence via geometric mean of boundary softmax probabilities
                start_probs = F.softmax(start_logits[:seq_l], dim=-1)
                end_probs = F.softmax(end_logits[:seq_l], dim=-1)
                p_start = float(start_probs[best_start].item()) if best_start < seq_l else 0.0
                p_end = float(end_probs[best_end].item()) if best_end < seq_l else 0.0
                confidence = float(max(0.01, min(1.0, math.sqrt(p_start * p_end))))

                results[q.id] = ExtractResult(
                    text=extracted_text,
                    span=(best_start, best_end),
                    confidence=confidence,
                )

        return [
            LeethDecisionBatch(
                state_preview=state[:120],
                results=per_state_results[s_idx],
                elapsed_ms=elapsed_ms,
                device=self.device,
                state_nll=state_nlls[s_idx],
            )
            for s_idx, state in enumerate(states)
        ]

    def _state_nlls(
        self,
        outputs: Dict[str, torch.Tensor],
        tensor_ids: torch.Tensor,
        tensor_masks: torch.Tensor,
        n_states: int,
        n_questions: int,
    ) -> List[Optional[float]]:
        """Mean MLM NLL per state over content tokens (low = familiar territory)."""
        try:
            mlm = outputs.get("mlm_logits")
            if mlm is None:
                return [None] * n_states
            bsz, seq_len, vocab = mlm.shape
            labels = tensor_ids[:, :seq_len].clone()
            labels[tensor_masks[:, :seq_len] == 0] = -100
            # Exclude special/decision tokens: they are constant decorations.
            labels[labels < len(self.tokenizer.special_tokens)] = -100
            if (labels != -100).sum() == 0:
                return [None] * n_states
            nll = F.cross_entropy(
                mlm.reshape(-1, vocab).float(),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(bsz, seq_len)
            token_mask = (labels != -100).float()
            row_nll = (nll * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1.0)
            per_state = []
            for s in range(n_states):
                rows = row_nll[s * n_questions:(s + 1) * n_questions]
                per_state.append(float(rows.mean().item()))
            return per_state
        except Exception as e:
            logger.debug("state_nll computation failed: %s", e)
            return [None] * n_states


leeth = LeethRuntime.get_instance()
