"""
src/leeth/dopt_solver.py — Runtime-солвер Universal Task Solver (DoPT v7 DAgger).

Играет в задачи формата «состояние → вопрос → варианты» (змейка, лабиринт,
крестики-нолики, любые новые домены с идеальным экспертом): выбирает оптимальный
ход argmax'ом по опциям. Метрики v7: snake 99.3%, maze 100%, ttt 99.9%,
zero-shot 12x12 94.9%, win-rate роллаута 90% (эксперт 93%).

Инференс: единый ONNX-граф (.argent/models/dopt/dopt_v2.onnx, см. dopt_export.py)
через onnxruntime (CPU, ~0.1-0.3 с/ход) с фолбэком на torch-чекпоинт.

Формат промпта — байт-в-байт как в кернеле kernel_dopt (иначе модель вне
распределения): "[STATE] {state} [QUESTION] {question} [OPT] {opt}...".
Маркер [OPT] = 3 BPE-токена, позиции ищутся подпоследовательностью.

Использование:
    from src.leeth.dopt_solver import DoptSolver
    solver = DoptSolver()                      # авто-поиск ONNX, фолбэк torch
    result = solver.decide(state_text, question, ["вверх", "вниз", "влево", "вправо"])
    result.choice      # "влево"
    result.confidence  # 0.87
    result.probabilities  # {"вверх": 0.02, ...}
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from logger import get_logger

logger = get_logger("leeth_dopt_solver")

DEFAULT_ONNX = Path(".argent/models/dopt/dopt_v2.onnx")
DEFAULT_CKPT = Path(".argent/models/mmbert_dynamic_opt_v2.pt")
DEFAULT_BASE = Path(".argent/models/mmbert-argent")
DEFAULT_TOKENIZER = Path(".argent/models/mmbert-argent")
MAX_LEN = 256


def build_text(state_text: str, question: str, options: List[str]) -> str:
    """Единый формат промпта (байт-в-байт как build_text в kernel_dopt)."""
    return f"[STATE] {state_text} [QUESTION] {question} " + " ".join(f"[OPT] {o}" for o in options)


def find_marker_positions(ids: List[int], marker_ids: List[int]) -> List[int]:
    """Все вхождения подпоследовательности маркера [OPT] в токенах."""
    L = len(marker_ids)
    return [i for i in range(len(ids) - L + 1) if ids[i : i + L] == marker_ids]


class DoptResult:
    """Результат выбора опции солвером DoPT."""

    def __init__(self, choice: str, confidence: float, probabilities: Dict[str, float], logits: Optional[List[float]] = None):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities
        self.logits = logits

    def __repr__(self):
        return f"DoptResult(choice={self.choice!r}, confidence={self.confidence:.3f})"


class DoptSolver:
    """Universal Task Solver: zero-shot выбор оптимальной опции по состоянию.

    Приоритет инференса: ONNX (быстрый CPU) → torch-чекпоинт (фолбэк).
    Формат промпта и постпроцессинг идентичны обучению (kernel_dopt).
    """

    def __init__(
        self,
        onnx_path: Optional[Path] = None,
        ckpt_path: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        tokenizer_dir: Optional[Path] = None,
        max_length: int = MAX_LEN,
    ):
        self.onnx_path = Path(onnx_path) if onnx_path else DEFAULT_ONNX
        self.ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE
        self.tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else DEFAULT_TOKENIZER
        self.max_length = max_length
        self._sess = None
        self._torch = None  # (base, head, marker_ids)
        self._tok = None
        self.backend = None  # "onnx" | "torch"
        self._load()

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------

    def _load(self):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(self.tokenizer_dir))
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 8
                opts.inter_op_num_threads = 1
                self._sess = ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.backend = "onnx"
                logger.info("DoPT solver: ONNX %s (%.0f MB)", self.onnx_path.name, self.onnx_path.stat().st_size / 1e6)
                return
            except Exception as e:
                logger.warning("DoPT ONNX failed to load (%s); falling back to torch.", e)
        if self.ckpt_path.exists():
            from src.leeth.dopt_export import DynamicOptHead, load_dopt_torch

            import torch

            base, head, tok, marker_ids, _ = load_dopt_torch(self.ckpt_path, self.base_dir)
            self._torch = (base, head, marker_ids)
            self._tok = tok
            self.backend = "torch"
            logger.info("DoPT solver: torch checkpoint %s", self.ckpt_path.name)
            return
        raise FileNotFoundError(f"DoPT solver: ни ONNX ({self.onnx_path}), ни чекпоинт ({self.ckpt_path}) не найдены")

    @property
    def is_available(self) -> bool:
        return self.backend in ("onnx", "torch")

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    def _encode(self, texts: List[str]):
        """Тексты → тензоры графа (input_ids, attention_mask, state_mask, opt_positions)."""
        import torch

        tok = self._tok
        enc = tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        B, L = enc["input_ids"].shape
        ids_list = enc["input_ids"].tolist()
        marker_ids = self._marker_ids()
        pos_lists = [find_marker_positions(ids, marker_ids) for ids in ids_list]
        K = max(len(p) for p in pos_lists)
        state_mask = torch.zeros(B, L, dtype=torch.float32)
        opt_positions = torch.full((B, K), -1, dtype=torch.long)
        for j, pos in enumerate(pos_lists):
            if not pos:
                raise ValueError(f"маркер [OPT] не найден в промпте #{j} (маркер_ids={marker_ids})")
            state_mask[j, : pos[0]] = 1.0
            opt_positions[j, : len(pos)] = torch.tensor(pos, dtype=torch.long)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "state_mask": state_mask,
            "opt_positions": opt_positions,
        }

    def _marker_ids(self) -> List[int]:
        if self.backend == "onnx":
            if self._marker_ids_cache is None:
                self._marker_ids_cache = list(self._tok.encode("[OPT]", add_special_tokens=False))
            return self._marker_ids_cache
        return self._torch[2]

    _marker_ids_cache = None

    def _infer_onnx(self, tensors) -> np.ndarray:
        out = self._sess.run(None, {k: v.numpy() for k, v in tensors.items()})
        return out[0]

    def _infer_torch(self, tensors) -> np.ndarray:
        import torch

        from src.leeth.dopt_export import DoptOnnxWrapper

        base, head, _ = self._torch
        wrapper = DoptOnnxWrapper(base, head).eval()
        with torch.no_grad():
            logits = wrapper(**tensors)
        return logits.numpy()

    def decide_batch(self, states: List[str], question: str, options_per_state: List[List[str]]) -> List[DoptResult]:
        """Батч-инференс: N состояний × (свой список опций). Опции шафлить НЕ нужно —
        модель сопоставляет текст опции с состоянием (обучена на шафле)."""
        if not states:
            return []
        texts = [build_text(s, question, opts) for s, opts in zip(states, options_per_state)]
        tensors = self._encode(texts)
        logits = self._infer_onnx(tensors) if self.backend == "onnx" else self._infer_torch(tensors)
        results = []
        for j, opts in enumerate(options_per_state):
            lg = logits[j, : len(opts)]
            probs = np.exp(lg - lg.max())
            probs = probs / probs.sum()
            best = int(np.argmax(lg))
            results.append(DoptResult(
                choice=opts[best],
                confidence=float(probs[best]),
                probabilities={o: float(p) for o, p in zip(opts, probs)},
                logits=[float(x) for x in lg],
            ))
        return results

    def decide(self, state_text: str, question: str, options: List[str]) -> DoptResult:
        """Выбор оптимальной опции для одного состояния."""
        return self.decide_batch([state_text], question, [options])[0]


def main():
    """Smoke: солвер решает по одному сэмплу каждого домена (task_domains)."""
    import random
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from task_domains import GENERATORS

    logging.basicConfig(level=logging.INFO)
    solver = DoptSolver()
    rng = random.Random(1)
    for domain, gen in GENERATORS.items():
        s = None
        while s is None:
            s = gen(rng)
        r = solver.decide(s.state_text, s.question, s.options)
        ok = s.optimal_set and (s.options.index(r.choice) in s.optimal_set)
        print(f"[{domain}] choice={r.choice!r} conf={r.confidence:.3f} optimal={ok} "
              f"probs={ {k: round(v, 3) for k, v in r.probabilities.items()} }")


if __name__ == "__main__":
    main()