"""
Local inference engine and agent delegation module for Argent Brain.

Provides:
- Resource auditing (RAM, VRAM, CPU) to safely choose model tiers.
- Concurrency limiting via semaphores.
- Support for Cloud Tier (T0) via free OpenRouter / auxiliary providers.
- Local Tier (T1-T3) execution via Ollama or llama_cpp with automatic Cloud Fallback.
- delegate_local tool implementation.
"""

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from typing import Any

import config
from logger import get_logger

logger = get_logger("local_inference")

# Tool schema for Argent LLM
DELEGATE_LOCAL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_local",
        "description": (
            "Delegates a self-contained, narrow subtask (e.g. data extraction, text rewriting, "
            "formula verification, code snippet drafting, unit test boilerplate, regex generation) "
            "to an auxiliary small model (local or cloud). Does not consume main reasoning context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Clear and specific task/instruction for the auxiliary model."
                },
                "context": {
                    "type": "string",
                    "description": "Optional compact context or data to process (keep concise)."
                },
                "tier": {
                    "type": "integer",
                    "description": "Preferred model tier (-1: Needle micro — CPU extraction in <50ms, 0: Cloud auxiliary model, 1: <=0.5B fast, 2: <=1.5B balanced, 3: <=3B). Defaults to current configuration.",
                    "enum": [-1, 0, 1, 2, 3]
                }
            },
            "required": ["task"]
        }
    }
}


# ── System Resource Auditor ──────────────────────────────────────────

class SystemResourceAuditor:
    """Inspects host RAM, VRAM, and CPU to prevent out-of-memory crashes."""

    @staticmethod
    def get_free_ram_gb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 ** 3)
        except Exception:
            return 4.0

    @staticmethod
    def get_total_ram_gb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return 16.0

    @staticmethod
    def get_cpu_percent() -> float:
        try:
            import psutil
            return psutil.cpu_percent(interval=None) or 0.0
        except Exception:
            return 0.0

    @staticmethod
    def get_gpu_vram_info() -> dict[str, Any]:
        """Queries NVIDIA GPU VRAM via nvidia-smi if available."""
        res = {
            "available": False,
            "total_mb": 0,
            "free_mb": 0,
            "used_mb": 0,
            "is_occupied": False,
        }
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=2).decode("utf-8").strip()
            if out:
                line = out.splitlines()[0]
                total, free, used = [int(v.strip()) for v in line.split(",")]
                res["available"] = True
                res["total_mb"] = total
                res["free_mb"] = free
                res["used_mb"] = used
                # If less than 1GB free or >85% VRAM used, GPU is occupied by a heavy model
                res["is_occupied"] = (free < 1024) or (total > 0 and (used / total > 0.85))
        except Exception:
            pass
        return res

    @classmethod
    def get_inference_device(cls) -> str:
        """Decide whether to offload local models to GPU or keep CPU-only."""
        gpu = cls.get_gpu_vram_info()
        if gpu["available"] and not gpu["is_occupied"] and gpu["free_mb"] >= 1500:
            return "gpu"
        return "cpu"

    @classmethod
    def available_tier(cls, max_user_tier: int = None) -> int | None:
        """Determines the highest safe tier runnable right now.
        Returns:
            0: Cloud tier (always safe, 0 MB local RAM)
            1: T1 (<=0.5B, ~350MB RAM)
            2: T2 (0.5-1.5B, ~1GB RAM)
            3: T3 (1.5-3B, ~2GB RAM)
            None: System overloaded and cloud fallback disabled
        """
        if max_user_tier is None:
            max_user_tier = config.get_local_model_tier()

        if max_user_tier == 0:
            return 0

        free_ram = cls.get_free_ram_gb()
        cpu = cls.get_cpu_percent()

        if free_ram < 1.0 or cpu > 95:
            if config.is_local_agent_cloud_fallback():
                return 0
            return None

        if free_ram < 1.8:
            max_res = 1
        elif free_ram < 3.2:
            max_res = 2
        else:
            max_res = 3

        return min(max_res, max_user_tier)

    @classmethod
    def max_concurrent_agents(cls, tier: int = 1) -> int:
        if tier == 0:
            return 4
        free_ram = cls.get_free_ram_gb()
        avail = max(0.0, free_ram - 2.0)  # Reserve 2GB for OS and Argent
        req = {1: 0.35, 2: 1.0, 3: 2.0}.get(tier, 1.0)
        slots = int(avail / req)
        return max(1, min(4, slots))


# ── Concurrency & Lifecycle ──────────────────────────────────────────

_semaphore_lock = threading.Lock()
_agent_semaphore = threading.Semaphore(2)

def _acquire_slot(timeout: float = 5.0) -> bool:
    return _agent_semaphore.acquire(timeout=timeout)

def _release_slot():
    try:
        _agent_semaphore.release()
    except ValueError:
        pass


# ── Cloud Tier Runner (T0) ───────────────────────────────────────────

def _run_cloud_tier(task: str, context: str = "", timeout: float = 30.0) -> str:
    """Executes task using configured cloud model (e.g. OpenRouter free / auxiliary)."""
    from providers import create_provider

    provider_name = config.get_local_agent_cloud_provider()
    model_name = config.get_local_agent_cloud_model()

    system_prompt = (
        "You are a fast, precise auxiliary sub-agent. "
        "Perform the requested task directly and concisely. "
        "Return ONLY the requested result without conversational filler."
    )

    user_content = f"Task:\n{task}"
    if context and context.strip():
        user_content += f"\n\nContext:\n{context.strip()}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        provider = create_provider(provider_name)
        result = provider.sync_chat(
            model=model_name,
            messages=messages,
            temperature=0.2,
            timeout=timeout,
        )
        return result.strip()
    except Exception as e:
        return f"[ERROR T0-Cloud ({provider_name}/{model_name})]: {e}"


# ── Local Tier Runner (T1-T3) ────────────────────────────────────────

def _ollama_url() -> str:
    from src.agent.local_backend import url_for
    return url_for("ollama").rstrip("/")


def _is_ollama_online(timeout: float = 1.0) -> bool:
    """Quick check if the Ollama daemon answers (at OLLAMA_HOST, default port 11434)."""
    try:
        req = urllib.request.Request(_ollama_url() + "/api/tags", headers={"User-Agent": "Argent"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _run_local_ollama(task: str, context: str = "", tier: int = 1, timeout: float = 30.0) -> str:
    """Executes task via local Ollama API."""
    tag_map = {
        1: "smollm2:360m",
        2: "qwen2.5:1.5b",
        3: "qwen2.5:3b",
    }
    model_tag = tag_map.get(tier, "smollm2:360m")

    prompt = (
        f"You are a concise sub-agent.\n"
        f"Task: {task}\n"
    )
    if context and context.strip():
        prompt += f"Context: {context.strip()}\n"
    prompt += "Answer directly:"

    payload = json.dumps({
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        }
    }).encode("utf-8")

    from src.agent import governor
    from src.agent.local_backend import classify

    req = urllib.request.Request(
        _ollama_url() + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Argent"},
    )

    try:
        # The sub-agent shares the server with the main model: it waits its turn in the same lane.
        with governor.slot(classify("ollama", model_tag), f"delegate_local:{model_tag}"):
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama execution failed: {e}")


_gguf_lock = threading.Lock()
_gguf_cached = None          # (model_path, n_gpu_layers, Llama)


def _gguf_model(model_path: str, n_gpu: int):
    """The loaded model, kept between calls: loading a GGUF is seconds of disk and gigabytes of RAM, and used to
    happen on every single delegate_local call."""
    global _gguf_cached
    from llama_cpp import Llama  # type: ignore
    with _gguf_lock:
        if _gguf_cached and _gguf_cached[0] == model_path and _gguf_cached[1] == n_gpu:
            return _gguf_cached[2]
        _gguf_cached = None                      # free the previous one before loading the next
        llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=n_gpu, verbose=False)
        _gguf_cached = (model_path, n_gpu, llm)
        return llm


def _run_local_gguf(task: str, context: str = "", model_path: str = None, timeout: float = 30.0) -> str:
    """Executes task directly via llama_cpp if installed and GGUF path is provided."""
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        raise RuntimeError("llama-cpp-python is not installed")

    device = SystemResourceAuditor.get_inference_device()
    n_gpu = 99 if device == "gpu" else 0

    from src.agent import governor

    prompt = f"Task: {task}\nContext: {context}\nAnswer:"
    with governor.slot(governor.GGUF, f"delegate_local:{os.path.basename(model_path or '')}"):
        llm = _gguf_model(model_path, n_gpu)
        res = llm.create_completion(
            prompt,
            max_tokens=512,
            temperature=0.2,
            stop=["\n\n\n"],
        )
    return res["choices"][0]["text"].strip()


# ── Main Entry Point: delegate_local ─────────────────────────────────

def delegate_local(task: str, context: str = "", tier: int = None) -> str:
    """Executes a delegated task with resource checking, concurrency limits, and timeouts."""
    if not task or not task.strip():
        return "Error: 'task' parameter is required for delegate_local."

    # Needle 3 micro tier: in-process CPU extraction without network
    if tier == -1:
        return _run_needle_micro(task, context)

    timeout = config.get_local_agent_timeout()

    # Determine requested tier
    if tier is None:
        tier = config.get_local_model_tier()

    # Resource audit
    effective_tier = SystemResourceAuditor.available_tier(tier)

    # 1. Cloud Tier requested or downgraded to T0
    if effective_tier == 0:
        return _run_cloud_tier(task, context, timeout=timeout)

    # 2. Local resource exhaustion and no fallback
    if effective_tier is None:
        return "Недостаточно ресурсов системы (RAM < 1GB) для запуска локальной модели. Настройте /local_model tier cloud или /local_model fallback on."

    # 3. Concurrency slot acquisition
    acquired = _acquire_slot(timeout=5.0)
    if not acquired:
        if config.is_local_agent_cloud_fallback():
            return "[Auto-Cloud Fallback (Все локальные слоты заняты)]: " + _run_cloud_tier(task, context, timeout=timeout)
        return "Все слоты локальных агентов заняты. Попробуйте позже."

    try:
        custom_path = config.get_local_model_path()
        if custom_path:
            return _run_local_gguf(task, context, model_path=custom_path, timeout=timeout)

        if _is_ollama_online():
            return _run_local_ollama(task, context, tier=effective_tier, timeout=timeout)

        # Neither custom GGUF nor Ollama is online:
        if config.is_local_agent_cloud_fallback():
            return "[Auto-Cloud Fallback (Локальный бэкенд Ollama не запущен)]: " + _run_cloud_tier(task, context, timeout=timeout)

        return (
            "[LOCAL INFERENCE UNAVAILABLE]: Локальный бэкенд Ollama не обнаружен (http://localhost:11434 не отвечает). "
            "Запустите Ollama, либо переключитесь на облачный тир: /local_model tier cloud"
        )
    except Exception as e:
        if config.is_local_agent_cloud_fallback():
            return f"[Auto-Cloud Fallback (Ошибка локального запуска: {e})]: " + _run_cloud_tier(task, context, timeout=timeout)
        return f"[ERROR] Ошибка локальной модели: {e}"
    finally:
        _release_slot()


def _run_needle_micro(task: str, context: str = "") -> str:
    """Run structured extraction via Needle 3 in-process (~20 MB, CPU-only, <50ms)."""
    try:
        from src.agent.needle_coprocessor import coprocessor
        if not coprocessor.is_available:
            logger.info("Needle micro tier requested but cactus-needle not installed, falling back to tier 1.")
            return delegate_local(task=task, context=context, tier=1)
        if not coprocessor.initialize():
            logger.info(f"Needle init failed ({coprocessor._init_error}), falling back to tier 1.")
            return delegate_local(task=task, context=context, tier=1)

        import time
        t0 = time.perf_counter()
        prompt = f"{task}\n\n{context}".strip() if context else task
        result = coprocessor.try_fast_dispatch(prompt)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result:
            return (
                f"[Needle 3 micro: {elapsed_ms:.0f}ms] "
                f"Tool={result.tool_name}, Args={result.arguments}, Confidence={result.confidence:.2f}"
            )
        # Extraction mode: use needle.extract with raw text
        extracted = coprocessor.extract_structured(prompt, {"answer": str})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if extracted:
            return f"[Needle 3 micro: {elapsed_ms:.0f}ms] {extracted}"

        logger.debug("Needle micro could not handle task, falling back to tier 1.")
        return delegate_local(task=task, context=context, tier=1)
    except Exception as e:
        logger.warning(f"Needle micro error: {e}, falling back to tier 1.")
        return delegate_local(task=task, context=context, tier=1)
