"""How much of Ultracode this model on this PC can carry.

The mode's machinery (a planner, workers, a critic) costs several model calls per step. On a cloud model that is
money; on a local one it is minutes of a PC that is also the person's workstation, and on a small model the plan
itself is unreliable. So the mode does not run "fully" or "not at all": it picks a level —

    3  full      planner, workers side by side, critic, advisor        cloud, or a strong local setup
    2  pipeline  planner, workers ONE at a time, critic, advisor        a typical local model
    1  light     a plan the model itself follows, one critic pass       slow or small machine
    0  plain     the ordinary agent                                    the mode would not help

— from a short measurement (first-token latency, tokens per second) and the facts about the machine. The
measurement is cached per (provider, model, hardware), so it is paid once, not on every start.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from logger import get_logger

log = get_logger("calibration")

CACHE_TTL = 14 * 24 * 3600
LEVEL_NAMES = {3: "полный рой", 2: "конвейер", 1: "облегчённый", 0: "обычный агент"}


@dataclass
class Hardware:
    cores: int = 0
    ram_gb: float = 0.0
    vram_gb: Optional[float] = None

    @property
    def fingerprint(self) -> str:
        vram = f"{self.vram_gb:.0f}" if self.vram_gb else "0"
        return f"{self.cores}c-{self.ram_gb:.0f}g-{vram}v"


@dataclass
class Profile:
    provider: str
    model: str
    local: bool
    size_category: str = "medium"
    tok_per_s: Optional[float] = None
    ttft_s: Optional[float] = None
    parallelism: int = 1
    hardware: Optional[dict] = None
    measured_at: float = 0.0


def read_hardware() -> Hardware:
    hw = Hardware()
    try:
        import psutil
        hw.cores = int(psutil.cpu_count(logical=False) or psutil.cpu_count() or 0)
        hw.ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        hw.cores = os.cpu_count() or 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        totals = [float(x) for x in out.decode().split() if x.replace(".", "", 1).isdigit()]
        if totals:
            hw.vram_gb = round(max(totals) / 1024, 1)
    except Exception:
        pass
    return hw


# ── the measurement ─────────────────────────────────────────────────────────────

Probe = Callable[[str, str], Tuple[Optional[float], Optional[float]]]


def probe_model(provider: str, model: str) -> Tuple[Optional[float], Optional[float]]:
    """(seconds to the first token, tokens per second) of one short request, or (None, None)."""
    try:
        from providers import create_provider
        from src.agent import governor
        prompt = [{"role": "user", "content": "Count from 1 to 30, separated by commas. Output only the numbers."}]
        started = time.monotonic()
        first = None
        chars = 0
        tokens = None
        with governor.priority(governor.P_ADVISOR):
            for chunk in create_provider(provider).stream_chat(model, prompt, None, None, 0.0):
                text = chunk.get("content") or ""
                if text and first is None:
                    first = time.monotonic()
                chars += len(text)
                usage = chunk.get("usage")
                if usage and usage.get("completion"):
                    tokens = float(usage["completion"])
                if chars > 600:
                    break
        end = time.monotonic()
        if first is None:
            return None, None
        if tokens is None:
            tokens = max(1.0, chars / 3.2)
        span = max(0.05, end - first)
        return round(first - started, 2), round(tokens / span, 1)
    except Exception as e:
        log.info("calibration probe failed: %s", e)
        return None, None


def _cache_path() -> Path:
    return Path.home() / ".argent" / "run" / "calibration.json"


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _store_cache(key: str, profile: Profile) -> None:
    data = _load_cache()
    data[key] = asdict(profile)
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    except OSError as e:
        log.debug("calibration cache not saved: %s", e)


def calibrate(provider: str, model: str, force: bool = False, probe: Optional[Probe] = None,
              now: Callable[[], float] = time.time) -> Profile:
    """The profile of (provider, model) on this machine; measured once and cached."""
    from src.agent.local_backend import classify
    import config

    backend = classify(provider, model)
    hw = read_hardware()
    category = config.get_model_size_category(model)
    try:
        parallelism = config.get_local_parallelism()
    except Exception:
        parallelism = 1
    profile = Profile(provider=provider, model=model, local=backend.local, size_category=category,
                      parallelism=parallelism, hardware=asdict(hw), measured_at=now())
    if not backend.local:
        return profile                              # a cloud model costs the PC nothing: nothing to measure

    key = f"{provider}:{model}:{hw.fingerprint}"
    cached = _load_cache().get(key)
    if cached and not force and now() - float(cached.get("measured_at") or 0) < CACHE_TTL:
        try:
            profile.tok_per_s, profile.ttft_s = cached.get("tok_per_s"), cached.get("ttft_s")
            profile.measured_at = float(cached.get("measured_at") or 0)
            return profile
        except (TypeError, ValueError):
            pass
    ttft, speed = (probe or probe_model)(provider, model)
    profile.ttft_s, profile.tok_per_s = ttft, speed
    if speed is not None:
        _store_cache(key, profile)
    return profile


# ── the decision ────────────────────────────────────────────────────────────────

def choose_level(profile: Profile, forced=None) -> Tuple[int, str]:
    """(level, why). `forced` is the person's own setting: an int 0..3, or None / "auto"."""
    if isinstance(forced, str) and forced.strip().isdigit():
        forced = int(forced)
    if isinstance(forced, int) and 0 <= forced <= 3:
        return forced, "уровень задан вручную"
    if not profile.local:
        return 3, "модель облачная: нагрузки на ПК нет"

    hw = profile.hardware or {}
    ram = float(hw.get("ram_gb") or 0)
    speed = profile.tok_per_s
    if profile.size_category == "tiny":
        return 0, "модель слишком маленькая, чтобы надёжно планировать и проверять (<3B)"
    if ram and ram < 8:
        return 1, f"мало оперативной памяти ({ram:.0f} ГБ)"
    if speed is not None and speed < 4:
        return 1, f"модель медленная ({speed:.1f} ток/с): полный конвейер занял бы часы"
    if profile.size_category == "small":
        return 1, "малая модель (3–7B): план сама она составит, но параллельные шаги и разбор патчей ей не по силам"
    if (profile.parallelism >= 2 and speed is not None and speed >= 15 and ram >= 16
            and profile.size_category in ("medium", "large", "moe")):
        return 3, f"{speed:.0f} ток/с и сервер принимает {profile.parallelism} запросов одновременно"
    return 2, ("локальная модель: шаги выполняются по одному, основной диалог не страдает"
               + (f" ({speed:.0f} ток/с)" if speed is not None else ""))


def parallelism_for(level: int, profile: Profile) -> int:
    """How many workers may run at once."""
    if level >= 3:
        return max(1, min(4, profile.parallelism if profile.local else 3))
    return 1
