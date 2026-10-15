"""Which model server a request goes to, and whether it is the person's own machine.

A cloud model costs the PC nothing; a local one costs all of it. Everything that protects the machine from a
local model (the lanes in `lanes.py`, the pressure monitor, the lowered priority of the server process) first
needs to know which of the two a request is, and for a local one WHICH server: two Argent windows on one Ollama
share one GPU, so they must share one lane.

The answer comes from the address, not from the provider's name: `ollama` may point at another computer on the
LAN (still somebody else's CPU), a "cloud" provider may be a gateway on localhost, and an Ollama model tagged
`:cloud` runs remotely although the server is local.

Server families are recognised by their default port and nothing else is assumed about them, so a new local
server (LM Studio, vLLM, a bare llama-server) is governed the moment its URL is configured:

    11434 ollama, 5001 koboldcpp, 1234 lmstudio, 8080 llama-server, anything else on this machine "local"

`ServerTuner` finds the server's processes by the PORT they listen on (and their children: Ollama starts a
separate runner process per model), so it needs no per-server knowledge either.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from urllib.parse import urlparse

from logger import get_logger

log = get_logger("local_backend")

_FAMILY_BY_PORT = {11434: "ollama", 5001: "koboldcpp", 1234: "lmstudio", 8080: "llama-server",
                   8000: "vllm", 5000: "openai-compatible"}
_CLOUD_TAG = re.compile(r"[:\-_]cloud\b", re.I)


@dataclass(frozen=True)
class Backend:
    """Where one request will run."""

    local: bool            # True: this machine's CPU/GPU/RAM does the work
    family: str            # ollama | koboldcpp | lmstudio | llama-server | vllm | local | cloud
    host: str = ""
    port: int = 0

    @property
    def lane_id(self) -> str:
        """One lane per server: the same id in every Argent process on this machine."""
        if not self.local:
            return ""
        return re.sub(r"[^a-z0-9]+", "-", f"{self.host}-{self.port}".lower()).strip("-") or "local"


CLOUD = Backend(local=False, family="cloud")


def _is_this_machine(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _ollama_host() -> str:
    raw = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    return raw if "://" in raw else "http://" + raw


def url_for(provider: str) -> str:
    """The base URL configured for a provider, or '' when it has none (a fixed cloud endpoint)."""
    name = (provider or "").lower()
    if name == "ollama":
        return _ollama_host()
    try:
        import config
        getter = getattr(config, f"get_{name}_url", None)
        if name == "zai":
            getter = getattr(config, "get_zai_endpoint", None)
        return str(getter() or "") if getter else ""
    except Exception:
        return ""


def classify(provider: str, model: str = "", base_url: Optional[str] = None) -> Backend:
    """The backend for a request. `base_url` overrides the configured one (tests, explicit callers)."""
    if model and _CLOUD_TAG.search(model):
        return CLOUD                                  # an Ollama ':cloud' model runs on somebody else's machine
    url = base_url if base_url is not None else url_for(provider)
    if not url:
        return CLOUD
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return CLOUD
    if not _is_this_machine(host):
        return CLOUD
    if (provider or "").lower() == "ollama":
        family = "ollama"
    else:
        family = _FAMILY_BY_PORT.get(port, "local")
    return Backend(local=True, family=family, host="localhost", port=port)


# ── the server process ──────────────────────────────────────────────────────

def listener_pids(port: int) -> List[int]:
    """Pids that listen on `port` on this machine (empty when psutil cannot see or nothing listens)."""
    try:
        import psutil
        pids = set()
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port and conn.pid:
                pids.add(conn.pid)
        return sorted(pids)
    except Exception:
        return []


class ServerTuner:
    """Lowers the priority of a local model server so that the PC stays usable while it generates.

    Opt-in (`lower_model_server_priority`): another program's priority is not ours to change by default. What is
    changed is remembered and put back by `restore` (also at exit), so Argent never leaves the server slowed
    after it is gone."""

    def __init__(self):
        self._saved: Dict[int, int] = {}
        self._at_exit = False

    def _processes(self, port: int):
        import psutil
        out = {}
        for pid in listener_pids(port):
            try:
                root = psutil.Process(pid)
                out[root.pid] = root
                for child in root.children(recursive=True):
                    out[child.pid] = child
            except Exception:
                continue
        return list(out.values())

    def lower(self, backend: Backend) -> List[int]:
        """Put the server (and its runners) at below-normal priority. Returns the pids changed."""
        if not backend.local or not backend.port:
            return []
        try:
            import psutil
        except Exception:
            return []
        target = psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10
        changed = []
        for proc in self._processes(backend.port):
            try:
                current = proc.nice()
                if proc.pid in self._saved or current == target:
                    continue
                proc.nice(target)
                self._saved[proc.pid] = current
                changed.append(proc.pid)
            except Exception as e:                      # another user's process, already gone, no rights
                log.debug("cannot lower priority of %s: %s", proc.pid, e)
        if changed:
            log.info("lowered the priority of %s (port %s)", changed, backend.port)
            self._restore_at_exit()
        return changed

    def _restore_at_exit(self) -> None:
        """Argent must never leave the person's model server slowed after it is gone."""
        if not self._at_exit:
            import atexit
            atexit.register(self.restore)
            self._at_exit = True

    def restore(self) -> int:
        try:
            import psutil
        except Exception:
            return 0
        done = 0
        for pid, value in list(self._saved.items()):
            try:
                psutil.Process(pid).nice(value)
                done += 1
            except Exception:
                pass
            self._saved.pop(pid, None)
        return done


TUNER = ServerTuner()


def physical_cores() -> int:
    try:
        import psutil
        return int(psutil.cpu_count(logical=False) or psutil.cpu_count() or 4)
    except Exception:
        return os.cpu_count() or 4


def thread_budget(reserve: int = 2) -> int:
    """How many CPU threads a model server should use so that `reserve` cores stay for the person."""
    return max(1, physical_cores() - max(0, reserve))
