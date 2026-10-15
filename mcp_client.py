"""
MCP (Model Context Protocol) Client for Argent.
Supports three transport types:
  - stdio: JSON-RPC 2.0 over subprocess stdin/stdout (standard MCP)
  - sse: HTTP + Server-Sent Events (remote MCP)
  - rest: Custom REST APIs (unity_bridge, standard)
"""

import json
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
import requests

import queue
from logger import get_logger
from project_paths import find_repo_root, project_root_or_cwd
from src.agent.shell import descendant_processes, kill_process_tree

# Was logging.getLogger("argent.mcp"), which has no handler — every MCP event
# went nowhere. A call that hung for ten minutes left not one line on disk, so
# afterwards there was nothing to diagnose it with. get_logger writes to
# ~/.argent/logs/mcp.log like every other subsystem.
log = get_logger("mcp")

# Above this a call is worth flagging even when it eventually succeeds: it is
# the early warning for the stall you would otherwise only meet at the timeout.
SLOW_CALL_SECONDS = 20

# Server messages that mean "you got the arguments wrong". Matching on text is
# crude, but MCP has no error taxonomy for this and the payoff is high: the
# model gets the real signature instead of guessing a second time.
_ARGUMENT_ERROR_CUES = (
    "parameter validation", "invalid parameter", "unknown parameter",
    "is required", "are required", "missing required", "invalid arguments",
)


def _signature(tool: dict) -> str:
    """`name(required, optional?) — first sentence`, the way a docstring reads."""
    schema = tool.get("inputSchema") or tool.get("parameters") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = ", ".join(p if p in required else f"{p}?" for p in props)
    desc = (tool.get("description") or "").strip().split(".")[0][:140]
    return f"{tool.get('name', '?')}({params})" + (f" — {desc}" if desc else "")


def _closest_names(name: str, candidates, limit: int = 3):
    """Nearest tool names, for a call that named something that does not exist."""
    import difflib
    return difflib.get_close_matches(name, [c for c in candidates if c], n=limit, cutoff=0.6)


def _wait_exit(process, timeout: float) -> bool:
    """True once `process` has exited; never raises."""
    try:
        process.wait(timeout=timeout)
        return True
    except Exception:
        return False


def _close_pipes(process) -> None:
    for pipe in (process.stdout, process.stderr, process.stdin):
        try:
            if pipe is not None:
                pipe.close()
        except Exception:
            pass


def _shutdown_stdio_process(process, readers, grace: float = 2.0) -> None:
    """Stop an MCP stdio server and release everything it held. See StdioTransport.stop."""
    descendants = descendant_processes(process)     # before the parent can vanish

    try:
        if process.stdin is not None:
            process.stdin.close()
    except Exception:
        pass

    if not _wait_exit(process, grace):
        try:
            process.terminate()
        except Exception:
            pass
        _wait_exit(process, grace)

    kill_process_tree(process, descendants)

    for reader in readers:
        reader.join(timeout=2)

    if any(reader.is_alive() for reader in readers):
        # A reader outlived the tree (a process in an uninterruptible state, a
        # descendant that escaped). Closing under it would block, so let a
        # throw-away thread take that risk instead of the caller.
        closer = threading.Thread(target=_close_pipes, args=(process,), daemon=True,
                                  name="mcp-pipe-closer")
        closer.start()
        closer.join(timeout=1)
    else:
        _close_pipes(process)


# ═══════════════════════════════════════════════════════════════
# Transport Layer
# ═══════════════════════════════════════════════════════════════

class MCPTransportType(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STANDARD = "standard"
    UNITY_BRIDGE = "unity_bridge"


class _BaseTransport(ABC):
    @abstractmethod
    def start(self) -> bool: ...
    @abstractmethod
    def stop(self): ...
    @abstractmethod
    def send_request(self, method: str, params: dict = None, timeout: float = 30) -> dict: ...


class StdioTransport(_BaseTransport):
    """JSON-RPC 2.0 over subprocess stdin/stdout — standard MCP protocol."""

    def __init__(self, command: str, args: List[str] = None, env: dict = None, cwd=None):
        self._command = command
        self._args = args or []
        self._env = env
        self._cwd = cwd
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._initialized = False
        self._pending_requests: Dict[int, queue.Queue] = {}
        self._readers: List[threading.Thread] = []
        self._write_lock = threading.Lock()

    def _resolve_command(self, env: dict) -> str:
        """Find the executable on the user's PATH, the way their shell would.

        On Windows CreateProcess looks in the directory of the *calling*
        executable before PATH. Argent started from `.venv\\Scripts\\python.exe`
        (Argent.bat, the Rust TUI) therefore turned a bare `python` into the
        bare interpreter behind that venv — one without the server's own
        dependencies — while the same config worked from a shell, where
        `python` is whatever PATH names first. Resolving through PATH makes the
        server start identically from every front-end. `which` also honours
        PATHEXT, so `npx` finds `npx.cmd`.
        """
        import shutil
        return shutil.which(self._command, path=env.get("PATH")) or self._command

    def start(self) -> bool:
        try:
            import os
            env = os.environ.copy()
            if self._env:
                env.update({k: str(v) for k, v in self._env.items()})

            process = subprocess.Popen(
                [self._resolve_command(env)] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self._cwd,
                bufsize=0,
            )
            self._process = process

            # Readers are bound to THIS process: after a restart a late reader
            # of the old one must not touch the new one's requests.
            self._readers = [
                threading.Thread(target=self._stderr_reader, args=(process,),
                                 daemon=True, name="mcp-stderr"),
                threading.Thread(target=self._stdout_reader, args=(process,),
                                 daemon=True, name="mcp-stdout"),
            ]
            for reader in self._readers:
                reader.start()

            result = self.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "argent-coder", "version": "1.0.0"}
            }, timeout=10)

            if "result" in result:
                self.send_notification("notifications/initialized", {})
                self._initialized = True
                log.info("Stdio MCP server initialized: %s", result.get("result", {}).get("serverInfo", {}))
                return True
            else:
                log.error("Stdio MCP init failed: %s", result)
                self.stop()
                return False
        except Exception as e:
            log.error("Stdio transport start error: %s", e)
            self.stop()
            return False

    def _stderr_reader(self, process):
        try:
            while True:
                line = process.stderr.readline()
                if not line or not isinstance(line, (bytes, str)):
                    break
                text = line.decode(errors='replace') if isinstance(line, bytes) else str(line)
                log.debug("MCP stderr: %s", text.rstrip())
        except Exception:
            pass

    def _stdout_reader(self, process):
        try:
            while True:
                line = process.stdout.readline()
                if not line or not isinstance(line, (bytes, str)):
                    break
                try:
                    text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                    data = json.loads(text)
                except Exception:
                    continue    # a banner or a log line, not JSON-RPC
                if not isinstance(data, dict):
                    continue
                if "method" in data:
                    # The server is asking us something (or just telling us).
                    # It must never be mistaken for the answer to OUR request
                    # with the same id: both sides count from 1.
                    self._answer_server_message(process, data)
                    continue
                req_id = data.get("id")
                if req_id is not None:
                    with self._lock:
                        waiter = self._pending_requests.get(req_id)
                    if waiter is not None:
                        waiter.put(data)
        except Exception:
            pass
        finally:
            # EOF: the server is gone. Anyone still waiting would otherwise sit
            # out the whole request timeout (up to minutes) for an answer that
            # can no longer come.
            if self._process is None or self._process is process:
                self._fail_pending("MCP server closed the connection")

    def _answer_server_message(self, process, data: dict) -> None:
        """Reply to a server-initiated request; ignore a notification.

        `ping` must be answered or a server that health-checks us gives up on
        us. Anything else is a capability we never advertised, so the honest
        reply is JSON-RPC's "method not found".
        """
        req_id = data.get("id")
        if req_id is None:
            return
        method = data.get("method")
        if method == "ping":
            reply = {"jsonrpc": "2.0", "id": req_id, "result": {}}
        else:
            reply = {"jsonrpc": "2.0", "id": req_id,
                     "error": {"code": -32601, "message": f"Method not found: {method}"}}
        self._write(process, reply)

    def _write(self, process, message: dict) -> None:
        payload = (json.dumps(message) + "\n").encode("utf-8")
        with self._write_lock:
            process.stdin.write(payload)
            process.stdin.flush()

    def _fail_pending(self, message: str) -> None:
        with self._lock:
            waiters = list(self._pending_requests.values())
        for waiter in waiters:
            try:
                waiter.put({"error": {"message": message}})
            except Exception:
                pass

    def stop(self):
        """Shut the server down. Returns in bounded time, whatever the server does.

        Order matters, and getting it wrong hangs the caller:

        1. Close the server's stdin: EOF is MCP's own "please exit".
        2. Wait briefly, then terminate(), then kill the WHOLE tree. Launchers
           (`uv`, `npx`, `cmd`) leave the real server alive behind them, and it
           keeps our pipes open.
        3. Join the readers - they return once every writer is gone.
        4. Only then close our ends. Closing a Windows pipe while a thread is
           still blocked reading it blocks the closer too, which is how quitting
           Argent with a launcher-based server used to hang for good.
        """
        process, self._process = self._process, None
        readers, self._readers = getattr(self, "_readers", []), []
        self._initialized = False
        if process is not None:
            _shutdown_stdio_process(process, readers)
        with self._lock:
            waiters = list(self._pending_requests.values())
            self._pending_requests.clear()
        for waiter in waiters:
            try:
                waiter.put({"error": {"message": "MCP server was stopped"}})
            except Exception:
                pass

    def send_request(self, method: str, params: dict = None, timeout: float = 30) -> dict:
        if not self._process or self._process.poll() is not None:
            return {"error": {"message": "MCP server process is not running"}}

        q = queue.Queue()
        with self._lock:
            self._request_id += 1
            req_id = self._request_id
            self._pending_requests[req_id] = q

        msg = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        try:
            self._write(self._process, msg)
        except Exception as e:
            with self._lock:
                self._pending_requests.pop(req_id, None)
            return {"error": {"message": f"Failed to write to MCP server: {e}"}}

        try:
            res = q.get(timeout=timeout)
            with self._lock:
                self._pending_requests.pop(req_id, None)
            return res
        except queue.Empty:
            with self._lock:
                self._pending_requests.pop(req_id, None)
            return {"error": {"message": f"MCP server request timed out after {timeout} seconds"}}
        except Exception as e:
            with self._lock:
                self._pending_requests.pop(req_id, None)
            return {"error": {"message": f"Read error from MCP server: {e}"}}

    def send_notification(self, method: str, params: dict = None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        try:
            self._write(self._process, msg)
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None and self._initialized


class SSETransport(_BaseTransport):
    """HTTP-based MCP with Server-Sent Events."""

    def __init__(self, url: str, headers: dict = None):
        self._url = url.rstrip("/")
        self._headers = headers or {}
        self._session = requests.Session()
        self._session.headers.update(self._headers)
        self._session.headers.update({"Content-Type": "application/json"})

    def start(self) -> bool:
        try:
            result = self.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "argent-coder", "version": "1.0.0"}
            }, timeout=10)
            return "result" in result
        except Exception as e:
            log.error("SSE transport start error: %s", e)
            return False

    def stop(self):
        try:
            self._session.close()
        except Exception:
            pass

    def send_request(self, method: str, params: dict = None, timeout: float = 30) -> dict:
        msg = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method}
        if params is not None:
            msg["params"] = params
        try:
            resp = self._session.post(f"{self._url}/message", json=msg, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": {"message": str(e)}}

    @property
    def is_running(self) -> bool:
        return True


class RESTTransport(_BaseTransport):
    """Custom REST API transport for non-standard servers."""

    def __init__(self, url: str, server_type: MCPTransportType):
        self._url = url.rstrip("/")
        self._server_type = server_type

    def start(self) -> bool:
        return True

    def stop(self):
        pass

    def send_request(self, method: str, params: dict = None, timeout: float = 30) -> dict:
        try:
            if method == "tools/list":
                return self._list_tools(timeout)
            elif method == "tools/call":
                return self._call_tool(params, timeout)
            else:
                return {"error": {"message": f"REST transport does not support method '{method}'"}}
        except Exception as e:
            return {"error": {"message": str(e)}}

    def _list_tools(self, timeout: float) -> dict:
        if self._server_type == MCPTransportType.UNITY_BRIDGE:
            resp = requests.get(f"{self._url}/api/tools", timeout=timeout)
            resp.raise_for_status()
            tools = resp.json().get("tools", [])
        else:
            resp = requests.get(f"{self._url}/tools", timeout=timeout)
            resp.raise_for_status()
            tools = resp.json().get("tools", [])
        return {"result": {"tools": tools}}

    def _call_tool(self, params: dict, timeout: float) -> dict:
        tool_name = (params or {}).get("name", "")
        arguments = (params or {}).get("arguments", {})

        if self._server_type == MCPTransportType.UNITY_BRIDGE:
            resp = requests.post(f"{self._url}/api/tool", json={"tool": tool_name, "args": arguments}, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            if not result.get("success", False):
                return {"result": {"content": [{"type": "text", "text": f"Error: {result.get('error', 'Unknown')}"}], "isError": True}}
            data = result.get("data")
            text = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, (dict, list)) else str(data)
            return {"result": {"content": [{"type": "text", "text": text}]}}
        else:
            resp = requests.post(f"{self._url}/call", json={"name": tool_name, "arguments": arguments}, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            if result.get("isError"):
                return {"result": {"content": [{"type": "text", "text": str(result.get("content"))}], "isError": True}}
            return {"result": {"content": [{"type": "text", "text": str(result.get("content"))}]}}

    @property
    def is_running(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════
# Server Instance
# ═══════════════════════════════════════════════════════════════

class MCPServer:
    """Manages a single MCP server connection with its transport."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.transport: Optional[_BaseTransport] = None
        self._tools_cache: List[dict] = []
        self._tools_unavailable = False

    @property
    def server_type(self) -> MCPTransportType:
        return MCPTransportType(self.config.get("type", "stdio"))

    def start(self) -> bool:
        stype = self.server_type
        if stype == MCPTransportType.STDIO:
            self.transport = StdioTransport(
                command=self.config["command"],
                args=self.config.get("args", []),
                env=self.config.get("env"),
                # stdio servers resolve resources relative to their CWD (e.g.
                # `unity mcp` discovers the Editor via <cwd>/Library/Pipeline/...).
                # Anchor to the repo root first: Argent drops `.argent` in its
                # launch folder (often Assets/ inside a Unity project), so the
                # plain project root would just be that folder. A per-server
                # "cwd" config entry wins if set.
                cwd=self.config.get("cwd") or find_repo_root() or project_root_or_cwd(),
            )
        elif stype == MCPTransportType.SSE:
            self.transport = SSETransport(
                url=self.config["url"],
                headers=self.config.get("headers"),
            )
        else:
            self.transport = RESTTransport(
                url=self.config["url"],
                server_type=stype,
            )

        success = self.transport.start()
        if success:
            self._refresh_tools()
        return success

    def stop(self):
        if self.transport:
            self.transport.stop()
            self.transport = None
        self._tools_cache = []
        self._tools_unavailable = False      # a restart deserves a fresh attempt

    def list_tools(self) -> List[dict]:
        if self._tools_cache:
            return self._tools_cache
        # A FAILED listing is remembered too. This is called while building the
        # system prompt, i.e. once per turn: without this an unreachable server
        # cost a 10s timeout every turn, forever, and the section text flipped
        # between the tool list and "could not fetch" — churning the prefix
        # cache. Recovery is via /mcp (stop/start clears the flag).
        if self._tools_unavailable:
            return []
        self._refresh_tools()
        return self._tools_cache

    def _refresh_tools(self):
        if not self.transport:
            return
        started = time.monotonic()
        result = self.transport.send_request("tools/list", timeout=10)
        tools = result.get("result", {}).get("tools", [])
        if tools and "error" not in tools[0]:
            self._tools_cache = tools
            log.info("mcp %s: listed %d tools in %.1fs",
                     self.name, len(tools), time.monotonic() - started)
        else:
            self._tools_unavailable = True
            log.warning("mcp %s: tools/list failed after %.1fs (%s) — the server "
                        "will be reported as unreachable until /mcp restarts it",
                        self.name, time.monotonic() - started,
                        str(result.get("error", "no tools returned"))[:150])

    def _find_tool(self, tool_name: str):
        for tool in self._tools_cache:
            if tool.get("name") == tool_name:
                return tool
        return None

    def preflight(self, tool_name: str, arguments: dict):
        """Catch a call the server is certain to reject, and answer with the
        signature instead of a round-trip to a 400.

        The signatures are not in the system prompt — one server can expose
        hundreds — so a model that skips `list_mcp_tools` invents plausible
        parameter names. Observed: find_assets(filter=, search=, max_results=)
        when the real ones are type/name/label/search_in/limit. The schema is
        already cached here, so the correction costs nothing until it is needed.

        Deliberately narrow: only a wrong tool name, an unknown argument, or a
        missing REQUIRED argument. Servers accept plenty that a schema does not
        describe, and blocking a call that would have worked is worse than the
        failure this prevents. Returns an error string, or None to proceed.
        """
        if not self._tools_cache:
            return None                   # nothing to validate against

        tool = self._find_tool(tool_name)
        if tool is None:
            close = _closest_names(tool_name, [t.get("name", "") for t in self._tools_cache])
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return (f"MCP Error: server '{self.name}' has no tool '{tool_name}'.{hint} "
                    f"Use list_mcp_tools('{self.name}', '<topic>') to find the right one.")

        schema = tool.get("inputSchema") or tool.get("parameters") or {}
        props = schema.get("properties") or {}
        if not props or not isinstance(arguments, dict):
            return None                   # loose schema — let the server judge

        unknown = [k for k in arguments if k not in props]
        missing = [k for k in (schema.get("required") or []) if k not in arguments]
        if not unknown and not missing:
            return None

        problems = []
        if unknown:
            problems.append(f"unknown parameter(s): {', '.join(sorted(unknown))}")
        if missing:
            problems.append(f"missing required: {', '.join(missing)}")
        log.warning("mcp %s.%s REJECTED before sending: %s",
                    self.name, tool_name, "; ".join(problems))
        return (f"MCP Error: wrong arguments for '{tool_name}' — {'; '.join(problems)}.\n"
                f"Correct signature: {_signature(tool)}\n"
                f"Re-send the call using these parameter names.")

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self.transport:
            return f"Error: Server '{self.name}' is not started."

        rejection = self.preflight(tool_name, arguments)
        if rejection is not None:
            return rejection

        from config import get_mcp_call_timeout
        timeout = get_mcp_call_timeout()

        # Every call is timed and recorded. A call that hangs is the one thing
        # you cannot debug afterwards from anything else: the terminal shows a
        # spinner, the server keeps no history, and once the turn is over the
        # only evidence left is the user's memory of how long they waited.
        # Logged BEFORE the call as well: if it never returns and the session is
        # killed, the "->" line is the only proof the call was ever made.
        log.debug("mcp %s.%s -> %s (timeout %gs)", self.name, tool_name,
                  json.dumps(arguments, ensure_ascii=False)[:200], timeout)
        started = time.monotonic()
        result = self.transport.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        }, timeout=timeout)
        elapsed = time.monotonic() - started

        if "error" in result:
            err = result["error"]
            message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            log.warning("mcp %s.%s FAILED after %.1fs: %s",
                        self.name, tool_name, elapsed, message[:200])
            if "timed out" in message:
                # The request is still in flight on the server side. Say so, or
                # the model re-sends it and queues a second copy behind the first.
                return (f"MCP Error: '{tool_name}' on server '{self.name}' did not answer "
                        f"within {timeout:g}s. The server may still be working on it — "
                        f"check that the application is responsive before retrying, and "
                        f"do NOT immediately repeat the same call.")
            return f"MCP Error: {self._explain(tool_name, message)}"

        content = result.get("result", {}).get("content", [])
        is_error = result.get("result", {}).get("isError", False)

        # A tool-level failure is a failure. It arrives inside a successful
        # JSON-RPC response, so logging on the transport outcome alone recorded
        # "ok in 0.1s" for calls the server had flatly rejected — which is how a
        # run of bad-argument errors left a log that looked perfectly healthy.
        if is_error:
            log.warning("mcp %s.%s tool error after %.1fs", self.name, tool_name, elapsed)
        else:
            log.info("mcp %s.%s ok in %.1fs", self.name, tool_name, elapsed)
        if elapsed > SLOW_CALL_SECONDS:
            log.warning("mcp %s.%s took %.1fs — the server is slow to respond",
                        self.name, tool_name, elapsed)

        texts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", json.dumps(item, ensure_ascii=False))
                texts.append(text)
            else:
                texts.append(str(item))

        output = "\n".join(texts)
        if is_error:
            return f"MCP Tool Error: {self._explain(tool_name, output)}"
        return output

    def _explain(self, tool_name: str, message: str) -> str:
        """Attach the real signature when the server complains about arguments.

        Preflight cannot catch everything — a rule like "at least one of type,
        name or label" lives in the server, not in the JSON schema. When the
        rejection does come back, answering with the signature turns a second
        guess into a correction.
        """
        lowered = (message or "").lower()
        if not any(cue in lowered for cue in _ARGUMENT_ERROR_CUES):
            return message
        tool = self._find_tool(tool_name)
        if tool is None:
            return message
        return f"{message}\nSignature: {_signature(tool)}"

    @property
    def is_running(self) -> bool:
        return self.transport is not None and self.transport.is_running

    def get_display_url(self) -> str:
        stype = self.server_type
        if stype == MCPTransportType.STDIO:
            cmd = self.config.get("command", "")
            args = " ".join(self.config.get("args", []))
            return f"{cmd} {args}".strip()
        return self.config.get("url", "unknown")


# ═══════════════════════════════════════════════════════════════
# Client Orchestrator
# ═══════════════════════════════════════════════════════════════

class MCPClient:
    """Orchestrates multiple MCP server connections."""

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}

    def register_and_start(self, name: str, config: dict) -> str:
        if name in self.servers:
            self.servers[name].stop()

        server = MCPServer(name, config)
        success = server.start()
        if success:
            self.servers[name] = server
            tools = server.list_tools()
            tool_count = len(tools)
            log.info("MCP server '%s' started with %d tools", name, tool_count)
            return f"Started MCP server '{name}' ({config.get('type', 'stdio')}). {tool_count} tools loaded."
        else:
            log.error("Failed to start MCP server '%s'", name)
            return f"Error: Failed to start MCP server '{name}'. Check command/path and logs."

    def register_rest(self, name: str, url: str, server_type: str) -> str:
        config = {"type": server_type, "url": url}
        return self.register_and_start(name, config)

    def unregister_server(self, name: str) -> str:
        if name in self.servers:
            self.servers[name].stop()
            del self.servers[name]
            return f"Stopped and removed MCP server '{name}'."
        return f"Server '{name}' not found."

    def stop_all(self):
        """Stop every server at once: exiting must cost one shutdown, not N."""
        servers = list(self.servers.values())
        self.servers.clear()

        def _stop(srv):
            try:
                srv.stop()
            except Exception:
                pass

        threads = [threading.Thread(target=_stop, args=(srv,), daemon=True,
                                    name="mcp-stop") for srv in servers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

    def get_servers(self) -> List[dict]:
        result = []
        for name, srv in self.servers.items():
            info = {
                "name": name,
                "type": srv.config.get("type", "stdio"),
                "running": srv.is_running,
                "endpoint": srv.get_display_url(),
                "tools": len(srv.list_tools()),
            }
            result.append(info)
        return result

    def test_connection(self, name: str) -> str:
        if name not in self.servers:
            return f"Error: Server '{name}' not registered."

        srv = self.servers[name]
        if not srv.is_running:
            started = srv.start()
            if not started:
                return f"Server '{name}' is not running and failed to start."

        tools = srv.list_tools()
        if not tools:
            return f"Server '{name}' is running but returned no tools."

        lines = [f"Server '{name}' ({srv.config.get('type')}) — {len(tools)} tools:"]
        for t in tools:
            tname = t.get("name", "?")
            desc = t.get("description", "").split(".")[0][:80]
            lines.append(f"  - {tname}: {desc}")
        return "\n".join(lines)

    def list_tools(self, name: str) -> List[dict]:
        if name not in self.servers:
            return [{"error": f"Server '{name}' not registered."}]
        return self.servers[name].list_tools()

    def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        if server_name not in self.servers:
            return f"Error: Server '{server_name}' not registered. Use /mcp to see available servers."

        srv = self.servers[server_name]
        if not srv.is_running and srv.server_type == MCPTransportType.STDIO:
            started = srv.start()
            if not started:
                return f"Error: Server '{server_name}' is not running and could not be restarted."

        return srv.call_tool(tool_name, arguments)


mcp_client = MCPClient()
