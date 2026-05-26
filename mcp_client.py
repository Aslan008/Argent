"""
MCP (Model Context Protocol) Client for Argent.
Supports three transport types:
  - stdio: JSON-RPC 2.0 over subprocess stdin/stdout (standard MCP)
  - sse: HTTP + Server-Sent Events (remote MCP)
  - rest: Custom REST APIs (unity_bridge, standard)
"""

import json
import logging
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
import requests

import queue
log = logging.getLogger("argent.mcp")


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

    def __init__(self, command: str, args: List[str] = None, env: dict = None):
        self._command = command
        self._args = args or []
        self._env = env
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._request_id = 0
        self._initialized = False
        self._pending_requests: Dict[int, queue.Queue] = {}

    def start(self) -> bool:
        try:
            import os
            env = os.environ.copy()
            if self._env:
                env.update({k: str(v) for k, v in self._env.items()})

            self._process = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )

            threading.Thread(target=self._stderr_reader, daemon=True).start()
            threading.Thread(target=self._stdout_reader, daemon=True).start()

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

    def _stderr_reader(self):
        try:
            while self._process and self._process.stderr:
                line = self._process.stderr.readline()
                if not line:
                    break
                log.debug("MCP stderr: %s", line.decode(errors='replace').rstrip())
        except Exception:
            pass

    def _stdout_reader(self):
        try:
            while self._process and self._process.stdout:
                line = self._process.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode("utf-8"))
                    req_id = data.get("id")
                    if req_id is not None:
                        with self._lock:
                            if req_id in self._pending_requests:
                                self._pending_requests[req_id].put(data)
                except Exception:
                    pass
        except Exception:
            pass

    def stop(self):
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._initialized = False
        with self._lock:
            self._pending_requests.clear()

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

        payload = json.dumps(msg) + "\n"
        try:
            self._process.stdin.write(payload.encode("utf-8"))
            self._process.stdin.flush()
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
        payload = json.dumps(msg) + "\n"
        try:
            self._process.stdin.write(payload.encode("utf-8"))
            self._process.stdin.flush()
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

    def list_tools(self) -> List[dict]:
        if self._tools_cache:
            return self._tools_cache
        self._refresh_tools()
        return self._tools_cache

    def _refresh_tools(self):
        if not self.transport:
            return
        result = self.transport.send_request("tools/list", timeout=10)
        tools = result.get("result", {}).get("tools", [])
        if tools and "error" not in tools[0]:
            self._tools_cache = tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self.transport:
            return f"Error: Server '{self.name}' is not started."

        result = self.transport.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        if "error" in result:
            err = result["error"]
            if isinstance(err, dict):
                return f"MCP Error: {err.get('message', str(err))}"
            return f"MCP Error: {err}"

        content = result.get("result", {}).get("content", [])
        is_error = result.get("result", {}).get("isError", False)

        texts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", json.dumps(item, ensure_ascii=False))
                texts.append(text)
            else:
                texts.append(str(item))

        output = "\n".join(texts)
        if is_error:
            return f"MCP Tool Error: {output}"
        return output

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
        for name in list(self.servers.keys()):
            try:
                self.servers[name].stop()
            except Exception:
                pass
        self.servers.clear()

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
            return f"Server '{name}' is not running. Attempting to start..."
            started = srv.start()
            if not started:
                return f"Failed to start server '{name}'."

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
