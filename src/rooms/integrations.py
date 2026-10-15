"""The nodes that reach outside a run: MCP servers, skills, plugins and Leeth.

Each is a thin step over code Argent already has — `call_mcp_tool`, `read_skill`, the
plugin hook manager, the Leeth tools — so a room does exactly what the model's own tool
call would do, under the same guards, and nothing is written twice. The class is
injected into the `NodeRunner`, so the engine and the runner stay testable without an
MCP server, a plugin directory or the Leeth weights.

Every method takes (node, state) and returns the node's result as text; the runner
applies `sets` / `writes` as it does for a tool node. Nothing here raises for an
expected failure: a missing server or skill comes back as ``Error: ...`` and the node's
`ok` is then False.

The catalog methods (`mcp_servers`, `skills`, `plugins`, `leeth_available`) answer "what
is there to pick from" for the editor and the validator.
"""

import contextlib
import hashlib
import io
import json
import re

from src.rooms import policy
from src.rooms.models import LEETH_MODES, POWER_MIN_DELAY

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _items(node, state) -> list:
    """The list a leeth / foreach node works on: the field `input_from` names (a list, JSON text or
    one item per line), else the node's own `args.items`."""
    value = None
    if node.input_from:
        key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
        value = state.data.get(key)
    if value is None:
        value = (node.args or {}).get("items")
    return as_list(value)


def as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass
    return [ln for ln in text.splitlines() if ln.strip()]


def click_indices(node, state) -> tuple:
    """(indices, problem) for a Leeth `click` node: what `leeth_click_many` should press.

    The source is the field `input_from` names (else `args.indices`). It may hold a plain list of
    numbers, the `yes` list of a scan (items with an `index`), or the whole scan result: posts then
    carry their own `buttons`, and `args.button` (a regular expression over the button's caption,
    e.g. "закладк|bookmark") says which button of each post to press. Without it posts are
    refused — pressing every button of a post is never what a person meant."""
    import re
    from src.rooms.runner import _parse_json_text
    a = node.args or {}
    value = None
    if node.input_from:
        key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
        value = state.data.get(key)
    if value is None:
        value = a.get("indices")
    if isinstance(value, str):
        text = value.strip()
        parsed = _parse_json_text(text) if text[:1] in "[{" or "{" in text else None
        if parsed is not None:
            value = parsed
        else:
            value = [w for w in re.split(r"[,;\s]+", text) if w]       # "7, 8" / "7 8" / one per line
    if isinstance(value, dict):                       # a whole scan result
        rows = list(value.get("yes") or [])
        if a.get("include_unsure"):
            rows += list(value.get("unsure") or [])
        if not rows and "yes" not in value:
            rows = [value]
    else:
        rows = as_list(value)
    pattern = None
    if a.get("button"):
        try:
            pattern = re.compile(str(a["button"]), re.I)
        except re.error as e:
            return [], f"args.button is not a valid pattern ({e})"
    out, posts_without_pattern = [], 0
    for r in rows:
        if isinstance(r, bool):
            continue
        if isinstance(r, int):
            out.append(r)
        elif isinstance(r, str) and r.strip().lstrip("-").isdigit():
            out.append(int(r.strip()))
        elif isinstance(r, dict) and isinstance(r.get("buttons"), list):
            if pattern is None:
                posts_without_pattern += 1
                continue
            for b in r["buttons"]:
                if isinstance(b, dict) and pattern.search(str(b.get("label") or "")) and b.get("index") is not None:
                    out.append(int(b["index"]))
                    break
        elif isinstance(r, dict) and r.get("index") is not None:
            out.append(int(r["index"]))
    if posts_without_pattern and not out:
        return [], ("the scanned items are posts: name the button to press in args.button "
                    "(a pattern over its caption, e.g. 'закладк|bookmark')")
    if not out:
        return [], "nothing to click: the list is empty or has no matching buttons"
    return list(dict.fromkeys(out)), ""


def _text_items(items: list) -> list:
    return [i if isinstance(i, str) else json.dumps(i, ensure_ascii=False) for i in items]


class Integrations:
    """Production implementation. Tests substitute a fake with the same methods."""

    # ── MCP ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _configured_servers() -> dict:
        try:
            from config import get_mcp_servers
            return {s.get("name"): s for s in get_mcp_servers() if s.get("name")}
        except Exception:
            return {}

    def ensure_server(self, name: str) -> str:
        """Make sure the server is connected, starting it from the configuration when it is not.
        Returns "" when it is ready, else the reason it is not."""
        from mcp_client import mcp_client
        srv = mcp_client.servers.get(name)
        if srv is not None and srv.is_running:
            return ""
        cfg = self._configured_servers().get(name)
        if cfg is None:
            known = ", ".join(sorted(self._configured_servers())) or "none configured"
            return f"Error: MCP server '{name}' is not configured (configured: {known}). Add it with /mcp."
        reply = mcp_client.register_and_start(name, {k: v for k, v in cfg.items() if k != "name"})
        if reply.startswith("Error"):
            return reply
        return ""

    def mcp(self, node, state) -> str:
        rules = policy.Policy(node.ignore)
        with policy.use(rules):
            blocked = policy.mcp_tool_blocked(node.server, (node.tool or "") if (node.mode or "call") == "call" else "")
        if blocked:
            return f"Error: {blocked}"
        problem = self.ensure_server(node.server)
        if problem:
            return problem
        from tools.schemas import AVAILABLE_TOOLS
        with policy.use(rules):
            if (node.mode or "call") == "list":
                return str(AVAILABLE_TOOLS["list_mcp_tools"](node.server, str((node.args or {}).get("filter", ""))))
            args = dict(node.args or {})
            if node.input_from:
                key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
                value = state.data.get(key)
                required = self._required_params(node.server, node.tool)
                free = [p for p in required if p not in args]
                if value is not None and free:
                    args[free[0]] = value
            return str(AVAILABLE_TOOLS["call_mcp_tool"](node.server, node.tool, json.dumps(args, ensure_ascii=False)))

    @staticmethod
    def _required_params(server: str, tool: str) -> list:
        try:
            from mcp_client import mcp_client
            for t in mcp_client.list_tools(server):
                if t.get("name") == tool:
                    schema = t.get("inputSchema") or t.get("parameters") or {}
                    return list(schema.get("required") or [])
        except Exception:
            pass
        return []

    # ── skills ───────────────────────────────────────────────────────────────────
    def skill(self, node, state) -> str:
        rules = policy.Policy(node.ignore)
        with policy.use(rules):
            blocked = policy.skill_blocked(node.skill)
            if blocked:
                return f"Error: {blocked}"
            from skill_manager import skill_manager
            text = skill_manager.read_skill(node.skill)
            if not text:
                return f"Error: skill '{node.skill}' not found."
            return policy.cut_skill(node.skill, text)

    # ── plugins ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _plugin_modules() -> list:
        try:
            from hook_manager import hook_manager
            return list(getattr(hook_manager, "_plugins", []) or [])
        except Exception:
            return []

    def plugin(self, node, state) -> str:
        from hook_manager import hook_manager
        mode = node.mode or "command"
        args = dict(node.args or {})
        if mode == "hook":
            results = hook_manager.call_hook(node.command, **args)
            shown = [str(r) for r in results if r is not None]
            return "\n".join(shown) if shown else "(hook handled, no result)"
        fn = hook_manager.get_custom_commands().get(node.command)
        if fn is None:
            return f"Error: no plugin command '{node.command}'."
        if node.plugin and getattr(fn, "__module__", "") != node.plugin:
            return f"Error: command '{node.command}' does not belong to the plugin '{node.plugin}'."
        words = [str(v) for v in args.values()]
        buf = io.StringIO()
        result = None
        try:
            from ui import console
            capture = console.capture()
        except Exception:
            capture = contextlib.nullcontext()
        with contextlib.redirect_stdout(buf):
            with capture as captured:
                result = fn(*words)
        said = _ANSI.sub("", buf.getvalue() + (captured.get() if hasattr(captured, "get") else "")).strip()
        parts = [p for p in (said, "" if result is None else str(result)) if p]
        return "\n".join(parts) if parts else "(the command printed nothing)"

    # ── Leeth ────────────────────────────────────────────────────────────────────
    @staticmethod
    def leeth_available() -> bool:
        """Whether Leeth can answer. Cheap by design — the editor asks on every page load, and
        asking the runtime itself loads the model (tens of seconds). When Leeth is already in
        this process its own answer is used; otherwise the weights being on disk is the proof."""
        import os
        import sys
        from pathlib import Path
        mod = sys.modules.get("src.leeth.dispatcher")
        if mod is not None:
            try:
                return bool(mod.dispatcher.is_available)
            except Exception:
                return False
        try:
            model_dir = Path(os.environ.get("LEETH_MODEL_DIR", "src/leeth/data_400m"))
            weights = (model_dir / "leeth_best.pt").exists() or (model_dir / "leeth_int8.onnx").exists()
            if weights and (model_dir / "leeth_vocab.json").exists():
                return True
            root = Path(__file__).resolve().parents[2]
            return (root / ".argent" / "models" / "leeth2" / "final").exists()
        except Exception:
            return False

    def leeth(self, node, state) -> str:
        from tools.schemas import AVAILABLE_TOOLS
        mode = node.mode or "delegate"
        a = node.args or {}
        task = (node.instruction or "").strip()
        if mode == "delegate":
            return str(AVAILABLE_TOOLS["delegate_to_leeth"](task))
        if mode in ("filter", "classify"):
            items = _items(node, state)
            if not items:
                return "Error: the list for Leeth is empty."
            kw = {"items": json.dumps(_text_items(items), ensure_ascii=False), "criterion": task}
            if mode == "filter":
                kw["mode"] = "yesno"
                for k in ("accept_above", "reject_below", "max_items"):
                    if a.get(k) is not None:
                        kw[k] = a[k]
                for k in ("positive_examples", "negative_examples"):
                    if a.get(k):
                        kw[k] = json.dumps(as_list(a[k]), ensure_ascii=False)
            else:
                kw["mode"] = "categories"
                kw["categories"] = json.dumps(as_list(a.get("categories")), ensure_ascii=False)
            return str(AVAILABLE_TOOLS["leeth_map"](**kw))
        if mode == "scan":
            kw = {"criterion": task, "mode": a.get("mode", "yesno"), "include": a.get("include", "links")}
            for k in ("accept_above", "reject_below", "max_items", "scroll_pages", "session"):
                if a.get(k) is not None:
                    kw[k] = a[k]
            if a.get("categories"):
                kw["categories"] = json.dumps(as_list(a["categories"]), ensure_ascii=False)
            for k in ("positive_examples", "negative_examples"):
                if a.get(k):
                    kw[k] = json.dumps(as_list(a[k]), ensure_ascii=False)
            return str(AVAILABLE_TOOLS["leeth_scan_page"](**kw))
        if mode == "fill":
            kw = {"instruction": task}
            for k in ("session", "dry_run", "submit", "min_confidence"):
                if a.get(k) is not None:
                    kw[k] = a[k]
            return str(AVAILABLE_TOOLS["leeth_fill_form"](**kw))
        if mode == "click":
            indices, problem = click_indices(node, state)
            if problem:
                return f"Error: {problem}"
            kw = {"indices": json.dumps(indices)}
            for k in ("session", "confirm"):
                if a.get(k) is not None:
                    kw[k] = a[k]
            return str(AVAILABLE_TOOLS["leeth_click_many"](**kw))
        return f"Error: unknown Leeth mode '{mode}' (use {list(LEETH_MODES)})."

    # ── the project's repository ─────────────────────────────────────────────────
    def git(self, node, state):
        """Look at, record or share the project's changes (see gitnode.py). The report text carries the
        fields it produced in `.values`; the runner stores them as `<id>.<name>`."""
        from src.rooms import gitnode
        return gitnode.run(node)

    # ── a web address ────────────────────────────────────────────────────────────
    def http(self, node, state):
        """One request to a web service (see httpnode.py); its fields come back in `.values`."""
        from src.rooms import httpnode
        return httpnode.run(node)

    # ── the computer itself ──────────────────────────────────────────────────────
    @staticmethod
    def power_command(mode: str, seconds: float = 60, message: str = "", platform: str = None) -> tuple:
        """(argv, how to undo it in words) for a power action on this system, or (None, reason).
        A shutdown or restart always waits (`seconds`, at least POWER_MIN_DELAY), so it can be cancelled."""
        import math
        import sys
        platform = platform or sys.platform
        delay = max(int(seconds or 0), POWER_MIN_DELAY) if mode in ("shutdown", "restart") else int(seconds or 0)
        note = (message or "Argent: комната завершила работу")[:120]
        if platform.startswith("win"):
            table = {
                "shutdown": (["shutdown", "/s", "/t", str(delay), "/c", note], "shutdown /a"),
                "restart": (["shutdown", "/r", "/t", str(delay), "/c", note], "shutdown /a"),
                "hibernate": (["shutdown", "/h"], ""),
                "logout": (["shutdown", "/l"], ""),
                "lock": (["rundll32.exe", "user32.dll,LockWorkStation"], ""),
                "sleep": (["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], ""),
                "cancel": (["shutdown", "/a"], ""),
            }
        elif platform == "darwin":
            minutes = max(1, math.ceil(delay / 60))
            table = {
                "shutdown": (["shutdown", "-h", f"+{minutes}", note], "sudo killall shutdown"),
                "restart": (["shutdown", "-r", f"+{minutes}", note], "sudo killall shutdown"),
                "sleep": (["pmset", "sleepnow"], ""),
                "lock": (["pmset", "displaysleepnow"], ""),
                "logout": (["osascript", "-e", 'tell application "System Events" to log out'], ""),
                "cancel": (["killall", "shutdown"], ""),
            }
        else:
            minutes = max(1, math.ceil(delay / 60))
            table = {
                "shutdown": (["shutdown", "-h", f"+{minutes}", note], "shutdown -c"),
                "restart": (["shutdown", "-r", f"+{minutes}", note], "shutdown -c"),
                "sleep": (["systemctl", "suspend"], ""),
                "hibernate": (["systemctl", "hibernate"], ""),
                "lock": (["loginctl", "lock-session"], ""),
                "cancel": (["shutdown", "-c"], ""),
            }
        if mode not in table:
            return None, f"'{mode}' is not available on this system ({platform})"
        return table[mode]

    def power(self, node, state) -> str:
        """Shut down / restart / sleep / lock ... this computer. `args.dry_run` only says what it
        would run. A shutdown or restart waits at least POWER_MIN_DELAY seconds and says how to cancel."""
        import subprocess
        mode = node.mode
        args = node.args or {}
        seconds = node.seconds if node.seconds is not None else (60 if mode in ("shutdown", "restart") else 0)
        argv, undo = self.power_command(mode, seconds, node.message or "")
        if argv is None:
            return f"Error: {undo}"
        shown = " ".join(argv)
        if args.get("dry_run"):
            return f"Dry run: would run `{shown}`" + (f"; cancel with `{undo}`" if undo else "")
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        except Exception as e:
            return f"Error: could not run `{shown}`: {str(e)[:200]}"
        if done.returncode != 0:
            return f"Error: `{shown}` exited with {done.returncode}: {(done.stderr or done.stdout or '').strip()[:300]}"
        when = f" in {int(max(seconds, POWER_MIN_DELAY))} s" if mode in ("shutdown", "restart") else ""
        return f"OK: {mode}{when}." + (f" Cancel with `{undo}`." if undo else "")

    # ── catalogs (for the editor and the validator) ──────────────────────────────
    def mcp_servers(self) -> dict:
        """{name: {"type": ..., "running": bool}} for every configured server."""
        out = {}
        try:
            from mcp_client import mcp_client
            running = {n: s.is_running for n, s in mcp_client.servers.items()}
        except Exception:
            running = {}
        for name, cfg in self._configured_servers().items():
            # `sig` changes when the server is pointed somewhere else, so a page can drop what it
            # remembered about its tools (the secrets in `env` are not part of it)
            shown = {k: v for k, v in cfg.items() if k != "env"}
            sig = hashlib.sha1(json.dumps(shown, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:10]
            out[name] = {"type": cfg.get("type", "stdio"), "running": bool(running.get(name)), "sig": sig}
        return out

    def mcp_tools(self, server: str) -> list:
        """The server's tools as [{name, description, params:[...]}], connecting it if need be."""
        problem = self.ensure_server(server)
        if problem:
            raise RuntimeError(problem)
        from mcp_client import mcp_client
        out = []
        for t in mcp_client.list_tools(server):
            if "name" not in t or "error" in t:
                continue
            schema = t.get("inputSchema") or t.get("parameters") or {}
            out.append({"name": t["name"], "description": (t.get("description") or "").strip()[:300],
                        "params": schema_params(schema)})
        return out

    def skills(self) -> dict:
        try:
            from skill_manager import skill_manager
            return {s["name"]: s.get("description", "") for s in skill_manager.list_skills()}
        except Exception:
            return {}

    def skill_headings(self, name: str) -> list:
        try:
            from skill_manager import skill_manager
            text = skill_manager.read_skill(name) or ""
        except Exception:
            return []
        fenced, out = False, []
        for line in text.splitlines():
            if line.strip().startswith(("```", "~~~")):
                fenced = not fenced
            m = None if fenced else re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
            if m:
                out.append({"level": len(m.group(1)), "title": m.group(2)})
        return out

    def plugins(self) -> dict:
        """{plugin: {"commands": [...], "hooks": [...]}} for the loaded plugins."""
        out = {}
        try:                                  # a plugin added or removed since the last look shows up
            from hook_manager import hook_manager
            hook_manager.reload_if_changed(quiet=True)
        except Exception:
            pass
        for mod in self._plugin_modules():
            name = getattr(mod, "__name__", "?")
            cmds = sorted(a[len("command_"):] for a in dir(mod) if a.startswith("command_") and callable(getattr(mod, a)))
            hooks = sorted(a for a in dir(mod) if not a.startswith(("_", "command_")) and callable(getattr(mod, a))
                           and getattr(getattr(mod, a), "__module__", None) == name)
            out[name] = {"commands": cmds, "hooks": hooks}
        return out


def schema_params(schema: dict) -> list:
    """A JSON-schema object as a flat list of parameters for a form."""
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    out = []
    for name, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        out.append({
            "name": name,
            "type": spec.get("type") if isinstance(spec.get("type"), str) else "string",
            "required": name in required,
            "description": (spec.get("description") or "")[:240],
            "enum": spec.get("enum") if isinstance(spec.get("enum"), list) else None,
            "default": spec.get("default"),
        })
    return out
