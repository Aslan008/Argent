"""Node runners: turn declarative nodes into real side effects.

The Engine decides WHICH node runs; the NodeRunner decides what running it means
— call a tool, run a bounded agent, prompt a human. Tool and agent execution are
injected, so engine+runner stay testable without an LLM; production wiring is a
thin adapter (argent_tool_executor).

Both executors return a dict of *proposed* state writes. The runner applies them
through the node's ``writes`` whitelist — a field the executor proposes but the
node did not declare is dropped. This is the verifier-first invariant in the
plumbing: an agent that self-reports ``tests_passed`` it wasn't granted simply
can't, so greenness stays owned by the verifier node that declares it. A plain
string return is treated as the node's ``.output`` only.

A tool node may also declare ``sets``: state fields the runner fills from the
tool's own result (the exit code of a command, the tail of its output). That is
the strongest form of a verdict — the process decides, not a model.

Text of a node (instruction, args, prompt) may carry ``{{field}}`` placeholders,
filled from the run state just before the node runs (see template.py).
"""

import inspect
import json
import re
import threading
import time

import math
from src.rooms import condition as cond_dsl
from src.rooms.parallel import resources_of
from src.rooms import policy as room_policy
from src.rooms import template
from src.rooms.models import CALLING_TYPES, FOREACH_MAX, LOOP_DEFAULT_MAX, LOOP_MAX, Node, State
from src.rooms.rules import Rules

_EXIT_CODE = re.compile(r"\s*Exit code:\s*(-?\d+)")
# What a tool (or an MCP server, or Leeth) says when it did not do the job. MCP Error and
# MCP Tool Error are the MCP client's own words; Leeth says why it did not answer in plain text.
_FAILURE_PREFIXES = ("Error", "Execution aborted", "Command BLOCKED", "MCP Error", "MCP Tool Error",
                     "Leeth runtime is currently unavailable", "Leeth declined", "Blocked")
#: How many trailing lines / characters of a failed command's output `tail` keeps.
TAIL_LINES = 40
TAIL_CHARS = 2500


_YES = {"да", "д", "yes", "y", "ok", "ок", "окей", "okay", "+", "go", "давай", "продолжай", "продолжить",
        "согласен", "согласна", "подтверждаю", "принять", "approve", "approved", "sure", "ага", "угу"}


def is_affirmative(answer) -> bool:
    """Whether a person's free-text answer to a question is a plain "yes"."""
    if answer is None:
        return False
    return str(answer).strip().strip(".!").lower() in _YES


def estimate_tokens(text) -> int:
    """A cheap, model-free token estimate (~3 characters per token). The budget needs a
    consistent yardstick, not an exact count."""
    return max(1, len(str(text)) // 3) if text else 0


class AgentText(str):
    """The final text of an agent node plus an estimate of the tokens it cost."""
    tokens: int = 0


def derive_result_values(result) -> dict:
    """The values a tool node's ``sets`` can name, computed from the tool's result."""
    text = result.get("output", "") if isinstance(result, dict) else ("" if result is None else str(result))
    text = text if isinstance(text, str) else str(text)
    m = _EXIT_CODE.match(text)
    code = int(m.group(1)) if m else None
    if code is not None:
        ok = code == 0
    else:
        ok = not text.lstrip().startswith(_FAILURE_PREFIXES)
    body = (text[m.end():] if m else text).strip()
    if body.startswith("OUTPUT:"):          # run_command's "Exit code: N / OUTPUT:" frame
        body = body[len("OUTPUT:"):].strip()
    tail = "" if ok else "\n".join(body.splitlines()[-TAIL_LINES:])[-TAIL_CHARS:]
    return {"ok": ok, "exit_code": code, "output": text, "body": body, "tail": tail}


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_YES_TEXT = {"true", "yes", "y", "да", "1", "ok", "ок"}


def _parse_json_text(text: str):
    """The JSON value in `text`: the whole text, or the last top-level object / array in it
    (an inner object of an earlier value is not a candidate)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    found, at = None, 0
    while at < len(text):
        if text[at] in "{[":
            end = _balanced_end(text, at)
            if end is not None:
                try:
                    found = json.loads(text[at:end])
                    at = end
                    continue
                except ValueError:
                    pass
        at += 1
    return found


def _balanced_end(text: str, start: int):
    pairs = {"{": "}", "[": "]"}
    open_ch, close_ch = text[start], pairs[text[start]]
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _json_path(value, path: str):
    """`a.b.0` inside a parsed JSON value; None when any step is missing."""
    cur = value
    for part in [p for p in path.split(".") if p != ""]:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if cur is None:
            return None
    return cur


def derive_value(spec: str, values: dict):
    """One value a node's ``sets`` asked for. `values` is `derive_result_values(result)`; the five
    base names are looked up there, the rest are computed from the result's body."""
    if spec in values:
        return values[spec]
    body = values.get("body", "")
    if spec == "lines":
        return [ln.strip() for ln in body.splitlines() if ln.strip()]
    if spec == "count":
        return len([ln for ln in body.splitlines() if ln.strip()])
    if spec == "first_line":
        return next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    if spec in ("int", "float"):
        m = _NUMBER.search(body)
        if not m:
            return None
        return int(float(m.group())) if spec == "int" else float(m.group())
    if spec == "bool":
        return body.strip().strip(".!").lower() in _YES_TEXT
    if spec == "json":
        return _parse_json_text(body)
    if spec.startswith("json:"):
        return _json_path(_parse_json_text(body), spec[5:].strip())
    if spec.startswith("regex:"):
        try:
            m = re.search(spec[6:], body, re.S)
        except re.error:
            return ""
        if not m:
            return ""
        return m.group(1) if m.groups() else m.group(0)
    return None


def _empty_value(spec: str):
    """What a field holds when its node did not run (declined, disabled)."""
    if spec == "ok" or spec == "bool":
        return False
    if spec in ("exit_code", "int", "float", "json") or spec.startswith("json:"):
        return None
    if spec == "count":
        return 0
    if spec == "lines":
        return []
    return ""


def _call_with(fn, prompt, **kwargs):
    """Call an injected agent runner with the keyword arguments it understands, so a runner written
    for the older `(prompt, max_iterations=None)` shape keeps working."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(prompt)
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(prompt, **kwargs)
    return fn(prompt, **{k: v for k, v in kwargs.items() if k in params})


class NodeRunner:
    def __init__(self, tool_executor=None, agent_executor=None, human_prompt=None,
                 spawn_handler=None, confirm=None, notify=None, room_handler=None,
                 integrations=None, sleep=None, compose=None):
        self._tool = tool_executor        # (node, state) -> dict | str
        self._agent = agent_executor      # (node, context: dict) -> dict
        self._human = human_prompt        # (node, state) -> None
        self._spawn = spawn_handler       # (node, state) -> None; may set state.route_to
        self._confirm = confirm           # (node, state) -> bool; None == nobody to ask
        self._notify = notify             # (node, state) -> None; None == only record the message
        self._room = room_handler         # (node, state, run_node, on_event) -> None
        # mcp / skill / plugin / leeth nodes: an object with those four methods (integrations.py)
        self._integrations = integrations
        self._sleep = sleep or time.sleep
        # (instruction, state_data) -> text: the model writes a message (notify / hint with `compose`)
        self._compose = compose
        self._on_event = None
        # Parallel branches call run_node from several threads: a question to the person is asked
        # one at a time, and steps that share the one browser page / desktop / Leeth model take turns.
        self._confirm_lock = threading.Lock()
        self._resource_locks = {name: threading.RLock() for name in ("leeth", "page", "screen")}

    def set_on_event(self, callback) -> None:
        """The run's observer, so a sub-room's nodes show up in the same progress and record."""
        self._on_event = callback

    def run_node(self, node: Node, state: State) -> None:
        resources = () if node.disabled else resources_of(node)
        if not resources:
            return self._run_node(node, state)
        locks = [self._resource_locks[name] for name in resources if name in self._resource_locks]
        for lock in locks:                  # resources_of() returns them sorted: one global order, no deadlock
            lock.acquire()
        try:
            return self._run_node(node, state)
        finally:
            for lock in reversed(locks):
                lock.release()

    def _run_node(self, node: Node, state: State) -> None:
        node = self._materialize(node, state)
        if node.disabled:
            # A switched-off step: the run passes through it, and what it would have set stays empty.
            state.data[f"{node.id}.output"] = ""
            state.data[f"{node.id}.skipped"] = True
            for field, which in node.sets.items():
                state.data[field] = _empty_value(which)
            return
        rules = Rules(state.rules)
        why = rules.node_block(node)
        if why:
            self._blocked(node, state, why)
            return
        if rules.ignore and node.type in ("agent", "mcp", "skill"):
            node = node.model_copy(update={"ignore": list(dict.fromkeys(list(node.ignore) + rules.ignore))})
        if node.type == "tool":
            if node.confirm and not self._confirmed(node, state):
                self._declined(node, state)
                return
            if self._tool:
                result = self._tool(node, state)
                self._apply(node, state, result)
                self._set_derived(node, state, result)
        elif node.type in ("mcp", "skill", "plugin", "leeth", "power"):
            if node.type in CALLING_TYPES and node.confirm and not self._confirmed(node, state):
                self._declined(node, state)
                return
            if self._integrations is None:
                raise RuntimeError(f"{node.type} node '{node.id}': no integrations are wired in")
            result = getattr(self._integrations, node.type)(node, state)
            self._apply(node, state, result)
            self._set_derived(node, state, result)
        elif node.type == "foreach":
            self._foreach(node, state)
        elif node.type == "loop":
            self._loop(node, state)
        elif node.type == "break":
            state.data[f"{node.loop}_break"] = True
            state.data[f"{node.id}.output"] = f"break out of '{node.loop}'"
        elif node.type == "continue":
            state.data[f"{node.id}.output"] = f"continue '{node.loop}'"
        elif node.type == "wait":
            self._sleep(float(node.seconds or 0))
        elif node.type == "hint":
            self._hint(node, state)
        elif node.type == "forget":
            self._forget(node, state)
        elif node.type == "assert":
            try:
                holds = cond_dsl.evaluate(node.condition, state.data)
            except cond_dsl.ConditionError as e:
                raise RuntimeError(f"assert '{node.id}': {e}")
            state.data[f"{node.id}.output"] = "ok" if holds else "failed"
            if not holds:
                raise AssertionError(node.message or f"assertion failed: {node.condition}")
        elif node.type == "agent":
            if self._agent:
                agent_node, brief = self._prepare_agent(node, state, rules)
                context = {ref: self._resolve(ref, state) for ref in agent_node.context}
                if hasattr(self._agent, "last_tokens"):
                    self._agent.last_tokens = 0
                if hasattr(self._agent, "last_text"):
                    self._agent.last_text = ""
                if getattr(self._agent, "accepts_brief", False):
                    result = self._agent(agent_node, context, brief=brief)
                else:
                    result = self._agent(agent_node, context)
                self._apply(agent_node, state, result or {})
                spent = getattr(self._agent, "last_tokens", 0) or 0
                if spent:
                    state.usage["tokens"] = state.usage.get("tokens", 0) + spent
                if brief.get("shared"):
                    self._remember(node, state, rules, getattr(self._agent, "last_text", "") or "")
        elif node.type == "human_pause":
            if self._human:
                self._human(node, state)
                if node.into:
                    answer = state.data.get(f"{node.id}.output")
                    state.data[node.into] = answer
                    state.data[f"{node.into}_approved"] = is_affirmative(answer)
        elif node.type == "spawn_room":
            if self._spawn:
                self._spawn(node, state)
        elif node.type == "set":
            for field, value in node.values.items():
                if isinstance(value, str) and value.lstrip().startswith("="):
                    try:
                        value = cond_dsl.evaluate_value(value.lstrip()[1:].strip(), state.data)
                    except cond_dsl.ConditionError as e:
                        raise RuntimeError(f"set '{node.id}', field '{field}': {e}")
                state.data[field] = value        # already rendered by _materialize
        elif node.type == "notify":
            message = node.message
            if node.compose and self._compose is not None:
                message = (self._compose(node.message, dict(state.data)) or node.message).strip() or node.message
                node = node.model_copy(update={"message": message})
            state.data[f"{node.id}.output"] = message
            if self._notify:
                delivered = self._notify(node, state)
                state.data[f"{node.id}.delivered"] = bool(delivered) if delivered is not None else True
        elif node.type == "room":
            if self._room is None:
                raise RuntimeError(f"room node '{node.id}': no sub-room runner is wired in")
            self._room(node, state, self.run_node, self._on_event)
        # condition: no side effect at this layer

    @staticmethod
    def _materialize(node: Node, state: State) -> Node:
        """The node with its `{{field}}` placeholders filled from the run state."""
        update = {}
        data = state.data
        if node.instruction and "{{" in node.instruction:
            update["instruction"] = template.render(node.instruction, data)
        if node.prompt and "{{" in node.prompt:
            update["prompt"] = template.render(node.prompt, data)
        if node.args and "{{" in repr(node.args):
            update["args"] = template.render_value(node.args, data)
        if node.message and "{{" in node.message:
            update["message"] = template.render(node.message, data)
        if node.title and "{{" in node.title:
            update["title"] = template.render(node.title, data)
        if node.values and "{{" in repr(node.values):
            update["values"] = template.render_value(node.values, data)
        if node.inputs and "{{" in repr(node.inputs):
            update["inputs"] = template.render_value(node.inputs, data)
        return node.model_copy(update=update) if update else node

    def _confirmed(self, node: Node, state: State) -> bool:
        if self._confirm is None:
            return False
        with self._confirm_lock:            # two branches must not both put a question on the screen
            try:
                return bool(self._confirm(node, state))
            except Exception:
                return False

    @staticmethod
    def _blocked(node: Node, state: State, why: str) -> None:
        """A step the room's rules forbid: it does not run, says why, and the run goes on."""
        state.data[f"{node.id}.output"] = f"Blocked: {why[0].upper() + why[1:]}."
        state.data[f"{node.id}.blocked"] = True
        for field, which in node.sets.items():
            state.data[field] = _empty_value(which)

    @staticmethod
    def _prepare_agent(node: Node, state: State, rules: Rules) -> tuple:
        """The agent step as it will really run — its tools cut by the room's rules, its model
        parameters (the room's, then its own) — and the brief it is given: hints and the memory
        of what earlier steps did."""
        update = {}
        if rules.restricts():
            update["tools"] = rules.agent_tools(node.tools)
            if rules.no_tools:
                update["mcp"] = []
        sampling = {**rules.sampling, **(node.sampling or {})}
        if sampling != (node.sampling or {}):
            update["sampling"] = sampling
        shared = (node.memory or rules.memory) == "shared"
        hints = []
        if rules.hint:
            hints.append(template.render(rules.hint, state.data))
        if node.hint:
            hints.append(template.render(node.hint, state.data))
        keep = []
        for h in state.hints:
            targets = h.get("targets")
            if targets and node.id not in targets:
                keep.append(h)
                continue
            hints.append(str(h.get("text", "")))
            if h.get("sticky"):
                keep.append(h)
        state.hints[:] = keep
        memory = ""
        if shared and state.memory:
            memory = "\n".join(state.memory)[-rules.memory_chars:]
        brief = {"hints": [h for h in hints if h.strip()], "memory": memory, "shared": shared}
        return (node.model_copy(update=update) if update else node), brief

    @staticmethod
    def _remember(node: Node, state: State, rules: Rules, text: str) -> None:
        """Add what an agent step did to the room's memory (a short digest, bounded in total)."""
        digest = " ".join(str(text).split())
        if not digest:
            return
        state.memory.append(f"[{node.id}] {digest[:700]}")
        while len(state.memory) > 1 and sum(len(m) + 1 for m in state.memory) > rules.memory_chars:
            state.memory.pop(0)

    def _hint(self, node: Node, state: State) -> None:
        mode = node.mode or "next"
        if mode == "clear":
            targets = node.targets
            state.hints[:] = [h for h in state.hints
                              if not (h.get("sticky") and (not targets or set(h.get("targets") or []) & set(targets)))]
            state.data[f"{node.id}.output"] = "hints cleared"
            return
        text = node.message
        if node.compose and self._compose is not None:
            text = (self._compose(node.message, dict(state.data)) or node.message).strip() or node.message
        state.hints.append({"text": text, "sticky": mode == "all", "targets": list(node.targets) if node.targets else None})
        state.data[f"{node.id}.output"] = text

    @staticmethod
    def _forget(node: Node, state: State) -> None:
        mode = node.mode or "all"
        gone = []
        # compared one by one: a literal ("all", "memory") reads as a tool list to tests/test_no_phantom_tools.py
        if (mode == "all" or mode == "memory") and state.memory:
            gone.append(f"{len(state.memory)} memory entries")
            state.memory.clear()
        if (mode == "all" or mode == "hints") and state.hints:
            gone.append(f"{len(state.hints)} hints")
            state.hints.clear()
        for name in node.clear:
            if name in state.data:
                del state.data[name]
                gone.append(f"field {name}")
        state.data[f"{node.id}.output"] = ("forgot: " + ", ".join(gone)) if gone else "nothing to forget"

    @staticmethod
    def _number(node: Node, value, state: State, default=None):
        if value is None or value == "":
            return default
        if isinstance(value, str):
            value = template.render(value, state.data).strip()
            if value.startswith("="):
                try:
                    value = cond_dsl.evaluate_value(value[1:].strip(), state.data)
                except cond_dsl.ConditionError as e:
                    raise RuntimeError(f"loop '{node.id}': {e}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise RuntimeError(f"loop '{node.id}': '{value}' is not a number")
        return int(number) if number == int(number) else number

    def _loop(self, node: Node, state: State) -> None:
        """One visit of a loop's header. It sets `<id>_done`: False (run the body, with the counter
        in `item`) or True (leave). `break` sets `<id>_break`, the next visit then leaves; `continue`
        is just the way back to the header. Bounded by `max_items` (and LOOP_MAX) in every mode."""
        mode = node.mode or "for"
        data = state.data
        counter = f"loop:{node.id}"
        at = state.loops.get(counter, 0)
        name = node.item or ("key" if mode == "for_in" else "i")
        cap = min(node.max_items or LOOP_DEFAULT_MAX, LOOP_MAX)
        done, capped = f"{node.id}_done", f"{node.id}_capped"

        def leave(was_capped=False):
            data[done], data[capped] = True, was_capped
            state.loops[counter] = 0
            data.pop(f"{node.id}_break", None)

        if data.get(f"{node.id}_break"):
            leave()
            data[f"{node.id}_broken"] = True
            return
        data[f"{node.id}_broken"] = False

        want, value, extra = False, at, None
        if mode == "for":
            start = self._number(node, node.start, state, 0)
            stop = self._number(node, node.stop, state)
            step = self._number(node, node.step, state, 1)
            if step == 0:
                raise RuntimeError(f"loop '{node.id}': step is 0")
            value = start + at * step
            want = value < stop if step > 0 else value > stop
            data[f"{node.id}_count"] = max(0, math.ceil((stop - start) / step))
        elif mode == "times":
            total = int(self._number(node, node.stop, state, 0))
            want, value = at < total, at
            data[f"{node.id}_count"] = max(total, 0)
        elif mode in ("while", "do_while", "until"):
            if mode != "while" and at == 0:
                want = True
            else:
                try:
                    holds = cond_dsl.evaluate(node.condition, data)
                except cond_dsl.ConditionError as e:
                    raise RuntimeError(f"loop '{node.id}': {e}")
                want = (not holds) if mode == "until" else holds
            value = at
        elif mode == "for_in":
            from src.rooms.integrations import as_list
            key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
            source = data.get(key)
            if isinstance(source, str) and source.strip().startswith("{"):
                parsed = _parse_json_text(source)
                source = parsed if isinstance(parsed, dict) else source
            if isinstance(source, dict):
                pairs = list(source.items())
            else:
                pairs = list(enumerate(as_list(source)))
            data[f"{node.id}_count"] = len(pairs)
            want = at < len(pairs)
            if want:
                value, extra = pairs[at]
        if want and at >= cap:
            leave(was_capped=True)
            return
        if not want:
            leave()
            return
        data[name] = value
        if mode == "for_in" and node.item_value:
            data[node.item_value] = extra
        data[f"{node.id}_index"] = at
        data[done], data[capped] = False, False
        state.loops[counter] = at + 1

    @staticmethod
    def _declined(node: Node, state: State) -> None:
        state.data[f"{node.id}.output"] = "Declined: the step was not approved, so it did not run."
        state.data[f"{node.id}.declined"] = True
        for field, which in node.sets.items():
            state.data[field] = _empty_value(which)

    @staticmethod
    def _set_derived(node: Node, state: State, result) -> None:
        if not node.sets:
            return
        values = derive_result_values(result)
        for field, which in node.sets.items():
            state.data[field] = derive_value(which, values)

    @staticmethod
    def _foreach(node: Node, state: State) -> None:
        """One visit = one item. `<id>_done` turns True when the list is used up (and the counter
        resets, so a later visit walks the list again). Bound by `max_items` and the room's steps."""
        from src.rooms.integrations import as_list
        key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
        if state.data.pop(f"{node.id}_break", False):          # a `break` inside the body
            state.data[f"{node.id}_done"] = True
            state.data[f"{node.id}_broken"] = True
            state.loops[f"foreach:{node.id}"] = 0
            return
        items = as_list(state.data.get(key))
        limit = min(node.max_items or FOREACH_MAX, FOREACH_MAX)
        items = items[:limit]
        name = node.item or "item"
        counter = f"foreach:{node.id}"
        at = state.loops.get(counter, 0)
        state.data[f"{node.id}_count"] = len(items)
        if at < len(items):
            state.data[name] = items[at]
            state.data[f"{node.id}_index"] = at
            state.data[f"{node.id}_done"] = False
            state.loops[counter] = at + 1
        else:
            state.data[f"{node.id}_done"] = True
            state.loops[counter] = 0

    @staticmethod
    def _resolve(ref: str, state: State):
        key = ref[len("state."):] if ref.startswith("state.") else ref
        return state.data.get(key)

    @staticmethod
    def _apply(node: Node, state: State, result) -> None:
        if not isinstance(result, dict):
            # bare output
            if result is not None:
                state.data[f"{node.id}.output"] = result
            return
        if "output" in result and result["output"] is not None:
            state.data[f"{node.id}.output"] = result["output"]
        allowed = set(node.writes)
        for key, value in result.items():
            if key == "output":
                continue
            if key in allowed:
                state.data[key] = value
            # keys outside the whitelist are silently dropped — bounded influence


def _extract_json_object(raw: str):
    """Pull the last balanced JSON object out of an agent's free-text answer.
    Returns a dict or None. Reuses Argent's balanced-brace extractor."""
    if not raw:
        return None
    import json
    from src.agent.parser import extract_balanced_json
    for start in reversed([i for i, ch in enumerate(raw) if ch == "{"]):
        blob = extract_balanced_json(raw, start)
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


class AgentExecutor:
    """Runs an agent node's bounded worker and returns the state fields it
    reported. The LLM call is injected (`run_agent(prompt) -> final_text`) so the
    prompt building and write parsing are testable without a model. The runner
    still enforces the node's `writes` whitelist on whatever comes back."""

    #: the NodeRunner passes `brief=` (hints and the room's memory) to executors that say so
    accepts_brief = True

    def __init__(self, run_agent):
        self._run_agent = run_agent
        # Per thread: two branches of a fork call this executor at the same time, and each must read
        # back what ITS call cost and said, not what the other's did.
        self._last = threading.local()

    @property
    def last_tokens(self) -> int:
        """estimated tokens the last call (of this thread) cost; the NodeRunner adds it to the run's usage"""
        return getattr(self._last, "tokens", 0)

    @last_tokens.setter
    def last_tokens(self, value: int) -> None:
        self._last.tokens = value

    @property
    def last_text(self) -> str:
        """what the worker last said (the NodeRunner keeps a digest of it as the room's memory)"""
        return getattr(self._last, "text", "")

    @last_text.setter
    def last_text(self, value: str) -> None:
        self._last.text = value

    def __call__(self, node: Node, context: dict, brief: dict = None) -> dict:
        prompt = self._build_prompt(node, context, brief)
        kwargs = {"max_iterations": node.budget.max_iterations if node.budget else None}
        if node.budget:
            kwargs["max_tokens"] = node.budget.max_tokens
        if node.tools is not None:
            kwargs["tools"] = list(node.tools)
        if node.role:
            kwargs["role"] = node.role
        if node.skills:
            kwargs["skills"] = list(node.skills)
        if node.mcp is not None:
            kwargs["mcp_servers"] = list(node.mcp)
        if node.ignore:
            kwargs["ignore"] = list(node.ignore)
        if node.sampling:
            kwargs["sampling"] = dict(node.sampling)
        raw = _call_with(self._run_agent, prompt, **kwargs)
        spent = getattr(raw, "tokens", 0) or 0
        self.last_tokens = spent or (estimate_tokens(prompt) + estimate_tokens(raw))
        self.last_text = str(raw or "")
        return _extract_json_object(raw) or {}

    @staticmethod
    def _build_prompt(node: Node, context: dict, brief: dict = None) -> str:
        parts = [node.instruction or "Complete your assigned task."]
        brief = brief or {}
        if brief.get("hints"):
            parts.append("\nHINTS (from the person who set this room up; follow them):")
            parts.extend(f"- {h}" for h in brief["hints"])
        if brief.get("memory"):
            parts.append("\nWHAT EARLIER STEPS OF THIS ROOM DID (your memory of the work so far):")
            parts.append(brief["memory"])
        if context:
            parts.append("\nCONTEXT:")
            for key, value in context.items():
                parts.append(f"- {key}: {value}")
        if node.writes:
            example = "{" + ", ".join(f'"{w}": ...' for w in node.writes) + "}"
            parts.append(
                "\nWhen finished, output a single JSON object on its own line with "
                f"exactly these fields (and no others): {node.writes}. "
                f"Example: {example}"
            )
        return "\n".join(parts)


def _consume_bounded(chunks, max_iterations=None, max_tokens=None) -> AgentText:
    """Drain a sub-agent's chunk stream into its final text, stopping after
    ``max_iterations`` tool-execution rounds or once the node has read and written
    about ``max_tokens``. This turns an agent node's budget into a real bound on the
    ReAct loop rather than a hope — without it a node could act far more than its
    declared budget. Content produced after the cap is not included. Isolated from
    ArgentSubAgent so it's testable without an LLM. The result carries ``.tokens``,
    the estimate of what the node cost (its own text plus the tool results it read)."""
    final = ""
    tool_rounds = 0
    spent = 0
    stop_note = ""
    for chunk in chunks:
        kind = chunk.get("type")
        if kind in ("content_stream", "content"):
            text = chunk.get("content", "")
            final += text
            spent += estimate_tokens(text)
        elif kind == "tool_end":
            tool_rounds += 1
            spent += estimate_tokens(chunk.get("result", ""))
            if max_iterations and tool_rounds >= max_iterations:
                stop_note = f"\n[Rooms: node iteration budget ({max_iterations}) reached — stopping.]"
                break
        if max_tokens and spent >= max_tokens:
            stop_note = f"\n[Rooms: node token budget ({max_tokens}) reached — stopping.]"
            break
    out = AgentText(final + stop_note)
    out.tokens = spent
    return out


def _skills_block(skills, rules) -> str:
    """The instructions of the skills a node was given, as one block for the worker's prompt
    (the rules cut out what the node was told to ignore)."""
    from skill_manager import skill_manager
    parts = []
    with room_policy.use(rules):
        for name in skills or ():
            if room_policy.skill_blocked(name):
                continue
            text = skill_manager.read_skill(name)
            if text:
                parts.append(f"=== SKILL: {name} ===\n{room_policy.cut_skill(name, text)}")
    return "\n\n".join(parts)


def argent_run_agent(instruction: str, tools_override=None, max_iterations=None,
                     role=None, tools=None, max_tokens=None, skills=None, mcp_servers=None,
                     ignore=None, sampling=None) -> str:
    """Production adapter: run an ArgentSubAgent worker bounded by the node's
    iteration and token budget, and return its final text. ``tools`` (a node's
    whitelist) limits what the worker may call; ``role`` picks its persona.

    ``skills`` are read into the worker's prompt, ``mcp_servers`` limits which MCP servers it
    may reach (None = all) and ``ignore`` lists what it must not see or use. The last two are
    enforced where the tools live (policy.py), for as long as the worker runs."""
    from agent import ArgentSubAgent
    from src.agent.subagents import FORBIDDEN_TOOLS
    from tools.schemas import AVAILABLE_TOOLS
    allowed = tools if tools is not None else tools_override
    if allowed is None:
        allowed = list(AVAILABLE_TOOLS)
    allowed = list(allowed)
    if mcp_servers:
        # a node that was given MCP servers must be able to call them
        allowed += [t for t in ("list_mcp_tools", "call_mcp_tool") if t not in allowed]
    # a room's worker never gets the tools that need a person, start more agents or another room
    allowed = [t for t in allowed if t not in FORBIDDEN_TOOLS]
    rules = room_policy.Policy(ignore or (), mcp_servers)
    extra = []
    block = _skills_block(skills, rules)
    if block:
        extra.append(block)
    told = rules.describe()
    if told:
        extra.append("RULES FOR THIS TASK:\n" + told)
    prompt = instruction if not extra else instruction + "\n\n" + "\n\n".join(extra)
    from config import generation_overrides
    sub = ArgentSubAgent(role or "Coder", prompt, tools_override=allowed)
    with room_policy.use(rules), generation_overrides(sampling):
        return _consume_bounded(
            sub.process_user_input(f"Start task: {prompt}", allowed_tools=allowed),
            max_iterations, max_tokens,
        )


def argent_compose(instruction: str, data: dict) -> str:
    """Production adapter for a `compose` message: the model writes one short text from the
    instruction and the run's facts. No tools, one round, a small budget."""
    facts = "\n".join(f"- {k}: {str(v)[:300]}" for k, v in data.items()
                      if not str(k).startswith("_") and k not in ("rooms_catalog",))[:3000]
    prompt = ("Write ONE short message for the person, following the instruction. "
              "Output only the message text — no preface, no quotes.\n"
              f"INSTRUCTION: {instruction}\nFACTS OF THE RUN:\n{facts}")
    return str(argent_run_agent(prompt, tools=[], max_iterations=1, max_tokens=800, role="Researcher")).strip()


def argent_tool_executor(node: Node, state: State):
    """Production adapter: run a real Argent tool for a tool node.

    `input_from` supplies the primary argument value, bound to the tool's first
    required parameter. Real tools return a string, which the runner stores as
    the node's `.output`; deriving structured flags from that output is the job
    of a following agent/condition node. Never raises — errors come back as text.
    """
    import inspect
    from tools.schemas import AVAILABLE_TOOLS

    fn = AVAILABLE_TOOLS.get(node.tool)
    if fn is None:
        return f"Error: tool '{node.tool}' is not available"

    kwargs = dict(node.args) if node.args else {}
    if node.input_from:
        key = node.input_from[len("state."):] if node.input_from.startswith("state.") else node.input_from
        value = state.data.get(key)
        if value is not None:
            required = [
                p for p in inspect.signature(fn).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and p.name not in kwargs
            ]
            if required:
                kwargs[required[0].name] = value
    try:
        return str(fn(**kwargs))
    except Exception as e:
        return f"Error running tool '{node.tool}': {e}"
