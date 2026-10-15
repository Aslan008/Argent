"""Pydantic schema for the rooms engine: State, Budget, Node, Edge, Room.

Rooms are declarative data validated against this schema. Structural, per-type
invariants that need no external context (an agent node must carry a budget, a
`while` edge must carry a counter, ...) are enforced here at construction time,
so a malformed room cannot even be built. Cross-cutting checks that need outside
knowledge (does the tool exist? is every node able to reach an EXIT?) live in the
RoomValidator, not here.
"""

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

NodeType = Literal[
    "tool", "agent", "condition", "human_pause", "spawn_room", "set", "notify", "room",
    "mcp", "skill", "plugin", "leeth", "foreach", "wait",
    "loop", "break", "continue", "power", "hint", "forget", "assert",
    "parallel", "join",
]

#: Node types that reach outside the run (a server, a plugin, another engine): they may change
#: things, so they are asked about when `confirm` is set and are never replayed blindly on resume.
CALLING_TYPES = ("tool", "mcp", "plugin", "leeth", "power")

#: What a `mcp` node does: call one tool of a server, or list the server's tools.
MCP_MODES = ("call", "list")
#: What a `plugin` node does: run the plugin's `command_<name>` or fire a hook event.
PLUGIN_MODES = ("command", "hook")
#: What a `leeth` node asks of Leeth: do a routine task (`delegate`), answer one yes/no question
#: about every item of a list (`filter`), sort a list into categories (`classify`), judge the
#: clickable things / posts of the open browser page (`scan`), fill the open page's form (`fill`),
#: click what the last scan found (`click`).
LEETH_MODES = ("delegate", "filter", "classify", "scan", "fill", "click")

#: What a `loop` node does. for: a number from `start` to before `stop` by `step`; times: `stop` times;
#: while: the body runs while the condition holds (checked first); do_while: the body runs once, then
#: while the condition holds; until: the body runs once, then until the condition holds (repeat-until);
#: for_in: over the keys (and values) of a dict, or the indices (and items) of a list.
LOOP_MODES = ("for", "times", "while", "do_while", "until", "for_in")
#: What a `power` node does to the computer.
POWER_MODES = ("shutdown", "restart", "sleep", "hibernate", "lock", "logout", "cancel")
#: A `hint` node: for the next agent step only, for every agent step until cleared, or clear the sticky ones.
HINT_MODES = ("next", "all", "clear")
#: What a `forget` node wipes: the memory of earlier steps, the hints, or both.
FORGET_MODES = ("all", "memory", "hints")
#: How an agent step relates to what earlier steps of the room did.
MEMORY_MODES = ("isolated", "shared")
#: A loop's safety ceiling when `max_items` is not given, and the most it may be given.
LOOP_DEFAULT_MAX = 100
LOOP_MAX = 1000
#: A shutdown / restart is never immediate: this many seconds at least, so it can be cancelled.
POWER_MIN_DELAY = 10
POWER_MAX_DELAY = 3600
#: Model parameters a step or a room may set, with their bounds (reasoning_effort is a word).
SAMPLING_BOUNDS = {
    "temperature": (0.0, 2.0), "top_p": (0.0, 1.0), "top_k": (1, 100000), "min_p": (0.0, 1.0),
    "repeat_penalty": (0.5, 2.0), "thinking_budget": (0, 1000000),
}
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")

#: A `parallel` node starts all of its outgoing arrows at once and the `join` node it names gathers them.
#: How many branches one fork may have, and how deep forks may be nested.
PARALLEL_MAX_BRANCHES = 8
PARALLEL_MAX_DEPTH = 3
#: What a `join` does when two branches wrote different values to the same field: stop the run (`error`),
#: keep the value of the first / last branch in the order the arrows were drawn, or keep all of them as a list.
MERGE_MODES = ("error", "first", "last", "list")
#: A `join` waits for every branch (`all`) or goes on as soon as the first one arrives (`any`; the others are stopped).
WAIT_FOR_MODES = ("all", "any")
#: A branch that ends otherwise than at the join (an exit, a failure) stops the whole run (`fail`) or is dropped (`continue`).
BRANCH_FAIL_MODES = ("fail", "continue")

#: How many items a `foreach` node may walk through at most.
FOREACH_MAX = 500
#: The longest a `wait` node may sleep, in seconds.
WAIT_MAX_SECONDS = 3600

_IGNORE_RULE = re.compile(r"^(mcp:[^\s:#]+|skill:[^\s:#]+(#.+)?)$")

#: Where a `notify` node speaks: the console, a desktop toast, the Telegram chat.
NOTIFY_CHANNELS = ("console", "desktop", "telegram")

#: What a tool node's ``sets`` can derive from the tool's result. ``ok`` and
#: ``exit_code`` come from the "Exit code: N" line of run_command; ``body`` is the
#: output without that line (empty when the command printed nothing); ``tail`` is
#: the last lines of the output and is empty when the command succeeded.
DERIVED_VALUES = ("ok", "exit_code", "output", "body", "tail")

#: More ways to take a value out of a result, for anything that returns more than a verdict:
#: ``lines`` (a list of the non-empty lines), ``count`` (how many), ``first_line``, ``int`` /
#: ``float`` (the first number), ``bool`` (a plain yes in the text), ``json`` (the parsed answer),
#: ``json:a.b.0`` (one value inside it) and ``regex:<pattern>`` (the first group, or the match).
EXTRA_DERIVED = ("lines", "count", "first_line", "int", "float", "bool", "json")


def is_derived_spec(spec: str) -> bool:
    """Whether `spec` is a value a node's ``sets`` may ask for."""
    if spec in DERIVED_VALUES or spec in EXTRA_DERIVED:
        return True
    if spec.startswith("json:"):
        return bool(spec[5:].strip())
    if spec.startswith("regex:"):
        try:
            re.compile(spec[6:])
        except re.error:
            return False
        return bool(spec[6:])
    return False


def sampling_problems(sampling) -> list:
    """What is wrong with a `sampling` dict, as readable text (empty when it is fine)."""
    problems = []
    if not isinstance(sampling, dict):
        return ["sampling must be an object like {\"temperature\": 0.2}"]
    for key, value in sampling.items():
        if key == "reasoning_effort":
            if value not in REASONING_EFFORTS:
                problems.append(f"reasoning_effort must be one of {list(REASONING_EFFORTS)}")
            continue
        if key not in SAMPLING_BOUNDS:
            problems.append(f"unknown model parameter '{key}' (use {sorted(SAMPLING_BOUNDS) + ['reasoning_effort']})")
            continue
        low, high = SAMPLING_BOUNDS[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            problems.append(f"{key} must be a number from {low} to {high}")
    return problems


def is_ignore_rule(rule: str) -> bool:
    """`mcp:<server>[.<tool pattern>]`, `skill:<name>` or `skill:<name>#<heading>`."""
    return bool(_IGNORE_RULE.match(str(rule).strip()))


class Budget(BaseModel):
    """Hard limits for a room or an agent node. Every agent node and every room
    must carry one — an unbounded loop is the failure mode this whole design
    exists to prevent. ``max_tokens`` is enforced (an estimate of what the agent
    nodes read and wrote); ``max_seconds`` is an optional wall-clock ceiling for
    a whole room."""
    model_config = ConfigDict(extra="forbid")
    max_iterations: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    max_seconds: Optional[int] = Field(default=None, gt=0)


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: NodeType

    # tool nodes
    tool: Optional[str] = None
    input_from: Optional[str] = None
    # Literal keyword arguments passed to the tool (deterministic). input_from,
    # when set, fills the first required parameter not already given here.
    args: dict = Field(default_factory=dict)
    instruction: Optional[str] = None
    # A non-idempotent tool (writes files, runs commands) must never be replayed
    # automatically on resume; the engine routes an indeterminate replay to a human.
    # (An mcp, plugin or leeth node reaches outside the run, so it defaults to False.)
    idempotent: bool = True
    # State fields the ENGINE fills from the tool's result — a verdict owned by the
    # process, not by anyone's opinion: {"tests_passed": "ok", "error_summary": "tail"}.
    sets: dict[str, str] = Field(default_factory=dict)
    # Ask the person before this node runs (a commit, a push, a deploy). With nobody
    # to ask, the node does NOT run.
    confirm: bool = False

    # agent nodes
    # Sub-agent role (Coder, Researcher, Reviewer, DocWriter, BrowserWorker, Critic).
    role: Optional[str] = None
    # Whitelist of tools the node's agent may use; None = the role's usual toolset.
    tools: Optional[list[str]] = None
    context: list[str] = Field(default_factory=list)
    budget: Optional[Budget] = None
    # Whitelist of state fields this agent node may write. The engine ignores
    # anything it writes outside this list, so a model's influence on routing is
    # bounded and auditable rather than "whatever it puts in state".
    writes: list[str] = Field(default_factory=list)

    # condition nodes
    condition: Optional[str] = None

    # human_pause nodes
    prompt: Optional[str] = None
    # State field that receives the person's answer; `<into>_approved` is set to
    # whether the answer was a "yes" (so an edge can branch on `state.<into>_approved`).
    into: Optional[str] = None

    # set nodes: literal values (strings may carry {{field}}) written into the run state
    values: dict[str, Any] = Field(default_factory=dict)

    # notify nodes
    message: Optional[str] = None
    title: Optional[str] = None
    channel: str = "console"

    # room nodes: run another room as a sub-routine. `inputs` become the child's state
    # ({{field}} filled from this run), `outputs` are the child's fields copied back, and
    # the child's exit name lands in `into` (default `<id>_exit`) for the edges to branch on.
    room: Optional[str] = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)

    # Run the node again (up to this many extra times) when it RAISES; see Edge.on_error.
    retries: int = Field(default=0, ge=0, le=5)

    # Any node: skip it (the run passes straight through, as if it were not there), and a
    # free-text note for whoever reads the room.
    disabled: bool = False
    note: Optional[str] = None

    # mcp nodes: which configured MCP server, and (with `tool`) which of its tools; `args` are the
    # tool's arguments. `mode` is "call" (default) or "list" (the server's tools become the output).
    server: Optional[str] = None
    mode: Optional[str] = None

    # skill nodes: the skill whose instructions are read into the run state.
    skill: Optional[str] = None

    # plugin nodes: the plugin's `command_<command>` (mode "command") or a hook event (mode "hook");
    # `args` become its positional / keyword arguments.
    plugin: Optional[str] = None
    command: Optional[str] = None

    # agent nodes: skills whose instructions the worker is given up front, MCP servers it may use
    # (None = all), and what to leave out of what it can see (see policy.py for the rules).
    skills: Optional[list[str]] = None
    mcp: Optional[list[str]] = None
    # agent, mcp and skill nodes: things to ignore — `mcp:<server>[.<tool pattern>]`,
    # `skill:<name>` or `skill:<name>#<heading>` (a section of that skill).
    ignore: list[str] = Field(default_factory=list)

    # foreach nodes: walk the list in `input_from`, one item per visit, into the field `item`
    # (also `<id>_index`, `<id>_count` and `<id>_done`); at most `max_items` of them.
    item: Optional[str] = None
    max_items: Optional[int] = None

    # wait nodes: how long to pause; power nodes: the delay before the action.
    seconds: Optional[float] = None

    # loop nodes (see LOOP_MODES). `start` / `stop` / `step` may be numbers or "{{field}}" text;
    # `item` is the counter (or the key), `item_value` the value of a for_in; `max_items` the ceiling.
    start: Optional[Any] = None
    stop: Optional[Any] = None
    step: Optional[Any] = None
    item_value: Optional[str] = None
    # break / continue nodes: the loop (a loop or foreach node id) they belong to. They must lead back to it.
    loop: Optional[str] = None

    # parallel nodes: the id of the join node that gathers the branches, and how many branches run at
    # once (None = all of them; 1 = one after another, which is handy for debugging).
    join: Optional[str] = None
    workers: Optional[int] = None
    # join nodes: how conflicting writes are merged, whether it waits for all branches, and what a
    # branch that does not arrive means (see MERGE_MODES, WAIT_FOR_MODES, BRANCH_FAIL_MODES).
    merge: Optional[str] = None
    wait_for: Optional[str] = None
    on_branch_fail: Optional[str] = None

    # agent nodes: `memory` shared = the worker is told what earlier steps of the room did; `hint` is a
    # note for this worker only; `sampling` are its model parameters (temperature, top_p, ...).
    memory: Optional[str] = None
    hint: Optional[str] = None
    sampling: dict = Field(default_factory=dict)
    # hint nodes: only these agent steps get the hint (None = any). forget nodes: state fields to wipe.
    targets: Optional[list[str]] = None
    clear: list[str] = Field(default_factory=list)
    # notify / hint nodes: `message` is an instruction, and the model writes the text from the state.
    compose: bool = False

    @model_validator(mode="after")
    def _require_per_type_fields(self) -> "Node":
        if self.type == "tool" and not self.tool:
            raise ValueError(f"tool node '{self.id}' requires a 'tool'")
        if self.type == "agent" and self.budget is None:
            raise ValueError(f"agent node '{self.id}' requires a 'budget'")
        if self.type == "condition" and not self.condition:
            raise ValueError(f"condition node '{self.id}' requires a 'condition'")
        if self.type == "set":
            if not self.values:
                raise ValueError(f"set node '{self.id}' requires 'values'")
            bad = sorted(k for k in self.values if not str(k).isidentifier())
            if bad:
                raise ValueError(f"set node '{self.id}': field names must be identifiers: {bad}")
        elif self.values:
            raise ValueError(f"node '{self.id}': 'values' is only for set nodes")
        if self.type == "notify":
            if not self.message:
                raise ValueError(f"notify node '{self.id}' requires a 'message'")
            if self.channel not in NOTIFY_CHANNELS:
                raise ValueError(f"notify node '{self.id}': channel must be one of {list(NOTIFY_CHANNELS)}")
        self._check_integrations()
        if self.type == "room":
            if not self.room:
                raise ValueError(f"room node '{self.id}' requires 'room' (the room to run)")
            bad = sorted(k for k in list(self.inputs) + list(self.outputs) if not str(k).isidentifier())
            if bad:
                raise ValueError(f"room node '{self.id}': inputs/outputs must be identifiers: {bad}")
        elif self.inputs or self.outputs:
            raise ValueError(f"node '{self.id}': 'inputs'/'outputs' are only for room nodes")
        if self.retries and self.type not in ("tool", "agent", "room", "mcp", "plugin", "leeth", "skill"):
            raise ValueError(f"node '{self.id}': 'retries' is only for tool, agent, room, mcp, skill, plugin and leeth nodes")
        if self.sets:
            if self.type not in ("tool", "mcp", "plugin", "leeth", "skill"):
                raise ValueError(f"node '{self.id}': 'sets' is only for tool, mcp, skill, plugin and leeth nodes")
            bad = sorted(v for v in self.sets.values() if not is_derived_spec(str(v)))
            if bad:
                raise ValueError(
                    f"node '{self.id}': unknown derived value {bad} (use {list(DERIVED_VALUES) + list(EXTRA_DERIVED)}, "
                    "json:<path> or regex:<pattern>)")
        if self.confirm and self.type not in CALLING_TYPES:
            raise ValueError(f"node '{self.id}': 'confirm' is only for tool, mcp, plugin and leeth nodes")
        if self.into and self.type not in ("human_pause", "room"):
            raise ValueError(f"node '{self.id}': 'into' is only for human_pause and room nodes")
        if (self.tools is not None or self.role) and self.type != "agent":
            raise ValueError(f"node '{self.id}': 'role'/'tools' are only for agent nodes")
        return self

    def _check_integrations(self) -> None:
        """Per-type requirements of the nodes that reach outside the run, and of the properties
        every node may carry."""
        t = self.type
        # Fields that belong to one kind of node only.
        only = {
            "server": ("mcp",),
            "skill": ("skill",),
            "plugin": ("plugin",),
            "command": ("plugin",),
            "skills": ("agent",),
            "mcp": ("agent",),
            "item": ("foreach", "loop"),
            "max_items": ("foreach", "loop"),
            "seconds": ("wait", "power"),
            "start": ("loop",),
            "stop": ("loop",),
            "step": ("loop",),
            "item_value": ("loop",),
            "loop": ("break", "continue"),
            "memory": ("agent",),
            "hint": ("agent",),
            "targets": ("hint",),
            "clear": ("forget",),
            "join": ("parallel",),
            "workers": ("parallel",),
            "merge": ("join",),
            "wait_for": ("join",),
            "on_branch_fail": ("join",),
        }
        for field, types in only.items():
            value = getattr(self, field)
            if value not in (None, [], "") and t not in types:
                raise ValueError(f"node '{self.id}': '{field}' is only for {', '.join(types)} nodes")
        if self.mode is not None and t not in ("mcp", "plugin", "leeth", "loop", "power", "hint", "forget"):
            raise ValueError(f"node '{self.id}': 'mode' is only for mcp, plugin, leeth, loop, power, hint and forget nodes")
        if self.sampling and t != "agent":
            raise ValueError(f"node '{self.id}': 'sampling' is only for agent nodes")
        if self.compose and t not in ("notify", "hint"):
            raise ValueError(f"node '{self.id}': 'compose' is only for notify and hint nodes")
        bad_sampling = sampling_problems(self.sampling) if self.sampling else []
        if bad_sampling:
            raise ValueError(f"node '{self.id}': {'; '.join(bad_sampling)}")
        if self.memory is not None and self.memory not in MEMORY_MODES:
            raise ValueError(f"node '{self.id}': memory must be one of {list(MEMORY_MODES)}")
        for field in ("item_value", "item"):
            value = getattr(self, field)
            if t == "loop" and value is not None and not str(value).isidentifier():
                raise ValueError(f"loop node '{self.id}': '{field}' must be a plain field name")
        for name in list(self.targets or []) + list(self.clear):
            if not str(name).replace(".", "_").isidentifier():
                raise ValueError(f"node '{self.id}': '{name}' is not a plain name")
        if self.ignore and t not in ("agent", "mcp", "skill"):
            raise ValueError(f"node '{self.id}': 'ignore' is only for agent, mcp and skill nodes")
        bad = [r for r in self.ignore if not is_ignore_rule(r)]
        if bad:
            raise ValueError(
                f"node '{self.id}': ignore rule {bad} is not understood — use mcp:<server>[.<tool>], "
                "skill:<name> or skill:<name>#<heading>")
        if t in ("mcp", "plugin", "leeth", "power") and "idempotent" not in self.model_fields_set:
            self.idempotent = False          # it reaches outside the run: never replay it blindly
        if self.args and t not in ("tool", "mcp", "plugin", "leeth", "power"):
            raise ValueError(f"node '{self.id}': 'args' is only for tool, mcp, plugin, leeth and power nodes")

        if t == "mcp":
            mode = self.mode or "call"
            if mode not in MCP_MODES:
                raise ValueError(f"mcp node '{self.id}': mode must be one of {list(MCP_MODES)}")
            if not self.server:
                raise ValueError(f"mcp node '{self.id}' requires 'server'")
            if mode == "call" and not self.tool:
                raise ValueError(f"mcp node '{self.id}' requires 'tool' (the server's tool to call)")
        elif t == "skill":
            if not self.skill:
                raise ValueError(f"skill node '{self.id}' requires 'skill'")
        elif t == "plugin":
            mode = self.mode or "command"
            if mode not in PLUGIN_MODES:
                raise ValueError(f"plugin node '{self.id}': mode must be one of {list(PLUGIN_MODES)}")
            if not self.command:
                raise ValueError(f"plugin node '{self.id}' requires 'command' (the command or hook event)")
        elif t == "leeth":
            mode = self.mode or "delegate"
            if mode not in LEETH_MODES:
                raise ValueError(f"leeth node '{self.id}': mode must be one of {list(LEETH_MODES)}")
            if mode == "delegate" and not (self.instruction or "").strip():
                raise ValueError(f"leeth node '{self.id}' (delegate) requires 'instruction' (the task for Leeth)")
            if mode in ("filter", "classify") and not (self.input_from or self.args.get("items")):
                raise ValueError(f"leeth node '{self.id}' ({mode}) requires 'input_from' (the field with the list) or args.items")
            if mode == "filter" and not (self.instruction or "").strip():
                raise ValueError(f"leeth node '{self.id}' (filter) requires 'instruction' (the yes/no question)")
            if mode == "classify" and len(self.args.get("categories") or []) < 2:
                raise ValueError(f"leeth node '{self.id}' (classify) requires args.categories with two or more names")
            if mode == "scan" and not (self.instruction or "").strip():
                raise ValueError(f"leeth node '{self.id}' (scan) requires 'instruction' (the question to judge the page by)")
            if mode == "fill" and not (self.instruction or "").strip():
                raise ValueError(f"leeth node '{self.id}' (fill) requires 'instruction' (what to type into the form)")
            if mode == "click" and not (self.input_from or self.args.get("indices")):
                raise ValueError(f"leeth node '{self.id}' (click) requires 'input_from' (the field with the scan result) or args.indices")
        elif t == "foreach":
            if not self.input_from:
                raise ValueError(f"foreach node '{self.id}' requires 'input_from' (the field that holds the list)")
            if self.item is not None and not str(self.item).isidentifier():
                raise ValueError(f"foreach node '{self.id}': 'item' must be a plain field name")
            if self.max_items is not None and not 0 < self.max_items <= FOREACH_MAX:
                raise ValueError(f"foreach node '{self.id}': 'max_items' must be between 1 and {FOREACH_MAX}")
        elif t == "wait":
            if self.seconds is None or not 0 < self.seconds <= WAIT_MAX_SECONDS:
                raise ValueError(f"wait node '{self.id}' requires 'seconds' between 0 and {WAIT_MAX_SECONDS}")
        elif t == "loop":
            mode = self.mode or "for"
            if mode not in LOOP_MODES:
                raise ValueError(f"loop node '{self.id}': mode must be one of {list(LOOP_MODES)}")
            if mode in ("for", "times") and self.stop in (None, ""):
                raise ValueError(f"loop node '{self.id}' ({mode}) requires 'stop' ({'where to stop' if mode == 'for' else 'how many times'})")
            if mode in ("while", "do_while", "until") and not (self.condition or "").strip():
                raise ValueError(f"loop node '{self.id}' ({mode}) requires a 'condition'")
            if mode == "for_in" and not self.input_from:
                raise ValueError(f"loop node '{self.id}' (for_in) requires 'input_from' (the field with the dict or list)")
            if self.step in (0, 0.0, "0"):
                raise ValueError(f"loop node '{self.id}': 'step' cannot be 0")
            if self.max_items is not None and not 0 < self.max_items <= LOOP_MAX:
                raise ValueError(f"loop node '{self.id}': 'max_items' must be between 1 and {LOOP_MAX}")
        elif t in ("break", "continue"):
            if not self.loop:
                raise ValueError(f"{t} node '{self.id}' requires 'loop' (the id of the loop it belongs to)")
        elif t == "power":
            if self.mode not in POWER_MODES:
                raise ValueError(f"power node '{self.id}' requires 'mode' (one of {list(POWER_MODES)})")
            if self.seconds is not None:
                if not 0 <= self.seconds <= POWER_MAX_DELAY:
                    raise ValueError(f"power node '{self.id}': 'seconds' must be between 0 and {POWER_MAX_DELAY}")
                if self.mode in ("shutdown", "restart") and self.seconds < POWER_MIN_DELAY:
                    raise ValueError(f"power node '{self.id}': {self.mode} waits at least {POWER_MIN_DELAY} seconds "
                                     "so that it can still be cancelled")
        elif t == "hint":
            if not (self.message or "").strip():
                raise ValueError(f"hint node '{self.id}' requires a 'message' (the hint, or with compose the instruction)")
            if (self.mode or "next") not in HINT_MODES:
                raise ValueError(f"hint node '{self.id}': mode must be one of {list(HINT_MODES)}")
        elif t == "forget":
            if (self.mode or "all") not in FORGET_MODES:
                raise ValueError(f"forget node '{self.id}': mode must be one of {list(FORGET_MODES)}")
        elif t == "assert":
            if not (self.condition or "").strip():
                raise ValueError(f"assert node '{self.id}' requires a 'condition'")
        elif t == "parallel":
            if not self.join:
                raise ValueError(f"parallel node '{self.id}' requires 'join' (the id of the join node that gathers its branches)")
            if self.workers is not None and not 1 <= self.workers <= PARALLEL_MAX_BRANCHES:
                raise ValueError(f"parallel node '{self.id}': 'workers' must be between 1 and {PARALLEL_MAX_BRANCHES}")
        elif t == "join":
            for field, allowed in (("merge", MERGE_MODES), ("wait_for", WAIT_FOR_MODES),
                                   ("on_branch_fail", BRANCH_FAIL_MODES)):
                value = getattr(self, field)
                if value is not None and value not in allowed:
                    raise ValueError(f"join node '{self.id}': {field} must be one of {list(allowed)}")


class Edge(BaseModel):
    """A transition. Kinds: unconditional (just `to`), `if` (condition ->),
    `else` (fallback), `while` (condition + mandatory counter). Reserved-word
    fields use JSON aliases (from/if/else/while)."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    if_: Optional[str] = Field(default=None, alias="if")
    else_: bool = Field(default=False, alias="else")
    while_: Optional[str] = Field(default=None, alias="while")
    # `while` edges MUST carry a counter id + ceiling — a loop without a bound is
    # rejected at construction.
    max: Optional[int] = None
    id: Optional[str] = None
    # An error edge is taken only when its node RAISED (after the node's `retries`); the message
    # is put in state `error` and the node's id in `error_node`.
    on_error: bool = False

    @model_validator(mode="after")
    def _check(self) -> "Edge":
        exclusive = [self.if_ is not None, self.else_, self.while_ is not None, self.on_error]
        if sum(1 for x in exclusive if x) > 1:
            raise ValueError(f"edge {self.from_}->{self.to} mixes if/else/while")
        if self.while_ is not None:
            if self.max is None or self.max <= 0:
                raise ValueError(f"while edge {self.from_}->{self.to} requires 'max' > 0")
            if not self.id:
                raise ValueError(f"while edge {self.from_}->{self.to} requires a counter 'id'")
        return self

    @property
    def kind(self) -> str:
        if self.on_error:
            return "error"
        if self.while_ is not None:
            return "while"
        if self.if_ is not None:
            return "if"
        if self.else_:
            return "else"
        return "uncond"

    def targets_exit(self) -> bool:
        return self.to.startswith("EXIT:")


class RoomRules(BaseModel):
    """What holds for every step of a room — set once instead of on each node. A restriction is
    ENFORCED where the step runs (and checked when the room is saved), not only said to the model."""
    model_config = ConfigDict(extra="forbid")
    no_tools: bool = False            # no step may call a tool; agent steps get none at all
    read_only: bool = False           # nothing that writes, deletes, runs commands or acts on a page
    deny_tools: list[str] = Field(default_factory=list)     # tool names that are switched off
    ignore: list[str] = Field(default_factory=list)         # mcp:... / skill:... rules for every step
    memory: str = "isolated"          # "shared": agent steps are told what earlier steps did
    memory_chars: int = Field(default=6000, ge=500, le=50000)
    sampling: dict = Field(default_factory=dict)            # default model parameters of the agent steps
    hint: Optional[str] = None        # a standing hint for every agent step

    @model_validator(mode="after")
    def _check_rules(self) -> "RoomRules":
        if self.memory not in MEMORY_MODES:
            raise ValueError(f"rules.memory must be one of {list(MEMORY_MODES)}")
        bad = [r for r in self.ignore if not is_ignore_rule(r)]
        if bad:
            raise ValueError(f"rules.ignore: {bad} is not understood — use mcp:<server>[.<tool>], skill:<name> or skill:<name>#<heading>")
        problems = sampling_problems(self.sampling) if self.sampling else []
        if problems:
            raise ValueError("rules.sampling: " + "; ".join(problems))
        return self

    def active(self) -> bool:
        return bool(self.no_tools or self.read_only or self.deny_tools or self.ignore or self.hint
                    or self.sampling or self.memory != "isolated")


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str
    description: str = ""
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)
    budget: Budget
    entry: Optional[str] = None  # defaults to the first node
    # Initial state fields (parameters with a default): {"test_command": "python -m pytest -q"}.
    # A run's `key=value` arguments override them; `{{key}}` in a node's text is replaced by the value.
    defaults: dict[str, Any] = Field(default_factory=dict)
    # Where this room's exits lead: {"planned": "debug_loop", "success": "END"}. A value is a room
    # name, "TRIAGE" or "END". Exits not listed follow the supervisor's usual rules.
    next: dict[str, str] = Field(default_factory=dict)
    # Words that make the router pick this room for a task (besides its name and description).
    tags: list[str] = Field(default_factory=list)
    # Text of the final report; `{{field}}` placeholders are filled from the final state.
    report: Optional[str] = None
    # Rules for the WHOLE room (and for every room it calls): what its steps may not do, how its
    # agents remember, the model parameters they work with.
    rules: Optional["RoomRules"] = None

    @model_validator(mode="after")
    def _default_entry(self) -> "Room":
        if self.entry is None and self.nodes:
            self.entry = self.nodes[0].id
        return self


class State(BaseModel):
    """Run state passed between nodes. `data` holds the working fields
    (findings, tests_passed, error_summary, ...); `loops` holds while-edge
    counters keyed by edge id (a reserved namespace)."""
    model_config = ConfigDict(extra="forbid")
    data: dict[str, Any] = Field(default_factory=dict)
    loops: dict[str, int] = Field(default_factory=dict)
    # Spent so far: {"tokens": estimated tokens read and written by agent nodes, "nodes": steps run}.
    usage: dict[str, float] = Field(default_factory=dict)
    # The rules in force for this run (the room's own, plus those of the rooms that called it).
    rules: dict = Field(default_factory=dict)
    # What earlier agent steps did (rules.memory = "shared"), and the hints given to the agents:
    # {"text": ..., "sticky": bool, "targets": [ids] | None}.
    memory: list[str] = Field(default_factory=list)
    hints: list[dict] = Field(default_factory=list)
