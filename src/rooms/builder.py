"""Build a room from a description in words — the chat way to author one.

The person says what the room should do and what to do in which case; a model with NO
tools turns that into the room's JSON; the same validator that guards every room checks
it, and the errors (if any) go back to the model to repair, a few times at most. What
comes out is a proposal: the caller shows the flow and the person decides whether it is
saved. Nothing here runs anything.

House rules applied AFTER the model has spoken, so a careless draft cannot get past
them: no `spawn_room`, every tool node that changes files or the outside world asks
first (`confirm`), a name that belongs to a packaged room is refused, and a verdict that
matters is expected to come from a process (the prompt says how; the validator cannot
judge that, the person's review of the flow can).
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.rooms.diagram import diff_rooms
from src.rooms.models import DERIVED_VALUES, NOTIFY_CHANNELS, Room
from src.rooms.validator import KNOWN_ROLES, validate_room

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
MAX_NODES = 25

GUIDE = """\
You write ONE "room": a small declarative graph that an engine executes. Output ONLY a JSON object.

Room: {"room": "<snake_case name>", "description": "<one or two sentences: what it does and when to use it>",
 "tags": ["<words a person would use to ask for it, RU and EN>"],
 "budget": {"max_iterations": <steps>, "max_tokens": <tokens>, "max_seconds": <optional>},
 "entry": "<first node id>", "defaults": {"<param>": "<default>"},
 "next": {"<exit name>": "<another room>" | "END" | "TRIAGE"}, "report": "<text with {{field}}>",
 "nodes": [...], "edges": [...]}

Node types (every node has "id" and "type"):
- tool: {"tool": "<tool name>", "args": {...}, "input_from": "state.<field>"}. Add "sets": {"<field>": "ok"|"exit_code"|"body"|"tail"}
  on run_command to make the PROCESS decide a verdict (ok = exit code 0; tail = last lines of a failed output).
  "confirm": true asks the person first (use for commit, push, delete, deploy). "retries": N retries it if it raises.
- agent: a bounded worker. {"role": "Researcher"|"Reviewer"|"Coder"|"DocWriter", "tools": [<names>] (give a READ-ONLY list to
  Researcher/Reviewer), "budget": {"max_iterations": N, "max_tokens": N}, "writes": ["<fields it may set>"],
  "context": ["<state fields it reads>"], "instruction": "<what to do; mention what JSON to answer with>"}.
  It can set ONLY the fields in "writes". Never let an agent declare that tests passed - use a tool node with "sets".
- condition: {"condition": "<expr>"} (no side effect; use edges with "if" instead when possible).
- human_pause: {"prompt": "<question>", "into": "<field>"} stores the answer in <field> and sets <field>_approved (true for yes).
- set: {"values": {"<field>": <literal or "text with {{field}}">}}.
- notify: {"message": "<text>", "title": "<optional>", "channel": "console"|"desktop"|"telegram"}.
- room: {"room": "<existing room>", "inputs": {"<field>": "{{field}}"}, "outputs": ["<field>"], "into": "<field for its exit name>"}
  runs another room as a sub-routine; its exit name is in state.<into> (default <id>_exit).
- mcp: {"server": "<configured MCP server>", "tool": "<its tool>", "args": {...}, "sets": {...}} calls ONE tool of an MCP server
  ("mode": "list" lists the server's tools instead). It reaches outside: "confirm": true unless it only reads.
- skill: {"skill": "<installed skill>", "ignore": ["skill:<name>#<heading>"]} reads the skill's instructions into
  state.<id>.output (use "sets": {"<field>": "body"} to keep them under a name).
- plugin: {"command": "<plugin command>", "args": {...}} runs a plugin command ("mode": "hook" fires a hook event).
- leeth: the local fast model. {"mode": "delegate", "instruction": "<routine task>"} | {"mode": "filter", "input_from":
  "<field with a list>", "instruction": "<yes/no question about one item>"} (answer: JSON with index lists; take them with
  "sets": {"yes": "json:yes"}) | {"mode": "classify", "input_from": ..., "args": {"categories": [..]}} | {"mode": "scan",
  "instruction": "<question>", "args": {"include": "links"|"posts"}} judges the open browser page | {"mode": "fill"} | {"mode": "click"}.
- foreach: {"input_from": "<field with a list>", "item": "<field for the current item>"} one item per visit; add the edge
  {"from": "<id>", "if": "not state.<id>_done", "to": "<body>"} and {"from": "<id>", "else": true, "to": "<after>"}; the body
  ends with an edge back to the foreach node.
- wait: {"seconds": N} pauses.
- loop: {"mode": "for", "item": "i", "start": 0, "stop": 5, "step": 1} | {"mode": "times", "stop": 3} | {"mode": "while" |
  "do_while" | "until", "condition": "state.<field>", "max_items": N} | {"mode": "for_in", "input_from": "<dict or list field>",
  "item": "<key var>", "item_value": "<value var>"}. Like foreach: edges {"from": "<id>", "if": "not state.<id>_done", "to":
  "<body>"} and {"from": "<id>", "else": true, "to": "<after>"}; the body ends with an edge back to the loop. Always bounded
  (max_items, default 100).
- break / continue: {"loop": "<loop or foreach id>"}; must have an edge back to that loop (break ends it, continue is the next round).
- hint: {"message": "<text for the agents>", "mode": "next" | "all" | "clear", "targets": ["<agent id>"], "compose": false} -
  puts a note into what the next agent step (or every agent step until cleared) is told. Use it when a loop is noticed
  (condition on state.<id>_visits) or when the model cannot know something.
- forget: {"mode": "all" | "memory" | "hints", "clear": ["<field>"]} wipes what the agents of the room remember.
- assert: {"condition": "state.<field> ...", "message": "<why it failed>"} stops (or takes the error edge) when false.
- parallel: {"join": "<join id>", "workers": 2} - every arrow leaving it is a branch; branches run at the same time, each on
  its own copy of the state, and must all reach the join. Arrows out of a parallel node are unconditional, 2 to 8 of them,
  and no branch may ask a person (human_pause). Write different fields in different branches.
- join: {"wait_for": "all" | "any", "merge": "error" | "first" | "last" | "list", "on_branch_fail": "fail" | "continue"} -
  gathers the branches of its parallel node; `merge` settles two branches writing different values to one field.
  Sets <join>_arrived and <join>_failed.
- power: {"mode": "shutdown" | "restart" | "sleep" | "hibernate" | "lock" | "logout" | "cancel", "seconds": 60} acts on the
  computer. Shutdown / restart wait at least 10 seconds. It always asks ("confirm": true) in a room you author.
- notify may carry "compose": true: its "message" is then an instruction and the model writes the text from the run state.
- An agent node may carry "sampling": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1,
  "reasoning_effort": "low"|"medium"|"high"|"xhigh"|"none"}, "memory": "shared"|"isolated" and "hint": "<note for it>".
- The room may carry "rules": {"no_tools": bool, "read_only": bool, "deny_tools": [names], "ignore": [mcp:/skill: rules],
  "memory": "shared"|"isolated", "sampling": {...}, "hint": "<text for every agent>"} - they bind every step and every
  room it calls. A "set" value that starts with "=" is computed: {"i": "= state.i + 1"} (numbers and + - * / // % only).
Built-in fields for conditions: state.<node id>_visits (how many times that step has started) and state._steps.
Any node may carry "disabled": true (skipped) and "note": "<text>". "sets" on tool/mcp/plugin/leeth/skill nodes also takes
"lines", "count", "first_line", "int", "float", "bool", "json", "json:<a.b.0>", "regex:<pattern with a group>".
An agent node may also have "skills": ["<name>"] (their text is given to it), "mcp": ["<server>"] (the only servers it may
use) and "ignore": ["mcp:<server>.<tool*>", "skill:<name>#<heading>"] (what it must not see or use).

Edges: {"from": "<id>", "to": "<id>" | "EXIT:<name>"} plus at most one of:
  "if": "<expr>" | "else": true | "while": "<expr>", "max": N, "id": "<counter>" (a bounded loop) | "on_error": true
  (taken only when the node raised, after its retries; the message is in state.error).
The first matching "if"/"while" wins; "else" or a plain edge is the fallback. EVERY node must be able to reach an EXIT.

Expressions: state.<field>, ==, !=, <, <=, >, >=, in [..], not in [..], and, or, not, literals. Nothing else.
Text may contain {{field}} placeholders (state fields, {{task}} is the person's request, "<node id>.output" is a node's raw
output). Never put {{task}} or a field an agent wrote inside a shell command.
Prefer reusing existing rooms with a "room" node over copying their steps. Keep it small: only the nodes the job needs.
"""

EXAMPLE = """\
Example (a short room):
{"room": "check_then_commit", "description": "Run the tests; commit when they pass, ask first.",
 "tags": ["тесты", "коммит", "commit"], "budget": {"max_iterations": 8, "max_tokens": 10000},
 "entry": "tests", "defaults": {"test_command": "python -m pytest -q"},
 "next": {"done": "END", "red": "END"}, "report": "Tests passed: {{tests_passed}}",
 "nodes": [
  {"id": "tests", "type": "tool", "tool": "run_command", "args": {"command": "{{test_command}}"},
   "sets": {"tests_passed": "ok", "error_summary": "tail"}},
  {"id": "commit", "type": "tool", "tool": "run_command", "args": {"command": "git commit -am \\"checked\\""},
   "confirm": true, "idempotent": false}],
 "edges": [{"from": "tests", "if": "state.tests_passed", "to": "commit"},
           {"from": "tests", "else": true, "to": "EXIT:red"},
           {"from": "commit", "to": "EXIT:done"}]}
"""


@dataclass
class BuildResult:
    room: Optional[Room] = None
    errors: list = field(default_factory=list)       # empty + room set == a valid proposal
    notes: list = field(default_factory=list)        # what the house rules changed
    attempts: int = 0
    raw: str = ""
    changes: list = field(default_factory=list)      # for an edit: what differs from the old room

    @property
    def ok(self) -> bool:
        return self.room is not None and not self.errors


def extract_room_json(raw: str):
    """The room object from a model's answer: the OUTERMOST balanced object that has `nodes`
    (the answer may be wrapped in prose or code fences, and rooms nest objects)."""
    if not raw:
        return None
    from src.agent.parser import extract_balanced_json
    found = None
    consumed = 0
    for start, ch in enumerate(raw):
        if ch != "{" or start < consumed:
            continue                      # inside an object already taken: not an outermost one
        blob = extract_balanced_json(raw, start)
        if not blob:
            continue
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        consumed = start + len(blob)
        if isinstance(obj, dict) and isinstance(obj.get("nodes"), list):
            found = obj                   # the LAST one wins: a model may show a draft before the answer
    return found


#: Leeth modes that only look (the others act on the open page or run a routine task).
_LEETH_READ_ONLY = ("filter", "classify", "scan")


def harden(room: Room, mutating: Callable[[str], bool]) -> list:
    """Apply the house rules in place; returns what was changed, in words."""
    notes = []
    for n in room.nodes:
        if n.type == "tool" and n.tool and mutating(n.tool) and not n.confirm:
            n.confirm = True
            notes.append(f"шаг «{n.id}» ({n.tool}) меняет файлы или внешний мир — теперь спрашивает перед запуском")
        elif n.type == "mcp" and (n.mode or "call") == "call" and not n.confirm:
            n.confirm = True
            notes.append(f"шаг «{n.id}» вызывает MCP {n.server}.{n.tool} — внешний сервер, теперь спрашивает перед запуском")
        elif n.type == "plugin" and not n.confirm:
            n.confirm = True
            notes.append(f"шаг «{n.id}» запускает плагин — теперь спрашивает перед запуском")
        elif n.type == "power" and n.mode != "cancel" and not n.confirm:
            n.confirm = True
            notes.append(f"шаг «{n.id}» ({n.mode} ПК) — теперь спрашивает перед запуском; снимите вопрос в редакторе, если комната должна работать без вас")
        elif n.type == "leeth" and (n.mode or "delegate") not in _LEETH_READ_ONLY and not n.confirm:
            n.confirm = True
            notes.append(f"шаг «{n.id}» поручает Leeth действие ({n.mode or 'delegate'}) — теперь спрашивает перед запуском")
    return notes


class RoomBuilder:
    def __init__(self, run_agent: Callable, library, tools, max_repairs: int = 2):
        self._run_agent = run_agent         # (prompt, max_iterations=..., tools=[], role=...) -> text
        self.library = library
        self.tools = set(tools)
        self.max_repairs = max_repairs

    # -- prompts ----------------------------------------------------------------------------------
    def _context(self) -> str:
        rooms = []
        for name in self.library.names():
            if name == "triage":
                continue
            r = self.library.rooms[name]
            params = f" (params: {', '.join(r.defaults)})" if r.defaults else ""
            rooms.append(f"- {name}: {' '.join(r.description.split())[:110]}{params}")
        return (
            f"{GUIDE}\n{EXAMPLE}\n"
            f"Existing rooms you may call with a room node:\n" + "\n".join(rooms) + "\n"
            f"Tool names you may use in tool nodes and in an agent's tools list: {', '.join(sorted(self.tools))}\n"
            f"Roles: {', '.join(sorted(KNOWN_ROLES))}. Notify channels: {', '.join(NOTIFY_CHANNELS)}. "
            f"Derived values for sets: {', '.join(DERIVED_VALUES)}. At most {MAX_NODES} nodes.\n"
        )

    def _prompt_new(self, description: str, name: Optional[str]) -> str:
        want = f'Use the room name "{name}". ' if name else "Choose a short snake_case name. "
        return (f"{self._context()}\nWrite the room for this request. {want}Describe exactly what to do in each "
                f"case the person mentioned (what if it fails, what if the answer is no).\n\nREQUEST:\n{description}\n")

    def _prompt_edit(self, old: Room, instruction: str, name: str) -> str:
        current = old.model_dump_json(indent=1, by_alias=True, exclude_none=True)
        return (f"{self._context()}\nHere is an existing room. Change it as requested and output the WHOLE changed room "
                f'with the name "{name}". Keep everything the request does not touch.\n\nCURRENT ROOM:\n{current}\n\n'
                f"CHANGE REQUEST:\n{instruction}\n")

    @staticmethod
    def _prompt_repair(previous: str, problems: list) -> str:
        return ("Your room was rejected:\n- " + "\n- ".join(problems[:12]) +
                "\nFix exactly these problems and output the corrected WHOLE room as a single JSON object.\n"
                f"PREVIOUS ANSWER:\n{previous[:6000]}")

    # -- the loop ---------------------------------------------------------------------------------
    def _attempt(self, prompt: str, name: Optional[str], taken: set) -> tuple:
        raw = self._run_agent(prompt, max_iterations=2, tools=[], role="Researcher")
        raw = raw if isinstance(raw, str) else str(raw or "")
        data = extract_room_json(raw)
        if data is None:
            return raw, None, ["the answer contained no room JSON object (it must have a \"nodes\" list)"]
        if name:
            data["room"] = name
        rname = str(data.get("room") or "")
        if not NAME_RE.match(rname):
            return raw, None, [f"room name '{rname}' must be snake_case (a-z, 0-9, _), starting with a letter"]
        if rname in taken:
            return raw, None, [f"room name '{rname}' is taken by a packaged room; choose another name"]
        try:
            room = Room(**data)
        except Exception as e:
            return raw, None, [f"schema: {str(e)[:700]}"]
        problems = []
        if len(room.nodes) > MAX_NODES:
            problems.append(f"too many nodes ({len(room.nodes)} > {MAX_NODES}); make it smaller or split with room nodes")
        problems += validate_room(room, self.tools, allow_spawn=False)
        if not problems:
            problems = self.library.reference_errors({**self.library.rooms, room.room: room}).get(room.room, [])
        return raw, room, problems

    def _run(self, first_prompt: str, name: Optional[str], old: Optional[Room]) -> BuildResult:
        from src.rooms.library import RoomLibrary  # noqa: F401  (type only)
        taken = {n for n, src in self.library.sources.items() if src == "starter"}
        if old is not None and name == old.room:
            taken.discard(name)
        result = BuildResult()
        prompt = first_prompt
        for attempt in range(1 + self.max_repairs):
            result.attempts = attempt + 1
            raw, room, problems = self._attempt(prompt, name, taken)
            result.raw = raw
            if not problems:
                from src.agent.plan_mode import _mutates
                result.room = room
                result.errors = []
                result.notes = harden(room, _mutates)
                if old is not None:
                    result.changes = diff_rooms(old, room)
                return result
            result.room, result.errors = room, problems
            prompt = self._prompt_repair(raw, problems)
        return result

    def build(self, description: str, name: Optional[str] = None) -> BuildResult:
        return self._run(self._prompt_new(description, name), name, None)

    def edit(self, old: Room, instruction: str, new_name: Optional[str] = None) -> BuildResult:
        name = new_name or old.room
        return self._run(self._prompt_edit(old, instruction, name), name, old)


def production_builder(library) -> RoomBuilder:
    """The builder wired to the real model (a tool-less sub-agent) and the real tool registry."""
    from src.rooms.runner import argent_run_agent
    from src.rooms.session import available_tool_names
    return RoomBuilder(argent_run_agent, library, available_tool_names())
