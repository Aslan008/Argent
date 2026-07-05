"""Pydantic schema for the rooms engine: State, Budget, Node, Edge, Room.

Rooms are declarative data validated against this schema. Structural, per-type
invariants that need no external context (an agent node must carry a budget, a
`while` edge must carry a counter, ...) are enforced here at construction time,
so a malformed room cannot even be built. Cross-cutting checks that need outside
knowledge (does the tool exist? is every node able to reach an EXIT?) live in the
RoomValidator, not here.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

NodeType = Literal["tool", "agent", "condition", "human_pause", "spawn_room"]


class Budget(BaseModel):
    """Hard limits for a room or an agent node. Every agent node and every room
    must carry one — an unbounded loop is the failure mode this whole design
    exists to prevent."""
    model_config = ConfigDict(extra="forbid")
    max_iterations: int = Field(gt=0)
    max_tokens: int = Field(gt=0)


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
    idempotent: bool = True

    # agent nodes
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

    @model_validator(mode="after")
    def _require_per_type_fields(self) -> "Node":
        if self.type == "tool" and not self.tool:
            raise ValueError(f"tool node '{self.id}' requires a 'tool'")
        if self.type == "agent" and self.budget is None:
            raise ValueError(f"agent node '{self.id}' requires a 'budget'")
        if self.type == "condition" and not self.condition:
            raise ValueError(f"condition node '{self.id}' requires a 'condition'")
        return self


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

    @model_validator(mode="after")
    def _check(self) -> "Edge":
        exclusive = [self.if_ is not None, self.else_, self.while_ is not None]
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
        if self.while_ is not None:
            return "while"
        if self.if_ is not None:
            return "if"
        if self.else_:
            return "else"
        return "uncond"

    def targets_exit(self) -> bool:
        return self.to.startswith("EXIT:")


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")
    room: str
    description: str = ""
    nodes: list[Node]
    edges: list[Edge] = Field(default_factory=list)
    budget: Budget
    entry: Optional[str] = None  # defaults to the first node

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
