"""Declarative "rooms and rails" agent engine.

Macro-level rails (a validated graph of rooms/nodes whose transitions the engine
decides) with micro-level freedom (agent nodes run their own bounded loop).
Rooms are DATA, never executable code. This package holds the schema (models),
the safe condition DSL, and the validator; the engine and library build on top.

Map of the package:
  models / condition / validator   the schema, the closed condition grammar, admission checks
  engine / journal                 deterministic execution with enforced budgets; crash-safe resume
  runner / template                what a node DOES (tools, bounded agents, `sets`, confirm, `{{field}}`)
  supervisor                       the meta-graph: room -> room, triage, `next` carried by the rooms
  library / router                 the rooms on disk, persistent stats + quarantine; room from words
  session                          production wiring: run_rooms, execute_room_run (progress+record+report)
  history / board                  run records and reports; the run on the person's screen
  cli / tool / hint                /rooms, the model's `rooms` tool, the one-line nudge
  spawn                            the model proposes a NEW room (off by default, validator + approval)
"""
