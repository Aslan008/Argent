"""Declarative "rooms and rails" agent engine.

Macro-level rails (a validated graph of rooms/nodes whose transitions the engine
decides) with micro-level freedom (agent nodes run their own bounded loop).
Rooms are DATA, never executable code. This package holds the schema (models),
the safe condition DSL, and the validator; the engine and library build on top.
"""
