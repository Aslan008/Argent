"""Language Server Protocol integration for Argent.

Provides instant code diagnostics, precise symbol navigation, and type info
through multilspy — closing the gap with Claude Code's code intelligence.

multilspy is an optional dependency. When not installed, every method here
gracefully returns None and Argent works exactly as before.
"""

from src.lsp.manager import lsp_manager

__all__ = ["lsp_manager"]