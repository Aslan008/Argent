"""
src/leeth/adapters — Host environment shims for Leeth co-processor.
"""
from src.leeth.adapters.claude_code_shim import ClaudeCodeShim, claude_shim

__all__ = ["ClaudeCodeShim", "claude_shim"]
