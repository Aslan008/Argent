"""
src/leeth/adapters/claude_code_shim.py — Zero-Shot Interceptor & Co-Processor Shim for Claude Code.

Enables seamless integration of Leeth into Claude Code, Cursor, OpenHands, and Aider:
1. Translates host environment tool schemas to Leeth Action Primitives:
   - Claude Code 'View' / 'FileRead'  <-> Leeth 'read_file'
   - Claude Code 'Grep' / 'Glob'      <-> Leeth 'grep_search' / 'search_files'
   - Claude Code 'Bash'               <-> Leeth 'run_command'
   - Claude Code 'Background'         <-> Leeth 'read_background_command'
2. Calibrated OOD Fail-Safe:
   - Evaluates certainty using Brier score and confidence gates.
   - If confidence >= 0.85, executes in 10-15 ms locally with 0 API tokens spent.
   - If confidence < 0.85 or complex reasoning is needed, immediately escalates to Claude 3.7 Sonnet.
3. Completely standalone without requiring pre-training specifically for Claude Code.
"""

import time
from typing import Any, Dict, Optional
from logger import get_logger

logger = get_logger("claude_code_shim")
CLAUDE_TO_LEETH_TOOL_MAP = {
    "View": "read_file",
    "FileRead": "read_file",
    "Grep": "grep_search",
    "Glob": "search_files",
    "Bash": "run_command",
    "Terminal": "run_command",
    "Background": "read_background_command",
}

LEETH_TO_CLAUDE_TOOL_MAP = {v: k for k, v in CLAUDE_TO_LEETH_TOOL_MAP.items()}


class ClaudeCodeShim:
    """Universal high-speed coprocessor bridge between Claude Code and Leeth."""

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        self._dispatcher = None

    @property
    def dispatcher(self):
        if self._dispatcher is None:
            from src.leeth.dispatcher import dispatcher
            self._dispatcher = dispatcher
        return self._dispatcher

    def intercept_user_query(
        self,
        query: str,
        active_tasks: Optional[list] = None,
        last_target_file: Optional[str] = None,
        last_tool_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates whether an incoming user prompt in Claude Code can be resolved locally.

        Returns:
            {
                "status": "fast_path",
                "tool": "View",
                "arguments": {"file_path": "..."},
                "confidence": 0.94,
                "latency_ms": 12.4
            }
            or
            {
                "status": "escalate_to_claude",
                "reason": "requires_deep_reasoning_or_unfamiliar"
            }
        """
        if not self.dispatcher.is_available:
            return {"status": "escalate_to_claude", "reason": "leeth_weights_unavailable"}

        context = {
            "active_tasks": active_tasks or [],
            "last_target": last_target_file,
            "last_tool": last_tool_used,
        }

        t0 = time.perf_counter()
        dispatch = self.dispatcher.try_fast_dispatch(query, context=context)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if dispatch is None or dispatch.confidence < self.confidence_threshold:
            return {
                "status": "escalate_to_claude",
                "reason": "below_confidence_threshold_or_reasoning",
                "latency_ms": elapsed_ms,
            }

        # Map Leeth tool back to Claude Code terminology
        claude_tool = LEETH_TO_CLAUDE_TOOL_MAP.get(dispatch.tool_name, dispatch.tool_name)
        claude_args = dict(dispatch.arguments)

        # Normalize argument keys for Claude Code
        if claude_tool == "View" and "file_path" in claude_args:
            claude_args = {"path": claude_args["file_path"]}
        elif claude_tool == "Bash" and "command" in claude_args:
            claude_args = {"command": claude_args["command"]}

        return {
            "status": "fast_path",
            "tool": claude_tool,
            "arguments": claude_args,
            "confidence": round(dispatch.confidence, 4),
            "latency_ms": round(elapsed_ms, 2),
            "leeth_internal_tool": dispatch.tool_name,
        }

    def get_tool_declaration(self) -> Dict[str, Any]:
        """Returns JSON schema declaration for exposing delegate_to_leeth to Claude Code."""
        return {
            "name": "delegate_to_leeth",
            "description": (
                "Fast local System One co-processor. Offloads routine inspections, grep searches, "
                "background job status, and simple commands in 10-15 ms with 0 API tokens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Routine subtask (e.g. 'read main.py', 'git status', 'check tests')",
                    }
                },
                "required": ["task"],
            },
        }


# Singleton export
claude_shim = ClaudeCodeShim()
