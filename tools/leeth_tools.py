"""
tools/leeth_tools.py — Native delegation coprocessor tool for large LLMs.

Enables Claude, GPT, and reasoning models to offload batches of routine tool actions
(reading files, grep searches, directory listings, calculations, command executions, background checks)
to the local System One Leeth decision engine (10–25 ms, 0 external API tokens).
"""

import json
import time

from logger import get_logger

logger = get_logger("leeth_tools")


def delegate_to_leeth(task: str) -> str:
    """Delegates a routine subtask or chain of actions to the local Leeth coprocessor.

    Args:
        task: A natural language instruction for a routine action or chain
              (e.g. 'read config.py', 'find file runtime.py', 'grep for LeethModel in src/', 'check background tasks').

    Returns:
        A concise execution report with results of all completed steps, or an escalation note
        if the task requires architectural planning or complex multi-file reasoning.
    """
    if not task or not task.strip():
        return "Error: Empty task provided to delegate_to_leeth."

    task_clean = task.strip()
    t0 = time.perf_counter()

    try:
        from src.leeth.dispatcher import dispatcher as leeth_disp
        if not leeth_disp.is_available:
            return "Leeth runtime is currently unavailable (model weights not loaded). Proceed with standard tools."

        # Execute fast dispatch
        dispatch = leeth_disp.try_fast_dispatch(task_clean)
        if dispatch is None:
            return f"Leeth declined task '{task_clean[:80]}' (requires complex LLM reasoning or human confirmation). Please proceed using native tools."

        steps = dispatch.steps if dispatch.steps else [dispatch]
        results = []

        from tools import AVAILABLE_TOOLS

        for idx, step in enumerate(steps, 1):
            tool_name = step.tool_name
            tool_args = step.arguments
            
            # Ground-truth safety check
            if tool_name not in AVAILABLE_TOOLS:
                return f"Step {idx}: Tool '{tool_name}' is not registered in system tools."

            try:
                fn = AVAILABLE_TOOLS[tool_name]
                step_res = fn(**tool_args)
            except Exception as exc:
                step_res = f"Error executing {tool_name}: {exc}"

            res_str = str(step_res)
            # Smart truncation for large outputs to keep context tight
            if len(res_str) > 8000:
                res_str = res_str[:4000] + f"\n\n[... truncated {len(res_str) - 8000} chars ...]\n\n" + res_str[-4000:]

            results.append(f"Step {idx} [{tool_name}({json.dumps(tool_args, ensure_ascii=False)})]:\n{res_str}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        header = f"Leeth Fast-Path completed {len(steps)} action(s) in {elapsed_ms:.1f} ms:\n"
        return header + "\n\n".join(results)

    except Exception as e:
        logger.exception("delegate_to_leeth error: %s", e)
        return f"Error in delegate_to_leeth execution: {e}"
