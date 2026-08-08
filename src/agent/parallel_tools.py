"""Concurrent execution of independent read-only tool calls.

When a model emits several tool calls in one response, Argent executed them
strictly one-by-one — even pure reads. On a local model every extra second of
wall time matters, and reads have no side effects to serialize for. This
module pre-executes a batch concurrently, but ONLY under the safest possible
conditions: every call in the batch is a whitelisted read-only tool with
cleanly resolvable arguments. Anything else returns None and the ordinary
sequential path runs untouched.

Note: user plugin hooks (on_tool_call) still run in the main dispatch loop and
can veto what the MODEL sees; pre-executing a side-effect-free read before the
veto is harmless by construction of the whitelist.
"""

import inspect
from concurrent.futures import ThreadPoolExecutor

# Pure reads only: no writes, no approvals, no process side effects.
READ_ONLY_PARALLEL = {
    "read_file", "grep_search", "search_files", "list_directory", "get_file_outline",
}
_MAX_WORKERS = 4


def _resolve_args(func, arguments, tool_name=None):
    """Filter hallucinated params; None when a required argument is missing.

    Types are coerced here too: every tool on the whitelist takes an integer
    (start_line, max_results), and measured, grep_search with max_results="3"
    reports "No matches found" for a pattern with three matches.
    """
    if not isinstance(arguments, dict):
        return None
    sig = inspect.signature(func)
    valid = set(sig.parameters)
    filtered = {k: v for k, v in arguments.items() if k in valid}
    missing = [p.name for p in sig.parameters.values()
               if p.default is inspect.Parameter.empty and p.name not in filtered]
    if missing:
        return None
    if tool_name:
        from src.agent.arg_coercion import coerce_and_log
        filtered = coerce_and_log(tool_name, filtered)
    return filtered


def precompute_readonly_parallel(tool_calls, current_tools, dedup=None):
    """Run a batch of read-only calls concurrently.

    Returns {original_index: result} when there are 2+ calls and EVERY one is
    whitelisted, known, and cleanly resolvable — otherwise None, and the
    sequential dispatch handles everything (including error hints).
    dedup, when given, is the agent's read_file short-circuit: (filtered_args)
    -> note | None; a note becomes the result without touching the disk.
    """
    if not tool_calls or len(tool_calls) < 2:
        return None

    jobs = []
    for i, tc in enumerate(tool_calls):
        name = (tc.get("function") or {}).get("name")
        if name not in READ_ONLY_PARALLEL or name not in current_tools:
            return None
        func = current_tools[name]
        filtered = _resolve_args(func, tc["function"].get("arguments", {}), name)
        if filtered is None:
            return None
        jobs.append((i, name, func, filtered))

    results = {}
    pending = []
    for i, name, func, filtered in jobs:
        note = dedup(filtered) if (dedup is not None and name == "read_file") else None
        if note is not None:
            results[i] = note
        else:
            pending.append((i, name, func, filtered))

    if pending:
        def _run(job):
            i, name, func, filtered = job
            try:
                return i, func(**filtered)
            except Exception as e:  # same shape the sequential path produces
                return i, f"Error executing tool {name}: {e}"

        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(pending))) as pool:
            for i, res in pool.map(_run, pending):
                results[i] = res
    return results
