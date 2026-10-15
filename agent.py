import os
import json
import inspect
import platform
import host_os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

from tools import TOOL_SCHEMAS, AVAILABLE_TOOLS, get_tool_schemas, get_available_tools
from config import (
    get_current_model,
    get_hooks_dir, get_autonomous_plugins_enabled,
    get_context_window, get_provider, get_mcp_servers
)
from providers import create_provider, ProviderError
from src.agent import governor
from logger import get_logger
from hook_manager import hook_manager
from prompt_compressor import (
    compress_system_prompt, compress_tool_result, drop_unavailable_tool_lines,
)
from tool_recovery import recover_tool_call
from memory_manager import memory
from config import get_model_size_category, is_moe_model

# Newly refactored modules
from src.agent.strategy import get_model_strategy 
from src.agent.parser import parse_raw_tool_call
from src.agent.trimmer import estimate_tokens
from src.agent.healing import detect_tool_failure, build_healing_hint, HEALING_TOOLS
from src.agent.constrained import build_step_schema, build_tool_catalog, StepStreamExtractor
from src.agent.loop_guard import LoopGuard, build_loop_note
from src.agent.semantic_loop_guard import (
    SemanticLoopGuard, build_semantic_loop_note, browser_eval_expression,
    is_browser_eval_call, is_page_changing_browser_action,
)
from src.agent.mcp_prompt import build_mcp_section
from src.agent.arg_coercion import coerce_and_log
from src.agent.context_limit import is_context_overflow, parse_context_limit

log = get_logger("agent")

# How many times a truncated generation may auto-continue within one turn
# before the turn is aborted (prevents the infinite regenerate-truncate loop
# on small context windows).
MAX_TRUNCATE_CONTINUES = 2

# How many times a truncated FILE WRITE may be salvaged + continued within one
# turn. Higher than MAX_TRUNCATE_CONTINUES because a genuinely large file needs
# many chunks, but still bounded so a stuck model can't append forever.
MAX_SALVAGE_CONTINUES = 12

# How many times to nudge a model that ended its turn with neither an answer
# nor a tool call (common with reasoning models that "think" then stop) before
# giving up — instead of silently dead-ending the turn.
MAX_NO_ACTION_CONTINUES = 2

# How often to say how long this is taking. Deliberately NOT a cap: measured
# over 1315 recorded turns, the long ones were all productive — 127 tool calls
# ending in a final answer, 106, 100 — and every turn that did not finish
# normally was already stopped by the loop guard or an error. A ceiling at any
# of those numbers would have cut finished work to prevent a runaway that the
# log does not contain. What the log does show is turns running 30-40 minutes,
# where the thing you actually lack is a way to notice and decide.
TURN_PROGRESS_EVERY = 25

# Sent (never stored) after the part of an answer the person already read, when the connection died mid-answer.
RESUME_NOTE = ("[Argent: соединение оборвалось на этом месте ответа. Продолжи ровно с места обрыва: "
               "не повторяй написанное, не извиняйся и не пересказывай начало.]")


def build_agents_memory(limit: int) -> List[str]:
    """Persistent-memory sections for the system prompt.

    Loads the GLOBAL file (~/.argent/AGENTS.md — standing preferences across
    every project) first, then the PROJECT file (.argent/AGENTS.md, or AGENTS.md
    in the project root). The two share a single `limit`-char budget (global
    first) so a small model's context isn't drowned. Returns a list of formatted
    prompt sections (empty when neither file exists)."""
    sources = []
    global_path = Path.home() / ".argent" / "AGENTS.md"
    if global_path.exists():
        sources.append((global_path, False))
    # Anchored to the project root, not the CWD: launching from a subfolder
    # (cd src) must not lose the project's instructions.
    from project_paths import project_root_or_cwd
    root = project_root_or_cwd()
    for p in (root / ".argent" / "AGENTS.md", root / "AGENTS.md"):
        if p.exists():
            sources.append((p, True))
            break

    sections, remaining = [], limit
    for p, is_project in sources:
        if remaining <= 0:
            break
        try:
            content = p.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        note = ""
        if len(content) > remaining:
            content = content[:remaining]
            note = f"\n\n[...truncated — keep {p} dense and under the tier limit.]"
        remaining -= len(content)
        if is_project:
            header = (
                f"## PROJECT INSTRUCTIONS (from {p})\n"
                f"This is your persistent project memory. Keep it accurate: when you learn "
                f"something durable about this codebase (architecture, conventions, commands), "
                f"update {p} with replace_in_file/write_file.\n\n"
            )
        else:
            header = (
                f"## GLOBAL INSTRUCTIONS (from {p})\n"
                f"Your user's standing preferences and instructions across ALL projects. "
                f"Honour them unless a project instruction overrides them.\n\n"
            )
        sections.append(f"{header}{content}{note}")
    return sections


_ACTIVE_SUBAGENTS: set = set()


class ArgentAgent:
    def build_system_prompt(self, available_tools=None) -> str:
        """Assemble the system prompt for this turn.

        `available_tools` is the set of tool names that will actually be sent
        with the request. When given, instructions naming tools outside it are
        dropped: the prompt is a description of the toolset, not an independent
        document, and letting the two disagree makes the model call names that
        do not exist. Passing None keeps every section (used by callers that
        only want the full text, e.g. diagnostics).
        """
        category = get_model_size_category(self.model_name)
        is_moe = category == "moe" or is_moe_model(self.model_name)
        if is_moe:
            category = "moe"
        is_small = category in ("tiny", "small") and not is_moe
        # The tier ladder only ever pointed down: weak models got a shorter
        # prompt, everyone else got the maximum. But most of this prompt is
        # guardrails written against failure modes of weak models, and a strong
        # one does better with them out of the way — it reads the codebase and
        # judges instead of following a rule that is wrong in the case at hand.
        # `/model category` overrides the tier when this guesses wrong.
        is_lean = category in ("large", "cloud")

        available = set(available_tools) if available_tools is not None else None
        prompt_parts = []

        def add(text: str, core: tuple = ()):
            """Append a section, minus any instruction the toolset can't honour.

            `core` names the tools a section exists FOR. Line filtering alone
            cannot tell that a section is pointless: strip the two `create_plugin`
            lines out of the plugin section and its remaining prose (hook paths,
            event names) survives as instructions for work the model can no
            longer do. Sections whose subject is the tool itself say so here.
            """
            if available is not None:
                if core and not set(core) <= available:
                    return
                text = drop_unavailable_tool_lines(text, available)
                if not text.strip():
                    return
            prompt_parts.append(text)

        prompt_parts.append(f"""# CRITICAL: LANGUAGE RULE
- You MUST respond and perform ALL internal reasoning (thinking process) in the EXACT SAME LANGUAGE the user used in their request. This is your highest priority rule.
- If the user writes in Russian, you THINK in Russian and REPLY in Russian.

# ROLE: Argent Coder
You are an autonomous AI software engineer. You design, build, and debug software with precision and speed on {host_os.system()}.""")

        # NOTE: volatile, per-turn facts (repository map, background processes)
        # deliberately do NOT live here — they change between turns, and the
        # system prompt is the PREFIX of every request: one changed byte here
        # invalidates the provider's whole KV/prefix cache and forces a full
        # re-prefill on local models. They are injected as an ephemeral tail
        # message instead (_build_ephemeral_context).

        if is_small:
            add(f"""## 1. OPERATIONAL PROTOCOL
- **Tool-First**: Invoke tools immediately via JSON when needed.
- **Ask Before Guessing**: Use `ask_user_questions` to clarify ambiguous requirements with structured options. When you must assume, say so explicitly in your reply.
- **Anti-Lazy**: Run commands and write/edit files yourself.
- **File Editing**: NEVER use write_file to overwrite existing large files (>150 lines). You MUST use replace_in_file (pass `changes` to apply several edits at once) to apply targeted patches. If you ALREADY have the file's current content in context (you just read it, or just proposed edits for it) and nothing changed it since, apply the edit DIRECTLY — do NOT read_file again first. Re-read only if the file may have changed.
- **Proactive Search**: Use `search_web` for technical info.
- **Persistence**: Do NOT stop after a single tool call. If the task requires multiple steps (read → edit → verify), execute ALL steps in a single response. Keep calling tools until the task is FULLY complete.
- **No Dummy Stop Phrases**: NEVER output placeholder phrases like 'No response requested', 'N/A', or 'Done' without explanation. If your tools finished, either proceed to the next verification step or report the final result to the user in Russian.
- **Strict Environment**: Use {host_os.system()}-native commands. On Windows the default shell is PowerShell — `&&`/`||` chains are auto-routed to cmd, so prefer `;` or separate calls. If a command fails, read the [DIAGNOSIS] line in the output.
- **File Paths**: Use forward slashes (/) in file paths — they work on Windows and avoid JSON backslash escaping. Example: `C:/Users/name/project/file.py`.""")
        else:
            add(f"""## 1. OPERATIONAL PROTOCOL
- **Tool-First**: YOU are the only one with tool access. Invoke tools immediately via JSON.
- **Ask Before Guessing**: If a user's request is ambiguous or lacks details, you MUST use the `ask_user_questions` tool to prompt them with structured options before writing code. Do NOT just ask questions in plain text chat.
- **Anti-Lazy**: Never ask the user to run code or copy-paste. Use `run_command` and `write_file` yourself.
- **File Editing**: NEVER use write_file to overwrite existing large files (>150 lines). You MUST use replace_in_file (pass `changes` to apply several edits at once) to apply targeted patches. If you ALREADY have the file's current content in context (you just read it, or just proposed edits for it) and nothing changed it since, apply the edit DIRECTLY with replace_in_file/write_file — do NOT call read_file again first. Re-read only if the file may have been modified since you last saw it.
- **Proactive Search**: Always use `search_web` for technical info, documentation, or current events.
- **Batch Reads**: independent lookups (read_file, grep_search, list_directory, get_file_outline, search_files) should be issued as MULTIPLE tool calls in ONE response — they execute in parallel, saving a full model round-trip each.
- **Persistence**: Do NOT stop after a single tool call. If the task requires multiple steps (read → edit → verify), execute ALL steps in a single response without waiting for user input. Keep calling tools until the task is FULLY complete.
- **No Dummy Stop Phrases**: NEVER output placeholder phrases like 'No response requested', 'N/A', or 'Done' without explanation. If your tools finished, either proceed to the next verification step or report the final result to the user in Russian.
- **Testing**: NEVER test logic or GUI apps by running `python app.py` via `run_command` (it will block). You MUST write and run `pytest` tests, or use `start_background_command`.
- **Self-Correction**: If a tool fails, analyze the error and fix it proactively. Do not apologize.
- **Strict Environment**: Use {host_os.system()}-native commands ONLY (NOT unix commands like 'ls' or 'grep'). On Windows the default shell is **PowerShell** (so `Select-Object`, `Get-ChildItem`, `$env:` work); `&&`/`||` chains are auto-routed to cmd, but prefer `;` or separate commands. If a command fails, READ the `[DIAGNOSIS]` line in its output before assuming the cause.
- **File Paths**: Use forward slashes (/) in file paths — they work on Windows and avoid JSON backslash escaping. Example: `C:/Users/name/project/file.py`.""")

        if category == "tiny" and self.provider == "ollama":
            add("""## RESPONSE FORMAT (STRICT JSON STEPS)
Every response is EXACTLY ONE JSON object, one of:
1. {"tool": {"name": "<tool_name>", "arguments": {...}}} — perform an action.
2. {"reply": "<final answer to the user>"} — ONLY when the task is fully complete or you must ask the user something.
Never mix plain text with JSON. Prefer "tool" steps until the task is done.
Example: {"tool": {"name": "read_file", "arguments": {"file_path": "main.py"}}}""")

        auto_plugins = get_autonomous_plugins_enabled()
        if auto_plugins or not is_small:
            hooks_dir = get_hooks_dir()
            add(f"""## 2. PLUGIN DEVELOPMENT
- **HOOKS_DIR**: `{hooks_dir}`
- **AUTONOMOUS_EXTENSION**: {'ENABLED' if auto_plugins else 'DISABLED'}
- **Standard Plugin Development**:
  1. Write `.py` plugin files into `{hooks_dir}` using `write_file` (they are automatically reloaded into Argent).
  2. Define a function `command_NAME(*args)` for slash commands (e.g., `command_hello` -> `/hello`).
  3. Use events: `on_startup()`, `pre_prompt(text)`, `on_tool_call(func_name, args)`, `post_response(text)`, `on_chat_saved(file_path)`.
  4. Always use `from ui import console` for output.""", core=("write_file",))

        add("""## 3. SKILLS SYSTEM
- You have access to instruction-based extensions stored in markdown files.
- Use `list_skills` to discover skills. Use `read_skill` to read and follow instructions.
- **Be proactive**: after you work out a non-trivial, repeatable workflow for THIS project (a build/deploy sequence, a multi-step fix pattern, project-specific conventions), persist it with `write_file` in skills/ so it can be reused. Don't wait to be asked.""",
            core=("list_skills", "read_skill"))

        if not is_small:
            add("""## 4. PLANNING MODE & ARTIFACTS
- **Blind Spot Pass first**: for ambiguous, large, or unfamiliar tasks, BEFORE planning: check the ground (`git status`, run the project's tests/build — was it ALREADY broken before you touched it?), then explore the affected code, then report FACTS / RISKS / ASSUMPTIONS and ask up to 3 structured questions via `ask_user_questions`. State assumptions explicitly ("Assuming X — скажи, если не так") instead of guessing silently. Full method: `read_skill("blind-spot-pass")`.
- **New features**: follow `read_skill("feature-workflow")` — a one-question-at-a-time interview (each answer shapes the next; architecture-changing ambiguities first), a spec/prototype approval gate, a deviation journal in .argent/artifacts/ while you build, then a review document + short comprehension quiz for the user.
- For complex changes, you MUST create an implementation plan before writing any code.
- Use `create_artifact("implementation_plan.md", content)` to present your plan to the user.
- Then, use `request_user_approval("I have created an implementation plan. Please review and approve.")` to PAUSE execution and wait for the user to confirm.
- NEVER start making massive changes without the user's explicit approval.
- Use `create_artifact("task.md", content)` to track your progress after approval.
- **Artifacts on disk**: Artifacts (plans, checklists, task logs) are stored in `.argent/artifacts/`. To inspect, verify, or resume an existing plan or task, read them with `read_file(".argent/artifacts/<filename>")` or inspect `.argent/artifacts/` with `list_directory`.""",
                core=("create_artifact", "request_user_approval"))

        add("""## 4b. TASK CHECKLIST
- For any job with three or more steps, call `todo_write` FIRST with the steps, then work through them. Keep exactly one step in_progress and mark each completed the moment it is really done (not in a batch at the end). The user watches this list.
- Skip it for a one-step request or a plain question.""", core=("todo_write",))

        add("""## 5. CODE STYLE
- Write clean, idiomatically correct code that matches the existing style, conventions, and dependencies of the project.""")

        if not is_small:
            add("""## 5b. REPLY STYLE
- Point at code as `path/to/file.py:42`, so the person can open the place.
- No emoji unless the person asks for them.
- Text right before a tool call ends with a period, not a colon.
- When you were wrong, say in one sentence what went wrong and fix it. No repeated apologies, and do not give up a correct answer under pressure.""")

        if is_small:
            add("""## 6. THINK & VERIFY PROTOCOL
- **Outcome Analysis**: Verify if tool results truly move you closer to the goal.
- **Proactive Verification**: After writing files or executing commands, verify they work as intended.
- **Self-Correction**: If stuck in a loop, stop, rethink, and explain to the user.""")
        elif is_lean:
            # "Analyze each result", "verify your work" and "self-correct" are
            # descriptions of what a capable model already does; what it cannot
            # know is that this environment lies about success, and what it must
            # be held to is not claiming completion it did not observe.
            add("""## 6. GROUND YOUR CLAIMS
- **False Success**: "Requirement already satisfied" or "Exit code: 0" does NOT always mean success here. If a tool reports success but the problem persists, try a different approach.
- **Prove It**: NEVER declare a task done without having RUN the relevant verification in the same turn (tests, a build, the command itself) and quoting its actual output. If verification is impossible from here (e.g. needs the Unity editor), say so explicitly instead of claiming success.""")
        else:
            add("""## 6. THINK & VERIFY PROTOCOL
- **Outcome Analysis**: After EACH tool call, analyze if the result truly moves you closer to the goal.
- **False Success**: "Requirement already satisfied" or "Exit code: 0" does NOT always mean success. If a tool reports success but the problem persists, try a different approach.
- **Proactive Verification**: After installing things or writing complex files, use `run_command` or `read_file` to VERIFY they work as intended.
- **Prove It**: NEVER declare a task done without having RUN the relevant verification in the same turn (tests, a build, the command itself) and quoting its actual output. If verification is impossible from here (e.g. needs the Unity editor), say so explicitly instead of claiming success.
- **Self-Correction**: If you are stuck in a loop, STOP. Rethink your strategy. Explain your new reasoning to the user.""")

        from config import get_thinking_rethink
        if get_thinking_rethink():
            add("""## DELIBERATE RETHINK PROTOCOL
When solving non-trivial problems, formulating plans, or taking actions:
1. First, explore ideas and draft your approach in `<think>...</think>`.
2. Before committing, perform an explicit adversarial check inside `<rethink>...</rethink>`:
   - Challenge assumptions: what unverified facts, gaps, or premature conclusions did you rely on?
   - Stress-test: what is the strongest counter-argument, failure mode, or overlooked alternative?
   - Self-correct: adapt and refine your conclusion if weaknesses were found.
3. Output your final response or tool call only after closing `</rethink>`.""")

        add("""## 7. COMMUNICATION
- **Direct Action**: Call tools directly when action is needed without unnecessary conversational preamble.
- **Length**: Keep replies concise — answer the question asked, without a recap of work the user just watched you do.""")

        add("""## 8. BROWSER AUTOMATION & WAITING PROTOCOL
- **Inspect & Target**: Open pages with `browser_open` or inspect them with `browser_inspect` (mode='dom_interactive' or 'screenshot') to see interactive elements. Target elements using index numbers or CSS selectors with `browser_action` (action='click', 'type', 'select', 'scroll', 'press_key', 'upload_file', 'wait_for').
- **Awaiting Web Events (Uploads, Dynamic Content, Buttons)**:
  - When waiting for dynamic pages, file/video upload completion, async rendering, or button activation (e.g. "Save" becoming enabled), use `browser_action(action="wait_for", text="Загрузка завершена")` or `browser_action(action="wait_for", selector="button.save")`.
  - NEVER run repeated `browser_action(action="eval_js")` busy-polling loops.
  - NEVER use `wait_heartbeat` for browser tasks or web pages: `wait_heartbeat` only monitors local OS files and background processes and CANNOT see browser DOM state!
- **No Reverse-Engineering of SPA Internals**:
  - Do NOT try to probe private React/Redux internals, fibers (`__reactFiber$*`), closures, or minified store variables.
  - Always interact with pages through standard UI controls (`browser_action`, `browser_inspect`, `browser_open`).
- **Many items, many fields**: do not click or type one by one. For "go through these posts / bookmarks and pick or remove those that ..." call `leeth_scan_page` once (include='posts' for feeds, `scroll_pages` for endless ones, and 2-3 `positive_examples` / `negative_examples` so the yes/no cut fits the question), then `leeth_click_many` on what it returned; for a plain list of texts use `leeth_map`. For a form with several fields call `leeth_fill_form(instruction)` once. Clicks that delete, pay or confirm need the user's approval first (`confirm=True`).""",
            core=("browser_open", "browser_action"))



        # Persistent memory injected every turn: the global ~/.argent/AGENTS.md
        # (standing preferences across all projects) then the project's
        # .argent/AGENTS.md (or AGENTS.md), sharing a tier-scaled budget so a
        # small model isn't drowned.
        AGENTS_MD_LIMIT = {"tiny": 2000, "small": 4000}.get(category, 12000)
        prompt_parts.extend(build_agents_memory(AGENTS_MD_LIMIT))

        # The long-term notes (src/memory): only the index goes in, a note is read on demand. Rewritten with the
        # prompt, so a note saved this turn is in the index from the next one.
        try:
            from src.memory import hub
            note = hub.prompt_section({"tiny": 900, "small": 1800}.get(category, 3500))
            if note:
                add(note, core=("memory",))
        except Exception as e:
            log.debug("memory index not added: %s", e)

        # Skipped wholesale rather than line-filtered when the model has no
        # `call_mcp_tool`: the per-server subsections would survive the line
        # filter as headings with an endpoint and no way to act on it. Checking
        # first also avoids the blocking tools/list round-trip below.
        try:
            mcp_servers = get_mcp_servers() if (available is None or "call_mcp_tool" in available) else []
            if mcp_servers:
                add(build_mcp_section(mcp_servers))
        except Exception:
            pass

        # Tell the model that indexed documentation exists and to consult it,
        # so a weak model actually retrieves "at the right moment" instead of
        # answering library/API questions from memory.
        try:
            from rag_engine import is_rag_enabled
            from config import get_external_kbs
            if is_rag_enabled():
                kb_names = [kb.get("name", kb.get("id")) for kb in get_external_kbs() if kb.get("enabled", True)]
                if kb_names:
                    add(
                        "## KNOWLEDGE BASES\n"
                        f"Indexed documentation is available: {', '.join(kb_names)}. "
                        "For ANY question about these libraries/APIs, call `semantic_search` FIRST and base your "
                        "answer on the returned snippets — do NOT answer API/method questions from memory, it leads "
                        "to hallucinated signatures.",
                        core=("semantic_search",)
                    )
        except Exception:
            pass

        # Current date — stable within a day, so it lives in the system prompt
        # (cache-friendly: invalidates once at midnight, not per-turn). Gives every
        # tier, including tiny models, day/month/year/weekday awareness for ~6 tokens.
        _now = datetime.now()
        add(f"## CURRENT DATE\nToday: {_now.strftime('%Y-%m-%d')} ({_now.strftime('%a')})")

        full_prompt = "\n\n".join(prompt_parts)
        # Pass the tier resolved above: building for one tier and compressing
        # for another would strip sections the prompt was written to include.
        compressed = compress_system_prompt(full_prompt, self.model_name, category)
        if available is not None:
            compressed = drop_unavailable_tool_lines(compressed, available)
        return compressed

    def effective_tool_names(self, allowed_tools=None, mode=None) -> set:
        """The tool names this turn's request will actually carry.

        Mirrors the filtering applied when the request is assembled (config
        denylist, mode allowlist, tier slimming) so the system prompt can be
        built against the same set. Stable within a mode, so deriving the
        prompt from it does not churn the prefix cache."""
        from tool_profiles import slim_tools_for_category

        names = [t["function"]["name"]
                 for t in get_tool_schemas(include_hidden=(allowed_tools is not None))]
        if allowed_tools is not None:
            # A hand-picked list is exempt from tier slimming — see the same
            # decision where the request is assembled.
            return set(n for n in names if n in allowed_tools)
        category = get_model_size_category(self.model_name)
        if category == "moe" or is_moe_model(self.model_name):
            return set(names)
        return set(slim_tools_for_category(names, category, mode=mode))

    def _log_turn_end(self, reason: str, detail: str = "") -> None:
        """Record WHY a turn stopped.

        A turn can end nine different ways — a final answer, the loop guard, a
        provider error, three flavours of truncation, exhausted nudges — and
        none of them used to leave a line on disk. When a run visibly cut short,
        agent.log held nothing at all for that window, so the only honest answer
        to "why did it stop?" was "unknowable". One line each fixes that for
        every future occurrence.
        """
        if getattr(self, "_turn_end_logged", False):
            return          # first reason wins; the wrapper's finally is a backstop
        self._turn_end_logged = True
        started = getattr(self, "_turn_started", None)
        elapsed = (time.monotonic() - started) if started else 0.0
        log.info("turn ended: %s after %d tool call(s) in %.1fs%s",
                 reason, getattr(self, "_turn_tool_calls", 0), elapsed,
                 f" — {detail[:300]}" if detail else "")
        # Phase 4: close the turn's trajectory with the end reason.
        try:
            from src.leeth import trajectory as leeth_trajectory
            leeth_trajectory.end_trajectory(reason[:60])
        except Exception:
            pass

    def _refresh_system_prompt(self, allowed_tools=None) -> bool:
        """Rebuild the system prompt only if it actually changed.

        The system prompt is the prefix of every request; replacing it with a
        byte-identical copy is harmless, but replacing it with a CHANGED one
        invalidates the provider's whole prefix cache. Stable inputs (tier,
        AGENTS.md, MCP list) rarely change, so most turns this is a no-op and
        the KV cache stays warm. Returns True when the prompt was replaced."""
        new_prompt = self.build_system_prompt(self.effective_tool_names(allowed_tools))
        if self.messages and self.messages[0].get("role") == "system":
            if self.messages[0].get("content") == new_prompt:
                return False
            self.messages[0]["content"] = new_prompt
        elif self.messages:
            self.messages.insert(0, {"role": "system", "content": new_prompt})
        else:
            self.messages = [{"role": "system", "content": new_prompt}]
        return True

    def _build_ephemeral_context(self) -> str | None:
        """Per-turn volatile facts, injected as a TAIL system message (never
        stored in history): the repository map and live background processes.
        Tail position keeps the request prefix stable for the KV cache — and is
        where small models attend best anyway."""
        parts = []
        # Precise time for large/cloud models only — small models get the date
        # from the system prompt and don't need per-turn time polluting their
        # limited context. Time changes every second, so it lives in the
        # ephemeral tail (not the cached system prompt prefix).
        if get_model_size_category(self.model_name) in ("large", "cloud"):
            _now = datetime.now()
            _offset = _now.astimezone().strftime('%z')  # e.g. '+0500'
            _tz = f"UTC{_offset[:3]}:{_offset[3:]}" if _offset else 'local'
            parts.append(f"## CURRENT TIME\n{_now.strftime('%H:%M:%S')} {_tz}")

        try:
            cwd = Path.cwd()
            dirs, files = [], []
            for item in cwd.iterdir():
                if item.name in ('.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode'):
                    continue
                (dirs if item.is_dir() else files).append(item.name + ("/" if item.is_dir() else ""))
            dirs.sort()
            files.sort()
            all_items = dirs + files
            if not all_items:
                map_str = "(Empty directory)"
            elif len(all_items) > 30:
                map_str = "\n".join(f"- {x}" for x in all_items[:30])
                map_str += f"\n- ... and {len(all_items) - 30} more items. Use list_directory to see all."
            else:
                map_str = "\n".join(f"- {x}" for x in all_items)
            parts.append(
                f"## REPOSITORY MAP (Current Directory: {cwd.as_posix()})\n"
                f"{map_str}\n"
                f"(Note: Ambient workspace file structure for reference if needed. Do NOT summarize or comment on this repository map unless the user explicitly asks about it.)"
            )
        except Exception:
            pass

        # Live background processes: the model must know they exist (and can
        # recover PIDs) even after summarization drops the original results.
        try:
            from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
            with ACTIVE_PROCESSES_LOCK:
                bg_count = len(ACTIVE_PROCESSES)
            if bg_count:
                parts.append(
                    f"## BACKGROUND PROCESSES\n"
                    f"You have {bg_count} background process(es) running this session. "
                    f"Call `list_background_commands` to see their PIDs, status and commands "
                    f"before reading output or stopping them — do NOT guess a PID."
                )
        except Exception:
            pass

        try:
            category = get_model_size_category(self.model_name)
            is_moe = category == "moe" or is_moe_model(self.model_name)
            if is_moe:
                parts.append(
                    "## TASK PERSISTENCE DIRECTIVE\n"
                    "- Complete all intermediate steps and verify the result before stopping.\n"
                    "- Never emit machine stop tokens like 'No response requested' or 'N/A'. If human action is required, clearly explain it in Russian."
                )
        except Exception:
            pass

        try:
            from src.agent import plan_mode
            from src.agent.todos import TODOS, ephemeral_block
            if plan_mode.PLAN.active:
                parts.append(plan_mode.plan_instructions())
            todo_block = ephemeral_block(TODOS.scope())
            if todo_block:
                parts.append(todo_block)
            from src.agent.workflow import runtime as ultracode_runtime
            ultracode_note = ultracode_runtime.instructions()
            if ultracode_note:
                parts.append(ultracode_note)
            if not hasattr(self, "tools_override"):      # a sub-agent cannot start a room
                from src.rooms.hint import room_hint
                room_note = room_hint(self.messages)
                if room_note:
                    parts.append(room_note)
        except Exception:
            pass

        return "\n\n".join(parts) if parts else None

    def _pre_tool_block(self, func_name: str, args: dict) -> str | None:
        """Why this call must not run, or None. The single place that refuses a
        call before it starts: plan mode today, the person's own rules next."""
        from src.agent import plan_mode
        refused = plan_mode.violation(func_name, args)
        if refused:
            return refused
        refused = self._permission_block(func_name, args)
        if refused:
            return refused
        try:
            from src.agent.shell_hooks import SHELL_HOOKS
            return SHELL_HOOKS.pre_tool(func_name, args)
        except Exception as e:          # a broken hooks file must not stop the agent
            log.warning("PreToolUse hooks failed: %s", e)
            return None

    @staticmethod
    def _permission_block(func_name: str, args: dict) -> str | None:
        """The person's standing rules (permissions.json): a deny stops the call, an ask
        puts the question to them. Shell commands are asked about by the approval gate."""
        try:
            from src.agent.permissions import PERMISSIONS, COMMAND_TOOLS
            denied = PERMISSIONS.denied(func_name, args)
            if denied:
                return denied
            rule = PERMISSIONS.asked(func_name, args)
            if rule is not None and func_name not in COMMAND_TOOLS:
                from approval import request_approval
                if not request_approval(f"{func_name}: {args.get('file_path') or args.get('path') or args} "
                                        f"(правило ask: {rule.raw})", destructive=False):
                    return (f"The user declined '{func_name}' (their rule {rule.raw} asks first). "
                            "It was NOT done; ask what they want instead.")
        except Exception as e:
            log.warning("permission rules failed: %s", e)
        return None

    def _post_tool_hooks(self, func_name: str, args: dict, result):
        """The configured PostToolUse hooks' output, appended to what the model reads."""
        try:
            from src.agent.shell_hooks import SHELL_HOOKS
            extra = SHELL_HOOKS.post_tool(func_name, args, result)
        except Exception as e:
            log.warning("PostToolUse hooks failed: %s", e)
            return result
        return f"{result}\n\n{extra}" if extra else result

    def __init__(self, max_history_messages: int = None):
        self.model_name = get_current_model()
        self.provider = get_provider()
        self.refresh_tier()

        if max_history_messages is not None:
            self.max_history_messages = max_history_messages

        self.loop_guard = LoopGuard()
        self.semantic_loop_guard = SemanticLoopGuard()
        # The saved session this conversation IS, once it has one. Autosave
        # writes back over it instead of filing another snapshot of the same
        # chat — six copies of one conversation used to compete for the fifty
        # stored slots.
        self.session_id: str | None = None
        # Per-file signature of the last whole-file read, so a re-read of an
        # UNCHANGED file (whose content is still in history) is answered with a
        # compact pointer instead of a full re-dump. Kills the wasteful
        # "read again before editing" round-trip. {abs_path: {sig, stored}}.
        self._read_cache: Dict[str, Dict[str, Any]] = {}
        # Time machine: one auto-checkpoint per turn, made lazily before the
        # first file edit (see _maybe_turn_checkpoint).
        self._turn_checkpoint_done = False
        self._turn_label = ""
        self._current_turn_msg_index = 0
        self._checkpoint_turn_indices: Dict[str, int] = {}
        # "Back to before this message": what each of the last turns started from
        # (see _begin_turn_record / rewind_last_turn).
        self._turn_records: List[Dict[str, Any]] = []
        self._stop_requested = False
        self._turn_tool_calls = 0
        self._turn_local_agent_calls = 0
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]

    def request_stop(self) -> None:
        """Request the active turn/tool execution to abort gracefully (like Ctrl+C)."""
        self._stop_requested = True
        try:
            from src.agent.service_llm import request_cancel
            request_cancel()
        except Exception:
            pass
        try:
            from src.agent.workflow import control as ultracode_control
            ultracode_control.request_stop()      # a running workflow stops its workers and saves its state
        except Exception:
            pass
        for sub in list(_ACTIVE_SUBAGENTS):
            try:
                sub.request_stop()
            except Exception:
                pass

    def reset_stop(self) -> None:
        self._stop_requested = False
        for sub in list(_ACTIVE_SUBAGENTS):
            try:
                sub.reset_stop()
            except Exception:
                pass

    def refresh_tier(self):
        """Recompute every tier-dependent knob after a model/provider change.

        Switching tiny -> cloud (or back) mid-session must fully swap the
        behaviour profile: strategy, history budget, context size and the
        constrained-decoding availability flag. Nothing from the previous
        tier may leak into the new one.
        """
        self.strategy = get_model_strategy(self.model_name, self.provider)
        category = get_model_size_category(self.model_name)
        self.max_history_messages = self.strategy.get_max_history_messages(category)
        self.max_context_tokens = get_context_window()
        # A new model/provider may support what the previous one didn't.
        self._constrained_unsupported = False
        # Set when a provider rejects native tool calls (e.g. an OpenRouter
        # free model with no tool-use endpoint) — then we fall back to prompted
        # tool-calling (in-context catalog + raw-JSON parsing) for this model.
        self._native_tools_unsupported = False

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.refresh_tier()

    def set_provider(self, provider_name: str):
        self.provider = provider_name
        self.model_name = get_current_model(provider_name)
        self.refresh_tier()

    def _read_dedup_note(self, filtered_args: dict) -> str | None:
        """Short-circuit a re-read of an UNCHANGED whole file still in history.

        Returns a compact pointer (so the model edits directly instead of
        re-dumping the file) only when it is provably safe: the file is a whole
        file (no line range), its on-disk mtime+size match the last read, AND
        that earlier content is still present in the conversation (not trimmed).
        Otherwise returns None and a real read happens.
        """
        if filtered_args.get("start_line") is not None or filtered_args.get("end_line") is not None:
            return None
        fp = filtered_args.get("file_path")
        if not fp:
            return None
        try:
            from tools._helpers import _resolve_path
            path = _resolve_path(fp)
            if not path.is_file():
                return None
            st = path.stat()
            sig = [st.st_mtime_ns, st.st_size]
            cached = self._read_cache.get(str(path))
            if not cached or cached.get("sig") != sig:
                return None
            stored = cached.get("stored")
            if not stored or not any(stored in (m.get("content") or "") for m in self.messages):
                return None
            return (
                f"[read_file] '{path.name}' is UNCHANGED since you read it earlier "
                f"(same size and modification time). Its content is already in the "
                f"conversation above — do NOT re-read it. Apply your edit directly "
                f"with replace_in_file or write_file."
            )
        except Exception:
            return None

    def _record_read(self, filtered_args: dict, stored_result: str) -> None:
        """Remember the on-disk signature + exact stored text of a real whole-file read."""
        if filtered_args.get("start_line") is not None or filtered_args.get("end_line") is not None:
            return
        fp = filtered_args.get("file_path")
        if not fp or not isinstance(stored_result, str):
            return
        try:
            from tools._helpers import _resolve_path
            path = _resolve_path(fp)
            if not path.is_file():
                return
            st = path.stat()
            self._read_cache[str(path)] = {"sig": [st.st_mtime_ns, st.st_size], "stored": stored_result}
        except Exception:
            pass

    _EDIT_TOOLS = {"replace_in_file", "write_file"}
    # Everything that changes the tree through Argent's own tools. The turn's safety
    # checkpoint is taken before the first of these; delete/move/copy/mkdir are not
    # "edits" (no read-cache bookkeeping applies to them) but a rewind must cover them.
    _TREE_MUTATING_TOOLS = _EDIT_TOOLS | {"delete_file", "move_file", "copy_file", "create_directory"}
    MAX_TURN_RECORDS = 12

    def _external_change_note(self, func_name: str, filtered_args: dict) -> str | None:
        """Warn when an edit targets a file that changed on disk AFTER the last
        read — i.e. it was edited outside this session (the user, Unity, a
        formatter). The model's mental copy is stale, and a fuzzy match could
        land the edit in the wrong place. Own edits don't trigger this: a
        successful edit drops the file's read-cache entry."""
        if func_name not in self._EDIT_TOOLS:
            return None
        fp = filtered_args.get("file_path")
        if not fp:
            return None
        try:
            from tools._helpers import _resolve_path
            path = _resolve_path(fp)
            cached = self._read_cache.get(str(path))
            if not cached or not path.is_file():
                return None
            st = path.stat()
            if cached.get("sig") == [st.st_mtime_ns, st.st_size]:
                return None
            return (
                "\n\n[Note]: this file changed on disk AFTER you last read it (edited outside "
                "this session — e.g. by the user, Unity, or a formatter). The content you saw "
                "may be stale; re-read the file to confirm the edit landed where intended."
            )
        except Exception:
            return None

    def _maybe_turn_checkpoint(self, func_name: str) -> dict | None:
        """Time machine: before the FIRST file edit of a turn, snapshot the
        pre-turn tree as an Argent Checkpoint so /rewind (or git_rollback) can
        return to any turn. One attempt per turn; a failed snapshot never
        blocks the edit. Returns a 'checkpoint' chunk to yield, or None."""
        if func_name not in self._TREE_MUTATING_TOOLS or self._turn_checkpoint_done:
            return None
        self._turn_checkpoint_done = True
        from src.agent.checkpoints import auto_checkpoint_detailed
        label = self._turn_label or func_name
        result = auto_checkpoint_detailed(label)
        records = getattr(self, "_turn_records", None)
        rec = records[-1] if records else None
        if rec is not None:
            rec.update(edited=True, checkpoint=result.sha, clean_start=result.clean, skip_reason=result.reason)
        if result.sha:
            self._checkpoint_turn_indices[result.sha] = self._current_turn_msg_index
            return {"type": "checkpoint", "sha": result.sha, "label": label}
        if result.warn:
            # Editing without a safety net is exactly the thing the user must
            # not discover only when /rewind turns up empty.
            return {"type": "error",
                    "content": f"\n[System: {result.reason}]"}
        return None

    # -- "back to before this message" ------------------------------------------------

    def _begin_turn_record(self, user_text: str) -> None:
        """Remember what this turn starts from: the whole conversation and the git HEAD.

        The conversation is deep-copied BEFORE the turn touches it (the system prompt
        refresh, the user message, history trimming all happen afterwards), so going back
        restores it exactly, not "roughly the same after truncation". The tree is covered
        by the safety checkpoint taken before the first edit (see _maybe_turn_checkpoint)."""
        import copy
        rec: Dict[str, Any] = {"text": user_text, "messages": None, "head": None, "edited": False,
                               "checkpoint": None, "clean_start": False, "skip_reason": None}
        try:
            rec["messages"] = copy.deepcopy(self.messages)
        except Exception as e:
            log.warning("could not snapshot the conversation for rewind: %s", e)
        try:
            from src.agent.checkpoints import current_head
            rec["head"] = current_head()
        except Exception:
            pass
        records = self.__dict__.setdefault("_turn_records", [])   # also for agents built without __init__ (test doubles)
        records.append(rec)
        del records[:-self.MAX_TURN_RECORDS]

    def reset_turn_records(self) -> None:
        """The conversation was replaced (cleared, loaded, compacted, rewound by other means):
        what the recorded turns started from no longer describes it."""
        getattr(self, "_turn_records", []).clear()

    def plan_last_turn_rewind(self) -> Dict[str, Any]:
        """What "back to before the last message" would do, without doing it.

        {"ok", "text", "files": "restore" | "none" | "unavailable", "detail"} — or
        {"ok": False, "error"}. The caller shows it so the person decides knowing the cost."""
        if not getattr(self, "_turn_records", None):
            return {"ok": False, "error": "Нет сообщения, к которому можно вернуться (оно было до этого сеанса или история сброшена)."}
        rec = self._turn_records[-1]
        if rec.get("messages") is None:
            return {"ok": False, "error": "Диалог этого хода не удалось сохранить — вернуться нельзя."}
        text = rec.get("text") or ""
        if not rec.get("edited"):
            return {"ok": True, "text": text, "files": "none",
                    "detail": "Этот ход не менял файлы через инструменты Argent — вернётся только диалог."}
        if rec.get("checkpoint"):
            return {"ok": True, "text": text, "files": "restore",
                    "detail": f"Файлы вернутся к состоянию до сообщения (чекпоинт {rec['checkpoint']}); "
                              "правки хода сохранятся в git stash."}
        if rec.get("clean_start") and rec.get("head"):
            return {"ok": True, "text": text, "files": "restore",
                    "detail": "Файлы вернутся к последнему коммиту (до сообщения дерево было чистым); "
                              "правки хода сохранятся в git stash."}
        return {"ok": True, "text": text, "files": "unavailable",
                "detail": "Файлы вернуть нельзя: " + (rec.get("skip_reason") or "для этого хода нет git-чекпоинта")
                          + ". Вернётся только диалог."}

    def rewind_last_turn(self) -> Dict[str, Any]:
        """Return the conversation AND the files to the moment before the last message.

        All-or-nothing on the part that can be refused: if git will not rewind (a real
        commit was made since, a merge is half done) nothing is touched. When there is
        simply no checkpoint to return the files to, the conversation still goes back and
        the result says so. Consent is the caller's job."""
        plan = self.plan_last_turn_rewind()
        if not plan.get("ok"):
            return plan
        rec = self._turn_records[-1]
        files_note = plan["detail"]
        if plan["files"] == "restore":
            from src.agent.checkpoints import CheckpointError, rewind_to, rewind_to_base
            try:
                files_note = rewind_to(rec["checkpoint"]) if rec.get("checkpoint") else rewind_to_base(rec["head"])
            except CheckpointError as e:
                return {"ok": False, "error": f"Файлы не удалось вернуть, ничего не изменено: {e}"}
        self.messages = rec["messages"]
        self._turn_records.pop()
        self._read_cache.clear()
        self._turn_checkpoint_done = False
        self._checkpoint_turn_indices = {sha: i for sha, i in self._checkpoint_turn_indices.items()
                                         if i < len(self.messages)}
        return {"ok": True, "text": plan["text"], "files": plan["files"], "detail": files_note}

    def on_rewind(self, target_sha: str, label: str = "") -> None:
        """Synchronize in-memory agent state after a Git tree rewind."""
        self._read_cache.clear()
        self._turn_checkpoint_done = False
        getattr(self, "_turn_records", []).clear()

        target_idx = None
        target_lower = (target_sha or "").strip().lower()
        for sha, idx in self._checkpoint_turn_indices.items():
            if sha.lower().startswith(target_lower) or target_lower.startswith(sha.lower()):
                target_idx = idx
                break

        lbl_text = f" — '{label}'" if label else ""
        if target_idx is not None and 0 < target_idx <= len(self.messages):
            self.messages = self.messages[:target_idx]
            self.messages.append({
                "role": "system",
                "content": (
                    f"[System: Дерево проекта откачено к чекпоинту {target_sha}{lbl_text}. "
                    f"Последующие правки и сообщения отменены. Актуальное состояние файлов на диске восстановлено.]"
                ),
            })
        else:
            self.messages.append({
                "role": "system",
                "content": (
                    f"[System: Дерево проекта откачено к чекпоинту {target_sha}{lbl_text}. "
                    f"Все последующие правки файлов отменены. Обязательно перепроверяйте файлы на диске (read_file), "
                    f"не полагайтесь на предыдущие ответы.]"
                ),
            })

    def _drop_read_cache(self, filtered_args: dict) -> None:
        """Forget a file's read signature after we edited it ourselves, so the
        next edit isn't misreported as an external change."""
        if not isinstance(filtered_args, dict):
            return
        fps = []
        fp = filtered_args.get("file_path") or filtered_args.get("path")
        if fp:
            fps.append(fp)
        changes_json = filtered_args.get("changes_json")
        if changes_json:
            raw_changes = changes_json
            if isinstance(raw_changes, str):
                import json
                try:
                    raw_changes = json.loads(raw_changes)
                except Exception:
                    raw_changes = []
            if isinstance(raw_changes, dict):
                raw_changes = [raw_changes]
            if isinstance(raw_changes, list):
                for item in raw_changes:
                    if isinstance(item, dict) and item.get("file_path"):
                        fps.append(item.get("file_path"))

        try:
            from tools._helpers import _resolve_path
            for p in fps:
                self._read_cache.pop(str(_resolve_path(p)), None)
        except Exception:
            pass

    def _get_fast_dispatch_context(self) -> Dict[str, Any]:
        """Gathers runtime session facts for System One decision models:
        active background tasks, last target file/url, last tool, and recent turns.
        """
        ctx: Dict[str, Any] = {
            "active_tasks": [],
            "last_target": None,
            "last_tool": None,
            "prev_turn": None,
        }
        try:
            from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
            with ACTIVE_PROCESSES_LOCK:
                for pid, info in ACTIVE_PROCESSES.items():
                    started = info.get("started", 0)
                    elapsed_s = int(time.time() - started) if started else 0
                    ctx["active_tasks"].append({
                        "pid": pid,
                        "command": info.get("command", ""),
                        "status": "running",
                        "elapsed": f"{elapsed_s}s",
                    })
        except Exception:
            pass

        try:
            if hasattr(self, "_read_cache") and self._read_cache:
                ctx["last_target"] = next(reversed(self._read_cache))

            for msg in reversed(getattr(self, "messages", [])[-6:]):
                if msg.get("role") in ("assistant", "model"):
                    if not ctx["prev_turn"]:
                        txt = msg.get("content") or ""
                        ctx["prev_turn"] = txt.splitlines()[0][:120] if txt else None
                    if not ctx["last_tool"]:
                        tc = msg.get("tool_calls")
                        if tc and isinstance(tc, list) and tc[0].get("function"):
                            ctx["last_tool"] = tc[0]["function"].get("name")
        except Exception:
            pass

        return ctx

    def _estimate_tokens(self, text: str) -> int:
        return estimate_tokens(text, self.model_name, self.provider)

    def _estimate_message(self, message) -> int:
        """One message, counting an attached image as an image rather than as
        the length of its base64 — see estimate_message_tokens."""
        from src.agent.trimmer import estimate_message_tokens
        return estimate_message_tokens(message, self.model_name, self.provider)

    def _history_load(self):
        """(history tokens, message count, budget) of the history as it would be sent."""
        self.max_context_tokens = get_context_window()

        from config import get_strip_reasoning, get_max_generation_tokens
        from src.agent.trimmer import clean_messages_for_llm
        from src.agent.context_limit import effective_history_budget
        cleaned = clean_messages_for_llm(self.messages, get_strip_reasoning())
        sys_tokens = self._estimate_tokens(str(self.messages[0])) if self.messages else 0
        response_reserve = min(get_max_generation_tokens() or 1024, 2048)
        budget = effective_history_budget(self.max_context_tokens, sys_tokens, response_reserve)
        return sum(self._estimate_message(m) for m in cleaned[1:]), len(cleaned) - 1, budget

    def _precompute_summary(self) -> None:
        """Between turns: when the history is close to its limit, start writing the summary the next compaction
        will need, in the background (src/agent/summary_precompute.py). Never raises, never waits."""
        try:
            if os.environ.get("ARGENT_LANE_PRIORITY") or self in _ACTIVE_SUBAGENTS:
                return                                    # a worker's own history is short and not worth a request
            from src.agent import summary_precompute
            if not summary_precompute.enabled() or not self.strategy.summarizes_history():
                return
            tokens, count, budget = self._history_load()
            start = summary_precompute.START
            if tokens < start * budget and count < start * self.max_history_messages:
                return
            from src.agent.trimmer import precompute_summary
            precompute_summary(self.messages, self.model_name, max(1024, int(budget * 0.75)))
        except Exception:
            log.debug("summary precompute skipped", exc_info=True)

    def _trim_history(self):
        self.max_context_tokens = get_context_window()

        from config import get_strip_reasoning, get_max_generation_tokens
        from src.agent.trimmer import clean_messages_for_llm, age_tool_results
        from src.agent.context_limit import effective_history_budget
        strip_enabled = get_strip_reasoning()

        # Continuous prevention: age old tool outputs before checking budget
        self.messages = age_tool_results(self.messages)

        cleaned = clean_messages_for_llm(self.messages, strip_enabled)

        # Trim history into the room left AFTER the system prompt, the tool
        # schemas (sent alongside) and the model's response — not against the
        # whole window — so the assembled request fits under a hard limit.
        sys_tokens = self._estimate_tokens(str(self.messages[0])) if self.messages else 0
        response_reserve = min(get_max_generation_tokens() or 1024, 2048)
        budget = effective_history_budget(self.max_context_tokens, sys_tokens, response_reserve)

        history_tokens = sum(self._estimate_message(m) for m in cleaned[1:])
        msg_count = len(cleaned) - 1

        if msg_count <= self.max_history_messages and history_tokens <= budget:
            return

        # Hysteresis: trigger at the full limit, but trim DOWN to lower targets.
        # Trimming rewrites the request prefix and invalidates the provider's
        # KV/prefix cache, so trimming a little every turn (the old behaviour
        # once at the limit) forced a full re-prefill on every turn. Cutting
        # deeper once buys many append-only (cache-warm) turns between trims.
        target_msgs = max(4, int(self.max_history_messages * 0.6))
        target_budget = max(1024, int(budget * 0.75))
        self.messages = self.strategy.trim_history(
            self.messages, self.model_name, target_msgs, target_budget
        )

    def _parse_raw_tool_call(self, content: str) -> dict | None:
        return parse_raw_tool_call(content)

    def _salvage_truncated_write(self, tool_calls_accumulator):
        """If the truncated tool call was a file write, persist the partial
        content directly to disk and return (file_path, lines_written, tail).
        Returns None if nothing could be salvaged."""
        from src.agent.salvage import (
            SALVAGEABLE_TOOLS, extract_partial_write, trim_to_last_line, last_lines,
        )
        if not tool_calls_accumulator:
            return None
        fn = tool_calls_accumulator[0].get("function", {})
        if fn.get("name") not in SALVAGEABLE_TOOLS:
            return None
        extracted = extract_partial_write(fn.get("arguments", ""))
        if not extracted:
            return None
        file_path, partial = extracted
        kept = trim_to_last_line(partial)
        if not kept.strip():
            return None
        try:
            from tools._helpers import _resolve_path
            p = _resolve_path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8", newline="") as f:
                f.write(kept)
        except Exception as e:
            log.warning("Salvage append crashed: %s", e)
            return None
        return file_path, kept.count("\n") or 1, last_lines(kept)

    def _maybe_auto_retrieve(self, user_text: str):
        """Proactive RAG: when auto_retrieve is on and RAG is active, run
        semantic_search on the query and inject the top snippets as context, so
        a weak model retrieves "at the right moment" without having to remember
        to call the tool. Returns silently on any failure."""
        try:
            from config import get_auto_retrieve
            if not get_auto_retrieve() or len(user_text.strip()) < 8:
                return
            from rag_engine import is_rag_enabled, semantic_search
            if not is_rag_enabled():
                return
            results = semantic_search(user_text, n_results=4)
            if not results or results.startswith("Error") or "No relevant" in results:
                return
            self.messages.append({
                "role": "system",
                "content": (
                    "## RETRIEVED CONTEXT (auto)\n"
                    "Snippets relevant to the user's query, pulled automatically from the knowledge base. "
                    "Use them to ground your answer; verify before relying.\n\n" + results
                ),
            })
            log.info("auto-retrieve injected context for query: %.60s", user_text)
        except Exception as e:
            log.warning("auto-retrieve failed: %s", e)

    def _build_objective_anchor(self) -> str | None:
        """Compact trailing summary of factual work done (files edited, recent errors) for long turns."""
        d = memory.data
        completed = d.get("completed") or []
        errors = d.get("errors_encountered") or []
        touched = d.get("files_modified") or []
        factual_only = getattr(self.strategy, "wants_factual_anchor_only", lambda: False)()

        if factual_only:
            if not touched:
                return None
        elif not (completed or errors or touched):
            return None

        parts = []
        if touched:
            unique = list(dict.fromkeys(touched))[-8:]
            parts.append("## RECENT WORKING CONTEXT\n- FILES ALREADY EDITED: " + "; ".join(unique))
        if not factual_only:
            if completed:
                parts.append("- COMPLETED: " + "; ".join(completed[-3:]))
            if errors:
                parts.append("- RECENT ERRORS: " + "; ".join(errors[-2:]))
        if parts:
            parts.append("- LANGUAGE NOTE: Keep responding and reasoning in the user's chosen language.")
        return "\n".join(parts) if parts else None

    def _execute_fast_tool(self, tool_name: str, args: dict) -> str:
        """Executes one fast-path tool call inline (no LLM, no approval flow).

        Shared by the single-dispatch path and the multi-intent chain loop.
        """
        from src.agent import plan_mode
        refused = plan_mode.violation(tool_name, args)
        if refused:
            return refused
        try:
            from src.agent.permissions import PERMISSIONS
            refused = PERMISSIONS.denied(tool_name, args)
        except Exception as e:
            log.warning("permission rules failed: %s", e)
            refused = None
        if refused:
            return refused
        if tool_name == "read_file":
            from tools.file_ops import read_file
            return read_file(**args)
        elif tool_name == "grep_search":
            from tools.search_ops import grep_search as _grep
            return _grep(**args)
        elif tool_name == "search_files":
            from tools.search_ops import search_files as _sfiles
            return _sfiles(**args)
        elif tool_name == "list_directory":
            from tools.file_ops import list_directory as _ldir
            return _ldir(**args)
        elif tool_name == "calculate":
            from tools.misc_tools import calculate as _calc
            return _calc(**args)
        elif tool_name == "run_command":
            from src.agent.shell import run_text
            cmd = args.get("command", "")
            cp = run_text(
                cmd, shell=True, capture_output=True, timeout=15
            )
            return (cp.stdout or "") + (cp.stderr or "")
        elif tool_name == "read_background_command":
            from tools.command_ops import read_background_command
            return read_background_command(**args)
        elif tool_name == "browser_open":
            from tools.browser_tools import browser_open
            return browser_open(**args)
        elif tool_name in ("browser_click", "browser_type", "browser_act"):
            from tools.stagehand_tools import browser_act
            instr = args.get("instruction") or args.get("text") or args.get("selector") or ""
            return browser_act(instruction=instr)
        return f"Fast-path not implemented for tool: {tool_name}"

    def process_user_input(self, user_text: str, allowed_tools: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """Run one turn, and guarantee the reason it ended is on disk.

        The wrapper exists for the ending no break-site can see: the caller
        stopping consumption. Ctrl+C, a closing UI or an abandoned generator
        closes this one, every logging site below is skipped, and the turn
        vanishes without a trace — which is exactly what "it just cut off"
        looks like from the outside.
        """
        self._turn_end_logged = False
        try:
            yield from self._run_turn(user_text, allowed_tools)
        except GeneratorExit:
            self._log_turn_end("interrupted — the caller stopped reading")
            raise
        except BaseException as e:
            self._log_turn_end("turn raised", f"{type(e).__name__}: {e}")
            raise
        finally:
            try:
                from config import get_speculative_lookahead_enabled
                if get_speculative_lookahead_enabled():
                    from src.micromodels.lookahead_observer import lookahead_observer
                    lookahead_observer.reset_turn()
            except Exception:
                pass
            # Reached when the loop simply ran out without passing a break site.
            self._log_turn_end("ended without a recorded reason")
            self._persist_progress(force=True)
            self._precompute_summary()

    def _run_turn(self, user_text: str, allowed_tools: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Process the user input and yield chunks of response or tool activity.
        Supports streaming generation.
        """
        self._begin_turn_record(user_text)
        from src.agent.todos import TODOS
        TODOS.begin_turn()
        try:                                   # a new word from the person: the series of desktop actions starts over
            from src.automation import desktop_guard
            desktop_guard.SERIES.begin_turn()
        except Exception:
            pass
        try:                                   # the word "ultracode" in the message: the mode is on for this session
            if not (os.environ.get("ARGENT_LANE_PRIORITY") or self in _ACTIVE_SUBAGENTS):
                from src.agent.workflow import trigger as ultracode_trigger
                ultracode_trigger.on_user_message(user_text)
        except Exception:
            pass
        self._refresh_system_prompt(allowed_tools)
        from vision import build_image_message, drain_images
        initial_queued = drain_images()
        if initial_queued:
            self.messages.append(build_image_message(
                [(img["data"], img["media_type"]) for img in initial_queued],
                user_text
            ))
        else:
            self.messages.append({"role": "user", "content": user_text})
        
        self._current_turn_msg_index = max(0, len(self.messages) - 1)
        if not user_text.startswith("/"):
            self._maybe_auto_retrieve(user_text)

        self._trim_history()
        self._persist_progress()
        self._turn_checkpoint_done = False
        _label_lines = (user_text or "").strip().splitlines()
        self._turn_label = _label_lines[0][:60] if _label_lines else ""
        self._truncate_continues = 0
        self._salvage_continues = 0
        self._ctx_overflow_retries = 0
        self._no_action_continues = 0
        self._turn_started = time.monotonic()
        self._turn_tool_calls = 0
        self._turn_local_agent_calls = 0

        # Phase 4: one trajectory per turn — links every Leeth decision and
        # outcome (tool results, approvals, exit codes) of this turn together.
        try:
            from src.leeth import trajectory as leeth_trajectory
            leeth_trajectory.start_trajectory(user_text)
        except Exception:
            pass

        # One provider instance per turn: tool-result formatting and retries
        # below reuse it instead of re-creating a provider on every call.
        try:
            provider = create_provider(self.provider)
            validation_error = provider.validate_config()
            if validation_error:
                self._log_turn_end("provider config invalid", validation_error)
                yield {"type": "error", "content": validation_error}
                return
        except Exception as e:
            self._log_turn_end("provider could not be created", str(e))
            yield {"type": "error", "content": f"Provider error: {e}"}
            return

        # Grammar-constrained steps: tiny models on Ollama get the step schema
        # enforced at the decoder level — malformed tool JSON becomes impossible.
        step_schema = None
        if (self.strategy.wants_constrained_decoding()
                and provider.supports_constrained_decoding()
                and not getattr(self, "_constrained_unsupported", False)):
            step_schema = build_step_schema()

        self._stop_requested = False

        # ── Speculative CoT Lookahead Initialization ─────────────
        from config import get_speculative_lookahead_enabled
        speculative_enabled = get_speculative_lookahead_enabled()
        if speculative_enabled:
            try:
                from src.micromodels.lookahead_observer import lookahead_observer
                from src.micromodels.speculative_cache import speculative_cache
                lookahead_observer.reset_turn()
            except Exception as _e:
                log.debug(f"Speculative lookahead init skipped: {_e}")
                speculative_enabled = False

        # ── Fast-Path Interception (MicroModel / Needle 3) ─────────
        # Routes routine tool calls (read_file, grep_search, run_command, list_directory, calculate)
        # in ~1.5 ms directly in-process on CPU without consuming LLM tokens or network latency.
        from config import get_fast_dispatch_backend, get_needle_enabled, get_needle_fast_path
        # A sub-agent is handed a task by its supervisor, not a request by a person: the router that
        # guesses what a person wants ("Start task: read X" → answer with the file) must not run for it.
        backend = get_fast_dispatch_backend() if getattr(self, "fast_path_allowed", True) else "off"

        is_forced_leeth = bool(user_text and (user_text.startswith("!fast ") or user_text.startswith("/leeth ")))
        query_for_dispatch = user_text[6:].strip() if is_forced_leeth else user_text

        if backend != "off" and (is_forced_leeth or (user_text and not user_text.startswith("/") and not user_text.startswith("!"))):
            try:
                dispatch = None
                badge_title = ""

                if backend == "leeth":
                    from src.leeth.dispatcher import dispatcher as leeth_disp
                    from src.leeth import trajectory as leeth_trajectory
                    session_ctx = self._get_fast_dispatch_context()
                    dispatch = leeth_disp.try_fast_dispatch(
                        query_for_dispatch,
                        trajectory_id=leeth_trajectory.current_trajectory_id(),
                        context=session_ctx,
                    )
                    badge_title = "Leeth Fast-Path (Forced)" if is_forced_leeth else "Leeth Fast-Path"
                elif backend == "micromodel":
                    from src.micromodels.dispatcher import dispatcher as micro_disp
                    dispatch = micro_disp.try_fast_dispatch(user_text)
                    badge_title = "MicroModel Fast-Path"
                elif backend == "needle" and get_needle_enabled() and get_needle_fast_path():
                    from src.agent.needle_coprocessor import coprocessor as needle_disp
                    dispatch = needle_disp.try_fast_dispatch(user_text)
                    badge_title = "Needle 3 Fast-Path"

                if dispatch is not None:
                    # Ground-truth verification: read_file only dispatches if file exists on disk
                    # (applies to every step of a multi-intent chain as well)
                    _chain_steps = getattr(dispatch, "steps", None)
                    if _chain_steps:
                        for _st in _chain_steps:
                            if _st.tool_name == "read_file":
                                _fp = _st.arguments.get("file_path", "")
                                if not _fp or not os.path.exists(_fp):
                                    log.debug(f"Fast-path chain read_file rejected non-existent path '{_fp}', delegating to LLM")
                                    dispatch = None
                                    break
                    elif dispatch.tool_name == "read_file":
                        fpath = dispatch.arguments.get("file_path", "")
                        if not fpath or not os.path.exists(fpath):
                            log.debug(f"Fast-path read_file rejected non-existent path '{fpath}', delegating to LLM")
                            dispatch = None

                if dispatch is not None and backend == "leeth":
                    # ── Active-Learning Review Gate ──────────────────────────
                    # Every Leeth proposal is reviewed before execution. The
                    # verdict (accept/reject/partial + correction) is recorded
                    # as training feedback for the next fine-tune cycle.
                    from config import get_leeth_review_mode
                    _review_mode = get_leeth_review_mode()
                    _decision_id = getattr(dispatch, "decision_id", None)
                    if not is_forced_leeth and _review_mode != "off" and _decision_id:
                        yield {"type": "content_stream", "content": f"\n[Ревью предложения Leeth: {dispatch.tool_name} {json.dumps(dispatch.arguments, ensure_ascii=False)}]\n"}
                        _verdict = self._review_leeth_proposal(provider, user_text, dispatch)
                        if _verdict is not None:
                            try:
                                from src.leeth import feedback as leeth_feedback
                                leeth_feedback.submit_verdict(
                                    _decision_id,
                                    _verdict.get("verdict", "reject"),
                                    correction_tool=_verdict.get("correction_tool"),
                                    correction_param=_verdict.get("correction_param"),
                                    source=(_review_mode if _review_mode in ("agent", "human") else "agent"),
                                    reason=_verdict.get("reason", ""),
                                )
                            except Exception as fb_err:
                                log.debug(f"Feedback verdict submit failed: {fb_err}")
                            if _verdict.get("verdict") == "reject":
                                log.info("Leeth proposal rejected by reviewer: %s", _verdict.get("reason", ""))
                                dispatch = None
                            elif _verdict.get("verdict") == "partial":
                                _corr_tool = _verdict.get("correction_tool") or dispatch.tool_name
                                _corr_param = _verdict.get("correction_param")
                                _new_args = leeth_disp._build_arguments(_corr_tool, _corr_param) if (_corr_param or _verdict.get("correction_tool")) else None
                                if _new_args:
                                    log.info("Leeth proposal corrected by reviewer: %s -> %s %s", dispatch.tool_name, _corr_tool, _corr_param)
                                    dispatch.tool_name = _corr_tool
                                    dispatch.arguments = _new_args
                                else:
                                    dispatch = None
                        else:
                            # Review could not be produced — fail safe: delegate to LLM.
                            dispatch = None

                if dispatch is not None and backend == "leeth" and getattr(dispatch, "steps", None):
                    # ── Chain Review Gate: each step through the same verdict flow ──
                    # A reject at ANY step aborts the whole chain -> delegate to LLM.
                    _review_mode_c = get_leeth_review_mode()
                    if not is_forced_leeth and _review_mode_c != "off":
                        _kept_steps = []
                        _chain_rejected = False
                        for _st in dispatch.steps:
                            if not _st.decision_id:
                                _kept_steps.append(_st)
                                continue
                            yield {"type": "content_stream", "content": f"\n[Ревью шага цепочки: {_st.tool_name} {json.dumps(_st.arguments, ensure_ascii=False)}]\n"}
                            _verdict = self._review_leeth_proposal(provider, user_text, _st)
                            if _verdict is not None:
                                try:
                                    from src.leeth import feedback as leeth_feedback
                                    leeth_feedback.submit_verdict(
                                        _st.decision_id,
                                        _verdict.get("verdict", "reject"),
                                        correction_tool=_verdict.get("correction_tool"),
                                        correction_param=_verdict.get("correction_param"),
                                        source=(_review_mode_c if _review_mode_c in ("agent", "human") else "agent"),
                                        reason=_verdict.get("reason", ""),
                                    )
                                except Exception as fb_err:
                                    log.debug(f"Feedback verdict submit failed: {fb_err}")
                                if _verdict.get("verdict") == "reject":
                                    log.info("Chain step rejected by reviewer: %s", _verdict.get("reason", ""))
                                    _chain_rejected = True
                                    break
                                elif _verdict.get("verdict") == "partial":
                                    _corr_tool = _verdict.get("correction_tool") or _st.tool_name
                                    _corr_param = _verdict.get("correction_param")
                                    _new_args = leeth_disp._build_arguments(_corr_tool, _corr_param) if (_corr_param or _verdict.get("correction_tool")) else None
                                    if _new_args:
                                        log.info("Chain step corrected by reviewer: %s -> %s %s", _st.tool_name, _corr_tool, _corr_param)
                                        _st.tool_name = _corr_tool
                                        _st.arguments = _new_args
                                    else:
                                        _chain_rejected = True
                                        break
                            else:
                                # Review could not be produced — fail safe.
                                _chain_rejected = True
                                break
                            _kept_steps.append(_st)
                        if _chain_rejected:
                            dispatch = None
                        else:
                            dispatch.steps = _kept_steps

                if dispatch is not None:
                    # Phase 4: link every dispatched decision to the turn trajectory.
                    try:
                        from src.leeth import trajectory as leeth_trajectory
                        _traj = leeth_trajectory.current_trajectory_id()
                        if _traj:
                            if getattr(dispatch, "decision_id", None):
                                leeth_trajectory.attach_decision(dispatch.decision_id, _traj)
                            for _st in (getattr(dispatch, "steps", None) or []):
                                if getattr(_st, "decision_id", None):
                                    leeth_trajectory.attach_decision(_st.decision_id, _traj)
                    except Exception:
                        pass
                    _chain_steps = getattr(dispatch, "steps", None)
                    if _chain_steps:
                        # ── Multi-intent chain execution ────────────────────
                        chain_results = []
                        chain_aborted = False
                        for _si, _st in enumerate(_chain_steps, 1):
                            _tn, _ta = _st.tool_name, _st.arguments
                            yield {"type": "tool_start", "name": _tn, "args": _ta}
                            try:
                                _res = self._execute_fast_tool(_tn, _ta)
                            except Exception as e:
                                _res = f"Error executing {_tn}: {e}"
                            yield {"type": "tool_end", "name": _tn, "result": _res}
                            try:
                                from src.leeth import trajectory as leeth_trajectory, feedback as leeth_feedback
                                _ok = not (isinstance(_res, str) and _res.startswith("Error executing"))
                                leeth_trajectory.record_event("tool_exec", tool=_tn, ok=_ok, detail=str(_res)[:200])
                                if _st.decision_id:
                                    leeth_feedback.record_outcome(_st.decision_id, "executed_ok" if _ok else "executed_error", {"tool": _tn})
                            except Exception:
                                pass
                            if isinstance(_res, str) and _res.startswith("Error executing"):
                                chain_aborted = True
                            chain_results.append((_tn, _res))
                            if chain_aborted:
                                break
                        badge = f"[{badge_title}: {dispatch.elapsed_ms:.1f} ms — цепочка {len(chain_results)}/{len(_chain_steps)} шагов]"
                        yield {"type": "content_stream", "content": f"\n{badge}\n"}
                        _hist = "\n".join(
                            f"{i}) Tool: {tn}\nResult:\n{res}" for i, (tn, res) in enumerate(chain_results, 1)
                        )
                        self.messages.append({
                            "role": "assistant",
                            "content": f"{badge}\nВыполнено шагов: {len(chain_results)} из {len(_chain_steps)}\n{_hist}",
                        })
                        if chain_aborted:
                            # Partial execution: the completed steps stay in history;
                            # the remainder delegates to the LLM (fall through).
                            self._log_turn_end(f"{backend} fast-path chain aborted", f"{len(chain_results)}/{len(_chain_steps)} steps")
                        else:
                            self._log_turn_end(f"{backend} fast-path chain", f"{len(_chain_steps)} steps in {dispatch.elapsed_ms:.1f}ms")
                            return
                    else:
                        tool_name = dispatch.tool_name
                        args = dispatch.arguments
                        badge = f"[{badge_title}: {dispatch.elapsed_ms:.1f} ms]"
                        yield {"type": "tool_start", "name": tool_name, "args": args}

                        try:
                            result = self._execute_fast_tool(tool_name, args)
                        except Exception as e:
                            result = f"Error executing {tool_name}: {e}"

                        yield {"type": "tool_end", "name": tool_name, "result": result}
                        # Phase 4: the dispatched decision's real-world outcome.
                        try:
                            from src.leeth import trajectory as leeth_trajectory, feedback as leeth_feedback
                            _ok = not (isinstance(result, str) and result.startswith("Error executing"))
                            leeth_trajectory.record_event("tool_exec", tool=tool_name, ok=_ok, detail=str(result)[:200])
                            if getattr(dispatch, "decision_id", None):
                                leeth_feedback.record_outcome(dispatch.decision_id, "executed_ok" if _ok else "executed_error", {"tool": tool_name})
                        except Exception:
                            pass
                        yield {"type": "content_stream", "content": f"\n{badge}\n"}

                        # Record tool result in history so the model has context
                        self.messages.append({
                            "role": "assistant",
                            "content": f"{badge}\nTool: {tool_name}\nResult:\n{result}"
                        })
                        self._log_turn_end(f"{backend} fast-path", f"{tool_name} in {dispatch.elapsed_ms:.1f}ms")
                        return
            except Exception as e:
                log.debug(f"Fast-path interception skipped: {e}")

        # Phase 4: mark the turn as delegated to the LLM (fast-path fell through
        # or was off) — the LLM's tool calls below are the correction signal.
        try:
            from src.leeth import trajectory as leeth_trajectory
            if leeth_trajectory.current_trajectory_id():
                leeth_trajectory.record_event("delegated_llm", backend=backend)
        except Exception:
            pass

        # A lost connection or a busy provider does not end the turn: the same request is sent again once it
        # is worth sending (src/agent/netstate.py). `net_streak` counts the failures of the request in flight;
        # `resume_prefix` is the answer text the person has ALREADY seen from an attempt that died mid-stream.
        from src.agent import netstate
        net_streak = netstate.Streak()
        resume_prefix = ""

        while True:
            if self._stop_requested:
                self._log_turn_end("turn stopped by user request")
                yield {"type": "error", "content": "\n[System: Выполнение прервано пользователем (Ctrl+C / Telegram)]"}
                self.repair_history()
                return

            # Variables to accumulate the streamed response
            full_content = resume_prefix
            full_reasoning = ""
            full_rethink = ""
            tool_calls_accumulator = []
            is_building_raw_tool = False
            raw_tool_buffer = ""
            raw_tool_char_count = 0
            reasoning_tag_active = False
            rethink_tag_active = False
            tag_lookahead_buffer = ""
            is_truncated = False
            constrained_extractor = StepStreamExtractor() if step_schema is not None else None
            loop_guard_stop = False
            START_TAGS = ("<thought>", "<think>", "<reasoning>")
            END_TAGS = ("</thought>", "</think>", "</reasoning>")
            RETHINK_START_TAGS = ("<rethink>", "<reflection>")
            RETHINK_END_TAGS = ("</rethink>", "</reflection>")
            ALL_STREAM_TAGS = START_TAGS + END_TAGS + RETHINK_START_TAGS + RETHINK_END_TAGS
            MAX_STREAM_TAG_LEN = max(len(t) for t in ALL_STREAM_TAGS)
            ALL_STREAM_TAG_PREFIXES = {t[:i] for t in ALL_STREAM_TAGS for i in range(1, len(t))}
            
            from config import get_strip_reasoning
            from src.agent.trimmer import clean_messages_for_llm
            strip_enabled = get_strip_reasoning()
            cleaned_messages = clean_messages_for_llm(self.messages, strip_enabled)
            
            try:
                active_tools = get_tool_schemas(include_hidden=(allowed_tools is not None))
                if allowed_tools is not None:
                    active_tools = [t for t in active_tools if t["function"]["name"] in allowed_tools]

                # Tier-based slimming: weak models (tiny/small) get only the core
                # toolset, cutting the schema budget (~6.5k tokens) and sharpening
                # tool choice. Larger models keep the full set. An explicit
                # allowlist is exempt — it is already short, and quietly deleting
                # a tool the caller hand-picked leaves the task unable to do its
                # job with nothing in the log to explain why.
                category = get_model_size_category(self.model_name)
                is_moe = category == "moe" or is_moe_model(self.model_name)
                if is_moe:
                    category = "moe"
                if allowed_tools is None and not is_moe:
                    from tool_profiles import slim_tools_for_category
                    _slim_names = set(slim_tools_for_category([t["function"]["name"] for t in active_tools], category))
                    active_tools = [t for t in active_tools if t["function"]["name"] in _slim_names]

                # Schema minification for small/moe models: shorten verbose descriptions
                # to save token context and reduce distraction while preserving parameter keys/types.
                wants_minify = getattr(self.strategy, "wants_minified_schemas", lambda: False)()
                if (category in ("tiny", "small", "moe") or wants_minify) and active_tools:
                    from tool_profiles import minify_tool_schemas
                    active_tools = minify_tool_schemas(active_tools)

                # No native tool channel — either the strategy never had one
                # (tiny/local) or a cloud model rejected native tools at runtime
                # (e.g. an OpenRouter free model). Give the model an in-context
                # tool catalog and rely on raw-JSON tool-call parsing instead of
                # sending native tool schemas.
                if not self.strategy.supports_native_tools() or self._native_tools_unsupported:
                    if active_tools:
                        cleaned_messages = cleaned_messages + [
                            {"role": "system", "content": build_tool_catalog(active_tools)}
                        ]
                    active_tools = None

                # Per-turn volatile facts (repo map, background processes) ride
                # as an ephemeral tail message at the start of a user turn.
                # NEVER append a system message after an intermediate tool result:
                # 1) It violates the OpenAI tool-call schema (tool messages must be directly followed by assistant).
                # 2) It confuses small/local models (recency bias causes them to comment on the repo map instead of synthesizing tool output).
                is_tool_followup = bool(cleaned_messages and cleaned_messages[-1].get("role") == "tool")

                if not is_tool_followup:
                    ephemeral = self._build_ephemeral_context()
                    if ephemeral:
                        cleaned_messages = cleaned_messages + [{"role": "system", "content": ephemeral}]

                    # Objective anchor: re-pin the goal at the end of long
                    # histories, where small models actually look.
                    if self.strategy.wants_objective_anchor() and len(cleaned_messages) > 8:
                        anchor = self._build_objective_anchor()
                        if anchor:
                            cleaned_messages = cleaned_messages + [{"role": "system", "content": anchor}]

                from config import get_temperature
                temp = get_temperature()

                if resume_prefix:
                    # The previous attempt died after the person had read this much. Not stored in the history:
                    # the continuation is joined to it below, and the stored message is one whole answer.
                    cleaned_messages = cleaned_messages + [
                        {"role": "assistant", "content": resume_prefix},
                        {"role": "user", "content": RESUME_NOTE},
                    ]

                response_stream = provider.stream_chat(
                    model=self.model_name,
                    messages=cleaned_messages,
                    tools=active_tools,
                    context_window=self.max_context_tokens if self.provider not in ("zai", "openrouter", "claude", "deepseek", "xkiro", "wormsoft") else None,
                    temperature=temp,
                    format_schema=step_schema,
                )

                for chunk in response_stream:
                    if self._stop_requested:
                        self._log_turn_end("turn stopped by user request during stream")
                        yield {"type": "error", "content": "\n[System: Выполнение прервано пользователем (Ctrl+C / Telegram)]"}
                        self.repair_history()
                        return

                    if chunk.get("truncated"):
                        is_truncated = True

                    usage_data = chunk.get("usage")
                    if usage_data:
                        yield {"type": "usage", "data": usage_data}

                    thinking_chunk = chunk.get("thinking", "")
                    if thinking_chunk:
                        full_reasoning += thinking_chunk
                        yield {"type": "thinking_stream", "content": thinking_chunk}
                        if speculative_enabled:
                            lookahead_observer.feed(thinking_chunk)

                    for tc_delta in chunk.get("tool_call_deltas", []):
                        index = tc_delta["index"]
                        while len(tool_calls_accumulator) <= index:
                            tool_calls_accumulator.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc_delta.get("id"):
                            tool_calls_accumulator[index]["id"] = tc_delta["id"]
                        if tc_delta.get("function_name_delta"):
                            tool_calls_accumulator[index]["function"]["name"] += tc_delta["function_name_delta"]
                        if tc_delta.get("function_arguments_delta"):
                            tool_calls_accumulator[index]["function"]["arguments"] += tc_delta["function_arguments_delta"]
                            # Yield a progress signal so the UI can show a spinner
                            yield {"type": "tool_generating", "name": tool_calls_accumulator[index]["function"]["name"], "bytes": len(tool_calls_accumulator[index]["function"]["arguments"]), "delta": tc_delta["function_arguments_delta"]}

                    content_chunk = chunk.get("content", "")
                    if content_chunk and constrained_extractor is not None:
                        # Constrained mode: the chunk is part of the step JSON.
                        # Stream the decoded "reply" text; buffer tool steps.
                        emitted = constrained_extractor.feed(content_chunk)
                        if emitted:
                            full_content += emitted
                            yield {"type": "content_stream", "content": emitted}
                        elif constrained_extractor.mode == "tool":
                            yield {"type": "tool_generating", "name": "?", "bytes": constrained_extractor.size, "delta": content_chunk}
                    elif content_chunk:
                        # Deterministic stream tag FSM with lookahead prefix buffering
                        # to handle tags split across chunk boundaries (L2 Boundary) and
                        # preserve strict chronological tag precedence (L5 State Machine).
                        text_to_process = tag_lookahead_buffer + content_chunk
                        tag_lookahead_buffer = ""

                        while text_to_process:
                            earliest_idx = -1
                            earliest_tag = None
                            for stag in ALL_STREAM_TAGS:
                                sidx = text_to_process.find(stag)
                                if sidx != -1:
                                    if earliest_idx == -1 or sidx < earliest_idx:
                                        earliest_idx = sidx
                                        earliest_tag = stag

                            if earliest_tag is not None:
                                before_tag = text_to_process[:earliest_idx]
                                if before_tag:
                                    if rethink_tag_active:
                                        full_rethink += before_tag
                                        full_reasoning += before_tag
                                        yield {"type": "rethink_stream", "content": before_tag}
                                    elif reasoning_tag_active:
                                        full_reasoning += before_tag
                                        yield {"type": "thinking_stream", "content": before_tag}
                                        if speculative_enabled:
                                            lookahead_observer.feed(before_tag)
                                    else:
                                        full_content += before_tag
                                        raw_tool_buffer += before_tag
                                        if '```json' in raw_tool_buffer or '{\n  "name":' in raw_tool_buffer:
                                            is_building_raw_tool = True
                                        if not is_building_raw_tool:
                                            yield {"type": "content_stream", "content": before_tag}
                                        else:
                                            yield {"type": "tool_generating", "name": "?", "bytes": len(raw_tool_buffer), "delta": before_tag}

                                if earliest_tag in RETHINK_START_TAGS:
                                    rethink_tag_active = True
                                    reasoning_tag_active = False
                                elif earliest_tag in RETHINK_END_TAGS:
                                    rethink_tag_active = False
                                    reasoning_tag_active = False
                                elif earliest_tag in START_TAGS:
                                    reasoning_tag_active = True
                                    rethink_tag_active = False
                                elif earliest_tag in END_TAGS:
                                    reasoning_tag_active = False
                                    rethink_tag_active = False

                                text_to_process = text_to_process[earliest_idx + len(earliest_tag):]
                            else:
                                # Check if suffix could be a partial tag across chunk boundary
                                longest_prefix_len = 0
                                for plen in range(min(len(text_to_process), MAX_STREAM_TAG_LEN - 1), 0, -1):
                                    candidate = text_to_process[-plen:]
                                    if candidate in ALL_STREAM_TAG_PREFIXES:
                                        longest_prefix_len = plen
                                        break

                                if longest_prefix_len > 0:
                                    emit_part = text_to_process[:-longest_prefix_len]
                                    tag_lookahead_buffer = text_to_process[-longest_prefix_len:]
                                else:
                                    emit_part = text_to_process

                                if emit_part:
                                    if rethink_tag_active:
                                        full_rethink += emit_part
                                        full_reasoning += emit_part
                                        yield {"type": "rethink_stream", "content": emit_part}
                                    elif reasoning_tag_active:
                                        full_reasoning += emit_part
                                        yield {"type": "thinking_stream", "content": emit_part}
                                        if speculative_enabled:
                                            lookahead_observer.feed(emit_part)
                                    else:
                                        full_content += emit_part
                                        raw_tool_buffer += emit_part
                                        if '```json' in raw_tool_buffer or '{\n  "name":' in raw_tool_buffer:
                                            is_building_raw_tool = True
                                        if not is_building_raw_tool:
                                            yield {"type": "content_stream", "content": emit_part}
                                        else:
                                            yield {"type": "tool_generating", "name": "?", "bytes": len(raw_tool_buffer), "delta": emit_part}
                                break

                # Flush any leftover lookahead buffer at the end of the stream
                if tag_lookahead_buffer:
                    if rethink_tag_active:
                        full_rethink += tag_lookahead_buffer
                        full_reasoning += tag_lookahead_buffer
                        yield {"type": "rethink_stream", "content": tag_lookahead_buffer}
                    elif reasoning_tag_active:
                        full_reasoning += tag_lookahead_buffer
                        yield {"type": "thinking_stream", "content": tag_lookahead_buffer}
                        if speculative_enabled:
                            lookahead_observer.feed(tag_lookahead_buffer)
                    else:
                        full_content += tag_lookahead_buffer
                        raw_tool_buffer += tag_lookahead_buffer
                        if '```json' in raw_tool_buffer or '{\n  "name":' in raw_tool_buffer:
                            is_building_raw_tool = True
                        if not is_building_raw_tool:
                            yield {"type": "content_stream", "content": tag_lookahead_buffer}
                        else:
                            yield {"type": "tool_generating", "name": "?", "bytes": len(raw_tool_buffer), "delta": tag_lookahead_buffer}
                    tag_lookahead_buffer = ""

                if speculative_enabled:
                    lookahead_observer.flush()
                            
            except Exception as e:
                error_str = str(e).lower()
                if any(s in error_str for s in (
                    "does not support tools", "element type",
                    "support tool use", "tool use is not supported",
                    "no endpoints found that support tool",
                )):
                    if not self._native_tools_unsupported:
                        # Persistently switch this model to prompted tool-calling
                        # (in-context catalog + raw-JSON parsing) so it can still
                        # USE tools, not just answer in prose. Retry the turn.
                        log.warning("No native tool channel (%s); switching to prompted tool-calling", e)
                        self._native_tools_unsupported = True
                        yield {"type": "error", "content": (
                            "\n[Argent: this model has no native tool support — "
                            "switching to in-prompt tool calling and retrying...]"
                        )}
                        continue
                    # Already prompted but tools still rejected — answer without
                    # tools so the turn doesn't crash.
                    try:
                        fallback_stream = provider.stream_chat(
                            model=self.model_name,
                            messages=cleaned_messages,
                            tools=None,
                            context_window=self.max_context_tokens if self.provider not in ("zai", "openrouter", "claude", "deepseek", "xkiro", "wormsoft") else None,
                            temperature=temp,
                        )
                        for chunk in fallback_stream:
                            content_chunk = chunk.get("content", "")
                            if content_chunk:
                                full_content += content_chunk
                                yield {"type": "content_stream", "content": content_chunk}
                    except Exception as retry_e:
                        self._log_turn_end("fallback stream failed", str(retry_e))
                        yield {"type": "error", "content": f"Fallback Error: {retry_e}"}
                        break
                elif "thought_signature" in error_str or "functioncall" in error_str:
                    self._flatten_tool_messages()
                    continue
                elif step_schema is not None and any(k in error_str for k in ("format", "schema", "grammar")):
                    # Older Ollama versions reject schema-constrained format —
                    # degrade gracefully to the raw-JSON parsing path for good.
                    log.warning("Constrained decoding unsupported by provider, falling back: %s", e)
                    self._constrained_unsupported = True
                    step_schema = None
                    continue
                elif isinstance(e, ProviderError) and is_context_overflow(str(e)):
                    # The assembled prompt overflowed the model's real context
                    # window. Use the size it reports (if any) to right-size our
                    # budget, re-trim hard and retry — instead of failing.
                    self._ctx_overflow_retries = getattr(self, "_ctx_overflow_retries", 0) + 1
                    if self._ctx_overflow_retries > 3:
                        self._log_turn_end("context overflow after 3 shrinks", str(e))
                        yield {"type": "error", "content": (
                            f"{e}\n[Argent: the prompt still won't fit after shrinking the window "
                            f"3×. Start a fresh chat with /clear, or lower it with /context.]"
                        )}
                        break
                    real_ctx = parse_context_limit(str(e))
                    from config import set_context_window
                    if real_ctx:
                        new_window = max(2048, int(real_ctx * 0.9))
                    else:
                        new_window = max(2048, int(self.max_context_tokens * 0.7))
                    set_context_window(new_window)   # learn the real window for next turns
                    self.max_context_tokens = new_window
                    self._trim_history()
                    yield {"type": "error", "content": (
                        f"\n[Argent: prompt exceeded the model's context window — trimmed history "
                        f"and retrying (window now ~{new_window} tokens)...]"
                    )}
                    continue
                elif self._wait_plan(e) is not None and netstate.patience_seconds() > 0:
                    # Connection lost, provider busy or rate-limited: wait it out, then send the same request
                    # again. Nothing was executed yet (tools run only after the stream is whole), so repeating
                    # is safe; what the person already read is continued, not shown twice.
                    kind, retry_after, detail = self._wait_plan(e)
                    can_continue = bool(
                        full_content.strip() and not tool_calls_accumulator and not is_building_raw_tool
                        and not reasoning_tag_active and not rethink_tag_active and constrained_extractor is None)
                    resume_prefix = full_content if can_continue else ""
                    waiter = netstate.RecoveryWait(
                        provider=provider, streak=net_streak, patience=netstate.patience_seconds(),
                        kind=kind, retry_after=retry_after, detail=detail,
                        stop=lambda: self._stop_requested)
                    for note in waiter.run():
                        yield {"type": "error", "content": note}
                    if waiter.recovered:
                        continue
                    if waiter.reason == "stopped":
                        self._log_turn_end("turn stopped by user request while waiting for the network")
                        yield {"type": "error", "content": "\n[System: Выполнение прервано пользователем (Ctrl+C / Telegram)]"}
                        self.repair_history()
                        return
                    self._log_turn_end("network lost", f"{waiter.reason}: {detail}")
                    waited = netstate.clock_text(net_streak.waited(time.time()))
                    yield {"type": "error", "content": (
                        f"{e}\n[Argent: связь с провайдером не восстановилась за {waited} — ход остановлен. "
                        f"Диалог сохранён: когда связь вернётся, напишите «продолжай», и работа пойдёт с этого места. "
                        f"Срок ожидания: /network.]")}
                    self.repair_history(reason="network")
                    self._persist_progress(force=True)
                    break
                elif isinstance(e, ProviderError):
                    error_lower = str(e).lower()
                    is_vision_rejection = any(term in error_lower for term in (
                        "image input is not supported",
                        "need to provide the mmproj",
                        "does not support image",
                        "image_url is not supported",
                        "vision is not supported",
                        "multimodal not supported",
                    ))
                    has_images = any(
                        (isinstance(m, dict) and bool(m.get("images")))
                        for m in self.messages
                    )
                    if is_vision_rejection and has_images:
                        from vision import mark_vision_unsupported
                        mark_vision_unsupported(self.provider, self.model_name)
                        log.warning(
                            "Provider %s rejected image input for %s: %s; stripping images and retrying turn",
                            self.provider, self.model_name, e
                        )
                        for m in self.messages:
                            if isinstance(m, dict) and "images" in m:
                                m.pop("images", None)
                        for m in cleaned_messages:
                            if isinstance(m, dict) and "images" in m:
                                m.pop("images", None)
                        yield {"type": "error", "content": (
                            f"\n[Argent: модель '{self.model_name}' ({self.provider}) не поддерживает чтение картинок "
                            f"(отсутствует mmproj или модель чисто текстовая). Изображение откреплено, повтор запроса в текстовом режиме...]"
                        )}
                        continue

                    self._log_turn_end("provider error", str(e))
                    yield {"type": "error", "content": str(e)}
                    break
                else:
                    self._log_turn_end("unhandled error", f"{type(e).__name__}: {e}")
                    log.exception("turn aborted by an unhandled exception")
                    yield {"type": "error", "content": f"Error: {e}"}
                    break

            # A whole response arrived: the failure streak (if any) is over, and what was resumed is now part of
            # full_content (it started from the prefix), so the stored message is one answer.
            net_streak.reset()
            resume_prefix = ""

            # End of stream — parse tool call arguments from strings to dicts.
            if tool_calls_accumulator and not is_truncated:
                for tc in tool_calls_accumulator:
                    if isinstance(tc["function"]["arguments"], str):
                        raw_args = tc["function"]["arguments"]
                        try:
                            tc["function"]["arguments"] = json.loads(raw_args)
                        except json.JSONDecodeError as e:
                            # Falling back to {} runs the tool with no arguments
                            # at all — read_file() with no path, run_command()
                            # with no command — and the model is told nothing.
                            # Marking it instead lets the dispatch answer with a
                            # real error the model can correct.
                            log.warning("Failed to parse tool args for %s (%s): %r",
                                        tc["function"]["name"], e, raw_args[:200])
                            tc["function"]["arguments"] = {
                                "__unparsable__": raw_args[:2000], "__error__": str(e)}

            # Constrained mode: materialize the final step from the JSON buffer.
            if constrained_extractor is not None and not is_truncated and not tool_calls_accumulator:
                final_step = constrained_extractor.finalize()
                if final_step and "tool" in final_step:
                    tool_calls_accumulator.append({
                        "type": "function",
                        "function": {
                            "name": final_step["tool"]["name"],
                            "arguments": final_step["tool"]["arguments"],
                        },
                    })
                elif final_step and "reply" in final_step and not full_content:
                    # Streaming extraction missed the reply (unusual key order).
                    full_content = final_step["reply"]
                    yield {"type": "content_replace", "content": full_content}

            # --- FALLBACK: Parse tool calls from raw JSON in content ---
            # Models like Qwen2.5-Coder, nanbeige, etc. often write tool calls
            # as raw JSON strings instead of using the native tool_calls mechanism.
            if not tool_calls_accumulator and not is_truncated and constrained_extractor is None:
                clean_content = full_content.strip()
                parsed_tool = self._parse_raw_tool_call(clean_content)
                
                if parsed_tool:
                    tool_calls_accumulator.append({"type": "function", "function": parsed_tool["parsed"]})
                    # Clean the raw JSON out of full_content
                    if parsed_tool.get("match_str"):
                        full_content = full_content.replace(parsed_tool["match_str"], "").strip()
                    else:
                        full_content = ""
                    yield {"type": "content_replace", "content": full_content}
            # --- END FALLBACK ---
            
            # If the model was truncated mid-generation (max_tokens limit reached)
            if is_truncated:
                was_building_tool = bool(
                    tool_calls_accumulator
                    or is_building_raw_tool
                    or (constrained_extractor is not None and constrained_extractor.mode == "tool")
                )

                # Path 1 — salvageable file write: persist the part that WAS
                # generated and ask the model to continue via append. This turns
                # truncation into incremental progress, so a file of any size can
                # be produced. It has its own, higher limit (large files take many
                # chunks) but is still bounded so a stuck model can't loop forever.
                salvaged = self._salvage_truncated_write(tool_calls_accumulator) if was_building_tool else None
                if was_building_tool:
                    tool_calls_accumulator = []

                if salvaged:
                    self._salvage_continues += 1
                    file_path, written_lines, tail = salvaged
                    if self._salvage_continues > MAX_SALVAGE_CONTINUES:
                        # Final compile check: the file was cut off mid-generation,
                        # so tell the user whether what's on disk is actually usable
                        # instead of a syntactically broken stub.
                        from src.agent.salvage import verify_salvaged_file
                        ok, detail = verify_salvaged_file(file_path)
                        status = ("The saved file parses cleanly."
                                  if ok else f"WARNING — the saved file is INCOMPLETE: {detail}")
                        self._log_turn_end("salvage continuations exhausted",
                                           f"{file_path}: {status}")
                        yield {"type": "error", "content": (
                            f"\n[System: '{file_path}' kept hitting the length limit after "
                            f"{MAX_SALVAGE_CONTINUES} continuations — stopping. The content "
                            f"generated so far is saved to the file. {status}]"
                        )}
                        break
                    yield {"type": "error", "content": f"\n[System: Saved {written_lines} line(s) to {file_path}; asking the model to continue the file...]"}
                    self.messages.append({
                        "role": "user",
                        "content": (
                            f"Your file write was cut off by the length limit, but I saved the "
                            f"{written_lines} line(s) you produced into '{file_path}'. The file is "
                            f"INCOMPLETE. Continue writing the REST of it now with "
                            f"replace_in_file('{file_path}', ...), or write_file('{file_path}', ..., overwrite=True), starting exactly after this tail:\n"
                            f"---\n{tail}\n---\n"
                            f"Do NOT repeat content that is already written. Keep going until the file is complete."
                        )
                    })
                    self._trim_history()
                    continue

                # Path 2 — non-salvageable truncation: bounded auto-continue.
                self._truncate_continues += 1
                if self._truncate_continues > MAX_TRUNCATE_CONTINUES:
                    if full_content and not was_building_tool:
                        self.messages.append({"role": "assistant", "content": full_content})
                    self._log_turn_end("length limit hit repeatedly",
                                       f"{self._truncate_continues} truncations in a row")
                    from config import get_max_generation_tokens
                    max_gen_val = get_max_generation_tokens() or 8192
                    yield {"type": "error", "content": (
                        f"\n[System: Generation reached the per-response limit ({max_gen_val} tokens) "
                        f"{self._truncate_continues} times in a row — auto-continue stopped. "
                        "Increase max_generation_tokens in config or split the task into smaller pieces.]"
                    )}
                    break

                if was_building_tool:
                    # A non-file tool call cannot be "continued" — steer to chunks.
                    yield {"type": "error", "content": "\n[System: Tool call was truncated by the length limit. Asking the model to work in smaller chunks...]"}
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Your tool call was cut off by the generation length limit. "
                            "Do NOT retry the same single huge call — it will be cut off again. "
                            "Produce the result in SMALLER STEPS instead: create the initial structure "
                            "with `write_file`, then populate sections with `replace_in_file`. Keep each call small."
                        )
                    })
                else:
                    msg_to_append = {"role": "assistant", "content": full_content}
                    if full_reasoning:
                        msg_to_append["thinking"] = full_reasoning
                    if full_rethink:
                        msg_to_append["rethink"] = full_rethink
                    self.messages.append(msg_to_append)
                    if reasoning_tag_active or rethink_tag_active:
                        yield {"type": "error", "content": "\n[System: Model reasoning reached the max_tokens limit. Auto-continuing to direct response...]"}
                        self.messages.append({
                            "role": "user",
                            "content": "Your preliminary reasoning was captured. Do NOT restart your thinking or re-open <think> or <rethink> tags. Continue directly where you left off with your response or required tool calls."
                        })
                    else:
                        yield {"type": "error", "content": "\n[System: Model generation was truncated by max_tokens limit. Auto-continuing...]"}
                        self.messages.append({
                            "role": "user",
                            "content": "Your previous response was cut off due to length limits. Please continue exactly where you left off, without any introductory text."
                        })

                # Relieve context pressure before retrying: on small windows the
                # partial output itself is what keeps re-triggering truncation.
                self._trim_history()
                continue
            
            if full_content or tool_calls_accumulator or full_reasoning or full_rethink:
                msg_to_append = {"role": "assistant", "content": full_content}
                if full_reasoning:
                    msg_to_append["thinking"] = full_reasoning
                if full_rethink:
                    msg_to_append["rethink"] = full_rethink
                if tool_calls_accumulator:
                    msg_to_append["tool_calls"] = tool_calls_accumulator
                self.messages.append(msg_to_append)
                # Recorded BEFORE the tools run: if the process dies inside one, the restored session shows the
                # call and repair_history() marks it "did not finish" instead of the step vanishing.
                self._persist_progress()

            # Now handle the fully assembled tool calls
            if tool_calls_accumulator:
                # Active Learning: record live user -> tool call for continuous micro-model self-improvement
                if user_text and not user_text.startswith("/"):
                    try:
                        from src.micromodels.data_collector import collector
                        _first_tc = tool_calls_accumulator[0].get("function", {})
                        collector.log_live_tool_call(
                            user_query=user_text,
                            tool_name=_first_tc.get("name", ""),
                            tool_arguments=_first_tc.get("arguments", {}),
                        )
                    except Exception:
                        pass

                current_tools = get_available_tools()
                if allowed_tools is not None:
                    current_tools = {name: func for name, func in current_tools.items() if name in allowed_tools}
                
                has_fatal_error = False

                # Independent read-only calls in the batch run concurrently
                # (pure reads have nothing to serialize for); anything else
                # falls through to the ordinary sequential dispatch below.
                from src.agent.parallel_tools import precompute_readonly_parallel
                _parallel_results = precompute_readonly_parallel(
                    tool_calls_accumulator, current_tools, dedup=self._read_dedup_note
                ) or {}

                has_auto_signal = False
                for _tc_index, tool_call in enumerate(tool_calls_accumulator):
                    if self._stop_requested:
                        self._log_turn_end("turn stopped by user request before tool call")
                        yield {"type": "error", "content": "\n[System: Выполнение прервано пользователем (Ctrl+C / Telegram)]"}
                        self.repair_history()
                        return

                    func_name = tool_call["function"]["name"]
                    arguments = tool_call["function"].get("arguments", {})
                    self._turn_tool_calls += 1
                    # Hard safety valve: a turn that makes this many tool calls
                    # is a runaway (e.g. a mock/tool contract bug looping the
                    # model) — it burns CPU, RAM and the live-dataset file.
                    if self._turn_tool_calls > 500:
                        self._log_turn_end("tool call limit",
                                           f"{self._turn_tool_calls} calls")
                        yield {"type": "error", "content": (
                            f"\n[System: превышен лимит {self._turn_tool_calls} "
                            f"вызовов инструментов в одном ходе — остановка.]")}
                        self.repair_history()
                        return
                    if func_name == "delegate_local":
                        self._turn_local_agent_calls = getattr(self, "_turn_local_agent_calls", 0) + 1
                    
                    if has_fatal_error:
                        result = "Error: Execution cancelled due to failure in a previous tool call."
                        yield {"type": "tool_start", "name": func_name, "args": arguments}
                        yield {"type": "tool_end", "name": func_name, "result": result}
                        self.messages.append(provider.format_tool_result(str(result), tool_call.get("id")))
                        continue
                    
                    # Name substitutions must be reported back (see below): the
                    # model asked for one tool and got another, and only it can
                    # judge whether the swap preserved its intent.
                    substituted_from = None
                    if func_name not in current_tools:
                        recovered = recover_tool_call(tool_call, current_tools)
                        if recovered:
                            substituted_from = recovered.get("renamed_from")
                            func_name = recovered["function"]["name"]
                            arguments = recovered["function"]["arguments"]
                            tool_call["function"]["name"] = func_name
                            tool_call["function"]["arguments"] = arguments

                    yield {"type": "tool_start", "name": func_name, "args": arguments}

                    if isinstance(arguments, dict) and "__unparsable__" in arguments:
                        # The stream produced arguments that are not JSON. Say so
                        # instead of running the tool empty-handed: the model can
                        # re-emit the call, and nobody can act on a silent no-op.
                        result = (
                            f"Error: the arguments for '{func_name}' were not valid JSON "
                            f"({arguments.get('__error__')}). Send the call again with one "
                            f"well-formed JSON object per tool call.")
                        yield {"type": "tool_end", "name": func_name, "result": result}
                        self.messages.append(
                            provider.format_tool_result(result, tool_call.get("id")))
                        continue

                    ran_tool = False
                    if func_name in current_tools:
                        try:
                            func = current_tools[func_name]
                            # Remap common argument aliases (e.g. old_string -> target_text)
                            # and coerce types before inspecting against tool signature
                            coerced_args = coerce_and_log(func_name, arguments)
                            sig = inspect.signature(func)
                            valid_params = set(sig.parameters.keys())
                            hallucinated_args = {k: v for k, v in coerced_args.items() if k not in valid_params}
                            filtered_args = {k: v for k, v in coerced_args.items() if k in valid_params}
                            
                            # Check for missing required arguments
                            missing_args = [
                                p.name for p in sig.parameters.values()
                                if p.default == inspect.Parameter.empty
                                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                                   inspect.Parameter.VAR_KEYWORD)
                                and p.name not in filtered_args
                            ]
                            
                            if missing_args:
                                provided_args = list(arguments.keys())
                                # Build a smart hint if the model clearly called the wrong tool
                                hint = ""
                                if hallucinated_args:
                                    hint = f"\n\n[HINT]: You passed parameters that do NOT exist in '{func_name}': {list(hallucinated_args.keys())}. "
                                    # Detect common confusion: read_file called with content= (meant write_file)
                                    if func_name == "read_file" and "content" in hallucinated_args:
                                        hint += "It looks like you want to WRITE content, not READ. Use `write_file(file_path, content)` to create or write a file."
                                    elif func_name == "write_file" and "target_text" in hallucinated_args:
                                        hint += "It looks like you want to EDIT part of a file. Use `replace_in_file(file_path, target_text, replacement_text)` instead."
                                    else:
                                        hint += f"Valid parameters for '{func_name}' are: {list(valid_params)}."
                                result = f"Error executing tool '{func_name}': Missing REQUIRED arguments: {missing_args}. You provided: {provided_args}.{hint}"
                            else:
                                # Confirmation for dangerous actions happens inside the tools
                                # themselves via the central `approval` module (single gate).
                                # Trigger on_tool_call hook. If any plugin returns False, we cancel the execution.
                                pre_block = self._pre_tool_block(func_name, filtered_args)
                                hook_results = [] if pre_block else hook_manager.call_hook(
                                    "on_tool_call", func_name, filtered_args)
                                if pre_block:
                                    result = pre_block
                                elif False in hook_results:
                                    result = f"Error: Execution of tool '{func_name}' was blocked by a user plugin."
                                else:
                                    # Time machine: snapshot the pre-turn tree
                                    # before the turn's first edit lands.
                                    cp_chunk = self._maybe_turn_checkpoint(func_name)
                                    if cp_chunk:
                                        yield cp_chunk
                                    # Staleness must be checked BEFORE executing:
                                    # after the edit the mtime change is our own.
                                    stale_note = self._external_change_note(func_name, filtered_args)
                                    pre = _parallel_results.get(_tc_index)
                                    spec_hit = False
                                    if pre is not None:
                                        result = pre
                                    else:
                                        if speculative_enabled:
                                            cached_val = speculative_cache.get(func_name, filtered_args)
                                            if cached_val is not None:
                                                result = cached_val
                                                spec_hit = True
                                        if not spec_hit:
                                            dedup = self._read_dedup_note(filtered_args) if func_name == "read_file" else None
                                            result = dedup if dedup is not None else func(**filtered_args)
                                    ran_tool = True
                                    if stale_note:
                                        result = str(result) + stale_note
                                    if spec_hit:
                                        yield {"type": "content_stream", "content": "\n[Speculative CoT Cache Hit: 0.1 ms]\n"}

                                # Invalidate speculative cache on workspace-modifying tools
                                if speculative_enabled:
                                    if func_name in ["write_file", "replace_in_file", "delete_file"]:
                                        target_file = filtered_args.get("file_path")
                                        if target_file:
                                            speculative_cache.invalidate(target_file)
                                    elif func_name in ["run_command", "git_rollback"]:
                                        speculative_cache.clear_turn()

                                # --- AUTO PLUGIN RELOAD ---
                                if func_name in ["write_file", "replace_in_file", "delete_file"]:
                                    target_file = filtered_args.get("file_path")
                                    if target_file and "Error" not in result:
                                        # Normalize to absolute path
                                        abs_target = os.path.abspath(target_file)
                                        hooks_dir = os.path.abspath(get_hooks_dir())
                                        if abs_target.startswith(hooks_dir):
                                            hook_manager.reload_plugins(hooks_dir)
                                            # Append a small notification to the tool result so the AI knows its new tool is ready
                                            result += f"\n\n[Argent]: Plugin system reloaded. Any new or modified commands in '{os.path.basename(abs_target)}' are now active."

                                # --- GIT ROLLBACK STATE SYNC ---
                                if func_name == "git_rollback" and "Rewound to checkpoint" in str(result):
                                    import re as _re
                                    _m = _re.search(r"Rewound to checkpoint ([0-9a-zA-Z]+)", str(result))
                                    _rb_sha = _m.group(1) if _m else ""
                                    self.on_rewind(_rb_sha)
                        except Exception as e:
                            result = f"Error executing tool {func_name}: {e}"
                    else:
                        import difflib as _dl
                        near = _dl.get_close_matches(func_name, current_tools.keys(), n=3, cutoff=0.4)
                        result = (f"Error: Tool '{func_name}' is not available."
                                  + (f" Closest available: {', '.join(near)}." if near else "")
                                  + " Call one of the tools you were given.")

                    if ran_tool:
                        result = self._post_tool_hooks(func_name, filtered_args, result)

                    # A recovered call ran under a DIFFERENT name than requested.
                    # Without this note the model reads the result as its own
                    # call succeeding — which is how 'append_to_file' resolving
                    # to 'read_file' silently discarded the content to write.
                    if substituted_from:
                        result = str(result) + (
                            f"\n\n[Argent]: no tool named '{substituted_from}' exists; the closest match "
                            f"`{func_name}` was run instead. If that does not match your intent, the action "
                            f"you wanted has NOT happened — call the correct tool now."
                        )

                    # Loop guard: deterministically catch verbatim-repeated calls.
                    # User preference from config ('on', 'off', 'auto') takes priority over model strategy.
                    import config as _cfg
                    lg_mode = getattr(_cfg, "get_loop_guard_mode", lambda: "auto")()
                    if lg_mode == "on":
                        guard_enabled = True
                    elif lg_mode == "off":
                        guard_enabled = False
                    else:
                        guard_enabled = getattr(self.strategy, "is_loop_guard_enabled", lambda: True)()
                    guard_level = self.loop_guard.record(func_name, arguments, str(result), enabled=guard_enabled, category=category)
                    if guard_level:
                        result = str(result) + build_loop_note(guard_level)
                        if guard_level == "stop":
                            loop_guard_stop = True
                        elif guard_level == "nudge":
                            log.warning("LoopGuard: injected sobering nudge for '%s' (%s)", func_name, category)

                    # Semantic loop guard: detect high-level edit-revert cycles and error ping-pong
                    if func_name in ("write_file", "replace_in_file") and not str(result).startswith("Error"):
                        target_file = filtered_args.get("file_path") or filtered_args.get("path")
                        if target_file:
                            sem_level = self.semantic_loop_guard.record_file_edit(target_file)
                            if sem_level:
                                result = str(result) + build_semantic_loop_note(sem_level, "edit_revert")
                                if sem_level == "stop":
                                    loop_guard_stop = True
                    elif func_name in ("run_command", "start_background_command"):
                        cmd_arg = arguments.get("command") if isinstance(arguments, dict) else ""
                        exit_code = 1 if ("Error" in str(result) or "FAILED" in str(result)) else 0
                        sem_level = self.semantic_loop_guard.record_command_result(cmd_arg, str(result), exit_code)
                        if sem_level:
                            result = str(result) + build_semantic_loop_note(sem_level, "error_ping_pong")
                            if sem_level == "stop":
                                loop_guard_stop = True
                    elif func_name == "browser_action" and is_browser_eval_call(arguments):
                        expr_arg = browser_eval_expression(arguments)
                        sem_level = self.semantic_loop_guard.record_browser_eval(expr_arg, str(result))
                        if sem_level:
                            result = str(result) + build_semantic_loop_note(sem_level, "browser_probing")
                            if sem_level == "stop":
                                loop_guard_stop = True

                    # State-changing tools reset the low-level verbatim loop guard and semantic browser tracker
                    _STATE_CHANGING_TOOLS = (
                        "write_file", "replace_in_file",
                        "delete_file", "create_directory", "move_file",
                        "browser_open",
                    )
                    if (func_name in _STATE_CHANGING_TOOLS
                            or (func_name == "browser_action" and is_page_changing_browser_action(arguments))
                            ) and not str(result).startswith("Error"):
                        self.loop_guard.reset()
                        self.semantic_loop_guard.reset_browser_eval()

                    # Auto-Healing Mechanism: trigger only on structured failure
                    # signals (exit codes / tool error prefixes), not on substrings.
                    healing_enabled = getattr(self.strategy, "is_auto_healing_enabled", lambda: True)()
                    if func_name in HEALING_TOOLS and healing_enabled:
                        cmd_arg = arguments.get("command") if isinstance(arguments, dict) else None
                        if detect_tool_failure(func_name, str(result), command=cmd_arg, category=category, enabled=healing_enabled):
                            has_fatal_error = True
                            self.error_retries = getattr(self, 'error_retries', 0) + 1
                            result += build_healing_hint(self.error_retries, func_name=func_name)
                        else:
                            self.error_retries = 0
                        
                    result = compress_tool_result(result, self.model_name, tool_name=func_name)

                    # Cache a real whole-file read (skip the dedup pointer and errors)
                    # so the next unchanged re-read can be short-circuited.
                    if func_name == "read_file" and not result.startswith(("[read_file]", "Error")):
                        self._record_read(filtered_args, result)
                    # Our own successful edit changes the on-disk signature; drop
                    # the entry so the next edit isn't flagged as external.
                    elif func_name in self._EDIT_TOOLS and result.startswith("Successfully"):
                        self._drop_read_cache(filtered_args)

                        # ── LSP auto-diagnostics ──────────────────────────
                        # After every successful edit, run the language server
                        # on the just-modified file and append any errors /
                        # warnings to the tool result.  This gives the model
                        # instant feedback — it can fix a typo before moving
                        # on, instead of discovering it ten calls later.
                        # Non-blocking: any failure is swallowed silently.
                        try:
                            from src.lsp.manager import lsp_manager
                            if lsp_manager.is_available():
                                target_fps = []
                                fp = filtered_args.get("file_path") or filtered_args.get("path")
                                if fp and fp not in target_fps:
                                    target_fps.append(fp)
                                raw_changes = filtered_args.get("changes") or filtered_args.get("changes_json")
                                if raw_changes:
                                    from tools.file_ops import _parse_multi_replace_changes
                                    items, _ = _parse_multi_replace_changes(raw_changes)
                                    if items:
                                        for it in items:
                                            if isinstance(it, dict) and it.get("file_path") and it["file_path"] not in target_fps:
                                                target_fps.append(it["file_path"])

                                for target_fp in target_fps:
                                    # Strict extension gate: skip .md, .txt, .json, .yaml, etc.
                                    language = lsp_manager.detect_language(target_fp)
                                    if not language:
                                        continue
                                    diags = lsp_manager.get_diagnostics(target_fp, timeout=2.0)
                                    if diags:
                                        crit_diags = [d for d in diags if d.get("severity", 1) in (1, 2)]
                                        if crit_diags:
                                            formatted = lsp_manager.format_diagnostics(target_fp, crit_diags)
                                            if formatted and "No issues" not in formatted:
                                                result = result + "\n\n" + formatted
                        except Exception:
                            pass

                    # Diff cards: file editors queue the unified diff of what
                    # they changed — forward each as a structured chunk.
                    from tools._helpers import drain_diff_events
                    for _d in drain_diff_events():
                        yield {"type": "diff", "file": _d["file"], "diff": _d["diff"]}

                    yield {"type": "tool_end", "name": func_name, "result": result}

                    # Phase 4: LLM-path tool outcomes feed the trajectory —
                    # this is the correction signal when the fast-path fell through.
                    try:
                        from src.leeth import trajectory as leeth_trajectory
                        if leeth_trajectory.current_trajectory_id():
                            _llm_ok = not str(result).startswith(("Error", "[DIAGNOSIS"))
                            _args_snap = {k: str(v)[:120] for k, v in list(filtered_args.items())[:4]} if isinstance(filtered_args, dict) else {}
                            leeth_trajectory.record_event("llm_tool_exec", tool=func_name, ok=_llm_ok, args=_args_snap, detail=str(result)[:200])
                    except Exception:
                        pass

                    if func_name == "wait_heartbeat" and isinstance(result, str) and "[HEARTBEAT_REQUEST:" in result:
                        has_auto_signal = True
                    if self._turn_tool_calls % TURN_PROGRESS_EVERY == 0:
                        mins = (time.monotonic() - self._turn_started) / 60
                        yield {"type": "error", "content":
                               f"[Argent: {self._turn_tool_calls} вызовов инструментов, "
                               f"{mins:.0f} мин работы над этим запросом. Ctrl+C — остановить.]"}

                    self.messages.append(provider.format_tool_result(str(result), tool_call.get("id")))

                    # A tool result is a string and cannot carry a picture, so
                    # view_image queues it and it becomes a real user message
                    # here — which is the only place the model can actually
                    # receive one.
                    from vision import build_image_message, drain_images
                    queued = drain_images()
                    if queued:
                        self.messages.append(build_image_message(
                            [(img["data"], img["media_type"]) for img in queued],
                            "Вот запрошенное изображение: "
                            + ", ".join(img["label"] for img in queued)))
                        yield {"type": "content_stream", "content": ""}
                
                self._persist_progress()

                if has_auto_signal:
                    self._log_turn_end("heartbeat_signal", "heartbeat wait requested")
                    break

                if loop_guard_stop:
                    self._log_turn_end("loop guard", "a tool call repeated verbatim")
                    yield {"type": "error", "content": "\n[Loop Guard]: повторяющийся цикл инструментов остановлен — ход завершён принудительно."}
                    break

                # Loop continues to let the model react to tool results
            else:
                # No tool call this turn.
                _normalized_reply = full_content.strip().lower().rstrip(".! \n\t")
                _is_dummy_stop = _normalized_reply in {
                    "no response requested",
                    "no response needed",
                    "no response required",
                    "n/a",
                    "null",
                    "none",
                    "done",
                    "finished",
                } and len(full_content.strip().split()) <= 4

                if full_content and full_content.strip() and not _is_dummy_stop:
                    # The one healthy ending. Logged too, so "it stopped early"
                    # can be told apart from "it finished" without guessing.
                    self._log_turn_end("final answer",
                                       f"{len(full_content)} chars")
                    break

                # Otherwise it produced only reasoning (or nothing) and neither
                # acted nor answered — a premature stop, common with reasoning
                # models. Nudge it to continue, bounded, instead of dead-ending.
                self._no_action_continues = getattr(self, "_no_action_continues", 0) + 1
                if self._no_action_continues > MAX_NO_ACTION_CONTINUES:
                    self._log_turn_end("no answer and no action",
                                       f"{MAX_NO_ACTION_CONTINUES} nudges ignored")
                    yield {"type": "error", "content": (
                        f"\n[System: the model produced no answer and no action after "
                        f"{MAX_NO_ACTION_CONTINUES} nudges — stopping.]"
                    )}
                    break
                yield {"type": "error", "content": (
                    "\n[System: no answer or action produced — nudging the model to continue...]"
                )}
                _nudge_prompt = (
                    f"You emitted a placeholder stop phrase ('{full_content.strip()}') instead of finishing the task. "
                    if _is_dummy_stop else
                    "You stopped after thinking, without producing anything. "
                )
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"{_nudge_prompt}Either CALL A TOOL to take the next concrete action, or "
                        f"WRITE your final answer to the user in Russian now. Do not stop again without doing one of these."
                    ),
                })
                self._trim_history()
                # loop continues — give the model another turn to act or answer.
    
    def _review_leeth_proposal(self, provider, user_text: str, dispatch) -> Optional[Dict[str, Any]]:
        """Active-learning review: ask the main agent LLM to verify a Leeth fast-path proposal.

        Returns {"verdict": accept|reject|partial, "correction_tool": ..., "correction_param": ..., "reason": ...}
        or None when the review could not be produced (the caller then fails safe
        by delegating the query to the normal LLM turn; the decision stays
        pending in the feedback store and is never used for training).
        """
        from src.leeth import feedback as leeth_feedback
        try:
            record = leeth_feedback.get_record(dispatch.decision_id)
            if record is None:
                log.warning("Leeth review: decision record %s not found.", dispatch.decision_id)
                return None
            review_messages = [
                {"role": "system", "content": (
                    "You are the reviewer for Leeth — Argent's fast-path tool-dispatch model. "
                    "Verify its proposal and answer with ONLY a JSON object. Be strict: reject "
                    "conversational, ambiguous or dangerous proposals; correct wrong tools or arguments."
                )},
                {"role": "user", "content": leeth_feedback.format_review_prompt(record)},
            ]
            text = ""
            stream = provider.stream_chat(
                model=self.model_name,
                messages=review_messages,
                tools=None,
                context_window=None,
                temperature=0.0,
            )
            for chunk in stream:
                if isinstance(chunk, dict):
                    text += chunk.get("content", "") or ""
            verdict = leeth_feedback.parse_review_response(text)
            if verdict is None:
                log.warning("Leeth review response could not be parsed (len=%d).", len(text))
            else:
                log.info("Leeth review verdict: %s (%s)", verdict.get("verdict"), verdict.get("reason", ""))
            return verdict
        except Exception as e:
            log.warning("Leeth review call failed: %s", e)
            return None

    def _flatten_tool_messages(self):
        """Convert structured tool_calls in message history to plain-text format.
        Fixes compatibility with models (like Gemini) that require fields 
        Ollama doesn't provide (e.g., thought_signature).
        
        Merges assistant+tool message pairs into a single assistant message
        with the tool interaction described as text."""
        new_messages = []
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                # Build a text representation of the tool calls and their results
                text_parts = []
                if msg.get("thinking"):
                    text_parts.append(f"[Reasoning process]:\n{msg['thinking']}")
                if msg.get("content"):
                    text_parts.append(msg["content"])
                
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", {})
                    text_parts.append(f"[Called tool: {name}({json.dumps(args, ensure_ascii=False)})]")
                
                # Consume following tool-role messages 
                j = i + 1
                while j < len(self.messages) and self.messages[j].get("role") == "tool":
                    tool_result = self.messages[j].get("content", "")
                    # Truncate long results to avoid context bloat
                    if len(tool_result) > 500:
                        tool_result = tool_result[:500] + "..."
                    text_parts.append(f"[Tool result: {tool_result}]")
                    j += 1
                
                new_messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts)
                })
                i = j
            else:
                new_messages.append(msg)
                i += 1
        
        self.messages = new_messages

    @staticmethod
    def _wait_plan(e):
        """(kind, retry_after, detail) when `e` is a failure worth WAITING out — a connection that is gone, a
        provider that is busy, a rate limit with a reset time — else None (bad key, unfit model, quota used up:
        waiting changes nothing, the ordinary error path tells the person)."""
        try:
            from src.agent import service_llm
            from src.agent.netstate import is_transport_error
            info = service_llm.classify(e)
            if is_transport_error(e):
                return "transport", info.retry_after, info.detail
            if info.kind is service_llm.Kind.RATE_LIMITED:
                return "rate_limited", info.retry_after, info.detail
            if info.kind is service_llm.Kind.BUSY:
                return "busy", info.retry_after, info.detail
        except Exception:
            log.debug("could not classify %r", e, exc_info=True)
        return None

    #: the conversation is written to disk as it grows, not only every few turns and at a clean exit — a crash,
    #: a power cut or a killed terminal then costs one step instead of the last several turns
    autosave_enabled = True
    _AUTOSAVE_MIN_GAP_S = 2.0
    _last_persist = None

    def session_state(self) -> Dict[str, Any]:
        """What belongs to the conversation but is not a message: today the todo list."""
        state: Dict[str, Any] = {}
        try:
            from src.agent.todos import TODOS
            todos = TODOS.snapshot("main")
            if todos:
                state["todos"] = todos
        except Exception:
            pass
        return state

    def _persist_progress(self, force: bool = False) -> None:
        """Save the session (throttled; `force` for the points that must not be skipped). Never raises."""
        if not getattr(self, "autosave_enabled", True) or not getattr(self, "messages", None):
            return
        now = time.monotonic()
        if not force and self._last_persist is not None and now - self._last_persist < self._AUTOSAVE_MIN_GAP_S:
            return
        self._last_persist = now
        try:
            from session import save_session
            saved = save_session(self.messages, {"model": self.model_name, "provider": self.provider},
                                 session_id=getattr(self, "session_id", None), state=self.session_state())
            if saved:
                self.session_id = saved
        except Exception as e:
            log.debug("autosave skipped: %s", e)

    def repair_history(self, reason: str = "user"):
        """Restore message-history invariants after an interrupted turn.

        If any assistant message carries tool_calls that lack matching
        tool results (e.g. user pressed Ctrl+C mid-execution or session
        was restored after crash/interruption), append synthetic results
        so the next request doesn't violate the API contract.

        `reason` words the note left when the history ends on an unanswered user message:
        "user" (Ctrl+C), "network" (the connection did not come back).
        Returns how many repairs were made (0: the history was whole).
        """
        if not self.messages:
            return 0

        try:
            from providers import create_provider
            provider = create_provider()
        except Exception:
            provider = None

        note = "Interrupted (session restored / cancelled). The tool did not finish."
        total_repaired = 0

        i = 0
        while i < len(self.messages):
            msg = self.messages[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
                j = i + 1
                seen_call_ids = set()
                while j < len(self.messages) and self.messages[j].get("role") == "tool":
                    tid = self.messages[j].get("tool_call_id")
                    if tid:
                        seen_call_ids.add(tid)
                    j += 1

                missing = [tc for tc in tool_calls if tc.get("id") and tc["id"] not in seen_call_ids]
                if not missing and len(seen_call_ids) < len(tool_calls):
                    missing = tool_calls[len(seen_call_ids):]

                if missing:
                    insert_pos = j
                    for tc in missing:
                        cid = tc.get("id") if isinstance(tc, dict) else None
                        if provider:
                            synth = provider.format_tool_result(note, cid)
                            if cid and isinstance(synth, dict) and "tool_call_id" not in synth:
                                synth["tool_call_id"] = cid
                        else:
                            synth = {"role": "tool", "content": note}
                            if cid:
                                synth["tool_call_id"] = cid
                        self.messages.insert(insert_pos, synth)
                        insert_pos += 1
                        total_repaired += 1
                    i = insert_pos - 1
            i += 1

        # Invariant: a history must never end on an un-responded user message,
        # otherwise the subsequent turn produces two consecutive user messages,
        # violating strict provider schemas (Anthropic, Mistral, Bedrock).
        if self.messages and self.messages[-1].get("role") == "user":
            note_text = ("[Ход прерван: пропала связь с провайдером, ответ не получен. Когда пользователь напишет "
                         "«продолжай» — продолжи выполнение его последнего запроса.]"
                         if reason == "network" else
                         "[Выполнение прервано пользователем (Ctrl+C / Cancel)]")
            self.messages.append({"role": "assistant", "content": note_text})
            total_repaired += 1

        if total_repaired:
            log.info("repair_history: appended %d synthetic tool result(s)", total_repaired)
        return total_repaired

    def clear_history(self):
        memory.clear()
        self.loop_guard.reset()
        if hasattr(self, "semantic_loop_guard"):
            try:
                self.semantic_loop_guard.reset()
            except Exception:
                pass
        self._read_cache.clear()
        self._checkpoint_turn_indices.clear()
        getattr(self, "_turn_records", []).clear()
        try:
            from src.agent.todos import TODOS
            TODOS.clear("main")
        except Exception:
            pass
        self._turn_checkpoint_done = False
        self._current_turn_msg_index = 0
        self._stop_requested = False

        # Drain unconsumed pending images so they don't leak into new chat
        try:
            from vision import drain_images
            drain_images()
        except Exception:
            pass

        # Build fresh system prompt from current AGENTS.md and active tools
        self.messages = [{"role": "system", "content": self.build_system_prompt()}]
        # The thrash guard counts compactions of ONE conversation; this is a new one.
        try:
            from src.agent.compaction_log import reset_compaction_history
            reset_compaction_history()
        except Exception:
            pass
        # A cleared history is a different conversation, so the next save must
        # start a new session rather than overwrite the one just abandoned.
        self.session_id = None

    def get_context_usage(self) -> Dict[str, Any]:
        """Calculates current context usage statistics."""
        history_tokens = sum(self._estimate_message(m) for m in self.messages)
        max_tokens = get_context_window()
        percent = (history_tokens / max_tokens) * 100 if max_tokens > 0 else 0
        result = {
            "tokens": history_tokens,
            "max": max_tokens,
            "percent": min(percent, 100),
            "messages": len(self.messages)
        }
        # Cache for the live bottom-toolbar context meter: it must read this on
        # every keystroke, so it reads this cached dict instead of recomputing
        # the O(history) token sum each time. Refreshed here, which already runs
        # once per turn.
        self._last_context_usage = result
        return result

    def get_context_breakdown(self) -> Dict[str, Any]:
        """Token breakdown of what the next request will carry: the system
        prompt (incl. AGENTS.md), the tool schemas actually sent for this tier,
        and the conversation history. Lets the user see where the budget goes."""
        import json
        from tool_profiles import slim_tools_for_category

        system_tokens = self._estimate_tokens(str(self.messages[0])) if self.messages else 0
        history_tokens = sum(self._estimate_message(m) for m in self.messages[1:])

        tool_tokens = 0
        tool_count = 0
        if self.strategy.supports_native_tools():
            schemas = get_tool_schemas(include_hidden=False)
            category = get_model_size_category(self.model_name)
            is_moe = category == "moe" or is_moe_model(self.model_name)
            if is_moe:
                category = "moe"
            if not is_moe:
                keep = set(slim_tools_for_category([t["function"]["name"] for t in schemas], category))
                sent = [t for t in schemas if t["function"]["name"] in keep]
            else:
                sent = schemas

            wants_minify = getattr(self.strategy, "wants_minified_schemas", lambda: False)()
            if (category in ("tiny", "small") or wants_minify) and sent:
                from tool_profiles import minify_tool_schemas
                sent = minify_tool_schemas(sent)

            tool_tokens = self._estimate_tokens(json.dumps(sent))
            tool_count = len(sent)

        max_tokens = get_context_window()
        total = system_tokens + tool_tokens + history_tokens
        return {
            "system": system_tokens,
            "tools": tool_tokens,
            "tool_count": tool_count,
            "history": history_tokens,
            "total": total,
            "max": max_tokens,
            "percent": min((total / max_tokens) * 100, 100) if max_tokens else 0,
        }

    def inject_context(self):
        """Clear conversation history but preserve the full system prompt.
        Used by Project Brain to give the model a fresh context window."""
        self.messages = [
            {"role": "system", "content": self.messages[0]["content"]}
        ]

class ArgentSubAgent(ArgentAgent):
    """
    A specialized, stateless sub-agent designed for isolated, precise tasks.
    Unlike the main agent, it doesn't maintain long-term chat history and 
    operates under strict tool restrictions.
    """
    fast_path_allowed = False
    autosave_enabled = False       # a helper's scratch conversation is not a session the person could resume

    def __init__(self, role: str, task: str, tools_override: List[str] = None,
                 model_override: str = None, provider_override: str = None):
        super().__init__()
        # Optionally run this sub-agent on a different model/provider than the
        # main agent (e.g. a stronger/independent critic). Refresh the tier so
        # strategy, context window and history limits match the chosen model.
        if provider_override:
            self.provider = provider_override
        if model_override:
            self.model_name = model_override
        if model_override or provider_override:
            self.refresh_tier()
        self.role = role
        self.task = task
        self.tools_override = tools_override
        
        # Override system prompt for specific role
        from src.agent.critic import CRITIC_SYSTEM
        from prompt_compressor import drop_unavailable_tool_lines
        role_prompts = {
            "Coder": (
                f"You are a specialized Coder sub-agent on {host_os.system()}. Your goal is to IMPLEMENT specific code as described.\n"
                "- NEVER use write_file to overwrite existing large files (>150 lines). You MUST use replace_in_file (pass `changes` to apply several edits at once) to apply targeted patches.\n"
                "- Use forward slashes (/) in file paths (e.g. C:/path/file.py).\n"
                "- Be concise and follow project style conventions."
            ),
            "Researcher": "You are a specialized Research sub-agent. Your goal is to gather technical information and documentation. Synthesize your findings into a clear report.",
            "Reviewer": "You are a specialized Code Reviewer. Your goal is to find bugs, security vulnerabilities, and architectural flaws in the provided code.",
            "DocWriter": "You are a specialized Documentation sub-agent. Your goal is to write clear, accurate markdown documentation for the project.",
            "BrowserWorker": (
                "You are a specialized BrowserWorker sub-agent. Your goal is to navigate web pages, inspect elements, and interact with the browser.\n"
                "- Open pages with `browser_open` and inspect interactive elements or screenshots with `browser_inspect`.\n"
                "- Interact with elements using `browser_action` (action='click', 'type', 'select', 'scroll', 'press_key', 'upload_file', 'wait_for').\n"
                "- Verify the resulting state and return a clear factual summary of your actions and findings."
            ),
            "Critic": CRITIC_SYSTEM,
        }
        
        language_rule = (
            "# CRITICAL: LANGUAGE RULE\n"
            "- You MUST respond and perform ALL internal reasoning (thinking process) in the EXACT SAME LANGUAGE as the user request or task.\n"
            "- If the task or user writes in Russian, you THINK in Russian and REPLY in Russian.\n"
        )
        
        role_body = role_prompts.get(role, f"You are a specialized {role} sub-agent.")
        custom_system = f"{language_rule}\n# ROLE: {role}\n{role_body}\n\n## PROTOCOL:\n- Focus ONLY on the assigned task.\n- Return a final summary to the Supervisor when finished.\n"
        
        if tools_override is not None:
            custom_system = drop_unavailable_tool_lines(custom_system, set(tools_override))
        
        # Replace main system prompt
        self.messages[0] = {"role": "system", "content": custom_system}

    def _refresh_system_prompt(self, allowed_tools=None) -> bool:
        """A sub-agent's system prompt IS its role, so it is never rebuilt.

        The inherited version regenerates the main agent's prompt and writes it
        over messages[0] — which happens on the sub-agent's very first turn,
        before it has produced anything. Every role (Critic, Reviewer, Coder…)
        was silently dissolved into a generic agent at that point.
        """
        return False

    def execute(self) -> str:
        """Run the sub-agent loop until completion and return the final report."""
        from ui import console
        console.print(f"[bold cyan]Sub-Agent ({self.role}) starting task...[/bold cyan]")
        
        _ACTIVE_SUBAGENTS.add(self)
        # Its checklist is its own: it must not overwrite the supervisor's (see src.agent.todos).
        from src.agent.todos import use_scope
        own_list = use_scope(f"sub-{id(self)}")
        own_list.__enter__()
        try:
            final_answer = ""
            last_error = ""
            # The person waits for this report inside their own turn: its requests are second in the model's
            # queue, right behind the main model's. A worker process stays a worker (`capped`).
            with governor.capped(governor.P_ADVISOR):
                for chunk in self.process_user_input(f"Start task: {self.task}", allowed_tools=self.tools_override):
                    ctype = chunk.get("type")
                    if ctype == "content_stream":
                        final_answer += chunk.get("content", "")
                    elif ctype == "content_replace":
                        final_answer = chunk.get("content", "")
                    elif ctype == "error":
                        content = str(chunk.get("content", "")).strip()
                        # Informational progress notices, auto-continue nudges, and loop guard warnings are non-fatal
                        if any(content.startswith(p) for p in ("[Argent:", "[System:", "[Loop Guard]:")):
                            console.print(f"[dim yellow]{content}[/dim yellow]")
                            continue
                        last_error = content

            if not final_answer.strip() and last_error:
                return f"Sub-Agent Error: {last_error}"

            return final_answer or "Sub-agent finished without producing text output."
        finally:
            _ACTIVE_SUBAGENTS.discard(self)
            own_list.__exit__(None, None, None)


def run_plan_critique(target: str, goal: str = None, what: str = "plan / idea") -> str:
    """Spawn an independent Critic sub-agent (cleared context, no tools) on the
    given material and return its findings.

    Uses the configured critic model/provider when set, else the current one. If
    the chosen critic model/provider fails to run, falls back to the current
    model so a misconfiguration never blocks the user.
    """
    from src.agent.critic import build_critique_task
    from config import get_critic_model, get_critic_provider

    task = build_critique_task(target, goal=goal, what=what)
    model = get_critic_model() or None
    provider = get_critic_provider() or None
    try:
        return ArgentSubAgent(
            "Critic", task, tools_override=[],
            model_override=model, provider_override=provider,
        ).execute()
    except Exception as e:
        if model or provider:
            from ui import console
            console.print(f"[dim yellow]Critic on {provider or ''}:{model} failed ({e}); falling back to current model...[/dim yellow]")
            log.warning("Critic on %s/%s failed (%s); falling back to current model",
                        provider, model, e)
            return ArgentSubAgent("Critic", task, tools_override=[]).execute()
        raise
