import sys
import os
import json
import inspect
import platform
from pathlib import Path
from typing import List, Dict, Any, Generator

from tools import TOOL_SCHEMAS, AVAILABLE_TOOLS, get_tool_schemas, get_available_tools
from config import (
    get_current_model, get_obsidian_vault, 
    get_hooks_dir, get_autonomous_plugins_enabled,
    get_context_window, get_provider, get_mcp_servers
)
from providers import create_provider, ProviderError
from logger import get_logger
from hook_manager import hook_manager
from prompt_compressor import compress_system_prompt, compress_tool_result
from tool_recovery import recover_tool_call
from memory_manager import memory
from config import get_model_size_category

# Newly refactored modules
from src.agent.strategy import get_model_strategy
from src.agent.parser import parse_raw_tool_call
from src.agent.trimmer import estimate_tokens
from src.agent.healing import detect_tool_failure, build_healing_hint, HEALING_TOOLS
from src.agent.constrained import build_step_schema, build_tool_catalog, StepStreamExtractor
from src.agent.loop_guard import LoopGuard, build_loop_note
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

class ArgentAgent:
    def build_system_prompt(self) -> str:
        category = get_model_size_category(self.model_name)
        is_small = category in ("tiny", "small")
        
        prompt_parts = []
        
        prompt_parts.append(f"""# CRITICAL: LANGUAGE RULE
- You MUST respond and perform ALL internal reasoning (thinking process) in the EXACT SAME LANGUAGE the user used in their request. This is your highest priority rule.
- If the user writes in Russian, you THINK in Russian and REPLY in Russian.

# ROLE: Argent Coder
You are an autonomous AI software engineer. You design, build, and debug software with precision and speed on {platform.system()}.""")

        try:
            cwd = Path.cwd()
            prompt_parts.append(f"## REPOSITORY MAP (Current Directory: {cwd})")
            
            items = []
            dirs = []
            files = []
            for item in cwd.iterdir():
                if item.name in ('.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode'):
                    continue
                if item.is_dir():
                    dirs.append(item.name + "/")
                else:
                    files.append(item.name)
            
            dirs.sort()
            files.sort()
            
            map_str = ""
            if not dirs and not files:
                map_str = "(Empty directory)"
            else:
                all_items = dirs + files
                if len(all_items) > 30:
                    map_str = "\n".join([f"- {x}" for x in all_items[:30]])
                    map_str += f"\n- ... and {len(all_items) - 30} more items. Use list_directory to see all."
                else:
                    map_str = "\n".join([f"- {x}" for x in all_items])
                    
            prompt_parts.append(map_str)
        except Exception:
            pass

        if is_small:
            prompt_parts.append(f"""## 1. OPERATIONAL PROTOCOL
- **Tool-First**: Invoke tools immediately via JSON when needed.
- **Ask Before Guessing**: Use `ask_user_questions` to clarify ambiguous requirements with structured options.
- **Anti-Lazy**: Run commands and write/edit files yourself.
- **File Editing**: NEVER use write_file to overwrite existing large files (>150 lines). You MUST use replace_in_file or multi_replace_in_file_chunk to apply targeted patches.
- **Proactive Search**: Use `search_web` for technical info.
- **Persistence**: Do NOT stop after a single tool call. If the task requires multiple steps (read → edit → verify), execute ALL steps in a single response. Keep calling tools until the task is FULLY complete.
- **Strict Environment**: Use {platform.system()}-native commands. On Windows the default shell is PowerShell — `&&`/`||` chains are auto-routed to cmd, so prefer `;` or separate calls. If a command fails, read the [DIAGNOSIS] line in the output.""")
        else:
            prompt_parts.append(f"""## 1. OPERATIONAL PROTOCOL
- **Tool-First**: YOU are the only one with tool access. Invoke tools immediately via JSON.
- **Ask Before Guessing**: If a user's request is ambiguous or lacks details, you MUST use the `ask_user_questions` tool to prompt them with structured options before writing code. Do NOT just ask questions in plain text chat.
- **Anti-Lazy**: Never ask the user to run code or copy-paste. Use `run_command` and `write_file` yourself.
- **File Editing**: NEVER use write_file to overwrite existing large files (>150 lines). You MUST use replace_in_file or multi_replace_in_file_chunk to apply targeted patches.
- **Proactive Search**: Always use `search_web` for technical info, documentation, or current events.
- **Persistence**: Do NOT stop after a single tool call. If the task requires multiple steps (read → edit → verify), execute ALL steps in a single response without waiting for user input. Keep calling tools until the task is FULLY complete.
- **Testing**: NEVER test logic or GUI apps by running `python app.py` via `run_command` (it will block). You MUST write and run `pytest` tests, or use `start_background_command`.
- **Self-Correction**: If a tool fails, analyze the error and fix it proactively. Do not apologize.
- **Strict Environment**: Use {platform.system()}-native commands ONLY (NOT unix commands like 'ls' or 'grep'). On Windows the default shell is **PowerShell** (so `Select-Object`, `Get-ChildItem`, `$env:` work); `&&`/`||` chains are auto-routed to cmd, but prefer `;` or separate commands. If a command fails, READ the `[DIAGNOSIS]` line in its output before assuming the cause.""")

        if category == "tiny" and self.provider == "ollama":
            prompt_parts.append("""## RESPONSE FORMAT (STRICT JSON STEPS)
Every response is EXACTLY ONE JSON object, one of:
1. {"tool": {"name": "<tool_name>", "arguments": {...}}} — perform an action.
2. {"reply": "<final answer to the user>"} — ONLY when the task is fully complete or you must ask the user something.
Never mix plain text with JSON. Prefer "tool" steps until the task is done.
Example: {"tool": {"name": "read_file", "arguments": {"file_path": "main.py"}}}""")

        auto_plugins = get_autonomous_plugins_enabled()
        if auto_plugins or not is_small:
            hooks_dir = get_hooks_dir()
            prompt_parts.append(f"""## 2. PLUGIN DEVELOPMENT
- **HOOKS_DIR**: `{hooks_dir}`
- **AUTONOMOUS_EXTENSION**: {'ENABLED' if auto_plugins else 'DISABLED'}
- **Standard Plugin Development**:
  1. Use `create_plugin` to write new logic and `delete_plugin` to remove it (handles ./plugins/ and reloading).
  2. Define a function `command_NAME(*args)` for slash commands (e.g., `command_hello` -> `/hello`).
  3. Use events: `on_startup()`, `pre_prompt(text)`, `on_tool_call(func_name, args)`, `post_response(text)`, `on_chat_saved(file_path)`.
  4. Always use `from ui import console` for output.""")

        prompt_parts.append("""## 3. SKILLS SYSTEM
- You have access to instruction-based extensions stored in markdown files.
- Use `list_skills` to discover skills. Use `read_skill` to read and follow instructions.
- **Be proactive**: after you work out a non-trivial, repeatable workflow for THIS project (a build/deploy sequence, a multi-step fix pattern, project-specific conventions), persist it with `create_skill` so it can be reused. Don't wait to be asked.""")

        if not is_small:
            prompt_parts.append("""## 4. PLANNING MODE & ARTIFACTS
- For complex changes, you MUST create an implementation plan before writing any code.
- Use `create_artifact("implementation_plan.md", content)` to present your plan to the user.
- Then, use `request_user_approval("I have created an implementation plan. Please review and approve.")` to PAUSE execution and wait for the user to confirm.
- NEVER start making massive changes without the user's explicit approval.
- Use `create_artifact("task.md", content)` to track your progress after approval.""")

        if not is_small:
            prompt_parts.append("""## 5. UI & TERMINOLOGY STANDARDS
- **"Panel"**: Always refers to `rich.panel.Panel` for terminal UI. NEVER start web servers or use web-dashboard libraries unless building a web app.
- **"Table"**: Always refers to `rich.table.Table`.
- **Output**: Use `console.print()` or `print_system()` for beautiful terminal results.""")

        if is_small:
            prompt_parts.append("""## 6. THINK & VERIFY PROTOCOL
- **Outcome Analysis**: Verify if tool results truly move you closer to the goal.
- **Proactive Verification**: After writing files or executing commands, verify they work as intended.
- **Self-Correction**: If stuck in a loop, stop, rethink, and explain to the user.""")
        else:
            prompt_parts.append("""## 6. THINK & VERIFY PROTOCOL
- **Outcome Analysis**: After EACH tool call, analyze if the result truly moves you closer to the goal.
- **False Success**: "Requirement already satisfied" or "Exit code: 0" does NOT always mean success. If a tool reports success but the problem persists, try a different approach.
- **Proactive Verification**: After installing things or writing complex files, use `run_command` or `read_file` to VERIFY they work as intended.
- **Self-Correction**: If you are stuck in a loop, STOP. Rethink your strategy. Explain your new reasoning to the user.""")

        prompt_parts.append("""## 7. COMMUNICATION
- **Language**: Follow the CRITICAL LANGUAGE RULE at the top of this prompt.
- **Transparency**: Briefly state your reasoning before executing tools.
- **Visuals**: Use `create_svg_image` to explain complex concepts or UI mockups via browser.""")



        # AGENTS.md is the project's persistent memory: loaded into context
        # every turn, editable by the user, and maintainable by the agent.
        # The cap scales with the tier so a small model isn't drowned by it.
        AGENTS_MD_LIMIT = {"tiny": 2000, "small": 4000}.get(category, 12000)
        agents_md_paths = [
            Path(".argent/AGENTS.md"),
            Path("AGENTS.md"),
        ]
        for p in agents_md_paths:
            if p.exists():
                try:
                    agents_content = p.read_text(encoding="utf-8").strip()
                    if agents_content:
                        truncated_note = ""
                        if len(agents_content) > AGENTS_MD_LIMIT:
                            agents_content = agents_content[:AGENTS_MD_LIMIT]
                            truncated_note = (
                                f"\n\n[...truncated at {AGENTS_MD_LIMIT} chars — this file is too long; "
                                f"trim it to keep it dense.]"
                            )
                        prompt_parts.append(
                            f"## PROJECT INSTRUCTIONS (from {p})\n"
                            f"This is your persistent project memory. Keep it accurate: when you learn "
                            f"something durable about this codebase (architecture, conventions, commands), "
                            f"update {p} with replace_in_file/write_file.\n\n"
                            f"{agents_content}{truncated_note}"
                        )
                    break
                except Exception:
                    pass

        try:
            from mcp_client import mcp_client
            mcp_servers = get_mcp_servers()
            if mcp_servers:
                mcp_section = "## MCP SERVERS (External Tool Integration)\n"
                mcp_section += "You have access to external tool servers via `call_mcp_tool(server_name, tool_name, arguments_json)`.\n"
                mcp_section += "When the user asks to do something related to these servers, use call_mcp_tool AUTOMATICALLY.\n\n"
                for srv_cfg in mcp_servers:
                    srv_name = srv_cfg["name"]
                    stype = srv_cfg.get("type", "stdio")
                    endpoint = srv_cfg.get("url") or f"{srv_cfg.get('command', '?')} {' '.join(srv_cfg.get('args', []))}".strip()
                    mcp_section += f"### Server: `{srv_name}` ({stype})\n"
                    mcp_section += f"- **Endpoint**: {endpoint}\n"
                    try:
                        tools = mcp_client.list_tools(srv_name)
                        valid_tools = [t for t in tools if "name" in t and "error" not in t]
                        if valid_tools:
                            mcp_section += "- **Tools**:\n"
                            for t in valid_tools:
                                tname = t["name"]
                                desc = t.get("description", "").split(".")[0]
                                schema = t.get("inputSchema", t.get("parameters", {}))
                                params = schema.get("properties", {})
                                param_str = ", ".join(f'"{p}": value' for p in params)
                                mcp_section += f"  - `{tname}`: {desc}\n"
                                if param_str:
                                    example_args = "{" + param_str + "}"
                                    mcp_section += f"    -> call_mcp_tool(\"{srv_name}\", \"{tname}\", '{example_args}')\n"
                                else:
                                    mcp_section += f"    -> call_mcp_tool(\"{srv_name}\", \"{tname}\", '{{}}')\n"
                        else:
                            mcp_section += "- **Tools**: (could not fetch — server may be offline)\n"
                    except Exception:
                        mcp_section += "- **Tools**: (connection failed — server may be offline)\n"
                    mcp_section += "\n"
                prompt_parts.append(mcp_section)
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
                    prompt_parts.append(
                        "## KNOWLEDGE BASES\n"
                        f"Indexed documentation is available: {', '.join(kb_names)}. "
                        "For ANY question about these libraries/APIs, call `semantic_search` FIRST and base your "
                        "answer on the returned snippets — do NOT answer API/method questions from memory, it leads "
                        "to hallucinated signatures."
                    )
        except Exception:
            pass

        # Lightweight reminder of live background processes so the model knows
        # they exist (and can recover PIDs) even after history summarization
        # drops the original "PID: N" tool results.
        try:
            from tools import ACTIVE_PROCESSES, ACTIVE_PROCESSES_LOCK
            with ACTIVE_PROCESSES_LOCK:
                bg_count = len(ACTIVE_PROCESSES)
            if bg_count:
                prompt_parts.append(
                    f"## BACKGROUND PROCESSES\n"
                    f"You have {bg_count} background process(es) running this session. "
                    f"Call `list_background_commands` to see their PIDs, status and commands "
                    f"before reading output or stopping them — do NOT guess a PID."
                )
        except Exception:
            pass

        full_prompt = "\n\n".join(prompt_parts)
        return compress_system_prompt(full_prompt, self.model_name)

    def __init__(self, max_history_messages: int = None):
        self.model_name = get_current_model()
        self.provider = get_provider()
        self.refresh_tier()

        if max_history_messages is not None:
            self.max_history_messages = max_history_messages

        self.loop_guard = LoopGuard()
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]

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
        self.refresh_tier()

    def _estimate_tokens(self, text: str) -> int:
        return estimate_tokens(text, self.model_name, self.provider)

    def _trim_history(self):
        self.max_context_tokens = get_context_window()

        from config import get_strip_reasoning, get_max_generation_tokens
        from src.agent.trimmer import clean_messages_for_llm
        from src.agent.context_limit import effective_history_budget
        strip_enabled = get_strip_reasoning()

        cleaned = clean_messages_for_llm(self.messages, strip_enabled)

        # Trim history into the room left AFTER the system prompt, the tool
        # schemas (sent alongside) and the model's response — not against the
        # whole window — so the assembled request fits under a hard limit.
        sys_tokens = self._estimate_tokens(str(self.messages[0])) if self.messages else 0
        response_reserve = min(get_max_generation_tokens() or 1024, 2048)
        budget = effective_history_budget(self.max_context_tokens, sys_tokens, response_reserve)

        history_tokens = sum(self._estimate_tokens(str(m)) for m in cleaned[1:])
        msg_count = len(cleaned) - 1

        if msg_count <= self.max_history_messages and history_tokens <= budget:
            return

        self.messages = self.strategy.trim_history(
            self.messages, self.model_name, self.max_history_messages, budget
        )

    def _parse_raw_tool_call(self, content: str) -> dict | None:
        return parse_raw_tool_call(content)

    def _salvage_truncated_write(self, tool_calls_accumulator):
        """If the truncated tool call was a file write, persist the partial
        content via append_to_file and return (file_path, lines_written, tail).
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
            from tools.file_ops import append_to_file
            result = append_to_file(file_path, kept)
            if isinstance(result, str) and result.startswith("Error"):
                log.warning("Salvage append failed: %s", result)
                return None
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
        """Compact trailing reminder for long contexts (local models).

        Beyond the goal, it re-pins a small working-memory slice — what's
        already been done and what has already failed — because the full
        working memory (memory.build_context_note) is only injected on a hard
        context reset, so on a long-but-not-overflowing conversation a weak
        model otherwise never sees it and redoes finished work or retries a
        known-bad approach. Kept deliberately short: it's re-injected on every
        long turn, so only the two highest-signal lists (recent completions,
        recent failures) are included, tightly bounded."""
        d = memory.data
        obj = d.get("objective")
        task = d.get("current_task")
        completed = d.get("completed") or []
        errors = d.get("errors_encountered") or []
        if not (obj or task or completed or errors):
            return None
        parts = ["[REMINDER — do not lose the goal]"]
        if obj:
            parts.append(f"OBJECTIVE: {obj}")
        if task and task != obj:
            parts.append(f"CURRENT TASK: {task}")
        if completed:
            parts.append("ALREADY DONE (do not redo): " + "; ".join(completed[-3:]))
        if errors:
            parts.append("KNOWN FAILURES (do not retry these): " + "; ".join(errors[-2:]))
        return "\n".join(parts)

    def process_user_input(self, user_text: str, allowed_tools: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Process the user input and yield chunks of response or tool activity.
        Supports streaming generation.
        """
        self.messages[0]["content"] = self.build_system_prompt()
        self.messages.append({"role": "user", "content": user_text})
        
        if not user_text.startswith("/"):
            if not memory.data.get("objective"):
                memory.set_objective(user_text)
            else:
                memory.set_current_task(user_text[:200])
            self._maybe_auto_retrieve(user_text)

        self._trim_history()
        self._truncate_continues = 0
        self._salvage_continues = 0
        self._ctx_overflow_retries = 0
        self._no_action_continues = 0

        # One provider instance per turn: tool-result formatting and retries
        # below reuse it instead of re-creating a provider on every call.
        try:
            provider = create_provider(self.provider)
            validation_error = provider.validate_config()
            if validation_error:
                yield {"type": "error", "content": validation_error}
                return
        except Exception as e:
            yield {"type": "error", "content": f"Provider error: {e}"}
            return

        # Grammar-constrained steps: tiny models on Ollama get the step schema
        # enforced at the decoder level — malformed tool JSON becomes impossible.
        step_schema = None
        if (self.strategy.wants_constrained_decoding()
                and provider.supports_constrained_decoding()
                and not getattr(self, "_constrained_unsupported", False)):
            step_schema = build_step_schema()

        while True:
            # Variables to accumulate the streamed response
            full_content = ""
            full_reasoning = ""
            tool_calls_accumulator = []
            is_building_raw_tool = False
            raw_tool_buffer = ""
            raw_tool_char_count = 0
            reasoning_tag_active = False
            is_truncated = False
            constrained_extractor = StepStreamExtractor() if step_schema is not None else None
            loop_guard_stop = False
            START_TAGS = ["<thought>", "<think>", "<reasoning>"]
            END_TAGS = ["</thought>", "</think>", "</reasoning>"]
            
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
                # tool choice. Larger models keep the full set.
                from tool_profiles import slim_tools_for_category
                category = get_model_size_category(self.model_name)
                _slim_names = set(slim_tools_for_category([t["function"]["name"] for t in active_tools], category))
                active_tools = [t for t in active_tools if t["function"]["name"] in _slim_names]

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

                # Objective anchor: re-pin the goal at the end of long
                # histories, where small models actually look.
                if self.strategy.wants_objective_anchor() and len(cleaned_messages) > 8:
                    anchor = self._build_objective_anchor()
                    if anchor:
                        cleaned_messages = cleaned_messages + [{"role": "system", "content": anchor}]

                from config import get_temperature
                temp = get_temperature()

                response_stream = provider.stream_chat(
                    model=self.model_name,
                    messages=cleaned_messages,
                    tools=active_tools,
                    context_window=self.max_context_tokens if self.provider not in ("zai", "openrouter") else None,
                    temperature=temp,
                    format_schema=step_schema,
                )

                for chunk in response_stream:
                    if chunk.get("truncated"):
                        is_truncated = True

                    usage_data = chunk.get("usage")
                    if usage_data:
                        yield {"type": "usage", "data": usage_data}

                    thinking_chunk = chunk.get("thinking", "")
                    if thinking_chunk:
                        full_reasoning += thinking_chunk
                        yield {"type": "thinking_stream", "content": thinking_chunk}

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
                        # Logic to handle tags that might be split across chunks
                        # and redirect content to reasoning if a tag is active.
                        temp_content = content_chunk
                        
                        # Check for START TAGS
                        for tag in START_TAGS:
                            if tag in temp_content:
                                # Split: everything before tag is content, anything after is reasoning
                                parts = temp_content.split(tag, 1)
                                if parts[0] and not reasoning_tag_active and not is_building_raw_tool:
                                    yield {"type": "content_stream", "content": parts[0]}
                                    full_content += parts[0]
                                
                                reasoning_tag_active = True
                                temp_content = parts[1]
                                break
                        
                        # Check for END TAGS
                        for tag in END_TAGS:
                            if tag in temp_content:
                                # Split: everything before tag is reasoning, after is content
                                parts = temp_content.split(tag, 1)
                                if parts[0] and reasoning_tag_active:
                                    full_reasoning += parts[0]
                                    yield {"type": "thinking_stream", "content": parts[0]}
                                
                                reasoning_tag_active = False
                                temp_content = parts[1]
                                break
                        
                        if reasoning_tag_active:
                            full_reasoning += temp_content
                            yield {"type": "thinking_stream", "content": temp_content}
                        else:
                            if temp_content:
                                full_content += temp_content
                                raw_tool_buffer += temp_content
                                
                                # Detect if we are building a raw JSON tool call
                                if '```json' in raw_tool_buffer or '{\n  "name":' in raw_tool_buffer:
                                    is_building_raw_tool = True
                                
                                if not is_building_raw_tool:
                                    yield {"type": "content_stream", "content": temp_content}
                                else:
                                    # Signal progress so the UI can show a spinner
                                    yield {"type": "tool_generating", "name": "?", "bytes": len(raw_tool_buffer), "delta": temp_content}
                            
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
                            context_window=self.max_context_tokens if self.provider not in ("zai", "openrouter") else None,
                            temperature=temp,
                        )
                        for chunk in fallback_stream:
                            content_chunk = chunk.get("content", "")
                            if content_chunk:
                                full_content += content_chunk
                                yield {"type": "content_stream", "content": content_chunk}
                    except Exception as retry_e:
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
                elif isinstance(e, ProviderError):
                    yield {"type": "error", "content": str(e)}
                    break
                else:
                    yield {"type": "error", "content": f"Error: {e}"}
                    break

            # End of stream — parse tool call arguments from strings to dicts.
            if tool_calls_accumulator and not is_truncated:
                for tc in tool_calls_accumulator:
                    if isinstance(tc["function"]["arguments"], str):
                        try:
                            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError as e:
                            log.warning("Failed to parse tool args for %s: %s", tc["function"]["name"], e)
                            tc["function"]["arguments"] = {}

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
                            f"append_to_file('{file_path}', ...), starting exactly after this tail:\n"
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
                    yield {"type": "error", "content": (
                        f"\n[System: Generation hit the length limit {self._truncate_continues} times in a row — "
                        "auto-continue stopped. Increase the context window or split the task into smaller pieces.]"
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
                            "Produce the result in SMALLER STEPS instead: create the file with "
                            "`write_file` containing only the FIRST part of the content, then "
                            "extend it with several `append_to_file` calls. Keep each call small."
                        )
                    })
                else:
                    msg_to_append = {"role": "assistant", "content": full_content}
                    if full_reasoning:
                        msg_to_append["thinking"] = full_reasoning
                    self.messages.append(msg_to_append)
                    yield {"type": "error", "content": "\n[System: Model generation was truncated by max_tokens limit. Auto-continuing...]"}
                    self.messages.append({
                        "role": "user",
                        "content": "Your previous response was cut off due to length limits. Please continue exactly where you left off, without any introductory text."
                    })

                # Relieve context pressure before retrying: on small windows the
                # partial output itself is what keeps re-triggering truncation.
                self._trim_history()
                continue
            
            if full_content or tool_calls_accumulator or full_reasoning:
                msg_to_append = {"role": "assistant", "content": full_content}
                if full_reasoning:
                    msg_to_append["thinking"] = full_reasoning
                if tool_calls_accumulator:
                    msg_to_append["tool_calls"] = tool_calls_accumulator
                self.messages.append(msg_to_append)
            
            # Now handle the fully assembled tool calls
            if tool_calls_accumulator:
                current_tools = get_available_tools()
                if allowed_tools is not None:
                    current_tools = {name: func for name, func in current_tools.items() if name in allowed_tools}
                
                has_fatal_error = False
                
                for tool_call in tool_calls_accumulator:
                    func_name = tool_call["function"]["name"]
                    arguments = tool_call["function"].get("arguments", {})
                    
                    if has_fatal_error:
                        result = "Error: Execution cancelled due to failure in a previous tool call."
                        yield {"type": "tool_start", "name": func_name, "args": arguments}
                        yield {"type": "tool_end", "name": func_name, "result": result}
                        self.messages.append(provider.format_tool_result(str(result), tool_call.get("id")))
                        continue
                    
                    if func_name not in current_tools:
                        recovered = recover_tool_call(tool_call, current_tools)
                        if recovered:
                            func_name = recovered["function"]["name"]
                            arguments = recovered["function"]["arguments"]
                            tool_call["function"]["name"] = func_name
                            tool_call["function"]["arguments"] = arguments
                    
                    yield {"type": "tool_start", "name": func_name, "args": arguments}
                        
                    if func_name in current_tools:
                        try:
                            func = current_tools[func_name]
                            # Filter out unknown parameters that the model may hallucinate
                            sig = inspect.signature(func)
                            valid_params = set(sig.parameters.keys())
                            hallucinated_args = {k: v for k, v in arguments.items() if k not in valid_params}
                            filtered_args = {k: v for k, v in arguments.items() if k in valid_params}
                            
                            # Check for missing required arguments
                            missing_args = [
                                p.name for p in sig.parameters.values() 
                                if p.default == inspect.Parameter.empty and p.name not in filtered_args
                            ]
                            
                            if missing_args:
                                provided_args = list(arguments.keys())
                                # Build a smart hint if the model clearly called the wrong tool
                                hint = ""
                                if hallucinated_args:
                                    hint = f"\n\n[HINT]: You passed parameters that do NOT exist in '{func_name}': {list(hallucinated_args.keys())}. "
                                    # Detect common confusion: read_file called with content= (meant write_file or append_to_file)
                                    if func_name == "read_file" and "content" in hallucinated_args:
                                        hint += "It looks like you want to WRITE content, not READ. Use `write_file(file_path, content)` to create a new file, or `append_to_file(file_path, content)` to add content to an existing file."
                                    elif func_name == "write_file" and "target_text" in hallucinated_args:
                                        hint += "It looks like you want to EDIT part of a file. Use `replace_in_file(file_path, target_text, replacement_text)` instead."
                                    else:
                                        hint += f"Valid parameters for '{func_name}' are: {list(valid_params)}."
                                result = f"Error executing tool '{func_name}': Missing REQUIRED arguments: {missing_args}. You provided: {provided_args}.{hint}"
                            else:
                                # Confirmation for dangerous actions happens inside the tools
                                # themselves via the central `approval` module (single gate).
                                # Trigger on_tool_call hook. If any plugin returns False, we cancel the execution.
                                hook_results = hook_manager.call_hook("on_tool_call", func_name, filtered_args)
                                if False in hook_results:
                                    result = f"Error: Execution of tool '{func_name}' was blocked by a user plugin."
                                else:
                                    result = func(**filtered_args)

                                # --- AUTO PLUGIN RELOAD ---
                                if func_name in ["write_file", "replace_in_file", "replace_python_function", "delete_file"]:
                                    target_file = filtered_args.get("file_path")
                                    if target_file and "Error" not in result:
                                        # Normalize to absolute path
                                        abs_target = os.path.abspath(target_file)
                                        hooks_dir = os.path.abspath(get_hooks_dir())
                                        if abs_target.startswith(hooks_dir):
                                            hook_manager.reload_plugins(hooks_dir)
                                            # Append a small notification to the tool result so the AI knows its new tool is ready
                                            result += f"\n\n[Argent]: Plugin system reloaded. Any new or modified commands in '{os.path.basename(abs_target)}' are now active."
                        except Exception as e:
                            result = f"Error executing tool {func_name}: {e}"
                    else:
                        result = f"Error: Tool {func_name} is not available."
                        
                    # Loop guard: deterministically catch verbatim-repeated calls
                    # that small models never notice on their own.
                    guard_level = self.loop_guard.record(func_name, arguments, str(result))
                    if guard_level:
                        result = str(result) + build_loop_note(guard_level)
                        if guard_level == "stop":
                            loop_guard_stop = True

                    # Auto-Healing Mechanism: trigger only on structured failure
                    # signals (exit codes / tool error prefixes), not on substrings.
                    if func_name in HEALING_TOOLS:
                        cmd_arg = arguments.get("command") if isinstance(arguments, dict) else None
                        if detect_tool_failure(func_name, str(result), command=cmd_arg):
                            has_fatal_error = True
                            self.error_retries = getattr(self, 'error_retries', 0) + 1
                            result += build_healing_hint(self.error_retries, func_name=func_name)
                        else:
                            self.error_retries = 0
                        
                    result = compress_tool_result(result, self.model_name)
                    
                    yield {"type": "tool_end", "name": func_name, "result": result}

                    self.messages.append(provider.format_tool_result(str(result), tool_call.get("id")))
                if loop_guard_stop:
                    yield {"type": "error", "content": "\n[Loop Guard]: повторяющийся цикл инструментов остановлен — ход завершён принудительно."}
                    break
                # Loop continues to let the model react to tool results
            else:
                # No tool call this turn.
                if full_content and full_content.strip():
                    break  # the model gave a final answer — turn is genuinely done.

                # Otherwise it produced only reasoning (or nothing) and neither
                # acted nor answered — a premature stop, common with reasoning
                # models. Nudge it to continue, bounded, instead of dead-ending.
                self._no_action_continues = getattr(self, "_no_action_continues", 0) + 1
                if self._no_action_continues > MAX_NO_ACTION_CONTINUES:
                    yield {"type": "error", "content": (
                        f"\n[System: the model produced no answer and no action after "
                        f"{MAX_NO_ACTION_CONTINUES} nudges — stopping.]"
                    )}
                    break
                yield {"type": "error", "content": (
                    "\n[System: no answer or action produced — nudging the model to continue...]"
                )}
                self.messages.append({
                    "role": "user",
                    "content": (
                        "You stopped after thinking, without producing anything. Either CALL A "
                        "TOOL to take the next concrete action, or WRITE your final answer to the "
                        "user now. Do not stop again without doing one of these."
                    ),
                })
                self._trim_history()
                # loop continues — give the model another turn to act or answer.
    
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

    def repair_history(self):
        """Restore message-history invariants after an interrupted turn.

        If the last assistant message carries tool_calls that lack matching
        tool results (e.g. the user pressed Ctrl+C mid-execution), append
        synthetic results so the next request doesn't violate the API contract.
        """
        if not self.messages:
            return
        # Find the last assistant message; only repair if it carries tool_calls.
        idx = None
        for i in range(len(self.messages) - 1, -1, -1):
            role = self.messages[i].get("role")
            if role == "assistant":
                if self.messages[i].get("tool_calls"):
                    idx = i
                break
            if role in ("user", "system"):
                break
        if idx is None:
            return

        tool_calls = self.messages[idx]["tool_calls"]
        results_after = sum(1 for m in self.messages[idx + 1:] if m.get("role") == "tool")
        missing = tool_calls[results_after:]
        if not missing:
            return

        try:
            provider = create_provider()
        except Exception:
            provider = None
        note = "Interrupted by user (Ctrl+C). The tool did not finish."
        for tc in missing:
            if provider:
                self.messages.append(provider.format_tool_result(note, tc.get("id")))
            else:
                self.messages.append({"role": "tool", "content": note})
        log.info("repair_history: appended %d synthetic tool result(s)", len(missing))

    def clear_history(self):
        memory.clear()
        self.loop_guard.reset()
        self.messages = [self.messages[0]]

    def get_context_usage(self) -> Dict[str, Any]:
        """Calculates current context usage statistics."""
        history_tokens = sum(self._estimate_tokens(str(m)) for m in self.messages)
        max_tokens = get_context_window()
        percent = (history_tokens / max_tokens) * 100 if max_tokens > 0 else 0
        return {
            "tokens": history_tokens,
            "max": max_tokens,
            "percent": min(percent, 100),
            "messages": len(self.messages)
        }

    def get_context_breakdown(self) -> Dict[str, Any]:
        """Token breakdown of what the next request will carry: the system
        prompt (incl. AGENTS.md), the tool schemas actually sent for this tier,
        and the conversation history. Lets the user see where the budget goes."""
        import json
        from tool_profiles import slim_tools_for_category

        system_tokens = self._estimate_tokens(str(self.messages[0])) if self.messages else 0
        history_tokens = sum(self._estimate_tokens(str(m)) for m in self.messages[1:])

        tool_tokens = 0
        tool_count = 0
        if self.strategy.supports_native_tools():
            schemas = get_tool_schemas(include_hidden=False)
            category = get_model_size_category(self.model_name)
            keep = set(slim_tools_for_category([t["function"]["name"] for t in schemas], category))
            sent = [t for t in schemas if t["function"]["name"] in keep]
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
        """Clear conversation history but preserve the full system prompt
        (including dynamic extensions like Obsidian vault config).
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
        role_prompts = {
            "Coder": "You are a specialized Coder sub-agent. Your goal is to IMPLEMENT specific code as described. Be concise and follow the style guide.",
            "Researcher": "You are a specialized Research sub-agent. Your goal is to gather technical information and documentation. Synthesize your findings into a clear report.",
            "Reviewer": "You are a specialized Code Reviewer. Your goal is to find bugs, security vulnerabilities, and architectural flaws in the provided code.",
            "DocWriter": "You are a specialized Documentation sub-agent. Your goal is to write clear, accurate markdown documentation for the project.",
            "Critic": CRITIC_SYSTEM,
        }
        
        custom_system = role_prompts.get(role, f"You are a specialized {role} sub-agent.")
        custom_system += f"\n\n## YOUR SPECIFIC TASK:\n{task}\n"
        custom_system += "\n## PROTOCOL:\n- Focus ONLY on your task.\n- Return a final summary to the Supervisor when finished.\n"
        
        # Replace main system prompt
        self.messages[0] = {"role": "system", "content": custom_system}

    def execute(self) -> str:
        """Run the sub-agent loop until completion and return the final report."""
        from ui import console
        console.print(f"[bold cyan]Sub-Agent ({self.role}) starting task...[/bold cyan]")
        
        final_answer = ""
        for chunk in self.process_user_input(f"Start task: {self.task}", allowed_tools=self.tools_override):
            if chunk["type"] == "content_stream":
                final_answer += chunk["content"]
            elif chunk["type"] == "error":
                return f"Sub-Agent Error: {chunk['content']}"

        return final_answer


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
            log.warning("Critic on %s/%s failed (%s); falling back to current model",
                        provider, model, e)
            return ArgentSubAgent("Critic", task, tools_override=[]).execute()
        raise
