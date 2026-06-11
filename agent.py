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

log = get_logger("agent")

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
- **Strict Environment**: Use {platform.system()}-native commands only (PowerShell/CMD on Windows).""")
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
- **Strict Environment**: Use {platform.system()}-native commands ONLY (e.g., PowerShell/CMD on Windows, NOT unix commands like 'ls' or 'grep').""")

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
- Use `list_skills` to discover skills. Use `read_skill` to read and follow instructions. Use `create_skill` to persist complex workflows.""")

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



        agents_md_paths = [
            Path(".argent/AGENTS.md"),
            Path("AGENTS.md"),
        ]
        for p in agents_md_paths:
            if p.exists():
                try:
                    agents_content = p.read_text(encoding="utf-8").strip()
                    if agents_content:
                        prompt_parts.append(f"## PROJECT INSTRUCTIONS (from {p})\n{agents_content}")
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

        full_prompt = "\n\n".join(prompt_parts)
        return compress_system_prompt(full_prompt, self.model_name)

    def __init__(self, max_history_messages: int = None):
        self.model_name = get_current_model()
        self.provider = get_provider()
        self.max_context_tokens = get_context_window()
        
        # Load Strategy Pattern
        self.strategy = get_model_strategy(self.model_name, self.provider)
        
        if max_history_messages is not None:
            self.max_history_messages = max_history_messages
        else:
            category = get_model_size_category(self.model_name)
            self.max_history_messages = self.strategy.get_max_history_messages(category)
            
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.strategy = get_model_strategy(self.model_name, self.provider)

    def _estimate_tokens(self, text: str) -> int:
        return estimate_tokens(text, self.model_name, self.provider)

    def _trim_history(self):
        self.max_context_tokens = get_context_window()
        
        from config import get_strip_reasoning
        from src.agent.trimmer import clean_messages_for_llm
        strip_enabled = get_strip_reasoning()
        
        cleaned = clean_messages_for_llm(self.messages, strip_enabled)
        history_tokens = sum(self._estimate_tokens(str(m)) for m in cleaned[1:])
        msg_count = len(cleaned) - 1
        
        if msg_count <= self.max_history_messages and history_tokens <= self.max_context_tokens:
            return

        self.messages = self.strategy.trim_history(
            self.messages, self.model_name, self.max_history_messages, self.max_context_tokens
        )

    def _parse_raw_tool_call(self, content: str) -> dict | None:
        return parse_raw_tool_call(content)

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
        
        self._trim_history()

        # One provider instance per turn: tool-result formatting and retries
        # below reuse it instead of re-creating a provider on every call.
        try:
            provider = create_provider()
            validation_error = provider.validate_config()
            if validation_error:
                yield {"type": "error", "content": validation_error}
                return
        except Exception as e:
            yield {"type": "error", "content": f"Provider error: {e}"}
            return

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
                
                # Check if the strategy supports native tools
                if not self.strategy.supports_native_tools():
                    active_tools = None

                from config import get_temperature
                temp = get_temperature()

                response_stream = provider.stream_chat(
                    model=self.model_name,
                    messages=cleaned_messages,
                    tools=active_tools,
                    context_window=self.max_context_tokens if self.provider != "zai" else None,
                    temperature=temp,
                )

                for chunk in response_stream:
                    if chunk.get("truncated"):
                        is_truncated = True

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
                    if content_chunk:
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
                if "does not support tools" in error_str or "element type" in error_str:
                    try:
                        fallback_stream = provider.stream_chat(
                            model=self.model_name,
                            messages=cleaned_messages,
                            tools=None,
                            context_window=self.max_context_tokens if self.provider != "zai" else None,
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

            # --- FALLBACK: Parse tool calls from raw JSON in content ---
            # Models like Qwen2.5-Coder, nanbeige, etc. often write tool calls
            # as raw JSON strings instead of using the native tool_calls mechanism.
            if not tool_calls_accumulator and not is_truncated:
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
                msg_to_append = {"role": "assistant", "content": full_content}
                if full_reasoning:
                    msg_to_append["thinking"] = full_reasoning
                if tool_calls_accumulator:
                    # Clear accumulator because it's incomplete JSON
                    tool_calls_accumulator = []
                self.messages.append(msg_to_append)
                
                # Yield a warning to the user
                yield {"type": "error", "content": "\n[System: Model generation was truncated by max_tokens limit. Auto-continuing...]"}
                
                # Append a system prompt asking the model to continue exactly where it left off
                self.messages.append({
                    "role": "user",
                    "content": "Your previous response was cut off due to length limits. Please continue exactly where you left off, without any introductory text."
                })
                # Loop continues to let the model finish
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
                        
                    # Auto-Healing Mechanism: trigger only on structured failure
                    # signals (exit codes / tool error prefixes), not on substrings.
                    if func_name in HEALING_TOOLS:
                        cmd_arg = arguments.get("command") if isinstance(arguments, dict) else None
                        if detect_tool_failure(func_name, str(result), command=cmd_arg):
                            has_fatal_error = True
                            self.error_retries = getattr(self, 'error_retries', 0) + 1
                            result += build_healing_hint(self.error_retries)
                        else:
                            self.error_retries = 0
                        
                    result = compress_tool_result(result, self.model_name)
                    
                    yield {"type": "tool_end", "name": func_name, "result": result}

                    self.messages.append(provider.format_tool_result(str(result), tool_call.get("id")))
                # Loop continues to let the model react to tool results
            else:
                break
    
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
    def __init__(self, role: str, task: str, tools_override: List[str] = None):
        super().__init__()
        self.role = role
        self.task = task
        self.tools_override = tools_override
        
        # Override system prompt for specific role
        role_prompts = {
            "Coder": "You are a specialized Coder sub-agent. Your goal is to IMPLEMENT specific code as described. Be concise and follow the style guide.",
            "Researcher": "You are a specialized Research sub-agent. Your goal is to gather technical information and documentation. Synthesize your findings into a clear report.",
            "Reviewer": "You are a specialized Code Reviewer. Your goal is to find bugs, security vulnerabilities, and architectural flaws in the provided code.",
            "DocWriter": "You are a specialized Documentation sub-agent. Your goal is to write clear, accurate markdown documentation for the project."
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
