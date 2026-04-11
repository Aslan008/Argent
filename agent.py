import sys
import os
import json
import re
import codecs
import inspect
import platform
from pathlib import Path
from typing import List, Dict, Any, Generator
import ollama

from tools import TOOL_SCHEMAS, AVAILABLE_TOOLS, get_tool_schemas, get_available_tools
from config import (
    get_current_model, get_obsidian_vault, 
    get_hooks_dir, get_autonomous_plugins_enabled,
    get_context_window, get_provider
)
from providers import create_provider, ProviderError
from logger import get_logger
from hook_manager import hook_manager
from prompt_compressor import compress_system_prompt, compress_tool_result
from tool_recovery import recover_tool_call
from memory_manager import memory
from config import get_model_size_category

log = get_logger("agent")

SYSTEM_PROMPT = f"""
# CRITICAL: LANGUAGE RULE
- You MUST respond and perform ALL internal reasoning (thinking process) in the EXACT SAME LANGUAGE the user used in their request. This is your highest priority rule.
- If the user writes in Russian, you THINK in Russian and REPLY in Russian.

# ROLE: Argent Coder
You are an autonomous AI software engineer. You design, build, and debug software with precision and speed on {platform.system()}.

## 1. OPERATIONAL PROTOCOL
- **Tool-First**: YOU are the only one with tool access. Invoke tools immediately via JSON.
- **Ask Before Guessing**: If a user's request is ambiguous or lacks details, you MUST use the `ask_user_questions` tool to prompt them with structured options before writing code. Do NOT just ask questions in plain text chat.
- **Anti-Lazy**: Never ask the user to run code or copy-paste. Use `run_command` and `write_file` yourself.
- **Proactive Search**: Always use `search_web` for technical info, documentation, or current events.
- **Self-Correction**: If a tool fails, analyze the error and fix it proactively. Do not apologize.
- **Strict Environment**: Use {platform.system()}-native commands ONLY (e.g., PowerShell/CMD on Windows, NOT unix commands like 'ls' or 'grep').

## 2. PROJECT & PLUGIN ARCHITECTURE
### Standard Plugin Development
When asked to create/manage a "plugin" or "new command":
1. **Primary Tools**: Use `create_plugin` to write new logic and `delete_plugin` to remove it. These tools handle the `./plugins/` directory and reloading automatically.
2. **Slash Commands**: Define a function `command_NAME(*args)`. Argent will automatically extract 'NAME' as a new slash command (e.g., `command_hello` becomes `/hello`).
3. **Internal Hooks**: Use these event names for automatic execution:
    - `on_startup()`: Runs when Argent starts.
    - `pre_prompt(text)`: Modifies user input before AI sees it.
    - `on_tool_call(func_name, args)`: Runs before tool execution. Return `False` to block.
    - `post_response(text)`: Runs after AI finishes speaking.
    - `on_chat_saved(file_path)`: Runs after chat log is saved.
4. **Implementation**: Always use `from ui import console` for output.

### Skills System
You have access to "Skills" — instruction-based extensions stored in markdown files.
1. **Discovery**: Use `list_skills` to see what specialized instructions are available.
2. **Usage**: If a user's request matches a skill's description, use `read_skill` to get the full instructions and FOLLOW THEM strictly.
3. **Persistence**: Use `create_skill` to save complex workflows, expert personas, or specific logic patterns for future use.

### Project Brain Mode
- Tools like `add_project_task`, `write_project_spec`, etc., are EXCLUSIVELY for massive multi-step projects.
- If these tools are not in your `allowed_tools` list, DO NOT attempt to call them. Use regular file tools instead.

## 3. UI & TERMINOLOGY STANDARDS
- **"Panel"**: Always refers to `rich.panel.Panel` for terminal UI. NEVER start web servers or use web-dashboard libraries (like HoloViz Panel) unless explicitly building a web app.
- **"Table"**: Always refers to `rich.table.Table`.
- **Output**: Use `console.print()` or `print_system()` for beautiful terminal results.

## 4. THINK & VERIFY PROTOCOL
- **Outcome Analysis**: After EACH tool call, analyze if the result truly moves you closer to the goal.
- **False Success**: "Requirement already satisfied" or "Exit code: 0" does NOT always mean success. If a tool reports success but the problem persists (e.g., a package still can't be imported), you MUST try a different approach (e.g., check paths, use `--force-reinstall`, or investigate environment).
- **Proactive Verification**: After installing things or writing complex files, use `run_command` or `read_file` to VERIFY they work as intended.
- **Self-Correction**: If you are stuck in a loop, STOP. Rethink your strategy. Explain your new reasoning to the user.

## 5. COMMUNICATION
- **Language**: Follow the CRITICAL LANGUAGE RULE at the top of this prompt.
- **Transparency**: Briefly state your reasoning before executing tools, especially if you are changing your plan.
- **Visuals**: If a complex concept, UI mockup, or architecture diagram would help the user, USE the `create_svg_image` tool. This will automatically open a browser window for the user to see your work.
"""

class ArgentAgent:
    def __init__(self, max_history_messages: int = None):
        self.model_name = get_current_model()
        self.provider = get_provider()
        self.max_context_tokens = get_context_window()
        
        if max_history_messages is not None:
            self.max_history_messages = max_history_messages
        else:
            category = get_model_size_category(self.model_name)
            self.max_history_messages = {
                "tiny": 6,
                "small": 10,
                "medium": 20,
                "large": 30,
                "cloud": 40,
            }.get(category, 20)
        system_prompt = SYSTEM_PROMPT
        
        vault = get_obsidian_vault()
        if vault:
            system_prompt += f"""
## 5. OBSIDIAN INTEGRATION
- **Active Vault**: `{vault}`
- **Protocols**:
    - Use `write_obsidian_note` for creating notes.
    - Use ABSOLUTE paths (e.g., `{vault}\\Note.md`) for `read_file` or `replace_in_file` on notes.
    - NEVER use `update_obsidian_properties` for text body edits; use `replace_in_file`.
"""
            
        # Plugin & Autonomous Awareness
        hooks_dir = get_hooks_dir()
        auto_plugins = get_autonomous_plugins_enabled()
        
        system_prompt += f"""
## 6. DYNAMIC CONFIGURATION
- **HOOKS_DIR**: `{hooks_dir}`
- **AUTONOMOUS_EXTENSION**: {'ENABLED' if auto_plugins else 'DISABLED'}
"""
        if auto_plugins:
            system_prompt += "- **Note**: You ARE allowed to autonomously create plugins to solve tasks more efficiently.\n"
        else:
            system_prompt += "- **Note**: You ARE NOT allowed to create plugins unless explicitly requested by the user.\n"

        # Load AGENTS.md project instructions if present
        agents_md_paths = [
            Path(".argent/AGENTS.md"),
            Path("AGENTS.md"),
        ]
        for p in agents_md_paths:
            if p.exists():
                try:
                    agents_content = p.read_text(encoding="utf-8").strip()
                    if agents_content:
                        system_prompt += f"\n## 7. PROJECT INSTRUCTIONS (from {p})\n{agents_content}\n"
                    break
                except Exception:
                    pass

        compressed_prompt = compress_system_prompt(system_prompt, self.model_name)
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": compressed_prompt}
        ]

    def set_model(self, model_name: str):
        self.model_name = model_name
        
    def _estimate_tokens(self, text: str) -> int:
        """Heuristic calculation for tokens (roughly 1 token = 4 chars)."""
        return len(text) // 4

    def _summarize_messages(self, msgs_to_summarize: List[Dict]) -> str:
        """Runs a fast, synchronous LLM call to summarize old context."""
        text_to_summarize = ""
        for m in msgs_to_summarize:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if not content and "tool_calls" in m:
                content = f"[Tool Calls: {m['tool_calls']}]"
            text_to_summarize += f"{role.upper()}: {content}\n\n"
            
        summary_prompt = (
            "You are a context-compression engine. Summarize the following past conversation strictly in 1-2 paragraphs. "
            "Focus entirely on the technical progress, code written, and facts established. "
            "Omit politeness and conversational filler. Here is the log:\n\n" + text_to_summarize
        )
        
        try:
            provider = create_provider()
            validation_error = provider.validate_config()
            if validation_error:
                return summary_prompt
            result = provider.sync_chat(
                model=self.model_name,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            return result or "[Compression Failed]"
        except Exception:
            return "Previous context omitted to save memory."

    def _trim_history(self):
        """Keeps history size manageable. Uses two strategies:
        - tiny/small models: hard reset with memory preservation (no LLM call needed)
        - medium/large/cloud: LLM summarization of old messages
        """
        self.max_context_tokens = get_context_window()
        
        history_tokens = sum(self._estimate_tokens(str(m)) for m in self.messages[1:])
        msg_count = len(self.messages) - 1
        
        if msg_count <= self.max_history_messages and history_tokens <= self.max_context_tokens:
            return

        category = get_model_size_category(self.model_name)

        if category in ("tiny", "small"):
            self._hard_reset_with_memory()
        else:
            self._soft_trim_with_summarization()

    def _hard_reset_with_memory(self):
        """Full context reset for small models.
        Preserves system prompt and injects structured memory note.
        No LLM call needed — memory is accumulated incrementally during conversation.
        """
        self._update_memory_from_messages()

        context_note = memory.build_context_note()

        system_content = self.messages[0]["content"]
        if context_note:
            sep = "\n\n=== PERSISTENT MEMORY (context was reset) ===\n"
            end = "\n=== END MEMORY ==="
            existing = system_content.find("=== PERSISTENT MEMORY")
            if existing != -1:
                end_marker = system_content.find("=== END MEMORY ===", existing)
                if end_marker != -1:
                    system_content = system_content[:existing] + sep + context_note + end + system_content[end_marker + len("=== END MEMORY ==="):]
                else:
                    system_content += sep + context_note + end
            else:
                system_content += sep + context_note + end

        self.messages = [{"role": "system", "content": system_content}]
        log.info("Hard context reset performed (model=%s, category=%s)", self.model_name, get_model_size_category(self.model_name))

    def _update_memory_from_messages(self):
        """Extract key information from recent messages before clearing them."""
        last_user_msg = ""
        last_assistant_action = ""

        for m in reversed(self.messages):
            if m.get("role") == "user" and not last_user_msg:
                content = m.get("content", "")
                if content and not content.startswith("/"):
                    last_user_msg = content
            elif m.get("role") == "assistant" and not last_assistant_action:
                content = m.get("content", "")
                if content:
                    first_line = content.strip().split("\n")[0][:200]
                    last_assistant_action = first_line

        if last_user_msg and not memory.data.get("objective"):
            memory.set_objective(last_user_msg)

        if last_assistant_action:
            memory.set_current_task(last_assistant_action)

    def _soft_trim_with_summarization(self):
        """LLM-based summarization for medium/large/cloud models."""
        pinned_indices = {0}
        for i, m in enumerate(self.messages):
            if m.get("role") == "system":
                content = m.get("content", "")
                if any(x in content for x in ["PROJECT SPECIFICATION", "ARCHITECTURE MAP", "PREVIOUS CONTEXT MEMORY", "PERSISTENT MEMORY"]):
                    pinned_indices.add(i)

        msgs_to_summarize = []
        indices_to_drop = []

        current_tokens = sum(self._estimate_tokens(str(m)) for m in self.messages[1:])
        for i in range(1, len(self.messages)):
            if i in pinned_indices:
                continue
            if len(self.messages) - len(indices_to_drop) <= self.max_history_messages and current_tokens <= (self.max_context_tokens * 0.7):
                break
            msgs_to_summarize.append(self.messages[i])
            indices_to_drop.append(i)
            current_tokens -= self._estimate_tokens(str(self.messages[i]))

        if not msgs_to_summarize:
            return

        from ui import console
        with console.status("[dim magenta]Оптимизация контекста...[/dim magenta]", spinner="dots"):
            summary = self._summarize_messages(msgs_to_summarize)
        
        memory_msg = {
            "role": "system",
            "content": f"=== PREVIOUS CONTEXT MEMORY ===\n{summary}\n=== END MEMORY ==="
        }

        new_messages = []
        memory_injected = False
        for i, m in enumerate(self.messages):
            if i in indices_to_drop:
                if not memory_injected:
                    new_messages.append(memory_msg)
                    memory_injected = True
                continue
            new_messages.append(m)

        self.messages = new_messages

    # Parameter name aliases for known tools
    _PARAM_ALIASES = {
        "write_file": {
            "filename": "file_path",
            "path": "file_path",
            "file": "file_path",
            "filepath": "file_path",
        },
        "read_file": {
            "filename": "file_path",
            "path": "file_path",
            "file": "file_path",
            "filepath": "file_path",
        },
        "delete_file": {
            "filename": "file_path",
            "path": "file_path",
            "file": "file_path",
            "filepath": "file_path",
        },
        "replace_in_file": {
            "filename": "file_path",
            "path": "file_path",
            "file": "file_path",
            "filepath": "file_path",
        },
        "list_directory": {
            "path": "dir_path",
            "directory": "dir_path",
            "dir": "dir_path",
            "directory_path": "dir_path",
        },
    }

    def _normalize_tool_params(self, parsed: dict) -> dict:
        """Normalize parameter names to match expected tool schemas."""
        tool_name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})
        aliases = self._PARAM_ALIASES.get(tool_name)
        if aliases:
            normalized = {}
            for key, value in arguments.items():
                canonical = aliases.get(key, key)
                normalized[canonical] = value
            parsed["arguments"] = normalized
        return parsed

    @staticmethod
    def _extract_balanced_json(text: str, start_pos: int) -> str | None:
        """Extract a balanced JSON object from text starting at start_pos.
        Uses brace-counting to handle nested objects correctly.
        Returns the complete JSON string or None if no balanced object found."""
        if start_pos >= len(text) or text[start_pos] != '{':
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start_pos, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start_pos:i + 1]
        return None

    def _decode_json_escapes(self, s: str) -> str:
        """Decode JSON escape sequences in a raw string extracted from malformed JSON."""
        # Try wrapping in quotes and parsing as JSON string for proper decoding
        try:
            return json.loads('"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"')
        except (json.JSONDecodeError, Exception):
            pass
        # Manual fallback for common escapes
        result = s
        result = result.replace('\\n', '\n')
        result = result.replace('\\t', '\t')
        result = result.replace('\\r', '\r')
        result = result.replace('\\"', '"')
        result = result.replace('\\\\', '\\')
        return result

    def _parse_raw_tool_call(self, content: str) -> dict | None:
        """Parse a raw tool call from AI-generated content.
        Returns {"parsed": {"name": ..., "arguments": {...}}, "match_str": ...} or None.
        
        Supports formats:
          1. ```json { ... } ``` or ``` { ... } ```  (markdown code blocks)
          2. {"name": "tool", "arguments": {...}}      (standard Ollama format)
          3. [{"name": "tool", "arguments": {...}}]    (array format)
          4. {"tool_name": {params}}                    (shorthand format)
        """
        clean = content.strip()
        
        # --- FORMAT 1: Markdown code blocks ---
        md_match = re.search(r'```(?:json)?\s*(\{.+)', clean, re.DOTALL)
        if md_match:
            # Extract balanced JSON from inside the code block
            inner_start = md_match.start(1)
            json_candidate = self._extract_balanced_json(clean, inner_start)
            if json_candidate:
                # Find the closing ``` to determine the full match string
                end_pos = inner_start + len(json_candidate)
                closing = clean.find('```', end_pos)
                if closing != -1:
                    match_str = clean[md_match.start():closing + 3]
                else:
                    match_str = clean[md_match.start():end_pos]
                
                result = self._try_parse_json_tool(json_candidate)
                if result:
                    result["match_str"] = match_str
                    return result

        # --- FORMAT 2, 3, 4: Find any JSON object in the content ---
        # Look for the first { that could be the start of a tool call JSON
        brace_pos = clean.find('{')
        if brace_pos != -1:
            json_candidate = self._extract_balanced_json(clean, brace_pos)
            if json_candidate:
                result = self._try_parse_json_tool(json_candidate)
                if result:
                    result["match_str"] = json_candidate
                    return result

        # --- LAST RESORT: Regex extraction for heavily malformed JSON ---
        # (e.g., write_file with actual newlines inside the content string)
        current_tools = get_available_tools()
        for tool_name in current_tools:
            if f'"{tool_name}"' in clean:
                result = self._try_recover_malformed_tool(clean, tool_name)
                if result:
                    return result
        
        return None

    def _try_parse_json_tool(self, json_str: str) -> dict | None:
        """Try to parse a JSON string as a tool call.
        Handles both {"name": ..., "arguments": {...}} and {"tool_name": {params}} formats."""
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return None
        
        if not isinstance(parsed, dict):
            return None
        
        # Format: {"name": ..., "arguments": {...}} or {"function": ..., "arguments": {...}}
        # Support both 'name' and 'function' as a top-level key for the tool name
        tool_id = parsed.get("name") or parsed.get("function")
        if tool_id and isinstance(tool_id, str) and "arguments" in parsed:
            parsed["name"] = tool_id # Canonicalize for the rest of the logic
            parsed = self._normalize_tool_params(parsed)
            return {"parsed": parsed}
        
        # Format: {"tool_name": {params}} (shorthand)
        current_tools = get_available_tools()
        for key, value in parsed.items():
            if key in current_tools and isinstance(value, dict):
                tool_parsed = {
                    "name": key,
                    "arguments": value
                }
                tool_parsed = self._normalize_tool_params(tool_parsed)
                return {"parsed": tool_parsed}
        
        return None
    
    def _try_recover_malformed_tool(self, content: str, tool_name: str) -> dict | None:
        """Last-resort recovery for malformed JSON (e.g., unescaped newlines in strings).
        Uses regex to extract path and content parameters directly."""
        # Supported tools for content recovery
        CONTENT_TOOLS = ["write_file", "write_obsidian_note", "replace_python_function", "replace_in_file"]
        if tool_name not in CONTENT_TOOLS:
            return None
        
        # Try to find file/note path under any common parameter name
        fp_match = re.search(
            r'"(?:file_path|filename|filepath|path|file|note_path)"\s*:\s*"([^"]+)"', content
        )
        # Find content: grab everything after "content": " until we can determine the end
        ct_match = re.search(r'"content"\s*:\s*"([\s\S]*)', content)
        
        if not fp_match or not ct_match:
            return None
        
        recovered = ct_match.group(1)
        # Strip trailing "} patterns (closing of JSON value + objects)
        # We need to remove the trailing: "  }  }  or "  } depending on nesting
        recovered = re.sub(r'"\s*\}\s*\}?\s*$', '', recovered)
        # Also strip a trailing lone quote if present
        recovered = recovered.rstrip().rstrip('"')
        
        # The recovered content has JSON escape sequences as literal characters.
        # Decode them properly (e.g., \n → newline, \" → quote)
        recovered = recovered.replace('\\n', '\n')
        recovered = recovered.replace('\\t', '\t')
        recovered = recovered.replace('\\"', '"')
        recovered = recovered.replace('\\\\', '\\')
        
        # Identify the correct parameter name for the path
        path_param = "file_path"
        if tool_name == "write_obsidian_note":
            path_param = "note_path"
            
        parsed = {
            "name": tool_name,
            "arguments": {
                path_param: fp_match.group(1),
                "content": recovered
            }
        }
        
        return {"parsed": parsed, "match_str": content}

    def process_user_input(self, user_text: str, allowed_tools: List[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Process the user input and yield chunks of response or tool activity.
        Supports streaming generation.
        """
        self.messages.append({"role": "user", "content": user_text})
        
        if not user_text.startswith("/"):
            if not memory.data.get("objective"):
                memory.set_objective(user_text)
            else:
                memory.set_current_task(user_text[:200])
        
        self._trim_history()
        
        while True:
            # Variables to accumulate the streamed response
            full_content = ""
            full_reasoning = ""
            tool_calls_accumulator = []
            is_building_raw_tool = False
            raw_tool_buffer = ""
            raw_tool_char_count = 0
            reasoning_tag_active = False
            START_TAGS = ["<thought>", "<think>", "<reasoning>"]
            END_TAGS = ["</thought>", "</think>", "</reasoning>"]
            
            try:
                active_tools = get_tool_schemas()
                if allowed_tools is not None:
                    active_tools = [t for t in active_tools if t["function"]["name"] in allowed_tools]

                provider = create_provider()
                validation_error = provider.validate_config()
                if validation_error:
                    yield {"type": "error", "content": validation_error}
                    return

                response_stream = provider.stream_chat(
                    model=self.model_name,
                    messages=self.messages,
                    tools=active_tools,
                    context_window=self.max_context_tokens if self.provider != "zai" else None,
                )

                for chunk in response_stream:
                    thinking_chunk = chunk.get("thinking", "")
                    if thinking_chunk:
                        full_reasoning += thinking_chunk
                        yield {"type": "thinking_stream", "content": thinking_chunk}

                    for tc_delta in chunk.get("tool_call_deltas", []):
                        index = tc_delta["index"]
                        while len(tool_calls_accumulator) <= index:
                            tool_calls_accumulator.append({"id": "", "function": {"name": "", "arguments": ""}})
                        if tc_delta.get("id"):
                            tool_calls_accumulator[index]["id"] = tc_delta["id"]
                        if tc_delta.get("function_name_delta"):
                            tool_calls_accumulator[index]["function"]["name"] += tc_delta["function_name_delta"]
                        if tc_delta.get("function_arguments_delta"):
                            tool_calls_accumulator[index]["function"]["arguments"] += tc_delta["function_arguments_delta"]
                            # Yield a progress signal so the UI can show a spinner
                            yield {"type": "tool_generating", "name": tool_calls_accumulator[index]["function"]["name"], "bytes": len(tool_calls_accumulator[index]["function"]["arguments"])}

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
                                stripped_buffer = raw_tool_buffer.lstrip()
                                if stripped_buffer.startswith('```json') or stripped_buffer.startswith('{'):
                                    is_building_raw_tool = True
                                
                                if not is_building_raw_tool:
                                    yield {"type": "content_stream", "content": temp_content}
                                else:
                                    # Signal progress so the UI can show a spinner
                                    yield {"type": "tool_generating", "name": "?", "bytes": len(raw_tool_buffer)}
                            
            except ollama.ResponseError as e:
                error_str = str(e).lower()
                if "does not support tools" in error_str:
                    try:
                        response_stream = ollama.chat(
                            model=self.model_name,
                            messages=self.messages,
                            options={"num_ctx": self.max_context_tokens},
                            stream=True
                        )
                        for chunk in response_stream:
                            content_chunk = chunk.get("message", {}).get("content", "")
                            if content_chunk:
                                full_content += content_chunk
                                yield {"type": "content_stream", "content": content_chunk}
                    except Exception as retry_e:
                        yield {"type": "error", "content": f"Fallback Error: {retry_e}"}
                        break
                elif "thought_signature" in error_str or "functioncall" in error_str:
                    self._flatten_tool_messages()
                    continue
                else:
                    yield {"type": "error", "content": f"Ollama Error: {e.error}"}
                    break
            except ProviderError as e:
                yield {"type": "error", "content": str(e)}
                break
            except Exception as e:
                log.error("Unexpected error in stream: %s", e, exc_info=True)
                yield {"type": "error", "content": f"Connection Error: {e}"}
                break

            # End of stream — parse tool call arguments from strings to dicts.
            if tool_calls_accumulator:
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
            if not tool_calls_accumulator:
                clean_content = full_content.strip()
                parsed_tool = self._parse_raw_tool_call(clean_content)
                
                if parsed_tool:
                    tool_calls_accumulator.append({"function": parsed_tool["parsed"]})
                    # Clean the raw JSON out of full_content
                    if parsed_tool.get("match_str"):
                        full_content = full_content.replace(parsed_tool["match_str"], "").strip()
                    else:
                        full_content = ""
                    yield {"type": "content_replace", "content": full_content}
            # --- END FALLBACK ---
            
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
                
                for tool_call in tool_calls_accumulator:
                    func_name = tool_call["function"]["name"]
                    arguments = tool_call["function"].get("arguments", {})
                    
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
                            filtered_args = {k: v for k, v in arguments.items() if k in valid_params}
                            
                            # Check for missing required arguments
                            missing_args = [
                                p.name for p in sig.parameters.values() 
                                if p.default == inspect.Parameter.empty and p.name not in filtered_args
                            ]
                            
                            if missing_args:
                                provided_args = list(arguments.keys())
                                result = f"Error executing tool '{func_name}': Missing REQUIRED arguments: {missing_args}. You provided: {provided_args}. Please check the tool schema and use the exact parameter names."
                            else:
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
                        
                    # Inject a forceful reminder on errors to prevent the model from reverting to raw python code output
                    if func_name in ["run_command", "write_file", "replace_in_file"] and ("Error" in result or "failed" in result.lower() or "Traceback" in result or "Requirement already satisfied" in result):
                        result += (
                            "\n\n[SYSTEM ADVICE]: If you see an error or a 'False Success' (like 'Requirement already satisfied' while the issue persists):"
                            "\n1. Do NOT just repeat the same command."
                            "\n2. VERIFY the state using other tools (e.g., check python versions, site-packages, or file contents)."
                            "\n3. Try alternative methods (e.g., --force-reinstall, checking environment variables)."
                            "\n4. If building code, ensure you didn't leave syntax errors from previous edits."
                        )
                        
                    result = compress_tool_result(result, self.model_name)
                    
                    yield {"type": "tool_end", "name": func_name, "result": result}
                    
                    try:
                        provider = create_provider()
                    except Exception:
                        provider = None
                    if provider:
                        tool_result_msg = provider.format_tool_result(str(result), tool_call.get("id"))
                    else:
                        tool_result_msg = {"role": "tool", "content": str(result)}
                    self.messages.append(tool_result_msg)
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
