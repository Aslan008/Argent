import sys
import os
import json
import re
import codecs
import inspect
import platform
from typing import List, Dict, Any, Generator
import ollama

from tools import TOOL_SCHEMAS, AVAILABLE_TOOLS, get_tool_schemas, get_available_tools
from config import (
    get_current_model, get_obsidian_vault, 
    get_hooks_dir, get_autonomous_plugins_enabled
)
from hook_manager import hook_manager

#If you suspect a typo or a non-existent entity, search the web to verify before correcting the user. Это если ИИ будет сопративляться и не искать информацию в интернете
SYSTEM_PROMPT = """You are Argent Coder, an autonomous AI programming agent.
Your objective to assist the user in designing, debugging, and building software. Act as a professional, deeply knowledgeable, and highly efficient AI engineer.
CRITICAL ENVIRONMENT INFO: The user is running on the """ + platform.system() + """ operating system. You MUST use terminal commands corresponding strictly to this OS (e.g., if Windows, use PowerShell or CMD commands; DO NOT use Unix commands like 'ls', 'rm', 'ps', or 'grep' on Windows).
- YOU are the ONLY one with access to tools (read_file, write_file, delete_file, list_directory, run_command, search_web, read_webpage). The user CANNOT use tools or run commands. DO NOT tell the user to run commands or use tools. YOU must invoke them yourself by outputting the JSON tool call!
- ANTI-LAZY RULE: NEVER output code blocks with shell commands (like `uvicorn --version`) and tell the user to run them. YOU MUST use `run_command` to run it yourself!
- NEVER write code blocks and tell the user to copy-paste them. YOU MUST use the `write_file` tool to save code directly to their disk.
- DO NOT use the `write_file` tool to answer conversational questions or provide general information (e.g., explaining history or recipes). Respond in plain text in the chat unless the user explicitly asks for a file or you are writing source code.
- If the information isn’t available in your existing knowledge or memory, use web search (search_web) before responding. Only say you don’t know if you still can’t find reliable information.
- If you don’t know something or you’re unsure about it, search_web is a must-have tool.
- Always keep in mind: Always try to use the search_web tool to find the most up-to-date information before answering the user.
- ANTI-REFUSAL: NEVER apologize and say you cannot browse the internet or perform web searches. YOU DO HAVE INTERNET ACCESS via the `search_web` tool. You MUST use it instead of saying you can't.
- CRITICAL SEARCH RULE: For time-sensitive information, news, current events, software versions, or release dates (e.g., "when does X release"), your internal knowledge is OUTDATED. You MUST use `search_web` to verify the current status even if you feel confident in your answer.
- SOURCE SELECTION RULE: Do not over-rely on Wikipedia. For niche topics (e.g., game characters, anime, pop culture), Wikipedia often lacks detailed info. You MUST prioritize specialized sites like Fandom wikis, official websites, Reddit, or dedicated forums found in the `search_web` results. If the first `read_webpage` attempt doesn't have the answer, read another link or refine your search query!
- If the user asks you to create a new file, use the `write_file` tool.
- If the user asks you to modify an existing file, strongly prefer using the `replace_in_file` tool to make localized changes without rewriting the entire file. Use `write_file` only for complete rewrites.
- If the user asks you to delete a file, use the `delete_file` tool immediately.
- If the user asks you to run a regular script, use the `run_command` tool.
- If the user asks you to perform an OS-level configuration (e.g. edit registry, set global environment variables, manage services) or run a command as Administrator, you MUST use the `run_admin_command` tool.
- CRITICAL: If the script is a server (e.g. `python server.py`, `uvicorn`, `npm start`), or if the user explicitly asks you to start a server, you MUST use `start_background_command` instead of `run_command`. `run_command` will hang forever on servers. After starting it, use `read_background_command` to verify it started successfully.
- If you use `search_web` and the snippets are insufficient, use `read_webpage` on the result URL to read the full documentation/forum thread.
- If a `run_command` tool fails with an error (stderr or non-zero exit code), you MUST read the error and autonomously invoke another tool to fix it. Do not just apologize; fix it proactively.
- CRITICAL: If you encounter a `ModuleNotFoundError` or `ImportError`, you MUST autonomously use the `run_command` tool to run `pip install <package>`. Do not ask the user to do it yourself.
- CRITICAL: When using file tools, NEVER hallucinate fake proxy paths like `/path/to/file`. Use relative paths to the current working directory, or use `list_directory` to find the exact filename first.
- If you need to use a tool, YOU MUST CALL THE TOOL IMMEDIATELY in the same response by outputting a JSON object.
- PROACTIVE INVESTIGATION: If a user asks a question about files or directories (e.g., "what do these scripts do?"), YOU MUST autonomously use `list_directory` and then `read_file` on relevant files to get the answer. DO NOT ask the user to provide the content of files you can access yourself.
- ANTI-PASSIVITY: NEVER say "Please provide the content of X" if X is in the current directory or accessible via tools. Use the tool immediately.
- If you suspect a typo or a non-existent entity, search the web to verify before correcting the user.
- PLUGIN DEVELOPMENT: When the user asks for a "plugin" or a "new command" (like `/mem`), you MUST write the implementation file to the specified hooks directory: **{get_hooks_dir()}**. 
- TERM CLARIFICATION: In the context of Argent, the term **"Panel"** refers to `rich.panel.Panel` for a beautiful Terminal UI. DO NOT use external web-dashboard libraries like `panel` (HoloViz) or start local web servers unless explicitly asked for a web-app.
- PLUGIN STRUCTURE: Plugins should use `from ui import console` and define functions named `command_NAME(*args)` to be automatically detected.
Do not explain what you are going to do and then stop. Output the correct tool call JSON directly!

CRITICAL LANGUAGE INSTRUCTION: You MUST respond to the user in the EXACT SAME LANGUAGE they used to address you. If the user speaks Russian, you MUST reply in Russian. If you reply in English to a Russian prompt, the system will fail.
"""

class ArgentAgent:
    def __init__(self, max_history_messages: int = 20):
        self.max_history_messages = max_history_messages
        self.model_name = get_current_model()
        system_prompt = SYSTEM_PROMPT
        
        vault = get_obsidian_vault()
        if vault:
            system_prompt += (
                f"\n\nOBSIDIAN INTEGRATION ACTIVE: The user has an Obsidian vault configured at '{vault}'. "
                f"When asked to create a note, YOU MUST use the `write_obsidian_note` tool. "
                f"Pass tags and aliases as direct parameters if requested (e.g. tags: ['idea', 'game']). "
                f"DO NOT use `write_file` for Obsidian notes. "
                f"CRITICAL: If you need to edit an existing Obsidian note using `replace_in_file` or `read_file`, you MUST use its ABSOLUTE path (e.g., '{vault}\\Note Name.md'), NOT its relative name."
                f"\nWARNING: If the user asks to rewrite, expand, or add examples to an Obsidian note's TEXT, ALWAYS use `replace_in_file` or rewrite it completely with `write_file`. DO NOT use `update_obsidian_properties` for text modifications!"
            )
            
        # Plugin & Autonomous Awareness
        hooks_dir = get_hooks_dir()
        auto_plugins = get_autonomous_plugins_enabled()
        
        system_prompt += (
            f"\n\nGLOBAL PLUGINS (HOOKS): You can extend your own functionality by creating new slash commands. "
            f"The global hooks directory is: '{hooks_dir}'. "
            "To create a new command, use `write_file` to create a .py file in that directory. "
            "The function name must start with 'command_'. After creating it, the command will be available after a restart or reload."
        )
        
        if auto_plugins:
            system_prompt += (
                "\nAUTONOMOUS EXTENSION ENABLED: You ARE allowed to autonomously create new plugins/tools "
                "if you decide it is a more efficient way to solve the user's task. For example, if no current tool "
                "can handle a specific file format or logic, write a specialized plugin for it."
            )
        else:
            system_prompt += (
                "\nAUTONOMOUS EXTENSION DISABLED: You ARE NOT allowed to autonomously create new plugins "
                "unless the user explicitly asks you to create a new command or plugin."
            )

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
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
            from ui import console
            # console.print("[dim]...Compressing context memory...[/dim]") # Removed to avoid spam
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": summary_prompt}]
            )
            return response.get("message", {}).get("content", "[Compression Failed]")
        except Exception:
            return "Previous context omitted to save memory."

    def _trim_history(self):
        """Keeps history size manageable using token heuristics and LLM summarization."""
        MAX_CONTEXT_TOKENS = 16000 # Safe budget
        
        # Calculate tokens of history (excluding system prompt at index 0)
        history_tokens = sum(self._estimate_tokens(str(m)) for m in self.messages[1:])
        msg_count = len(self.messages) - 1
        
        if msg_count <= self.max_history_messages and history_tokens <= MAX_CONTEXT_TOKENS:
            return
            
        # We need to trim. Calculate how many to drop.
        # We drop either because we hit max_messages, or because we hit token limit.
        excess_count = msg_count - self.max_history_messages
        excess_count = max(excess_count, 0)
        
        # If still over tokens after excess_count, drop more until we're under 50% of max tokens to give headroom
        drop_idx = 1 + excess_count
        while drop_idx < len(self.messages) and sum(self._estimate_tokens(str(m)) for m in self.messages[drop_idx:]) > (MAX_CONTEXT_TOKENS // 2):
            drop_idx += 1
            
        # We must keep pairs intact (tool_call <-> tool_result). For simplicity, we just chunk it.
        # The messages to summarize are from index 1 to drop_idx
        if drop_idx > 1:
            msgs_to_drop = self.messages[1:drop_idx]
            summary = self._summarize_messages(msgs_to_drop)
            
            # Create a memory injection message
            memory_msg = {
                "role": "system", 
                "content": f"=== PREVIOUS CONTEXT MEMORY ===\n{summary}\n=== END MEMORY ==="
            }
            
            # Replace the dropped messages with the memory message
            self.messages = [self.messages[0]] + [memory_msg] + self.messages[drop_idx:]

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
        
        # Format: [{"name": ..., "arguments": {...}}] (array)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
            if not isinstance(parsed, dict):
                return None
        
        # Format: {"name": ..., "arguments": {...}} (standard)
        if "name" in parsed and "arguments" in parsed:
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
        Uses regex to extract file_path and content parameters directly."""
        if tool_name != "write_file":
            return None
        
        # Try to find file path under any common parameter name
        fp_match = re.search(
            r'"(?:file_path|filename|filepath|path|file)"\s*:\s*"([^"]+)"', content
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
        
        parsed = {
            "name": "write_file",
            "arguments": {
                "file_path": fp_match.group(1),
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
        self._trim_history()
        
        while True:
            # Variables to accumulate the streamed response
            full_content = ""
            tool_calls_accumulator = []
            is_building_raw_tool = False
            raw_tool_buffer = ""
            
            try:
                # Filter tools if requested
                active_tools = get_tool_schemas()
                if allowed_tools is not None:
                    active_tools = [t for t in active_tools if t["function"]["name"] in allowed_tools]

                # Make the streaming call
                response_stream = ollama.chat(
                    model=self.model_name,
                    messages=self.messages,
                    tools=active_tools,
                    stream=True
                )
                
                # Iterate through the stream. We wrap this in a try/except because
                # the 400 error often happens on the first iteration.
                for chunk in response_stream:
                    msg_chunk = chunk.get("message", {})
                    
                    # 1. Handle native tool calls if the model supports them
                    if "tool_calls" in msg_chunk:
                        for tc in msg_chunk["tool_calls"]:
                            if tc not in tool_calls_accumulator:
                                tool_calls_accumulator.append(tc)
                    
                    # 2. Check for content stream
                    content_chunk = msg_chunk.get("content", "")
                    if content_chunk:
                        full_content += content_chunk
                        raw_tool_buffer += content_chunk
                        
                        # Detect if we are building a raw JSON tool call
                        stripped_buffer = raw_tool_buffer.lstrip()
                        if stripped_buffer.startswith('```json') or stripped_buffer.startswith('{'):
                            is_building_raw_tool = True
                        
                        if not is_building_raw_tool:
                            yield {"type": "content_stream", "content": content_chunk}
                            
            except ollama.ResponseError as e:
                error_str = str(e).lower()
                # If the model doesn't support tools natively, retry without tools.
                if "does not support tools" in error_str:
                    try:
                        response_stream = ollama.chat(
                            model=self.model_name,
                            messages=self.messages,
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
                    # Gemini cloud models require thought_signature in tool call history.
                    # Fix: convert structured tool_calls to plain text and retry.
                    self._flatten_tool_messages()
                    continue  # Retry the loop with flattened messages
                else:
                    yield {"type": "error", "content": f"Ollama Error: {e.error}"}
                    break
            except Exception as e:
                yield {"type": "error", "content": f"Connection Error: {e}. Is Ollama running?"}
                break

            # End of stream.
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
            
            if full_content or tool_calls_accumulator:
                msg_to_append = {"role": "assistant", "content": full_content}
                if tool_calls_accumulator:
                    msg_to_append["tool_calls"] = tool_calls_accumulator
                self.messages.append(msg_to_append)
            
            # Now handle the fully assembled tool calls
            if tool_calls_accumulator:
                for tool_call in tool_calls_accumulator:
                    func_name = tool_call["function"]["name"]
                    arguments = tool_call["function"].get("arguments", {})
                    
                    yield {"type": "tool_start", "name": func_name, "args": arguments}
                    
                    current_tools = get_available_tools()
                    # Enforce allowed_tools filter during execution as well
                    if allowed_tools is not None:
                        current_tools = {name: func for name, func in current_tools.items() if name in allowed_tools}
                        
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
                    if func_name in ["run_command", "write_file", "replace_in_file"] and ("Error" in result or "failed" in result.lower() or "Traceback" in result):
                        result += "\n\nCRITICAL REMINDER: You must resolve this error. Do NOT output raw code or text explanations without tool calls. If this is a missing package, use `run_command` to pip install it. If it is a code bug, use `replace_in_file` or `write_file` to fix the file."
                        
                    yield {"type": "tool_end", "name": func_name, "result": result}
                    
                    self.messages.append({
                        "role": "tool",
                        "content": str(result)
                    })
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
        """Clears the conversation history but keeps the system prompt."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def inject_context(self):
        """Clear conversation history but preserve the full system prompt
        (including dynamic extensions like Obsidian vault config).
        Used by Project Brain to give the model a fresh context window."""
        self.messages = [
            {"role": "system", "content": self.messages[0]["content"]}
        ]
