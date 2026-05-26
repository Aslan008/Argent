import re
from rich.live import Live
from rich.console import Group
from rich.text import Text
from rich.spinner import Spinner as RichSpinner
from ui import (
    console, print_system, print_error, print_tool_start, print_tool_end,
    s, c, create_content_panel, print_reasoning_header,
    create_final_panel, safe_print, print_context_usage
)
from config import get_verbose_status

def render_response_stream(agent, response_chunks, is_auto_mode=False) -> tuple[str, bool, int, str]:
    """
    Renders the streaming response from the agent in real time.
    Handles thinking, content panels with Rich Live, tool execution, and errors.
    Returns a tuple (streamed_text, final_is_auto_mode, auto_sleep_time, auto_wake_context).
    """
    chunk_iterator = iter(response_chunks)
    streamed_text = ""
    streamed_thinking = ""
    is_tool_executing = False
    current_tool_name = ""
    verbose = get_verbose_status()
    
    auto_sleep_time = 0
    auto_wake_context = ""
    final_is_auto_mode = is_auto_mode
    
    while True:
        try:
            done = False
            if not is_tool_executing:
                # Phase 1: Thinking/Reasoning Stream
                has_started_reasoning = False
                is_waiting_ttft = True
                while True:
                    try:
                        if is_waiting_ttft and verbose:
                            with console.status("[bold cyan]Обработка...[/bold cyan]", spinner="dots"):
                                chunk = next(chunk_iterator)
                            is_waiting_ttft = False
                        else:
                            chunk = next(chunk_iterator)
                            is_waiting_ttft = False
                    except StopIteration:
                        done = True
                        break
                    
                    type_ = chunk.get("type")
                    if type_ == "thinking_stream":
                        if not has_started_reasoning:
                            print_reasoning_header()
                            has_started_reasoning = True
                        
                        content = chunk["content"]
                        streamed_thinking += content
                        console.print(content, end="", style=c.get("reasoning_text", "dim white"))
                    elif type_ == "tool_generating":
                        if has_started_reasoning:
                            console.print("\n")
                            has_started_reasoning = False
                        
                        tool_gen_name = chunk.get("name", "?")
                        tool_gen_bytes = chunk.get("bytes", 0)
                        if verbose:
                            with console.status(f"[dim cyan]Генерация {tool_gen_name}... ({tool_gen_bytes} B)[/dim cyan]", spinner="dots") as status:
                                while True:
                                    try:
                                        chunk = next(chunk_iterator)
                                    except StopIteration:
                                        done = True
                                        break
                                    type_ = chunk.get("type")
                                    if type_ == "tool_generating":
                                        tool_gen_name = chunk.get("name", tool_gen_name)
                                        tool_gen_bytes = chunk.get("bytes", 0)
                                        status.update(f"[dim cyan]Генерация {tool_gen_name}... ({tool_gen_bytes} B)[/dim cyan]")
                                    else:
                                        break
                        else:
                            while True:
                                try:
                                    chunk = next(chunk_iterator)
                                except StopIteration:
                                    done = True
                                    break
                                type_ = chunk.get("type")
                                if type_ != "tool_generating":
                                    break
                        if done:
                            break
                        if type_ == "tool_start":
                            print_tool_start(chunk["name"], chunk.get("args", {}))
                            is_tool_executing = True
                            current_tool_name = chunk["name"]
                            break
                        else:
                            break
                    else:
                        if has_started_reasoning:
                            console.print("\n")
                        break
                    
                # Phase 2: Main Content/Tool Stream
                if not done and not is_tool_executing:
                    with Live(create_content_panel(""), console=console, refresh_per_second=10, transient=True) as live:
                        while True:
                            if type_ in ("content_stream", "content"):
                                streamed_text += chunk["content"]
                                live.update(create_content_panel(streamed_text))
                            elif type_ == "content_replace":
                                streamed_text = chunk["content"]
                                live.update(create_content_panel(streamed_text))
                            elif type_ == "tool_generating":
                                break
                            elif type_ not in ("thinking_stream"):
                                break
                            
                            if streamed_text:
                                live.update(Group(
                                    create_content_panel(streamed_text),
                                    RichSpinner("dots", text=Text(" Генерация...", style="dim cyan"))
                                ))
                            
                            try:
                                chunk = next(chunk_iterator)
                                type_ = chunk.get("type")
                                if type_ in ("content_stream", "content", "content_replace"):
                                    pass
                                elif streamed_text:
                                    live.update(create_content_panel(streamed_text))
                            except StopIteration:
                                done = True
                                break
                
                if not done and not is_tool_executing:
                    if type_ == "tool_start":
                        print_tool_start(chunk["name"], chunk.get("args", {}))
                        is_tool_executing = True
                        current_tool_name = chunk["name"]
                    elif type_ == "tool_generating":
                        pass
                    elif type_ == "tool_end":
                        res = chunk.get("result", "")
                        print_tool_end(chunk["name"], res)
                        if final_is_auto_mode:
                            if "[END_AUTO_MODE]" in str(res):
                                final_is_auto_mode = False
                                print_system("🏁 Автоматический режим завершен агентом.")
                            elif "[HEARTBEAT_REQUEST:" in str(res):
                                match = re.search(r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\]", str(res))
                                if match:
                                    auto_sleep_time = int(match.group(1))
                                    auto_wake_context = f"[Heartbeat пробуждение] Причина: {match.group(2)}"
                    elif type_ == "error":
                        print_error(chunk["content"])
            
            else:
                # Tool is executing
                interactive_tools = {
                    "ask_user_questions", "run_command", "run_admin_command",
                    "start_background_command", "delete_file", "plan_work_changes"
                }
                use_spinner = verbose and current_tool_name not in interactive_tools
                
                if use_spinner:
                    with console.status("[dim cyan]Выполнение...[/dim cyan]", spinner="dots"):
                        try:
                            chunk = next(chunk_iterator)
                        except StopIteration:
                            done = True
                else:
                    try:
                        chunk = next(chunk_iterator)
                    except StopIteration:
                        done = True
                      
                if not done:
                    type_ = chunk.get("type")
                    if type_ == "tool_end":
                        res = chunk.get("result", "")
                        print_tool_end(chunk["name"], res)
                        if final_is_auto_mode:
                            if "[END_AUTO_MODE]" in str(res):
                                final_is_auto_mode = False
                                print_system("🏁 Автоматический режим завершен агентом.")
                            elif "[HEARTBEAT_REQUEST:" in str(res):
                                match = re.search(r"\[HEARTBEAT_REQUEST:\s*(\d+)\s*\|\s*(.*?)\]", str(res))
                                if match:
                                    auto_sleep_time = int(match.group(1))
                                    auto_wake_context = f"[Heartbeat пробуждение] Причина: {match.group(2)}"
                        is_tool_executing = False
                        current_tool_name = ""
                    elif type_ == "tool_start":
                        print_tool_start(chunk["name"], chunk.get("args", {}))
                        current_tool_name = chunk["name"]
                    elif type_ == "error":
                        print_error(chunk["content"])
                
            if done:
                break
                         
        except StopIteration:
            break
        except Exception as e:
            print_error(f"Streaming error: {e}")
            break
            
    # Final render
    if streamed_text:
        for el in create_final_panel(streamed_text):
            safe_print(el)
            safe_print("")
            
    return streamed_text, final_is_auto_mode, auto_sleep_time, auto_wake_context
