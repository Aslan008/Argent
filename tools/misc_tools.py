import json
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
import questionary
from memory_manager import memory
from intelligence import intel
from mcp_client import mcp_client
from ui import console
from logger import get_logger

log = get_logger("tools")

def ask_user_questions(questions: list) -> str:
    """Ask the user a series of structured questions."""
    from prompt_toolkit import prompt as ptk_prompt
    
    if not isinstance(questions, list):
        return "Error: 'questions' must be a JSON array of objects."
        
    responses = {}
    console.print("\n[bold cyan]🔍 Уточнение требований:[/bold cyan]")
    
    for q in questions:
        q_type = q.get("type", "text")
        q_text = q.get("question", "Question?")
        options = q.get("options", [])
        
        console.print(f"\n[bold yellow]{q_text}[/bold yellow]")
        
        if q_type == "text":
            console.print("[dim](Введите текст и нажмите Enter)[/dim]")
            try:
                answer = ptk_prompt("Ваш ответ ❯ ")
                responses[q_text] = answer.strip() if answer.strip() else "No answer"
            except (KeyboardInterrupt, EOFError):
                responses[q_text] = "Skipped"
                
        elif q_type in ("single_choice", "multi_choice"):
            display_options = options.copy()
            if "✏ Свой вариант..." not in display_options:
                display_options.append("✏ Свой вариант...")
                
            if q_type == "single_choice":
                console.print("[dim](Выберите один вариант стрелками ↑↓ и нажмите Enter)[/dim]")
                try:
                    selected = questionary.select("Выберите:", choices=display_options).ask()
                    if selected == "✏ Свой вариант...":
                        custom = ptk_prompt("Введите свой вариант ❯ ")
                        responses[q_text] = custom.strip() if custom.strip() else "No answer"
                    elif selected:
                        responses[q_text] = selected
                    else:
                        responses[q_text] = "Skipped"
                except (KeyboardInterrupt, EOFError):
                    responses[q_text] = "Skipped"
            else:
                console.print("[dim](Выделите пробелом нужные варианты и нажмите Enter)[/dim]")
                try:
                    selected = questionary.checkbox("Выберите варианты:", choices=display_options).ask()
                    if selected and "✏ Свой вариант..." in selected:
                        selected.remove("✏ Свой вариант...")
                        custom = ptk_prompt("Введите свой(и) вариант(ы) через запятую ❯ ")
                        if custom.strip():
                            selected.append(custom.strip())
                    
                    responses[q_text] = ", ".join(selected) if selected else "No answer"
                except (KeyboardInterrupt, EOFError):
                    responses[q_text] = "Skipped"
    
    summary_lines = []
    for k, v in responses.items():
        summary_lines.append(f"- {k}: {v}")
        memory.add_fact(f"User preference on '{k}': {v}")
        
    return f"User responses:\n" + "\n".join(summary_lines)

def create_svg_image(svg_code: str, filename: str = None) -> str:
    """Creates an SVG image file from the provided SVG code and opens it in the default web browser."""
    try:
        from config import get_visuals_dir
        visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
        visuals_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"svg_{timestamp}.svg"
        
        if not filename.endswith(".svg"):
            filename += ".svg"
            
        file_path = visuals_dir / filename
        
        if "<?xml" not in svg_code:
            svg_code = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_code
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_code)
            
        webbrowser.open(f"file:///{file_path}")
        
        return f"Successfully created SVG image at '{file_path}'. It should now be open in your browser."
    except Exception as e:
        return f"Error creating SVG image: {e}"

def find_definition(file_path: str, line: int, column: int) -> str:
    """Find the definition of a symbol at the given line and column."""
    results = intel.find_definitions(file_path, line, column)
    if not results:
        return "No definitions found."
    if "error" in results[0]:
        return f"Error finding definitions: {results[0]['error']}"
    
    out = "Found definitions:\n"
    for d in results:
        out += f"- {d['name']} ({d['type']}) in {d['file_path']}:{d['line']}:{d['column']}\n"
        out += f"  {d['description']}\n"
    return out

def find_references(file_path: str, line: int, column: int) -> str:
    """Find all references to a symbol at the given line and column."""
    results = intel.find_references(file_path, line, column)
    if not results:
        return "No references found."
    if "error" in results[0]:
        return f"Error finding references: {results[0]['error']}"
    
    out = "Found references:\n"
    for r in results:
        out += f"- {r['name']} in {r['file_path']}:{r['line']}:{r['column']}\n"
    return out

def git_checkpoint(message: str) -> str:
    """Create a temporary git commit (checkpoint) to save state before an experiment."""
    try:
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        if res.returncode != 0:
            return "Error: Not a git repository. Checkpoints require git."
        
        subprocess.run(["git", "add", "."], check=True)
        res = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if res.returncode == 0:
            return "No changes to checkpoint."
            
        subprocess.run(["git", "commit", "-m", f"Argent Checkpoint: {message}"], check=True)
        return f"Checkpoint created: '{message}'"
    except Exception as e:
        return f"Error creating checkpoint: {e}"

def git_rollback() -> str:
    """Roll back the last checkpoint (git reset --hard HEAD~1)."""
    try:
        res = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
        if res.returncode != 0:
            return "Error: Not a git repository."
        
        res = subprocess.run(["git", "log", "-1", "--pretty=%B"], capture_output=True, text=True)
        last_msg = res.stdout.strip()
        
        if not last_msg.startswith("Argent Checkpoint:"):
            return f"Error: The last commit ('{last_msg}') was not an Argent Checkpoint. Rollback aborted for safety."
            
        from approval import request_approval
        approved = request_approval(
            f"откатить ВСЕ изменения до чекпоинта '{last_msg}' (git reset --hard)",
            destructive=True,
        )
        if not approved:
            return "Rollback aborted by user."

        # Stash unstaged and untracked changes for safety before hard reset
        status_res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if status_res.stdout.strip():
            console.print("\n[bold yellow]Unstaged or untracked changes detected. Stashing them for safety before rollback...[/bold yellow]")
            subprocess.run(["git", "stash", "push", "-u", "-m", f"Argent Auto-Save before Rollback to {last_msg}"], capture_output=True)
            
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        return f"Successfully rolled back: {last_msg}"
    except Exception as e:
        return f"Error rolling back: {e}"

def call_mcp_tool(server_name: str, tool_name: str, arguments_json: str) -> str:
    """Call a standardized tool from an MCP server. arguments_json must be a valid JSON string."""
    args = None
    try:
        args = json.loads(arguments_json)
    except json.JSONDecodeError:
        try:
            import json5
            args = json5.loads(arguments_json)
        except ImportError:
            pass # Если json5 не установлен, просто падаем ниже
        except Exception:
            pass # Если json5 тоже не справился, идем к ошибке

    if args is None:
        return (
            "Error: arguments_json is invalid JSON. "
            "Did you use single quotes instead of double quotes? "
            "Did you forget to escape newlines (\\n) or quotes (\\\") inside your prompt? "
            "Fix the syntax and try again."
        )

    try:
        return mcp_client.call_tool(server_name, tool_name, args)
    except Exception as e:
        return f"Error calling MCP tool: {e}"

def run_subagent(role: str, task: str, tools_json: str = None) -> str:
    """Spawn a specialized sub-agent for an isolated task."""
    from agent import ArgentSubAgent
    tools = None
    if tools_json:
        try:
            tools = json.loads(tools_json)
        except Exception:
            pass
    agent = ArgentSubAgent(role, task, tools)
    return agent.execute()

def create_artifact(filename: str, content: str) -> str:
    """Create a Markdown artifact in the .argent/artifacts/ directory. Useful for plans or long text."""
    try:
        path = Path(".argent") / "artifacts" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("create_artifact: %s", path)
        return f"Artifact created at {path.absolute()}. The user can now review it."
    except Exception as e:
        return f"Error creating artifact: {e}"

def request_user_approval(message: str) -> str:
    """Pause execution and ask the user for approval. Use this after generating an implementation plan."""
    from ui import console
    import questionary
    console.print(f"\n[bold yellow]Agent requests approval:[/bold yellow] {message}")
    approved = questionary.confirm("Do you approve this plan/action?").ask()
    if approved:
        memory.add_completed(f"Approved plan: {message}")
        return "User approved. Proceed with execution."
    else:
        memory.add_completed(f"Rejected plan: {message}")
        return "User REJECTED. Please ask the user for feedback or revise your plan."

def wait_heartbeat(delay_seconds: int, condition_to_check: str) -> str:
    """Schedules a delayed continuation in Auto Mode."""
    return f"Heartbeat scheduled. [HEARTBEAT_REQUEST: {delay_seconds}|{condition_to_check}]"

def end_auto_mode(reason: str) -> str:
    """Stops the experimental Auto Mode."""
    return f"Auto Mode finished. [END_AUTO_MODE] Reason: {reason}"
