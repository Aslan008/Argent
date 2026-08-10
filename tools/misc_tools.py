import ast
import json
import math
import operator
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


def _safe_pow(a, b):
    # Guard against DoS expressions like 9**9**9 that hang the process.
    if abs(b) > 10000 or (abs(a) > 1e6 and abs(b) > 100):
        raise ValueError("exponent too large for the calculator")
    return operator.pow(a, b)


_CALC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _safe_pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CALC_FUNCTIONS = {name: getattr(math, name) for name in (
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "log", "log2", "log10", "exp", "floor", "ceil", "fabs",
    "degrees", "radians", "factorial", "gcd",
)}
_CALC_FUNCTIONS.update({"abs": abs, "round": round, "min": min, "max": max})

_CALC_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _eval_calc_node(node):
    """Recursively evaluate a whitelisted arithmetic AST node. No code execution."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_eval_calc_node(node.left), _eval_calc_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_eval_calc_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CALC_CONSTANTS:
            return _CALC_CONSTANTS[node.id]
        raise ValueError(f"unknown identifier '{node.id}'")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCTIONS and not node.keywords:
            args = [_eval_calc_node(a) for a in node.args]
            return _CALC_FUNCTIONS[node.func.id](*args)
        raise ValueError("only these functions are allowed: " + ", ".join(sorted(_CALC_FUNCTIONS)))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression exactly. Two engines, one entry point:

    plain arithmetic runs on a fast whitelist evaluator, and anything symbolic
    (integrals, limits, derivatives, equations, sums, series) falls through to
    SymPy. Neither path ever executes code.
    """
    if not expression or not expression.strip():
        return "Error: expression is empty."
    # Models often write ^ meaning exponentiation.
    normalized = expression.strip().replace("^", "**")

    # --- Fast path: pure arithmetic, no dependencies, exact ints ---
    try:
        tree = ast.parse(normalized, mode="eval")
        result = _eval_calc_node(tree.body)
        if isinstance(result, float) and result.is_integer() and abs(result) < 1e15:
            result = int(result)
        log.info("calculate: %s = %s", expression, result)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return f"Error: division by zero in '{expression}'."
    except (ValueError, SyntaxError, OverflowError, TypeError) as numeric_error:
        pass

    # --- Symbolic path: the expression mentions variables or calculus ---
    from tools.symbolic_math import SymbolicError, evaluate_symbolic
    try:
        answer = evaluate_symbolic(expression)
    except SymbolicError as symbolic_error:
        return (
            f"Error: cannot evaluate '{expression}': {symbolic_error}. "
            f"Provide a plain arithmetic expression (e.g. '(1847 * 0.15) + sqrt(2)') or a "
            f"symbolic one (e.g. 'integrate(x**2, x)', 'limit(sin(x)/x, x, 0)', "
            f"'solve(x**2 - 4, x)')."
        )
    except Exception as e:  # sympy failed on a well-formed but hard expression
        return f"Error: cannot evaluate '{expression}': {e}."

    log.info("calculate (symbolic): %s = %s", expression, answer)
    return f"{expression} = {answer}"


def set_goal(objective: str = None, current_task: str = None) -> str:
    """Record the overall objective and/or the current sub-task as a persistent goal.

    Argent re-pins these at the end of the context as a reminder, so the model
    keeps sight of the goal on long runs. Does not touch any files.
    """
    changed = []
    if objective and objective.strip():
        memory.set_objective(objective.strip())
        changed.append(f"OBJECTIVE = {objective.strip()[:120]}")
    if current_task and current_task.strip():
        memory.set_current_task(current_task.strip())
        changed.append(f"CURRENT TASK = {current_task.strip()[:120]}")
    if not changed:
        return "Error: provide 'objective' and/or 'current_task' to set the goal."
    log.info("set_goal: %s", "; ".join(changed))
    return "Goal updated: " + "; ".join(changed)


# Keys a model plausibly uses when it writes an option as an object instead of
# the declared string. questionary renders any dict without "name" as the text
# "None", so an unrecognised shape does not fail — it produces a menu of four
# Nones that the user cannot answer and nothing anywhere records why.
_OPTION_TEXT_KEYS = ("name", "label", "title", "text", "option", "value",
                     "description", "choice")


def _option_text(option):
    """One option as displayable text, or None if there is nothing to show."""
    if isinstance(option, str):
        return option.strip() or None
    if isinstance(option, (int, float, bool)):
        return str(option)
    if isinstance(option, dict):
        for key in _OPTION_TEXT_KEYS:
            value = option.get(key)
            if isinstance(value, str) and value.strip():
                # A label plus its explanation reads better than the label
                # alone, and the model wrote the explanation for a reason.
                detail = option.get("description")
                if key != "description" and isinstance(detail, str) and detail.strip():
                    return f"{value.strip()} — {detail.strip()}"
                return value.strip()
        # A checklist written as {"Система HP": false} — the text is the KEY.
        # Only with exactly one entry, so there is no choosing between two
        # candidates; with more, we would be inventing which one the user sees.
        if len(option) == 1:
            only_key = next(iter(option))
            if isinstance(only_key, str) and only_key.strip():
                return only_key.strip()
        return None
    if isinstance(option, (list, tuple)):
        # ["Система HP", false] — a value paired with its state. Exactly one
        # string in it means there is nothing to disambiguate.
        strings = [x.strip() for x in option if isinstance(x, str) and x.strip()]
        if len(strings) == 1:
            return strings[0]
        return None
    return None


def _fit_options(texts):
    """Draw options that fit the terminal without losing what they say.

    questionary draws one line per choice and clips the overflow. Eliding the
    middle was worse than the clipping: on sentence-length options it produced

        Главная страница подписывается на BroadcastChannel и localStora...оя
        всё равно нужен экспорт YAML.

    which reads as damage, not as a choice. So when anything overflows, the
    FULL text is printed above the menu as a numbered list — the terminal wraps
    it properly there — and the menu carries "N. <beginning>…". The number ties
    the two together, and the value stays whole so the model still receives the
    complete answer.

    Returns (choices, listing) where listing is the block to print, or "".
    """
    import shutil

    width = max(40, shutil.get_terminal_size((110, 30)).columns)
    budget = width - 12          # questionary's pointer, padding and a margin
    if all(len(t) <= budget for t in texts):
        return [(t, t) for t in texts], ""

    numbered = budget - 5        # room for "N. " and the ellipsis
    choices, lines = [], []
    for i, text in enumerate(texts, 1):
        lines.append(f"  [bold]{i}.[/bold] {text}")
        label = text if len(text) <= numbered else f"{text[:numbered].rstrip()}…"
        choices.append((f"{i}. {label}", text))
    return choices, "\n".join(lines)


def _normalize_options(raw):
    """(usable options, the raw entries that could not be displayed).

    The rejects are returned rather than counted: knowing THAT options were
    malformed does not tell you what shape to support next, and the payload is
    gone by the time anyone reads the log. That gap cost a debugging session.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [line for line in raw.splitlines() if line.strip()]
    if not isinstance(raw, (list, tuple)):
        return [], [raw]
    texts, rejected = [], []
    for option in raw:
        text = _option_text(option)
        if text is None:
            rejected.append(option)
        elif text not in texts:
            texts.append(text)
    return texts, rejected


def ask_user_questions(questions: list) -> str:
    """Ask the user a series of structured questions."""
    from prompt_toolkit import prompt as ptk_prompt

    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except json.JSONDecodeError:
            return ("Error: 'questions' must be a JSON array of objects, e.g. "
                    '[{"type": "single_choice", "question": "...", '
                    '"options": ["A", "B"]}]')
    if isinstance(questions, dict):
        questions = [questions]          # one question, sent unwrapped
    if not isinstance(questions, list) or not questions:
        return "Error: 'questions' must be a non-empty JSON array of objects."

    responses = {}
    malformed = []
    console.print("\n[bold cyan]🔍 Уточнение требований:[/bold cyan]")

    for q in questions:
        if not isinstance(q, dict):
            malformed.append(str(q)[:60])
            continue
        q_type = q.get("type", "text")
        q_text = q.get("question", "Question?")
        raw_options = q.get("options", [])
        options, rejected = _normalize_options(raw_options)
        if rejected:
            malformed.append(q_text)
            # The payload, not just the fact: without it the next occurrence is
            # as unexplainable as this one was.
            log.warning("ask_user_questions: %d unusable option(s) in %r -> %s",
                        len(rejected), q_text[:60],
                        json.dumps(rejected, ensure_ascii=False, default=str)[:400])

        if q_type in ("single_choice", "multi_choice") and not options:
            # A choice with nothing to choose from is unanswerable. Asking it as
            # free text at least keeps the question, instead of showing an empty
            # menu the user has to escape out of.
            console.print(f"\n[bold yellow]{q_text}[/bold yellow]")
            console.print("[dim](варианты пришли в неподдерживаемом виде — "
                          "ответьте своими словами)[/dim]" if rejected else
                          "[dim](вариантов не пришло — ответьте своими словами)[/dim]")
            try:
                answer = ptk_prompt("Ваш ответ ❯ ")
                responses[q_text] = answer.strip() or "No answer"
            except (KeyboardInterrupt, EOFError):
                responses[q_text] = "Skipped"
            continue

        console.print(f"\n[bold yellow]{q_text}[/bold yellow]")

        if q_type == "text":
            console.print("[dim](Введите текст и нажмите Enter)[/dim]")
            try:
                answer = ptk_prompt("Ваш ответ ❯ ")
                responses[q_text] = answer.strip() if answer.strip() else "No answer"
            except (KeyboardInterrupt, EOFError):
                responses[q_text] = "Skipped"
                
        elif q_type in ("single_choice", "multi_choice"):
            # Drawn short enough to survive the terminal width, answered in
            # full: the value carries the whole text even when the label is
            # shortened, so the model never receives a truncated answer.
            fitted, listing = _fit_options(options)
            if listing:
                console.print(listing)
            display_options = [questionary.Choice(title=label, value=value)
                               for label, value in fitted]
            CUSTOM = "✏ Свой вариант..."
            display_options.append(questionary.Choice(title=CUSTOM, value=CUSTOM))

            if q_type == "single_choice":
                console.print("[dim](Выберите один вариант стрелками ↑↓ и нажмите Enter)[/dim]")
                try:
                    selected = questionary.select("Выберите:", choices=display_options).ask()
                    if selected == CUSTOM:
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
                    if selected and CUSTOM in selected:
                        selected.remove(CUSTOM)
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

    out = "User responses:\n" + "\n".join(summary_lines)
    if malformed:
        # Told back to the model, not just logged: it can only stop sending the
        # wrong shape if something says so within the turn.
        log.warning("ask_user_questions received malformed options in: %s", malformed)
        out += ("\n\n[Note: some options were not plain strings and had to be "
                "converted or dropped. `options` must be an array of STRINGS, "
                'e.g. "options": ["Only critical paths", "Full coverage"] — not '
                "objects. Re-check the schema before asking again.]")
    return out

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
    """Find the definition of a symbol at the given line and column.

    Uses LSP (multilspy) when available — cross-language, type-aware — and
    falls back to the Jedi-based intelligence module for Python.
    """
    # Try LSP first (cross-language, type-aware)
    try:
        from src.lsp.manager import lsp_manager
        if lsp_manager.is_available():
            results = lsp_manager.find_definition(file_path, line, column)
            if results:
                out = "Found definitions (LSP):\n"
                for d in results:
                    out += f"- {d['file_path']}:{d['line']}:{d['column']}\n"
                return out.rstrip()
    except Exception:
        pass

    # Fall back to Jedi (Python only)
    results = intel.find_definitions(file_path, line, column)
    if not results:
        return "No definitions found."
    if "error" in results[0]:
        return f"Error finding definitions: {results[0]['error']}"
    
    out = "Found definitions:\n"
    for d in results:
        out += f"- {d['name']} ({d['type']}) in {d['file_path']}:{d['line']}:{d['column']}\n"
        out += f"  {d['description']}\n"
    return out.rstrip()

def find_references(file_path: str, line: int, column: int) -> str:
    """Find all references to a symbol at the given line and column.

    Uses LSP (multilspy) when available — cross-language, type-aware — and
    falls back to the Jedi-based intelligence module for Python.
    """
    # Try LSP first (cross-language, type-aware)
    try:
        from src.lsp.manager import lsp_manager
        if lsp_manager.is_available():
            results = lsp_manager.find_references(file_path, line, column)
            if results:
                out = "Found references (LSP):\n"
                for r in results:
                    out += f"- {r['file_path']}:{r['line']}:{r['column']}\n"
                return out.rstrip()
    except Exception:
        pass

    # Fall back to Jedi (Python only)
    results = intel.find_references(file_path, line, column)
    if not results:
        return "No references found."
    if "error" in results[0]:
        return f"Error finding references: {results[0]['error']}"
    
    out = "Found references:\n"
    for r in results:
        out += f"- {r['name']} in {r['file_path']}:{r['line']}:{r['column']}\n"
    return out.rstrip()

def git_checkpoint(message: str) -> str:
    """Create a temporary git commit (checkpoint) to save state before an experiment."""
    from src.agent import checkpoints
    try:
        if not checkpoints.is_git_repo():
            return "Error: Not a git repository. Checkpoints require git."
        sha = checkpoints.create_checkpoint(message)
        if sha is None:
            return "No changes to checkpoint."
        return f"Checkpoint created: '{message}' ({sha})"
    except Exception as e:
        return f"Error creating checkpoint: {e}"

def git_rollback(to_checkpoint: str = None) -> str:
    """Roll back to an Argent Checkpoint. Without arguments — the most recent
    one; to_checkpoint picks an older checkpoint by short sha or a substring
    of its message (e.g. what the user asked to return to)."""
    from src.agent import checkpoints
    try:
        if not checkpoints.is_git_repo():
            return "Error: Not a git repository."

        available = checkpoints.list_checkpoints()
        if not available:
            return "Error: no Argent Checkpoints found in recent history. Nothing to roll back to."

        target = None
        if to_checkpoint and to_checkpoint.strip():
            needle = to_checkpoint.strip().lower()
            for cp in available:
                if cp["sha"].lower().startswith(needle) or needle in cp["label"].lower():
                    target = cp
                    break
            if target is None:
                listing = "\n".join(f"- {c['sha']}  {c['label']}  ({c['age']})" for c in available)
                return (f"Error: no checkpoint matches '{to_checkpoint}'. Available checkpoints:\n"
                        f"{listing}")
        else:
            target = available[0]

        from approval import request_approval
        approved = request_approval(
            f"откатить ВСЕ изменения до чекпоинта '{target['label']}' ({target['sha']}, git reset --hard)",
            destructive=True,
        )
        if not approved:
            return "Rollback aborted by user."

        return checkpoints.rewind_to(target["sha"])
    except Exception as e:
        return f"Error rolling back: {e}"

# Above this, dumping every signature is worse than useless: it buries the two
# tools the model actually wants. Unity's MCP server alone exposes 140.
MCP_DETAIL_LIMIT = 25


def _mcp_tool_line(tool: dict, detailed: bool) -> str:
    name = tool.get("name", "?")
    if not detailed:
        return f"  - {name}"
    schema = tool.get("inputSchema") or tool.get("parameters") or {}
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = ", ".join(p if p in required else f"{p}?" for p in list(props)[:8])
    desc = (tool.get("description") or "").strip().split(".")[0][:110]
    return f"  - {name}({params}) — {desc}"


def list_mcp_tools(server_name: str = "", filter: str = "") -> str:
    """Discover what an MCP server offers, instead of carrying its whole catalog.

    A server's tool list is fetched on demand and cached, so asking is cheap;
    pinning 140 signatures into every system prompt is not. The model gets a
    one-line map up front and comes here when it needs a signature.
    """
    from config import get_mcp_servers

    configured = get_mcp_servers()
    if not configured:
        return "No MCP servers are configured. Add one with /mcp."

    if not (server_name or "").strip():
        lines = ["Configured MCP servers:"]
        for srv in configured:
            name = srv["name"]
            tools = [t for t in mcp_client.list_tools(name) if "error" not in t]
            state = f"{len(tools)} tools" if tools else "unreachable"
            lines.append(f"  - {name} ({srv.get('type', 'stdio')}): {state}")
        lines.append("Call list_mcp_tools(server_name, filter) to see signatures.")
        return "\n".join(lines)

    server_name = server_name.strip()
    if server_name not in [s["name"] for s in configured]:
        known = ", ".join(s["name"] for s in configured)
        return f"Error: no MCP server named '{server_name}'. Configured: {known}."

    tools = mcp_client.list_tools(server_name)
    broken = [t for t in tools if "error" in t]
    tools = [t for t in tools if "name" in t and "error" not in t]
    if not tools:
        reason = broken[0].get("error") if broken else "server returned no tools"
        return f"Server '{server_name}' offers nothing right now: {reason}"

    needle = (filter or "").strip().lower()
    if needle:
        matched = [t for t in tools
                   if needle in t.get("name", "").lower()
                   or needle in (t.get("description") or "").lower()]
        if not matched:
            sample = ", ".join(t["name"] for t in tools[:15])
            return (f"No tool on '{server_name}' matches '{filter}'. "
                    f"{len(tools)} available, e.g.: {sample}")
    else:
        matched = tools

    # Signatures only when the list is short enough to read. Otherwise names,
    # plus the nudge that got the model here in the first place.
    detailed = len(matched) <= MCP_DETAIL_LIMIT
    head = f"'{server_name}' — {len(matched)} of {len(tools)} tools" if needle \
        else f"'{server_name}' — {len(tools)} tools"
    lines = [head + ":"]
    lines += [_mcp_tool_line(t, detailed) for t in matched[:120]]
    if len(matched) > 120:
        lines.append(f"  ... and {len(matched) - 120} more")
    if not detailed:
        lines.append("Names only — pass `filter` to get signatures for what you need.")
    return "\n".join(lines)


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
    console.print(f"\n[bold yellow]Агент просит подтверждения:[/bold yellow] {message}")
    console.print("[dim](y — принять, n — отклонить, Enter — принять)[/dim]")
    try:
        approved = questionary.confirm("Одобряете?", default=True).ask()
    except (KeyboardInterrupt, EOFError):
        approved = None
    if approved is None:
        # Escaped rather than answered. Treating that as approval would let a
        # Ctrl+C start the very work the user was declining to authorise.
        memory.add_completed(f"Approval dismissed: {message}")
        return ("User dismissed the approval prompt without answering. Treat it as a NO: "
                "do not proceed, ask what they want changed.")
    if approved:
        memory.add_completed(f"Approved plan: {message}")
        return "User approved. Proceed with execution."
    memory.add_completed(f"Rejected plan: {message}")
    return "User REJECTED. Please ask the user for feedback or revise your plan."

def wait_heartbeat(delay_seconds: int = 0, condition_to_check: str = "",
                   until: str = "", timeout_seconds: int = 600) -> str:
    """Sleep during Auto Mode, either for a fixed delay or UNTIL something is true.

    `until` is the cheap path: waking up costs a full model turn, so polling
    "is it done yet?" ten times costs ten turns. A condition is checked locally
    for free, and the agent wakes once — when it actually happened.
    """
    from src.agent.wait_conditions import (
        ConditionError, MAX_TIMEOUT_SECONDS, describe_predicates, evaluate,
    )

    if until and until.strip():
        # Fail fast on a malformed condition: the alternative is sleeping the
        # whole timeout and reporting a mystery.
        try:
            evaluate(until)
        except ConditionError as e:
            return (f"Error: {e}\nAvailable checks: {describe_predicates()}\n"
                    f"Example: wait_heartbeat(until='file_contains(\"build.log\", "
                    f"\"BUILD SUCCESSFUL\")', timeout_seconds=600)")
        try:
            timeout = max(1, min(int(timeout_seconds or 600), MAX_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            timeout = 600
        reason = condition_to_check or until
        return (f"Waiting until: {until} (timeout {timeout}s). "
                f"[HEARTBEAT_REQUEST: 0|{reason}|until={until}|timeout={timeout}]")

    try:
        delay = max(0, int(delay_seconds or 0))
    except (TypeError, ValueError):
        delay = 0
    if delay <= 0:
        return ("Error: provide either delay_seconds (a fixed sleep) or until "
                f"(a condition to wait for). Available checks: {describe_predicates()}")
    return f"Heartbeat scheduled. [HEARTBEAT_REQUEST: {delay}|{condition_to_check}]"

def view_image(file_path: str = "", question: str = "", path: str = "") -> str:
    """Attach an image to the conversation so the model can actually see it.

    ``path`` is accepted because browser_screenshot RETURNS its file under that
    name, so a model chaining the two writes view_image(path=…) — observed on
    the first real use. Rejecting it would be punishing the model for our own
    inconsistent naming.

    The picture cannot travel in a tool result — those are strings — so it is
    queued here and the turn loop attaches it as the next user message. The
    model therefore sees it on its NEXT step, which the return value says
    plainly: a model told "done" would otherwise answer about an image it has
    not received yet.
    """
    from config import get_current_model, get_provider
    from vision import encode_image, model_supports_vision, queue_image

    file_path = (file_path or path or "").strip()
    if not file_path:
        return "Error: give the image path, e.g. view_image('screenshot.png')."

    provider = get_provider()
    model = get_current_model()
    supports = model_supports_vision(model, provider)
    if supports is False:
        return (f"Error: model '{model}' cannot see images (Ollama reports no "
                f"'vision' capability). Switch to a vision model with /model, or "
                f"read the page as text with browser_get_content.")

    payload, media_type, note = encode_image(file_path)
    if payload is None:
        return f"Error: {note}"

    from tools._helpers import _resolve_path
    label = str(_resolve_path(file_path))
    queue_image(payload, media_type, label)

    detail = f" ({note})" if note else ""
    unsure = ("\nNote: this model's vision support could not be verified — if the "
              "next message looks like it has no image, it does not.") if supports is None else ""
    asked = f" You asked: {question}" if question else ""
    return (f"Image '{label}' attached{detail}. You will SEE it in the next "
            f"message, not in this result — continue and describe what is "
            f"actually there.{asked}{unsure}")


def end_auto_mode(reason: str) -> str:
    """Stops the experimental Auto Mode."""
    return f"Auto Mode finished. [END_AUTO_MODE] Reason: {reason}"


def filter_new_items(items: list, label: str = "") -> str:
    """Drop everything this task already reported on a previous run.

    Monitoring only means anything if it can say "this is new". Without a
    memory across runs, a scheduled job re-reports the same twenty results
    every time and you stop reading it by the third day.

    The items are staged, not committed: if the run fails afterwards they stay
    unseen and come back next time, because a silently swallowed item is worse
    than a repeated one.
    """
    from src.automation.memory import current_scope, filter_new

    if isinstance(items, str):
        # Small models sometimes send a JSON string or one item per line.
        text = items.strip()
        if text.startswith("["):
            try:
                items = json.loads(text)
            except json.JSONDecodeError:
                items = [line for line in text.splitlines() if line.strip()]
        else:
            items = [line for line in text.splitlines() if line.strip()]
    if not isinstance(items, list):
        return "Error: items must be a list of identifiers (URLs, ids or titles)."

    total = len(items)
    if total == 0:
        return "No items given, so nothing is new."

    fresh = filter_new(items)
    where = f" for '{current_scope()}'" if current_scope() else ""
    if not fresh:
        return (f"0 of {total} items are new{where} — everything here was already "
                f"reported. Say so instead of repeating the old list.")

    listed = "\n".join(f"- {item}" for item in fresh[:50])
    more = f"\n... and {len(fresh) - 50} more" if len(fresh) > 50 else ""
    header = f"{len(fresh)} of {total} items are new{where}"
    if label:
        header += f" ({label})"
    return f"{header}:\n{listed}{more}"
