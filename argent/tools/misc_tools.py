import ast
import json
import math
import operator
import re
from fractions import Fraction
from pathlib import Path
from typing import Any
import questionary
from argent.memory_manager import memory
from argent.intelligence import intel
from argent.mcp_client import mcp_client
from argent.ui import console
from argent.logger import get_logger

log = get_logger("tools")


class _CalcGuardError(Exception):
    """A DoS guard tripped (huge exponent / huge result). This is a definitive
    answer — report it, never reroute to another engine that might hang."""


def _safe_pow(a, b):
    # Guard against DoS expressions like 9**9**9 or 10**5000 that hang the process
    # or exceed Python's 4300-digit string conversion limit.
    try:
        if abs(b) > 10000:
            raise _CalcGuardError("exponent too large for the calculator")
        abs_a = abs(a)
        if abs_a > 1 and abs(b) > 0:
            digits = abs(b) * math.log10(abs_a)
            if digits > 2000:
                raise _CalcGuardError(
                    "power operation result exceeds maximum allowed digits (2000)")
    except (TypeError, ValueError, OverflowError):
        # Could not measure the operands (e.g. an unusual base); let pow try.
        pass
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


def _safe_factorial(value):
    """factorial with a size cap so a huge argument cannot hang the process."""
    if isinstance(value, Fraction):
        if value.denominator != 1:
            raise ValueError("factorial is only defined for integers")
        value = value.numerator
    if isinstance(value, int) and not isinstance(value, bool) and value > 20000:
        raise _CalcGuardError("factorial argument too large for the calculator")
    return math.factorial(value)


_CALC_FUNCTIONS["factorial"] = _safe_factorial

_CALC_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


class _NeedsSymbolic(Exception):
    """The expression is not plain arithmetic (a variable, calculus, a method
    call…) — hand it to SymPy instead of failing."""


def _is_exact(value) -> bool:
    """True for ints and Fractions — operands that divide without losing digits."""
    return isinstance(value, (int, Fraction)) and not isinstance(value, bool)


def _calc_divide(left, right):
    """Divide exactly when both operands are exact (int/Fraction) so that
    ``10**30 / 3`` stays a true rational instead of collapsing to a 16-digit
    float; fall back to float division only when a float is already involved."""
    if _is_exact(left) and _is_exact(right):
        if right == 0:
            raise ZeroDivisionError
        return Fraction(left) / Fraction(right)
    return operator.truediv(left, right)


def _eval_calc_node(node):
    """Recursively evaluate a whitelisted arithmetic AST node. No code execution.

    Raises _NeedsSymbolic for anything that is not plain arithmetic, and real
    errors (ZeroDivisionError, ValueError from a function…) for actual faults.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            raise ValueError(f"unsupported constant: {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # Keep a decimal literal as the exact number the user typed
            # (0.1 -> 1/10, not the binary 0.1000000000000000055…), so that
            # 0.1 + 0.2 = 0.3 rather than 0.30000000000000004.
            if math.isfinite(value):
                try:
                    return Fraction(str(value))
                except (ValueError, ZeroDivisionError):
                    return value
            return value
        raise ValueError(f"unsupported constant: {value!r}")

    if isinstance(node, ast.BinOp):
        left = _eval_calc_node(node.left)
        right = _eval_calc_node(node.right)
        if isinstance(node.op, ast.Div):
            return _calc_divide(left, right)
        handler = _CALC_OPERATORS.get(type(node.op))
        if handler is None:
            raise _NeedsSymbolic(f"operator {type(node.op).__name__}")
        return handler(left, right)

    if isinstance(node, ast.UnaryOp):
        handler = _CALC_OPERATORS.get(type(node.op))
        if handler is None:
            raise _NeedsSymbolic(f"operator {type(node.op).__name__}")
        return handler(_eval_calc_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CALC_CONSTANTS:
            return _CALC_CONSTANTS[node.id]
        raise _NeedsSymbolic(f"variable '{node.id}'")

    if isinstance(node, ast.List):
        return [_eval_calc_node(el) for el in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_eval_calc_node(el) for el in node.elts)

    if isinstance(node, ast.Call):
        # Only a bare whitelisted math function runs here. Everything else
        # (integrate, solve, f(x), obj.method()) belongs to the symbolic engine.
        if (isinstance(node.func, ast.Name) and node.func.id in _CALC_FUNCTIONS
                and not node.keywords):
            args = [_eval_calc_node(a) for a in node.args]
            return _CALC_FUNCTIONS[node.func.id](*args)
        raise _NeedsSymbolic("non-arithmetic call")

    raise _NeedsSymbolic(f"expression element {type(node).__name__}")


def _fraction_to_decimal(fr: Fraction):
    """Exact decimal string for a terminating fraction (denominator 2^a·5^b),
    otherwise None — e.g. 7/2 -> '3.5', 4456151/99 -> None."""
    denom = fr.denominator
    rest, twos, fives = denom, 0, 0
    while rest % 2 == 0:
        rest //= 2
        twos += 1
    while rest % 5 == 0:
        rest //= 5
        fives += 1
    if rest != 1:
        return None
    places = max(twos, fives)
    scaled = fr.numerator * (10 ** places // denom)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    if places:
        return f"{sign}{digits[:-places]}.{digits[-places:]}"
    return f"{sign}{digits}"


_MAX_SHOWN_DIGITS = 4000  # stay safely under CPython's 4300-digit int→str limit


def _int_text(n: int) -> str:
    """str() of an int, but never trip Python's 4300-digit string limit: past
    _MAX_SHOWN_DIGITS the size is reported instead of the digits themselves."""
    n = int(n)
    if n and int(abs(n).bit_length() * 0.3010299956639812) > _MAX_SHOWN_DIGITS:
        digits = int(abs(n).bit_length() * 0.3010299956639812) + 1
        return f"<a {digits}-digit integer>"
    return str(n)


def _format_number(value):
    """Render a numeric result for the model: ints plainly, terminating
    fractions as exact decimals, non-terminating fractions as 'p/q ≈ decimal'
    (so a ratio never hides its magnitude), and recurse into lists/tuples.
    Guaranteed never to raise: extreme magnitudes fall back to a size label."""
    if isinstance(value, (list, tuple)):
        left, right = ("[", "]") if isinstance(value, list) else ("(", ")")
        return left + ", ".join(_format_number(v) for v in value) + right
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return _int_text(value)
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return _int_text(value.numerator)
        try:
            decimal = _fraction_to_decimal(value)
        except (ValueError, OverflowError):
            decimal = None
        if decimal is not None:
            return decimal
        head = f"{_int_text(value.numerator)}/{_int_text(value.denominator)}"
        try:
            approx = float(value)
        except (OverflowError, ValueError):
            return head  # magnitude unrepresentable as a float; keep exact form
        return f"{head} ≈ {approx}"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return str(value)
    return str(value)


def _contains_nan(value) -> bool:
    """True if a float nan appears anywhere in a (possibly nested) result."""
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    return False


_UNDEFINED_RE = re.compile(r"(?<![A-Za-z0-9_])(zoo|nan)(?![A-Za-z0-9_])")


def _is_undefined(answer: str) -> bool:
    """A concrete expression that collapsed to a pole (zoo) or nan has no
    meaningful value — the caller reports an error instead of returning it."""
    return bool(_UNDEFINED_RE.search(answer))


# Implicit multiplication: a number or ')' directly followed by a letter or '('
# means a product. The lookbehind keeps it away from identifiers that merely end
# in digits (log10(, log2() and from scientific notation (2e5, 1e-3).
_IMPLICIT_NUM_RE = re.compile(
    # Atomic group (?>…) stops the engine from giving the exponent back to satisfy
    # the trailing lookahead — without it "2e5" would backtrack to "2" and become
    # "2*e5". The lookbehind keeps identifiers ending in digits (log10() intact.
    r"(?<![A-Za-z0-9_.])(?P<num>(?>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?))(?=[A-Za-z(])"
)
_IMPLICIT_CLOSE_RE = re.compile(r"(?<=\))(?=[A-Za-z0-9(])")


def _insert_implicit_multiplication(text: str) -> str:
    """Turn common implicit products into explicit ones: ``2x`` -> ``2*x``,
    ``2(x+1)`` -> ``2*(x+1)``, ``(a)(b)`` -> ``(a)*(b)``. Deliberately cautious:
    scientific notation (``2e5``, ``1e-3``), identifiers ending in digits
    (``log10(``), and bare names (``xy``) are left untouched."""
    text = _IMPLICIT_NUM_RE.sub(r"\g<num>*", text)
    return _IMPLICIT_CLOSE_RE.sub("*", text)


def _normalize_expression(expression: str) -> str:
    """Shared preprocessing for both engines: ^ means power, and implicit
    multiplication (2x) becomes explicit (2*x)."""
    return _insert_implicit_multiplication(expression.strip().replace("^", "**"))


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression exactly. Two engines, one entry point:

    plain arithmetic runs on a fast exact evaluator (integers and decimals stay
    exact, so 10**30/3 does not turn into a 16-digit float), and anything
    symbolic (variables, integrals, limits, derivatives, equations, sums,
    series) falls through to SymPy. Neither path ever executes code.
    """
    if not expression or not expression.strip():
        return "Error: expression is empty."
    # Models often write ^ for power and 2x for 2*x; normalise both notations.
    normalized = _normalize_expression(expression)

    # --- Fast path: pure arithmetic, exact values, no dependencies ---
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        tree = None
    if tree is not None:
        try:
            result = _eval_calc_node(tree.body)
        except _NeedsSymbolic:
            pass  # not arithmetic — let SymPy handle it below
        except _CalcGuardError as e:
            return f"Error: cannot evaluate '{expression}': {e}."
        except ZeroDivisionError:
            return f"Error: division by zero in '{expression}'."
        except RecursionError:
            return f"Error: cannot evaluate '{expression}': expression is too deeply nested."
        except (ValueError, OverflowError, TypeError):
            # A ValueError here is ambiguous and its wording differs across Python
            # versions: sqrt(-1) has the legitimate complex answer I, while
            # factorial(-1) is a genuine pole. So let SymPy decide — a real pole
            # surfaces as zoo/nan and is turned into an error just below.
            pass
        else:
            if _contains_nan(result):
                return f"Error: cannot evaluate '{expression}': result is undefined (nan)."
            try:
                rendered = _format_number(result)
            except Exception as e:  # formatting must never crash the tool
                return (
                    f"Error: cannot evaluate '{expression}': result could not be "
                    f"rendered ({type(e).__name__})."
                )
            log.info("calculate: %s = %s", expression, rendered)
            return f"{expression} = {rendered}"

    # --- Symbolic path: variables, calculus, complex domains ---
    from argent.tools.symbolic_math import SymbolicError, evaluate_symbolic
    try:
        answer = evaluate_symbolic(normalized)
    except SymbolicError as symbolic_error:
        return (
            f"Error: cannot evaluate '{expression}': {symbolic_error}. "
            f"Provide a plain arithmetic expression (e.g. '(1847 * 0.15) + sqrt(2)') or a "
            f"symbolic one (e.g. 'integrate(x**2, x)', 'limit(sin(x)/x, x, 0)', "
            f"'solve(x**2 - 4, x)')."
        )
    except Exception as e:  # sympy failed on a well-formed but hard expression
        return f"Error: cannot evaluate '{expression}': {e}."
    # A concrete expression that collapsed to a pole (zoo) or nan has no
    # meaningful value — report it instead of returning a cryptic 'zoo'.
    if _is_undefined(answer):
        return (
            f"Error: cannot evaluate '{expression}': result is undefined "
            f"(division by zero or a pole)."
        )

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


def _ask_text(prompt: str, default: str = "") -> str:
    """Prompt user for text, routing through ReplBridge modal if TUI is active."""
    import os
    if os.environ.get("ARGENT_TUI") == "1":
        try:
            from src.bridge.repl_bridge import ReplBridge
            res = ReplBridge.ask_active_select(prompt, options=[], default=default)
            return res if res is not None else ""
        except Exception:
            pass
    from prompt_toolkit import prompt as ptk_prompt
    return ptk_prompt(prompt, default=default)


def ask_user_questions(questions: list) -> str:
    """Ask the user a series of structured questions."""
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
    try:
        from src.cli.sound import play_sound
        play_sound("prompt")
    except Exception:
        pass
    console.print("\n[bold cyan]Уточнение требований:[/bold cyan]")


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
                answer = _ask_text("Ваш ответ ❯ ")
                responses[q_text] = answer.strip() or "No answer"
            except (KeyboardInterrupt, EOFError):
                responses[q_text] = "Skipped"
            continue

        console.print(f"\n[bold yellow]{q_text}[/bold yellow]")

        if q_type == "text":
            console.print("[dim](Введите текст и нажмите Enter)[/dim]")
            try:
                answer = _ask_text("Ваш ответ ❯ ")
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
                        custom = _ask_text("Введите свой вариант ❯ ")
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
                        custom = _ask_text("Введите свой(и) вариант(ы) через запятую ❯ ")
                        if custom.strip():
                            selected.append(custom.strip())
                    
                    responses[q_text] = ", ".join(selected) if selected else "No answer"
                except (KeyboardInterrupt, EOFError):
                    responses[q_text] = "Skipped"
    
    # The verdict goes first: a caller must not have to read three lines to find out
    # whether the person answered at all, and "Skipped" and "No answer" are different
    # facts — one means the prompt was dismissed, the other that nothing was entered.
    return _summarize_answers(questions, responses, malformed)


def _summarize_answers(questions: list, responses: dict, malformed: list = None) -> str:
    """What to tell the model, and what to remember.

    An empty answer is a fact about the prompt, not a preference. Recording it as
    "User preference on X: No answer" put an invented preference into long-term memory,
    where it would quietly shape later sessions.
    """
    answered = {k: v for k, v in responses.items() if v != "Skipped"}
    asked = len(questions)
    dismissed = [k for k, v in responses.items() if v == "Skipped"]
    if not responses:
        head = f"NO ANSWER: none of {asked} question(s) was reached."
    elif not answered and dismissed and len(dismissed) == len(responses):
        head = f"SKIPPED: none of {asked} question(s) was answered (the prompt was dismissed)."
    elif not answered:
        head = f"NO ANSWER: none of {asked} question(s) got an answer (nothing was entered)."
    elif len(answered) < asked:
        head = f"PARTLY ANSWERED: {len(answered)} of {asked}."
    else:
        # An empty answer still counts as an answer: the prompt was shown and accepted.
        # It is only reported as empty, never as a choice — see the line below.
        head = f"ANSWERED: all {asked}."

    summary_lines = []
    for k, v in responses.items():
        if v == "Skipped":
            summary_lines.append(f"- {k}: SKIPPED (dismissed without an answer)")
            continue
        if v in ("No answer", "") or v is None:
            summary_lines.append(f"- {k}: (empty — nothing was entered; do not treat as a choice)")
            continue
        summary_lines.append(f"- {k}: {v}")
        memory.add_fact(f"User preference on '{k}': {v}")

    out = head + "\n" + "\n".join(summary_lines)
    if malformed:
        # Told back to the model, not just logged: it can only stop sending the
        # wrong shape if something says so within the turn.
        log.warning("ask_user_questions received malformed options in: %s", malformed)
        out += ("\n\n[Note: some options were not plain strings and had to be "
                "converted or dropped. `options` must be an array of STRINGS, "
                'e.g. "options": ["Only critical paths", "Full coverage"] — not '
                "objects. Re-check the schema before asking again.]")
    return out


def git_checkpoint(message: str) -> str:
    """Snapshot the files (a checkpoint kept off the branch) before an experiment."""
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

        from argent.approval import request_approval
        approved = request_approval(
            f"откатить ВСЕ изменения до чекпоинта '{target['label']}' ({target['sha']}); текущие незакоммиченные правки уйдут в git stash",
            destructive=True,
        )
        if not approved:
            return "Rollback aborted by user."

        msg = checkpoints.rewind_to(target["sha"])
        try:
            from argent.file_tracker import clear_session_snapshots
            clear_session_snapshots()
        except Exception:
            pass
        return msg
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
    from argent.config import get_mcp_servers
    from src.rooms import policy as room_policy

    configured = get_mcp_servers()
    if not configured:
        return "No MCP servers are configured. Add one with /mcp."
    # A room's node may be told to ignore some servers / tools (see src/rooms/policy.py).
    configured = [s for s in configured if room_policy.mcp_tool_blocked(s["name"]) is None]
    if not configured:
        return "Every MCP server is ignored by this room's rules."

    if not (server_name or "").strip():
        lines = ["Configured MCP servers:"]
        for srv in configured:
            name = srv["name"]
            tools = [t for t in mcp_client.list_tools(name) if "error" not in t]
            if tools:
                state = f"{len(tools)} tools"
            else:
                # "No tools" and "no answer" are different problems: one is fixed in
                # the server, the other in the transport. Saying "unreachable" for
                # both sent people hunting for a crashed process that was running.
                state = ("NO TOOLS — the server answers but offers none; check the server"
                         if mcp_client.listing_state(name) == "empty"
                         else "unreachable (no answer to tools/list — check /mcp)")
            lines.append(f"  - {name} ({srv.get('type', 'stdio')}): {state}")
        lines.append("Call list_mcp_tools(server_name, filter) to see signatures.")
        return "\n".join(lines)

    server_name = server_name.strip()
    if server_name not in [s["name"] for s in configured]:
        known = ", ".join(s["name"] for s in configured)
        return f"Error: no MCP server named '{server_name}'. Configured: {known}."

    tools = mcp_client.list_tools(server_name)
    broken = [t for t in tools if "error" in t]
    tools = [t for t in tools if "name" in t and "error" not in t
             and room_policy.mcp_tool_blocked(server_name, t["name"]) is None]
    if not tools:
        if broken:
            reason = broken[0].get("error")
            return (f"Server '{server_name}' could not be reached: {reason}. "
                    f"Retry /mcp, or check the server process.")
        return (f"Server '{server_name}' is up but offers NO TOOLS. "
                f"Nothing to call here — check the server's own configuration or log.")

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


def _decode_mcp_arguments(raw):
    """The tool's arguments from whatever shape arrived: (args, note, error).

    Models produce three shapes in practice — the object the schema asks for, a JSON
    string (the old contract), and a *double*-encoded one, where the object was turned
    into a string and then into a string again. The third used to reach the server as
    a single string field and fail there with a message nobody could act on, so it is
    unwrapped here and reported plainly.
    """
    if raw is None or raw == "":
        return {}, "", ""
    if isinstance(raw, dict):
        # {"arguments": "{...}"} — the classic double wrap when a model fills the
        # parameter named in the description instead of the parameter itself.
        for key in ("arguments", "arguments_json", "args"):
            inner = raw.get(key)
            if len(raw) == 1 and isinstance(inner, (str, dict)):
                args, note, err = _decode_mcp_arguments(inner)
                if err:
                    return {}, "", err
                return args, (note or "arguments arrived wrapped in an object and were unwrapped"), ""
        return raw, "", ""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if not isinstance(raw, str):
        return {}, "", (f"Error: `arguments` must be an object, got {type(raw).__name__}. "
                        "Pass the tool's arguments as they appear in its schema.")
    text = raw.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import json5
            value = json5.loads(text)
        except Exception:
            return {}, "", ("Error: `arguments` is not valid JSON. Send it as an OBJECT "
                            "(e.g. {\"path\": \"a.py\"}), not a hand-written string; if you must "
                            "send a string, use double quotes and escape newlines (\\n) and "
                            "quotes (\\\").")
    if isinstance(value, str):
        args, note, err = _decode_mcp_arguments(value)
        if err:
            return {}, "", err
        return args, "arguments were encoded twice (an object inside a string inside a string) and were unwrapped", ""
    if not isinstance(value, dict):
        return {}, "", (f"Error: `arguments` decoded to {type(value).__name__}, not an object. "
                        "An MCP tool takes named arguments.")
    return value, "", ""


def call_mcp_tool(server_name: str, tool_name: str, arguments=None,
                  arguments_json: str = None) -> str:
    """Call a standardized tool from an MCP server.

    `arguments` is the tool's arguments as an OBJECT (the schema's shape). A JSON
    string is still accepted — and a string wrapped twice is unwrapped and reported —
    because old traces and habits keep sending one.
    """
    if arguments is None:
        arguments = arguments_json

    args, note, err = _decode_mcp_arguments(arguments)
    if err:
        return err

    from src.rooms import policy as room_policy
    blocked = room_policy.mcp_tool_blocked(server_name, tool_name)
    if blocked:
        return f"Error: {blocked}"

    try:
        result = mcp_client.call_tool(server_name, tool_name, args)
    except Exception as e:
        return f"Error calling MCP tool: {e}"
    # The note rides along: the call worked, but the caller should know its shape was
    # wrong, or it will keep sending that shape.
    return f"{result}\n\n[Note: {note}]" if note else result

def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def run_subagent(role: str, task: str, tools_json: str = None, isolation: str = "none",
                 background: bool = False, model: str = "", provider: str = "",
                 thinking: str = "") -> str:
    """Spawn a specialized sub-agent for a task.

    With neither isolation nor background it runs in this process and the call returns its
    report. `isolation="worktree"` gives it a private copy of the tree (its edits come back as a
    patch); `background=True` returns at once and the sub-agent keeps working beside you.

    `model`, `provider` and `thinking` say what it should run on, so routine work can be
    handed to a cheaper model or a lower thinking level. Empty means the configured
    default, and that default is the main agent itself — a helper never costs more than
    the agent that called it unless you ask. subagents(action='models') lists what may
    be passed here.
    """
    tools = None
    if tools_json:
        if isinstance(tools_json, (list, tuple)):
            tools = [str(t) for t in tools_json]
        elif isinstance(tools_json, str):
            try:
                parsed = json.loads(tools_json)
                if isinstance(parsed, list):
                    tools = [str(t) for t in parsed]
                elif isinstance(parsed, str):
                    tools = [parsed]
            except Exception:
                tools = [tools_json]

    # Whether the CALLER chose a target, as opposed to inheriting one from config.
    # Only an explicit choice is checked strictly: a person's own configured critic
    # keeps its existing fallback instead of being refused for a missing key.
    explicit_target = bool(model or provider)
    if role.strip().lower() == "critic":
        if tools is None:
            tools = []
        # The critic keeps its own configured model (it exists to be independent), but
        # an explicit choice by the caller wins over that configuration.
        if not explicit_target:
            from argent.config import get_critic_model, get_critic_provider
            model = get_critic_model() or ""
            provider = get_critic_provider() or ""

    from src.agent.subagents import resolve_target
    try:
        model, provider, thinking = resolve_target(model, provider, thinking,
                                                   strict=explicit_target)
    except ValueError as e:
        return f"Error: {e}. Available: subagents(action=\"models\")."

    isolation = (isolation or "none").strip().lower()
    background = _truthy(background)
    if isolation == "none" and not background:
        from argent.agent import ArgentSubAgent
        agent = ArgentSubAgent(
            role, task, tools_override=tools,
            model_override=model or None, provider_override=provider or None,
            thinking=thinking)
        return agent.execute()

    from src.agent.subagents import MANAGER, describe
    from src.agent.worktrees import WorktreeError
    try:
        run = MANAGER.spawn(role, task, tools=tools, isolation=isolation,
                            model=model, provider=provider, thinking=thinking)
    except (ValueError, WorktreeError) as e:
        return f"Error: {e}"
    if not background:
        MANAGER.wait(run)
        return describe(run)

    from src.agent.watches import WATCHES, async_enabled
    if async_enabled():
        try:
            WATCHES.add(f"sub-agent #{run.id} ({role})", until=f'subagent_finished("{run.id}")', timeout=3600)
        except ValueError:
            pass
        return (f"Sub-agent #{run.id} ({role}) started in the background. You will be woken when it finishes; "
                "meanwhile carry on with other work, or end your turn with a short note to the user.")
    return (f"Sub-agent #{run.id} ({role}) started in the background. When you need its result, call "
            f"subagents(action=\"result\", id={run.id}, wait=true), or wait_heartbeat("
            f"until='subagent_finished(\"{run.id}\")').")


def subagents(action: str = "list", id: int = None, wait: bool = False, timeout_seconds: int = 600) -> str:
    """Look at, wait for, stop, apply or discard the sub-agents started with run_subagent."""
    from src.agent.subagents import MANAGER, available_targets, describe, table

    action = (action or "list").strip().lower()
    if action == "models":
        return available_targets()
    if action == "list":
        return table(MANAGER.all())
    run = MANAGER.get(id)
    if run is None:
        return f"Error: there is no sub-agent #{id}. Known: " + (", ".join(f"#{r.id}" for r in MANAGER.all()) or "none")
    if action == "result":
        if run.running and _truthy(wait):
            MANAGER.wait(run, max(1, min(int(timeout_seconds or 600), 6 * 3600)))
        if not run.running:
            # The model has the report now; a wake-up for the same run would cost a turn to say nothing new.
            from src.agent.watches import WATCHES
            for watch in WATCHES.pending():
                if watch.until == f'subagent_finished("{run.id}")':
                    WATCHES.cancel(watch.id)
        return describe(run)
    if action == "stop":
        return (f"Sub-agent #{run.id} stopped." if MANAGER.stop(run) else f"Sub-agent #{run.id} is not running.")
    if action == "apply":
        problem = MANAGER.apply(run)
        return (f"Error: could not apply the patch of sub-agent #{run.id}: {problem}" if problem
                else f"Applied the changes of sub-agent #{run.id} to the working tree.")
    if action == "discard":
        MANAGER.discard(run)
        return f"Discarded the changes of sub-agent #{run.id}."
    return "Error: action must be one of list, models, result, stop, apply, discard."

def create_artifact(filename: str, content: str) -> str:
    """Create a Markdown artifact in the .argent/artifacts/ directory.

    The name is confined to that folder: `../` used to walk straight out of it
    (`create_artifact("../../AGENTS.md", ...)` overwrote a project file), and an
    absolute path replaced the folder entirely. Both are refused now, with the
    reason and the expected shape.
    """
    base = Path(".argent") / "artifacts"
    name = str(filename or "").strip()
    if not name:
        return "Error: filename is required (e.g. 'implementation_plan.md')."
    candidate = Path(name)
    if candidate.is_absolute() or candidate.drive or name.startswith(("/", "\\")):
        return (f"Error: filename must be relative to {base.as_posix()}, "
                f"not an absolute path ('{filename}').")
    parts = [p for p in candidate.parts if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return (f"Error: filename must stay inside {base.as_posix()} "
                f"('..' would walk out of it). Use a plain name like 'plan.md'.")
    try:
        path = base.joinpath(*parts)
        existed = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("create_artifact: %s", path)
        verb = "overwrote" if existed else "created"
        return (f"Artifact {verb} at {path.as_posix()}. The user can now review it. "
                f"To review or resume it later, use read_file('{path.as_posix()}').")
    except Exception as e:
        return f"Error creating artifact: {e}"

def request_user_approval(message: str) -> str:
    """Pause and wait for the person to approve a plan or an action.

    Goes through the same approval channel as every other gated action, so it also
    answers in the TUI, over Telegram and in an unattended run — it used to call
    questionary directly, which only ever worked on a terminal.

    The first line is the verdict: APPROVED / REJECTED: <reason> / CANCELLED.
    """
    from argent.approval import APPROVED, CANCELLED, request_approval_status
    status, reason = request_approval_status(f"Одобрить: {message}")
    if status == APPROVED:
        memory.add_completed(f"Approved plan: {message}")
        return "APPROVED — proceed with execution."
    if status == CANCELLED:
        # Nobody said no; nobody said yes either. Treating silence as consent is how
        # an unattended run starts the very work the person was declining to authorise.
        memory.add_completed(f"Approval cancelled, no answer: {message}")
        return ("CANCELLED: the prompt was dismissed without an answer (Esc, closed input "
                "or the person walked away). Treat it as NO: do not proceed. Say what you "
                "were about to do and ask whether to continue.")
    memory.add_completed(f"Rejected plan: {message}")
    return (f"REJECTED: {reason or 'no reason given'}. Do not proceed. "
            "Ask what to change, or revise the plan and ask again.")

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
    from src.agent.watches import async_enabled

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
            timeout = max(1, min(int(float(timeout_seconds or 600)), MAX_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            timeout = 600
        reason = condition_to_check or until
        if async_enabled():
            return _register_watch(reason, until=until, timeout=timeout)
        return (f"Waiting until: {until} (timeout {timeout}s). "
                f"[HEARTBEAT_REQUEST: 0|{reason}|until={until}|timeout={timeout}]")

    try:
        delay = max(0, int(float(delay_seconds or 0)))
    except (TypeError, ValueError):
        delay = 0
    if delay <= 0:
        return ("Error: provide either delay_seconds (a fixed sleep) or until "
                f"(a condition to wait for). Available checks: {describe_predicates()}")
    if async_enabled():
        return _register_watch(condition_to_check, delay=delay)
    return f"Heartbeat scheduled. [HEARTBEAT_REQUEST: {delay}|{condition_to_check}]"


def _register_watch(reason: str, until: str = "", timeout: int = 600, delay: int = 0) -> str:
    """Non-blocking wait: the frontend wakes the agent, so the agent can simply finish its turn."""
    from src.agent.watches import WATCHES
    try:
        watch = WATCHES.add(reason, until=until, timeout=timeout, delay=delay)
    except (ValueError, OSError) as e:
        return f"Error: {e}"
    what = f"until {until} (gives up after {watch.timeout}s)" if until else f"for {watch.timeout}s"
    return (f"Waiting {what}. This does NOT block anything: end your turn now with a short message to the "
            f"user saying what is running and what you are waiting for. You will be woken with a "
            f"'[Heartbeat ...]' message when it happens — do not poll or call wait_heartbeat again "
            f"for the same thing. [watch #{watch.id}]")

def view_image(path: str = "", question: str = "", crop: Any = None) -> str:
    """Attach an image to the conversation so the model can actually see it.

    `path` — the same name every other file tool uses, and the name a browser
    screenshot already returns with its file, so chaining the two tools needs no
    translation.

    ``crop`` allows zooming into a Region of Interest in 100% native resolution
    (crucial for reading fine console fonts, IDE text, or logs on 4K/2K screens).
    Format can be a dict ({"x": 100, "y": 200, "width": 800, "height": 400}),
    a box ([x, y, w, h] or [left, top, right, bottom]), or a string ("x,y,w,h").

    The picture cannot travel in a tool result — those are strings — so it is
    queued here and the turn loop attaches it as the next user message. The
    model therefore sees it on its NEXT step, which the return value says
    plainly: a model told "done" would otherwise answer about an image it has
    not received yet.
    """
    from argent.config import get_current_model, get_provider
    from argent.vision import encode_image, model_supports_vision, queue_image

    path = (path or "").strip()
    if not path:
        return "Error: give the image path, e.g. view_image('screenshot.png')."

    provider = get_provider()
    model = get_current_model()
    supports = model_supports_vision(model, provider)
    if supports is False:
        return (f"Error: model '{model}' cannot see images (Ollama reports no "
                f"'vision' capability). Switch to a vision model with /model, or "
                f"read the page as text with read_webpage.")

    payload, media_type, note = encode_image(path, crop=crop)
    if payload is None:
        return f"Error: {note}"

    from argent.tools._helpers import _resolve_path
    label = str(_resolve_path(path))
    queue_image(payload, media_type, label, path=label)

    detail = f" ({note})" if note else ""
    unsure = ("\nNote: this model's vision support could not be verified — if the "
              "next message looks like it has no image, it does not.") if supports is None else ""
    asked = f" You asked: {question}" if question else ""
    return (f"Image '{label}' attached{detail}. You will SEE it in the next "
            f"message, not in this result — continue and describe what is "
            f"actually there.{asked}{unsure}")


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


def request_blind_spot_tests(source_file: str, focus_layers: str = "L1,L2,L4,L4b,L7,L9", note: str = "") -> str:
    """Submit a request for comprehensive LMTrust blind-spot tests to the local agent bridge."""
    from src.bridge.lmtrust_bridge import create_test_request
    try:
        req = create_test_request(source_file, focus_layers=focus_layers, note=note)
        return (
            f"LMTrust testing request created successfully.\n"
            f"- Request ID: {req['id']}\n"
            f"- Target File: {req['source_file']}\n"
            f"- Focus Layers: {', '.join(req['focus_layers'])}\n"
            f"- Status: Pending Antigravity / Agent processing in .argent/lmtrust_bridge/inbox/"
        )
    except Exception as e:
        log.error("request_blind_spot_tests error: %s", e)
        return f"Error creating LMTrust request for '{source_file}': {e}"

