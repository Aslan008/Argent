"""/leeth: журнал решений Leeth — сводка и разметка руками человека.

Зачем: у Leeth есть активная разметка силами самого агента (`verdict_source='agent'`), но она пристрастна, а объективный
исход записывался редко. Боевые запросы Argent — единственный доступный набор боевой трудности, которого не хватает для
калибровки выбора на больших каталогах; без человеческих меток он бесполезен. Эта команда даёт метку в одно нажатие.

    /leeth            — сводка журнала
    /leeth review     — разметить до 10 последних неразмеченных решений
    /leeth review 50  — то же, до 50
"""

from typing import Any, Dict, List, Optional

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from argent.ui import print_error, print_system

KEYS = ("a", "p", "r", "s", "q")


def _print_summary() -> None:
    from src.leeth import feedback

    st = feedback.stats()
    by_verdict = st.get("by_verdict") or {}
    by_source = st.get("by_source") or {}
    labeled = st["total"] - by_verdict.get("pending", 0)
    print_system(
        "Журнал решений Leeth: всего {total}, размечено {labeled}, ждёт {pending}".format(
            total=st["total"], labeled=labeled, pending=by_verdict.get("pending", 0)
        )
    )
    if by_verdict:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_verdict.items()) if k != "pending")
        if parts:
            print_system(f"[dim]Вердикты: {parts}[/dim]")
    if by_source:
        parts = ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items()))
        print_system(f"[dim]Кто размечал: {parts}[/dim]")
    conflicts = [r for r in feedback.load_records()
                 if (r.get("arbitrator") or {}).get("sat") is False and not r.get("verdict")]
    if conflicts:
        print_system(f"[dim]Солвер возражает (идут первыми в очереди): {len(conflicts)}[/dim]")
    print_system(f"[dim]Хранилище: {st.get('store')}[/dim]")


def _ask(prompt: str) -> Optional[str]:
    """One line from the person; None when there is no terminal to ask (TUI runs, EOF, Ctrl-C)."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return None


def _label_one(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One record, one verdict. Returns None to skip it (or when the person stops)."""
    from src.leeth import feedback

    proposal = record.get("proposal") or {}
    dispatch = record.get("dispatch") or {}
    tool = dispatch.get("tool_name") or (proposal.get("tool") or {}).get("choice")
    conf = (proposal.get("tool") or {}).get("confidence")
    args = dispatch.get("arguments") or proposal.get("param")

    print_system(f"\n[bold]{str(record.get('query', ''))[:300]}[/bold]")
    print_system(f"[dim]Предложено: {tool} {args} | уверенность {conf} | уверенность в вызове "
                 f"{(proposal.get('need_tool') or {}).get('probability')}[/dim]")
    if record.get("skip_reason"):
        print_system(f"[dim]Диспетчер отказал: {record['skip_reason']}[/dim]")
    if record.get("outcome"):
        print_system(f"[dim]Исход: {record['outcome']}[/dim]")
    arb = record.get("arbitrator") or {}
    if arb.get("sat") is False:
        # The solver never blocks; here is where its disagreement is worth something.
        print_system(f"[yellow]Солвер возражает: {', '.join(arb.get('core') or []) or 'формальное противоречие'}[/yellow]")

    answer = _ask("[a] принять  [p] частично  [r] отклонить  [s] пропустить  [q] выход > ")
    if answer is None or answer.lower().startswith("q"):
        return {"stop": True}
    answer = answer.lower()[:1]
    if answer not in KEYS or answer == "s":
        return None
    verdict = {"a": "accept", "p": "partial", "r": "reject"}[answer]

    correction_tool = correction_param = None
    if verdict in ("partial", "reject"):
        correction_tool = _ask("Правильный инструмент (Enter = оставить как есть): ") or None
        correction_param = _ask("Правильный параметр (Enter = оставить): ") or None
    reason = _ask("Почему (Enter = без пояснения): ") or ""

    if feedback.submit_verdict(record.get("id", ""), verdict, correction_tool=correction_tool,
                               correction_param=correction_param, source="human", reason=reason):
        print_system(f"[dim]записано: {verdict}[/dim]")
    else:
        print_error(f"не удалось записать вердикт для {record.get('id')}")
    return {"verdict": verdict}


def _review(limit: int) -> None:
    from src.leeth import feedback

    pending: List[Dict[str, Any]] = [r for r in feedback.load_records() if not r.get("verdict")]
    if not pending:
        print_system("Неразмеченных решений нет.")
        return

    # A CDCL conflict first: the solver is right about every conflict that can be checked by re-running
    # the disputed call, and these are exactly the proposals the neural gates let through with high
    # confidence. Everything else follows, newest first: today's traffic is the difficulty that matters.
    def _conflict(rec: Dict[str, Any]) -> bool:
        return (rec.get("arbitrator") or {}).get("sat") is False

    conflicts = sorted((r for r in pending if _conflict(r)), key=lambda r: float(r.get("ts") or 0), reverse=True)
    rest = sorted((r for r in pending if not _conflict(r)), key=lambda r: float(r.get("ts") or 0), reverse=True)
    pending = conflicts + rest
    todo = pending[:limit]
    if conflicts:
        print_system(f"Спорных с солвером в очереди: {len(conflicts)} — они идут первыми.")
    print_system(f"К разметке: {len(todo)} из {len(pending)} неразмеченных.")
    done = 0
    for i, rec in enumerate(todo, 1):
        print_system(f"\n[dim]({i} из {len(todo)})[/dim]")
        result = _label_one(rec)
        if result is None:
            continue
        if result.get("stop"):
            break
        done += 1
    print_system(f"\nРазмечено за этот заход: {done}.")


@registry.register("/leeth", help_text="Журнал решений Leeth: сводка, или /leeth review [N] — разметить руками "
                                      "(это даёт калибровку на боевой трудности)", category="agent", is_prefix=True)
def handle_leeth(ctx: CommandContext) -> CommandResult:
    arg = ctx.user_input.strip()[len("/leeth"):].strip().lower()
    if arg.startswith("review"):
        rest = arg[len("review"):].strip()
        limit = 10
        if rest.isdigit():
            limit = max(1, min(200, int(rest)))
        _review(limit)
    else:
        _print_summary()
    return CommandResult.continue_loop()
