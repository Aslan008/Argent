"""/review: an independent read of the changes (see src.agent.review)."""

from rich.markdown import Markdown

from src.cli.commands.base import CommandContext, CommandResult
from src.cli.commands.registry import registry
from ui import console, print_error, print_system


@registry.register("/review", help_text="Независимое ревью изменений: /review [staged|коммит|A..B|путь] [на что "
                                       "смотреть] [--fix]",
                   category="project", is_prefix=True)
def handle_review(ctx: CommandContext) -> CommandResult:
    from src.agent import review

    arg = ctx.user_input.strip()[len("/review"):].strip()
    fix = "--fix" in arg.split()
    arg = " ".join(w for w in arg.split() if w != "--fix")
    target, focus = review.split_target(arg)

    print_system("[dim]Запускаю независимое ревью (чистый контекст, только чтение кода)…[/dim]")
    try:
        result = review.run_review(target, focus)
    except review.ReviewError as e:
        print_error(str(e))
        return CommandResult.continue_loop()

    console.print(Markdown(review.render(result)))
    saved = review.save(result)
    if saved:
        print_system(f"[dim]Сохранено: {saved}[/dim]")
    if fix and result.findings:
        print_system("Передаю находки агенту на исправление…")
        return CommandResult.pass_to_agent(new_input=review.fix_request(result))
    if result.findings and not fix:
        print_system("[dim]Чтобы агент сразу всё исправил: /review --fix[/dim]")
    return CommandResult.continue_loop()
