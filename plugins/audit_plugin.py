import os
import ollama
from skill_manager import skill_manager
from ui import console, print_markdown, print_error, print_system
from config import get_current_model

def command_audit(*args):
    """
    Performs a deep architectural and clean code audit of a file.
    Usage: /audit <file_path>
    """
    if not args:
        print_error("Usage: /audit <file_path>")
        return

    file_path = args[0]
    if not os.path.exists(file_path):
        print_error(f"File not found: {file_path}")
        return

    print_system(f"Starting architectural audit of: [bold cyan]{file_path}[/bold cyan]...")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()
    except Exception as e:
        print_error(f"Error reading file: {e}")
        return

    # Load the Expert Architect skill
    skill_instructions = skill_manager.read_skill("ExpertArchitect")
    if not skill_instructions:
        print_error("ExpertArchitect skill instructions not found.")
        return

    prompt = (
        f"You are a Senior Architect auditing the following code file: `{file_path}`.\n\n"
        f"### ARCHITECTURAL GUIDELINES (ExpertArchitect skill):\n"
        f"{skill_instructions}\n\n"
        f"### CODE TO AUDIT:\n"
        f"```python\n{code_content}\n```\n\n"
        "Your Task: Provide a brutal but fair architectural audit. Point out violations of KISS, DRY, SOLID, and YAGNI. "
        "Suggest specific improvements and refactoring strategies. Output in Markdown."
    )

    try:
        model = get_current_model()
        # Fast synchronous call for analysis
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        report = response.get("message", {}).get("content", "Error generating report.")
        
        console.rule(f"[bold green]Audit Results Specialist: Expert Architect[/bold green]")
        print_markdown(report)
        
    except Exception as e:
        print_error(f"Audit failed: {e}")

# Note: The 'command_audit' will be automatically picked up as '/audit' by HookManager.
