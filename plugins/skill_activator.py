import re
from skill_manager import skill_manager
from ui import console

# Keywords that trigger the Expert Architect skill
ARCHITECT_KEYWORDS = [
    r"\bbuild\b", r"\bdesign\b", r"\barchetecture\b", r"\bархитектура\b",
    r"\bпроектирование\b", r"\brefactor\b", r"\bрефакторинг\b",
    r"\bsolid\b", r"\bclean code\b", r"\bчистый код\b",
    r"\bpattern\b", r"\bпаттерн\b", r"\bструктура\b", r"\bstructure\b"
]

def pre_prompt(text: str) -> str:
    """
    Analyzes user prompt and automatically injects ExpertArchitect skill instructions
    if architectural context is detected.
    """
    text_lower = text.lower()
    
    # Check if any keyword matches
    should_activate = any(re.search(kw, text_lower) for kw in ARCHITECT_KEYWORDS)
    
    if should_activate:
        # Load the skill content
        skill_content = skill_manager.read_skill("ExpertArchitect")
        if skill_content:
            console.print("[dim cyan]ℹ Active Skill Activated: Expert Architect[/dim cyan]")
            
            # Inject at the beginning of the prompt to set the context
            injection = (
                "\n\n--- [ACTIVE SKILL: EXPERT ARCHITECT] ---\n"
                f"{skill_content}\n"
                "------------------------------------------\n\n"
            )
            return injection + text
            
    return text
