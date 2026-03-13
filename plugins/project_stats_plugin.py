import os
from ui import console

def on_startup():
    """Triggered every time Argent starts. Counts files in the current project root."""
    try:
        # Count only files (not directories) in the current working directory
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        file_count = len(files)
        
        console.print(f"[dim]📁 В проекте найдено [bold cyan]{file_count}[/bold cyan] файлов.[/dim]")
    except Exception as e:
        # Silent fail or subtle log on startup to not disrupt the user
        pass
