import psutil
from ui import console
from rich.panel import Panel

def command_mem(*args):
    """Shows RAM usage percentage in a beautiful Rich Panel."""
    try:
        mem = psutil.virtual_memory()
        percent = mem.percent
        
        # Color based on usage severity
        color = "green"
        if percent > 70: 
            color = "yellow"
        if percent > 90: 
            color = "red"
        
        # Build the info string
        info = (
            f"Total: {mem.total / (1024**3):.2f} GB\n"
            f"Available: {mem.available / (1024**3):.2f} GB\n"
            f"Used: {mem.used / (1024**3):.2f} GB\n"
            f"Usage: [bold {color}]{percent}%[/bold {color}]"
        )
        
        console.print(Panel(
            info,
            title="[bold cyan]Memory Status[/bold cyan]",
            border_style=color,
            expand=False,
            padding=(1, 2)
        ))
    except Exception as e:
        console.print(f"[red]Error getting memory info: {e}[/red]")

def on_startup():
    # Quietly loaded
    pass
