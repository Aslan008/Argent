from datetime import datetime
import requests
import platform

def command_inf(*args):
    """Displays current time and weather."""
    try:
        from ui import console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print("Error: Rich not found.")
        return

    # 1. Get Time
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%d.%m.%Y")
    
    # 2. Get Weather via wttr.in (text-based)
    weather_info = "Weather: Service unavailable"
    try:
        # We use a simple request to wttr.in with '?format=3' for a concise one-liner
        # or '?format=%C+%t' for Condition + Temp
        response = requests.get("https://wttr.in?format=%C+%t", timeout=5)
        if response.status_code == 200:
            weather_info = response.text.strip()
    except Exception as e:
        weather_info = f"Weather: Error ({str(e)})"

    # 3. Create UI
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold cyan]Date:[/bold cyan]", date_str)
    table.add_row("[bold cyan]Time:[/bold cyan]", f"[bold yellow]{time_str}[/bold yellow]")
    table.add_row("[bold cyan]Weather:[/bold cyan]", weather_info)

    console.print(Panel(
        table, 
        title="[bold green]Information[/bold green]", 
        expand=False,
        border_style="blue"
    ))

def on_startup():
    # Subtle log on startup
    pass
