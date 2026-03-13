import platform
import shutil
import os
import subprocess
from datetime import datetime

def command_system(*args):
    """Displays system information (CPU, RAM, Disk, OS)."""
    # Use Argent's console if available, otherwise fallback to print
    try:
        from ui import console
        from rich.table import Table
        from rich.panel import Panel
    except ImportError:
        print("Error: Rich not found. Using basic output.")
        _basic_output()
        return

    table = Table(show_header=False, box=None)
    
    # OS Info
    table.add_row("[bold cyan]OS:[/bold cyan]", f"{platform.system()} {platform.release()} ({platform.machine()})")
    
    # CPU Info (Basic)
    cpu_count = os.cpu_count()
    table.add_row("[bold cyan]CPU Cores:[/bold cyan]", str(cpu_count))
    
    # RAM Info (Windows specific using WMIC for simplicity without psutil)
    if platform.system() == "Windows":
        try:
            cmd = "wmic computersystem get TotalPhysicalMemory"
            total_mem = subprocess.check_output(cmd, shell=True).decode().split("\n")[1].strip()
            total_gb = round(int(total_mem) / (1024**3), 2)
            table.add_row("[bold cyan]Total RAM:[/bold cyan]", f"{total_gb} GB")
        except:
            table.add_row("[bold cyan]Total RAM:[/bold cyan]", "Unknown")
    
    # Disk Usage
    total, used, free = shutil.disk_usage("/")
    table.add_row("[bold cyan]Disk Size:[/bold cyan]", f"{total // (2**30)} GB")
    table.add_row("[bold cyan]Disk Used:[/bold cyan]", f"{used // (2**30)} GB ({round((used/total)*100, 1)}%)")
    table.add_row("[bold cyan]Disk Free:[/bold cyan]", f"{free // (2**30)} GB")
    
    # Time
    table.add_row("[bold cyan]Current Time:[/bold cyan]", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    console.print(Panel(table, title="[bold green]System Information[/bold green]", expand=False))

def _basic_output():
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU Cores: {os.cpu_count()}")
    total, used, free = shutil.disk_usage("/")
    print(f"Disk: {used // (2**30)}GB / {total // (2**30)}GB Used")
