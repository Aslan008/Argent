from ui import print_system, console

def command_hello(*args):
    """A simple hello command from a plugin."""
    console.print("\n[bold green]Hello from Argent Plugin System![/bold green]")
    if args:
        console.print(f"Arguments passed: {' '.join(args)}")
    else:
        console.print("No arguments provided. Try: /hello world")

def on_startup():
    print_system("Example plugin loaded successfully.")
