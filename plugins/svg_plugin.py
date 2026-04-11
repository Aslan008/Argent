import os
import webbrowser
from pathlib import Path
from ui import console, print_system
from config import get_visuals_dir

def command_visuals(*args):
    """List all generated SVG images in the visuals directory."""
    visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
    if not visuals_dir.exists():
        console.print("[yellow]No visuals directory found. No images have been created yet.[/yellow]")
        return
        
    files = list(visuals_dir.glob("*.svg"))
    if not files:
        console.print("[yellow]No SVG images found in the visuals directory.[/yellow]")
        return
        
    console.print("\n[bold cyan]Generated Visuals:[/bold cyan]")
    for i, f in enumerate(files, 1):
        console.print(f"  {i}. [green]{f.name}[/green] ([dim]{f.stat().st_size} bytes[/dim])")
    console.print("\nUse `/view <filename>` to open an image in your browser.")

def command_view(*args):
    """Open a specific SVG image in the browser. Example: /view my_diagram.svg"""
    if not args:
        console.print("[red]Error: Please specify the filename to view. Example: /view image.svg[/red]")
        return
        
    filename = args[0]
    if not filename.endswith(".svg"):
        filename += ".svg"
        
    visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
    file_path = visuals_dir / filename
    
    if file_path.exists():
        console.print(f"[green]Opening {filename} in your browser...[/green]")
        webbrowser.open(f"file:///{file_path}")
    else:
        console.print(f"[red]Error: File '{filename}' not found in {visuals_dir}.[/red]")

import re
import questionary
from datetime import datetime

def post_response(text):
    """Automatically detects SVG code in the response and offers to save/open it if found."""
    # Find SVG blocks using regex
    svg_blocks = re.findall(r'<svg[\s\S]*?<\/svg>', text)
    
    if svg_blocks:
        total = len(svg_blocks)
        console.print(f"\n[bold cyan]🎨 I detected {total} SVG image(s) in the response.[/bold cyan]")
        
        save_all = questionary.confirm("Would you like to save and open these visuals in your browser?").ask()
        
        if save_all:
            from config import get_visuals_dir
            visuals_dir = Path(get_visuals_dir()).expanduser().resolve()
            visuals_dir.mkdir(parents=True, exist_ok=True)
            
            for i, svg_code in enumerate(svg_blocks, 1):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detected_{timestamp}_{i}.svg"
                file_path = visuals_dir / filename
                
                # Ensure XML declaration
                if "<?xml" not in svg_code:
                    svg_code = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg_code
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(svg_code)
                
                console.print(f"  [green]Saved visual {i}/{total} to {filename}[/green]")
                webbrowser.open(f"file:///{file_path}")
        else:
            console.print("[dim]Tip: You can always use `/visuals` to see previously generated images.[/dim]")
