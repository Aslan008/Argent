import os
import shutil
import subprocess
import threading
from pathlib import Path
from ui import print_system, print_error, print_markdown

def get_sandbox_dir() -> Path:
    cwd = Path(os.getcwd())
    sandbox_dir = cwd / ".argent" / "sandbox"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    return sandbox_dir

class SandboxManager:
    def __init__(self):
        self.sandbox_dir = get_sandbox_dir()
        self.http_server_process = None
        
    def export_files(self, dest_dir: str):
        """Export files from sandbox to the destination directory."""
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for item in self.sandbox_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_path / item.name)
                count += 1
        print_system(f"Exported {count} files to {dest_dir}")
        
    def clean_sandbox(self):
        """Delete all files in the sandbox."""
        for item in self.sandbox_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print_system("Sandbox cleared.")

    def run_python(self) -> str:
        """Run sandbox_main.py in the sandbox."""
        target_file = self.sandbox_dir / "sandbox_main.py"
        if not target_file.exists():
            return "Error: sandbox_main.py not found in sandbox."
            
        print_system(f"Running {target_file.name}...")
        try:
            result = subprocess.run(
                ["python", "sandbox_main.py"],
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return output
        except Exception as e:
            return f"Execution failed: {e}"

    def run_csharp(self) -> str:
        """Compile and run simple C# files in the sandbox."""
        target_file = self.sandbox_dir / "SandboxMain.cs"
        if not target_file.exists():
            return "Error: SandboxMain.cs not found in sandbox."
            
        print_system(f"Compiling and running {target_file.name}...")
        try:
            # We assume 'csc' (C# compiler from .NET Framework or Mono) is in PATH, 
            # or we use 'dotnet run' if we initialize a project. 
            # Easiest way for a single file on Windows without full project is csc.
            compile_result = subprocess.run(
                ["csc", "SandboxMain.cs"],
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True
            )
            
            if compile_result.returncode != 0:
                output = f"Compilation failed (Exit {compile_result.returncode}):\n{compile_result.stdout}\n{compile_result.stderr}"
                return output
                
            # If compiled successfully, a SandboxMain.exe is created
            run_result = subprocess.run(
                ["SandboxMain.exe"],
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = f"Exit code: {run_result.returncode}\n"
            if run_result.stdout:
                output += f"STDOUT:\n{run_result.stdout}\n"
            if run_result.stderr:
                output += f"STDERR:\n{run_result.stderr}\n"
            return output
        except FileNotFoundError:
            return "Error: 'csc' compiler not found. Please ensure .NET SDK is installed and in your PATH."
        except Exception as e:
            return f"Execution failed: {e}"

    def run_js(self) -> str:
        """Run sandbox_main.js using Node.js."""
        target_file = self.sandbox_dir / "sandbox_main.js"
        if not target_file.exists():
            return "Error: sandbox_main.js not found in sandbox."
            
        print_system(f"Running {target_file.name} with Node.js...")
        try:
            result = subprocess.run(
                ["node", "sandbox_main.js"],
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            output = f"Exit code: {result.returncode}\n"
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return output
        except FileNotFoundError:
            return "Error: 'node' not found. Please ensure Node.js is installed."
        except Exception as e:
            return f"Execution failed: {e}"

    def run_web(self):
        """Start an HTTP server and open the browser."""
        import webbrowser
        
        if self.http_server_process:
            print_system("Web server is already running. Refreshing browser...")
            webbrowser.open("http://localhost:8000")
            return "Web server is already running on port 8000. Opened browser."
            
        print_system("Starting local HTTP server on port 8000...")
        try:
            self.http_server_process = subprocess.Popen(
                ["python", "-m", "http.server", "8000"],
                cwd=str(self.sandbox_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            webbrowser.open("http://localhost:8000")
            return "HTTP server started on port 8000. Opened in default browser."
        except Exception as e:
            return f"Failed to start web server: {e}"
            
    def stop_web(self):
        if self.http_server_process:
            self.http_server_process.terminate()
            self.http_server_process = None
            print_system("Stopped local HTTP server.")
