import sys
sys.path.insert(0, str(sys.argv[1]) if len(sys.argv) > 1 else ".")
from tools.plugin_tools import create_plugin

code = '''"""
Prompt Enhancer Plugin
======================
Automatically enriches user prompts with project context before sending to LLM.

Context includes:
  - Project directory structure (top-level overview)
  - README summary (first N lines)
  - Dependencies from requirements.txt
  - Key module files in src/, tests/, plugins/, tools/, skills/

This dramatically improves AI understanding of the codebase without
manual context injection.
"""

from ui import console, print_system
import os
import re
from pathlib import Path

# --- Configuration ---

README_MAX_LINES = 40          # How many lines of README to include
CONTEXT_CACHE = None           # Cached context string (reused per session)
CACHE_DIR = ''                 # Cached project directory


# --- Context Builders ---

def _get_project_structure(project_dir):
    """
    Scans top-level directory and returns a compact tree of files and subdirectories.
    Groups files by type for readability.
    """
    try:
        entries = sorted(os.listdir(project_dir))
        dirs = []
        files = []
        
        for entry in entries:
            full = os.path.join(project_dir, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                dirs.append(entry)
            elif os.path.isfile(full):
                files.append(entry)
        
        lines = []
        lines.append(f"[Project Directory: {Path(project_dir).name}]")
        
        if dirs:
            lines.append(f"  Directories ({len(dirs)}):")
            for d in dirs:
                count = len(os.listdir(os.path.join(project_dir, d)))
                lines.append(f"    📁 {d}/ ({count} items)")
        
        if files:
            ext_groups = {}
            others = []
            for f in files:
                ext = os.path.splitext(f)[1]
                if ext:
                    ext_groups.setdefault(ext, []).append(f)
                else:
                    others.append(f)
            
            lines.append(f"  Files ({len(files)}):")
            for ext, group in sorted(ext_groups.items()):
                lines.append(f"    {ext}: {', '.join(sorted(group))}")
            if others:
                lines.append(f"    no ext: {', '.join(sorted(others))}")
        
        return chr(10).join(lines)
    except Exception as e:
        return f"[Project structure scan failed: {e}]"


def _get_readme_summary(project_dir):
    """
    Reads the first N lines of README.md to capture project overview.
    """
    readme_path = os.path.join(project_dir, "README.md")
    if not os.path.exists(readme_path):
        return ""
    
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        summary_lines = lines[:README_MAX_LINES]
        summary = "".join(summary_lines)
        summary = summary.rstrip()
        truncated = "..." if len(lines) > README_MAX_LINES else ""
        
        return f"[README Summary]" + chr(10) + summary + chr(10) + truncated
    except Exception as e:
        return f"[README read failed: {e}]"


def _get_dependencies(project_dir):
    """
    Parses requirements.txt and returns a compact dependency list.
    """
    req_path = os.path.join(project_dir, "requirements.txt")
    if not os.path.exists(req_path):
        return ""
    
    try:
        with open(req_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        deps = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-r"):
                pkg = re.split(r"[><=!~\[\]]", line)[0].strip()
                if pkg:
                    deps.append(pkg)
        
        if deps:
            return f"[Dependencies ({len(deps)} packages)]" + chr(10) + f"  {', '.join(sorted(deps))}"
        return ""
    except Exception as e:
        return f"[Requirements parse failed: {e}]"


def _get_key_modules(project_dir):
    """
    Lists key Python modules in src/, tests/, plugins/, tools/, skills/ directories.
    """
    key_dirs = ["src", "tests", "plugins", "tools", "skills"]
    modules = []
    
    for subdir in key_dirs:
        dir_path = os.path.join(project_dir, subdir)
        if not os.path.isdir(dir_path):
            continue
        
        py_files = sorted([
            f for f in os.listdir(dir_path)
            if f.endswith(".py") and not f.startswith("_")
        ])
        
        if py_files:
            modules.append(f"  {subdir}/: {', '.join(py_files)}")
    
    if modules:
        return "[Key Modules]" + chr(10) + chr(10).join(modules)
    return ""


def _get_skills_list(project_dir):
    """
    Lists available skill markdown files.
    """
    skills_dir = os.path.join(project_dir, "skills")
    if not os.path.isdir(skills_dir):
        return ""
    
    md_files = sorted([
        f.replace(".md", "") for f in os.listdir(skills_dir)
        if f.endswith(".md")
    ])
    
    if md_files:
        return f"[Available Skills ({len(md_files)})]" + chr(10) + f"  {', '.join(md_files)}"
    return ""


def _get_plugins_list(project_dir):
    """
    Lists loaded plugin modules.
    """
    plugins_dir = os.path.join(project_dir, "plugins")
    if not os.path.isdir(plugins_dir):
        return ""
    
    py_files = sorted([
        f.replace(".py", "") for f in os.listdir(plugins_dir)
        if f.endswith(".py") and not f.startswith("_")
    ])
    
    if py_files:
        return f"[Loaded Plugins ({len(py_files)})]" + chr(10) + f"  {', '.join(py_files)}"
    return ""


# --- Context Assembly ---

def _build_full_context(project_dir):
    """
    Assembles all context sections into a single enriched block.
    Only includes sections that have content.
    """
    sections = [
        _get_project_structure(project_dir),
        _get_readme_summary(project_dir),
        _get_dependencies(project_dir),
        _get_key_modules(project_dir),
        _get_skills_list(project_dir),
        _get_plugins_list(project_dir),
    ]
    
    active_sections = [s for s in sections if s]
    
    if not active_sections:
        return ""
    
    return (chr(10) + chr(10)).join(active_sections)


# --- Pre-Prompt Hook ---

def pre_prompt(text):
    """
    Intercepts user prompt and appends project context.
    Uses caching to avoid re-scanning on every prompt within the same session.
    
    Context is injected at the END of the prompt (after user text) to preserve
    the user's original intent at the beginning, where small models attend best.
    """
    global CONTEXT_CACHE, CACHE_DIR
    
    project_dir = os.getcwd()
    
    if CONTEXT_CACHE is None or CACHE_DIR != project_dir:
        CONTEXT_CACHE = _build_full_context(project_dir)
        CACHE_DIR = project_dir
        
        if CONTEXT_CACHE:
            console.print("[dim]🔍 Project context loaded for prompt enhancement[/dim]")
    
    if CONTEXT_CACHE:
        injection = (
            chr(10) + chr(10) + "--- [PROJECT CONTEXT] ---" + chr(10)
            + CONTEXT_CACHE + chr(10)
            + "------------------------------------------" + chr(10) + chr(10)
        )
        return text + injection
    
    return text


# --- Commands ---

def command_context(*args):
    """
    Displays the current project context that is being injected into prompts.
    Usage: /context
    
    Args:
        refresh  - Force rebuild the context cache
    """
    global CONTEXT_CACHE, CACHE_DIR
    
    if "refresh" in args:
        CONTEXT_CACHE = None
        project_dir = os.getcwd()
        CONTEXT_CACHE = _build_full_context(project_dir)
        CACHE_DIR = project_dir
        console.print("[green]✓ Context cache refreshed[/green]")
    
    if CONTEXT_CACHE:
        console.print(chr(10) + "[bold cyan]📋 Current Project Context[/bold cyan]")
        console.print(CONTEXT_CACHE)
    else:
        console.print("[yellow]No context available. Run /context refresh to build it.[/yellow]")


def command_refresh_context(*args):
    """Alias for /context refresh"""
    command_context("refresh")
'''

result = create_plugin('prompt_enhancer_plugin', code)
print(result)
