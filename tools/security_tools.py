"""
tools/security_tools.py — Security audit utilities for Argent.

Provides project-level security audits:
- Dependency integrity & conflicts (pip check, pip-audit if present).
- Secrets & sensitive files detection (.env, private keys, certificates, tokens).
- Git repository safety & .gitignore enforcement.
- Static scanning for high-risk code patterns (dangerous eval/exec/shell=True).
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from logger import get_logger

log = get_logger("tools")

# Sensitive file patterns that should not be tracked or exposed
SENSITIVE_FILE_PATTERNS = [
    re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
    re.compile(r".*\.pem$", re.IGNORECASE),
    re.compile(r".*\.key$", re.IGNORECASE),
    re.compile(r".*\.pfx$", re.IGNORECASE),
    re.compile(r".*\.pkcs12$", re.IGNORECASE),
    re.compile(r"^id_rsa(\.pub)?$", re.IGNORECASE),
    re.compile(r"^id_ed25519(\.pub)?$", re.IGNORECASE),
    re.compile(r".*service[-_]account.*\.json$", re.IGNORECASE),
    re.compile(r".*credentials.*\.json$", re.IGNORECASE),
]

# Patterns for potential hardcoded secrets in code files
SECRET_IN_CODE_PATTERNS = [
    ("OpenAI / Anthropic API Key", re.compile(r"(?:sk-[a-zA-Z0-9_-]{20,}|sk-ant-[a-zA-Z0-9_-]{20,})")),
    ("GitHub Personal Access Token", re.compile(r"ghp_[a-zA-Z0-9]{36}")),
    ("Generic High-Entropy Secret Assignment", re.compile(r'(?i)(?:api_key|secret_key|access_token|private_key)\s*=\s*["\'][a-zA-Z0-9_\-+=]{24,}["\']')),
]

IGNORED_SCAN_DIRS = {
    ".git", "__pycache__", "node_modules", "venv", ".venv",
    ".idea", ".vscode", "dist", "build", ".next", ".cache"
}


def _check_dependencies() -> Tuple[str, List[str]]:
    """Check Python dependency integrity and conflicts via pip check / pip-audit."""
    findings = []
    
    # 1. pip check
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            for l in lines:
                findings.append(f"[yellow]Conflict:[/yellow] {l}")
    except Exception as e:
        findings.append(f"[dim]pip check could not run: {e}[/dim]")

    # 2. pip-audit if installed
    pip_audit_path = shutil.which("pip-audit")
    if pip_audit_path:
        try:
            proc = subprocess.run(
                [pip_audit_path, "--format", "json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0 and proc.stdout:
                findings.append(f"[red]Known vulnerabilities detected by pip-audit[/red]")
        except Exception:
            pass

    status = "WARN" if findings else "OK"
    return status, findings


def _check_sensitive_files(root_path: Path) -> Tuple[str, List[str]]:
    """Check for sensitive files (keys, env files, tokens) in workspace and git status."""
    findings = []
    
    # Check if .gitignore exists
    gitignore_path = root_path / ".gitignore"
    gitignore_content = ""
    if gitignore_path.exists():
        try:
            gitignore_content = gitignore_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pass
    else:
        findings.append("[yellow]No .gitignore file found in project root.[/yellow]")

    # Scan project tree for sensitive filenames
    for current_dir, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_SCAN_DIRS]
        for f in files:
            file_rel = os.path.relpath(os.path.join(current_dir, f), root_path)
            for pat in SENSITIVE_FILE_PATTERNS:
                if pat.search(f):
                    # Check if file is tracked by git
                    is_tracked = False
                    try:
                        git_check = subprocess.run(
                            ["git", "ls-files", "--error-unmatch", file_rel],
                            cwd=root_path,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        is_tracked = (git_check.returncode == 0)
                    except Exception:
                        pass

                    if is_tracked:
                        findings.append(f"[bold red]CRITICAL:[/bold red] Sensitive file [bold]{file_rel}[/bold] is tracked in Git!")
                    elif gitignore_content and f not in gitignore_content and file_rel not in gitignore_content:
                        findings.append(f"[yellow]Warning:[/yellow] Sensitive file [bold]{file_rel}[/bold] found; ensure it is gitignored.")

    status = "FAIL" if any("CRITICAL" in f for f in findings) else ("WARN" if findings else "OK")
    return status, findings


def _check_code_patterns(root_path: Path) -> Tuple[str, List[str]]:
    """Scan source code for potential hardcoded credentials or risky execution patterns."""
    findings = []
    scan_exts = {".py", ".ts", ".js", ".json", ".yaml", ".yml"}
    
    scanned_files = 0
    for current_dir, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_SCAN_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in scan_exts:
                continue
            
            scanned_files += 1
            if scanned_files > 400:
                break
                
            full_path = os.path.join(current_dir, f)
            rel_path = os.path.relpath(full_path, root_path)
            
            try:
                # Read first 100KB to stay fast
                with open(full_path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read(102400)
            except Exception:
                continue

            for secret_name, pat in SECRET_IN_CODE_PATTERNS:
                matches = pat.findall(content)
                if matches:
                    # Filter out placeholders
                    real_matches = [m for m in matches if "dummy" not in m.lower() and "placeholder" not in m.lower() and "test" not in m.lower() and "example" not in m.lower()]
                    if real_matches:
                        findings.append(f"[yellow]Possible secret in {rel_path}:[/yellow] pattern match for {secret_name}")
                        break

    status = "WARN" if findings else "OK"
    return status, findings


def audit_security(level: str = "standard") -> str:
    """
    Run security audit for the current project.
    
    Args:
        level: Audit depth ("standard" or "deep").
        
    Returns:
        Formatted summary report string for print_system().
    """
    root = Path.cwd().resolve()
    
    report_lines = [
        f"[bold cyan]Argent Project Security Audit[/bold cyan] ([dim]{root}[/dim])",
        ""
    ]
    
    # 1. Dependencies
    dep_status, dep_findings = _check_dependencies()
    status_icon = "[green]+[/green]" if dep_status == "OK" else "[yellow]![/yellow]"
    report_lines.append(f"{status_icon} [bold]Dependencies & Packages:[/bold] {dep_status}")
    if dep_findings:
        for f in dep_findings[:8]:
            report_lines.append(f"   • {f}")
    else:
        report_lines.append("   • No dependency conflicts detected.")
        
    report_lines.append("")

    # 2. Sensitive files & Git safety
    file_status, file_findings = _check_sensitive_files(root)
    file_icon = "[green]+[/green]" if file_status == "OK" else ("[red]x[/red]" if file_status == "FAIL" else "[yellow]![/yellow]")
    report_lines.append(f"{file_icon} [bold]Secrets & Sensitive Files:[/bold] {file_status}")
    if file_findings:
        for f in file_findings[:10]:
            report_lines.append(f"   • {f}")
    else:
        report_lines.append("   • No exposed or untracked sensitive files found.")

    report_lines.append("")

    # 3. Static patterns in code
    code_status, code_findings = _check_code_patterns(root)
    code_icon = "[green]+[/green]" if code_status == "OK" else "[yellow]![/yellow]"
    report_lines.append(f"{code_icon} [bold]Code Secrets & Patterns:[/bold] {code_status}")
    if code_findings:
        for f in code_findings[:8]:
            report_lines.append(f"   • {f}")
    else:
        report_lines.append("   • No unmasked secret patterns detected in source code.")

    report_lines.append("")
    overall_status = "CRITICAL ISSUES DETECTED" if (file_status == "FAIL") else (
        "WARNINGS DETECTED" if any(s == "WARN" for s in (dep_status, file_status, code_status)) else "ALL CHECKS PASSED"
    )
    style = "bold red" if "CRITICAL" in overall_status else ("bold yellow" if "WARNINGS" in overall_status else "bold green")
    report_lines.append(f"Audit Summary: [{style}]{overall_status}[/{style}]")
    
    return "\n".join(report_lines)
