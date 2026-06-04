import re
from pathlib import Path
from tools._helpers import _resolve_path
from logger import get_logger

log = get_logger("tools")

def search_files(directory: str = ".", pattern: str = "*", name_contains: str = None, content_contains: str = None, max_results: int = 50) -> str:
    """Recursively search for files matching criteria. Can filter by file pattern, name, and content."""
    try:
        start_path = _resolve_path(directory)
        if not start_path.exists():
            return f"Error: Directory '{directory}' does not exist."
        if not start_path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        results = []
        name_filter = name_contains.lower() if name_contains else None
        content_filter = content_contains.lower() if content_contains else None
        
        for file_path in start_path.rglob(pattern):
            if not file_path.is_file():
                continue
            
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if any(part in ['node_modules', '__pycache__', 'Library', 'Temp', 'obj', 'bin'] for part in file_path.parts):
                continue
            
            if name_filter and name_filter not in file_path.name.lower():
                continue
            
            snippet = None
            if content_filter:
                try:
                    if file_path.suffix.lower() in ['.exe', '.dll', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.mp3', '.mp4', '.wav', '.asset', '.meta', '.prefab', '.unity']:
                        continue
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    if content_filter not in content.lower():
                        continue
                    idx = content.lower().find(content_filter)
                    if idx != -1:
                        start = max(0, idx - 40)
                        end = min(len(content), idx + len(content_filter) + 60)
                        snippet = content[start:end].replace('\n', ' ').strip()
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                except Exception:
                    continue
            
            results.append({'path': str(file_path), 'name': file_path.name, 'snippet': snippet})
            
            if len(results) >= max_results:
                break
        
        if not results:
            conditions = []
            if pattern != '*':
                conditions.append(f"pattern='{pattern}'")
            if name_contains:
                conditions.append(f"name contains '{name_contains}'")
            if content_contains:
                conditions.append(f"content contains '{content_contains}'")
            condition_str = " and ".join(conditions) if conditions else "any file"
            return f"No files found matching: {condition_str}"
        
        output = [f"Found {len(results)} file(s):"]
        output.append("-" * 60)
        
        for r in results:
            output.append(f"{r['path']}")
            if r['snippet']:
                output.append(f"  >> {r['snippet']}")
        
        if len(results) >= max_results:
            output.append(f"(Results limited to {max_results}. Use max_results parameter to see more.)")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error searching files: {e}"

def _python_grep_search(directory: str, pattern: str, file_pattern: str = None, max_results: int = 30) -> str:
    """Search file contents using regex pattern (Fallback Python implementation)."""
    try:
        start_path = _resolve_path(directory)
        if not start_path.exists():
            return f"Error: Directory '{directory}' does not exist."
        if not start_path.is_dir():
            return f"Error: '{directory}' is not a directory."
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {e}"
        
        results = []
        glob_pattern = file_pattern or "*"
        skip_ext = {'.exe', '.dll', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.mp3', '.mp4', '.wav', '.asset', '.meta', '.prefab', '.unity', '.pdb', '.obj', '.bin'}
        skip_dirs = {'.git', '.argent', 'node_modules', '__pycache__', 'venv', 'env', 'Library', 'Temp', 'obj', 'bin', '.venv'}
        
        for file_path in start_path.rglob(glob_pattern):
            if not file_path.is_file():
                continue
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if any(part in skip_dirs for part in file_path.parts):
                continue
            if file_path.suffix.lower() in skip_ext:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
            except Exception:
                continue
            
            for line_num, line in enumerate(lines, 1):
                match = regex.search(line)
                if match:
                    matched_text = match.group(0)
                    line_stripped = line.rstrip()
                    results.append({
                        'file': str(file_path),
                        'line': line_num,
                        'text': line_stripped,
                        'match': matched_text
                    })
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break
        
        if not results:
            return f"No matches found for pattern '{pattern}' in '{directory}'."
        
        output = [f"Found {len(results)} match(es) for '{pattern}' (via Python):"]
        output.append("-" * 60)
        current_file = None
        for r in results:
            if r['file'] != current_file:
                current_file = r['file']
                output.append(f"\n{current_file}:")
            output.append(f"  {r['line']}: {r['text']}")
        
        if len(results) >= max_results:
            output.append(f"\n(Results limited to {max_results}. Use max_results to see more.)")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error in grep search: {e}"

def grep_search(directory: str, pattern: str, file_pattern: str = None, max_results: int = 30) -> str:
    """Search file contents using ripgrep (rg) with fallback to regex pattern on python. Faster and more precise than search_files for finding specific code."""
    import subprocess
    
    start_path = _resolve_path(directory)
    if not start_path.exists():
        return f"Error: Directory '{directory}' does not exist."
    if not start_path.is_dir():
        return f"Error: '{directory}' is not a directory."
        
    try:
        cmd = ["rg", "-n", "-i", "--no-heading", "-M", "200", "-m", str(max_results)]
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        cmd.extend(["-g", "!*.{exe,dll,png,jpg,jpeg,gif,pdf,zip,mp3,mp4,wav,asset,meta,prefab,unity,pdb,obj,bin}"])
        cmd.append(pattern)
        cmd.append(str(start_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if not lines:
                return f"No matches found for pattern '{pattern}' in '{directory}'."
                
            results = []
            for line in lines:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_p = parts[0]
                    line_num = parts[1]
                    text = parts[2].strip()
                    results.append({'file': file_p, 'line': line_num, 'text': text})
                    if len(results) >= max_results:
                        break
            
            output = [f"Found {len(results)} match(es) for '{pattern}' (via ripgrep):"]
            output.append("-" * 60)
            current_file = None
            for r in results:
                if r['file'] != current_file:
                    current_file = r['file']
                    output.append(f"\n{current_file}:")
                output.append(f"  {r['line']}: {r['text']}")
            
            if len(lines) >= max_results:
                output.append(f"\n(Results limited to {max_results}. Use max_results to see more.)")
                
            return "\n".join(output)
            
        elif result.returncode == 1:
            return f"No matches found for pattern '{pattern}' in '{directory}'."
        else:
            log.warning(f"ripgrep returned non-zero code {result.returncode}, falling back to python. Error: {result.stderr}")
            pass
            
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"ripgrep error: {e}, falling back to python")
        pass
        
    return _python_grep_search(directory, pattern, file_pattern, max_results)
