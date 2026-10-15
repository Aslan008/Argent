import re
from pathlib import Path
from tools._helpers import _resolve_path
from logger import get_logger

log = get_logger("tools")

def search_files(directory: str = ".", pattern: str = "*", name_contains: str = None, content_contains: str = None, max_results: int = 50) -> str:
    """Recursively search for files matching criteria. Can filter by file pattern, name, and content.
    Uses ripgrep (rg) when available for 10-30x faster searching, falling back to Python rglob.
    """
    import shutil
    import subprocess

    try:
        start_path = _resolve_path(directory)
        if not start_path.exists():
            return f"Error: Directory '{directory}' does not exist."
        if not start_path.is_dir():
            return f"Error: '{directory}' is not a directory."

        results = []
        name_filter = name_contains.lower() if name_contains else None
        content_filter = content_contains.lower() if content_contains else None

        # ── Step 1: Candidate file paths generation ─────────────────────────
        candidate_paths = []
        rg_bin = shutil.which("rg")
        rg_success = False
        if rg_bin:
            try:
                cmd = [rg_bin]
                # Exclude obvious noise directories
                cmd.extend([
                    "--glob", "!**/node_modules/**",
                    "--glob", "!**/__pycache__/**",
                    "--glob", "!**/library/**",
                    "--glob", "!**/obj/**",
                    "--glob", "!**/bin/**",
                    "--glob", "!**/.git/**",
                ])
                if pattern != "*":
                    cmd.extend(["--glob", pattern])

                if content_filter:
                    # Treat query as literal fixed string (-F), case-insensitive (-i), listing files (-l)
                    cmd.extend(["-F", "-l", "-i", content_filter, str(start_path)])
                else:
                    cmd.extend(["--files", str(start_path)])

                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="ignore")
                if proc.returncode in (0, 1):
                    rg_success = True
                    for line in proc.stdout.splitlines():
                        if line.strip():
                            candidate_paths.append(Path(line.strip()))
            except Exception as rg_err:
                log.debug("search_files rg candidate generation failed: %s", rg_err)
                candidate_paths = []
                rg_success = False

        if not rg_success:
            # Fallback candidate generation: rglob when rg is not installed or encountered an execution failure
            candidate_paths = list(start_path.rglob(pattern))

        # ── Step 2: Exact contract filtering ─────────────────────────────────
        skip_dirs = {"node_modules", "__pycache__", "library", "obj", "bin"}
        binary_exts = {
            ".exe", ".dll", ".png", ".jpg", ".jpeg", ".gif", ".pdf",
            ".zip", ".mp3", ".mp4", ".wav", ".asset", ".meta", ".prefab", ".unity",
        }

        for file_path in candidate_paths:
            if not file_path.is_file():
                continue

            # Dot-directories/files exclusion (e.g. .git, .argent)
            if any(part.startswith(".") for part in file_path.parts):
                continue

            parts_lower = [p.lower() for p in file_path.parts]
            if any(p in skip_dirs for p in parts_lower):
                continue
            if "temp" in parts_lower and "assets" in parts_lower:
                continue

            # Binary files are noise in an open listing and a target when they were
            # asked for: by an extension in the glob (`*.png`) or by NAME. The old
            # rule honoured only the glob, so `name_contains="Player"` could never
            # find Player.prefab, Player.asset or a screenshot — every Unity asset
            # and every image was invisible to a search by name. (A content search
            # skips them separately, below: their bytes mean nothing to it.)
            suffix = file_path.suffix.lower()
            if suffix in binary_exts and not name_filter and not pattern.lower().endswith(suffix):
                continue

            # Glob pattern check
            if pattern != "*" and not file_path.match(pattern):
                continue

            # Case-insensitive name_contains check
            if name_filter and name_filter not in file_path.name.lower():
                continue

            # Snippet extraction if content_filter
            snippet = None
            if content_filter:
                if file_path.suffix.lower() in binary_exts:
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content_filter not in content.lower():
                        continue
                    idx = content.lower().find(content_filter)
                    if idx != -1:
                        start = max(0, idx - 40)
                        end = min(len(content), idx + len(content_filter) + 60)
                        snippet = content[start:end].replace("\n", " ").strip()
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(content):
                            snippet = snippet + "..."
                except Exception:
                    continue

            results.append({"path": str(file_path), "name": file_path.name, "snippet": snippet})
            if len(results) >= max_results:
                break

        if not results:
            conditions = []
            if pattern != "*":
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
            if r["snippet"]:
                output.append(f"  >> {r['snippet']}")
        if len(results) >= max_results:
            output.append(f"(Results limited to {max_results}. Use max_results parameter to see more.)")
        return "\n".join(output)

    except Exception as e:
        return f"Error searching files: {e}"


def _normalize_grep_pattern(pattern: str) -> str:
    """Normalize grep/regex pattern received from LLM or user:
    1. Strip matching outer quotes ("..." or '...') if wrapped accidentally.
    2. Convert BRE-style alternation (\\|) to ERE/PCRE (|) unless inside character classes.
    """
    if pattern is None:
        return ""
    if not isinstance(pattern, str):
        pattern = str(pattern)
    if not pattern.strip():
        return ""
    p = pattern
    
    # 1. Strip matching outer quotes ("..." or '...') if wrapped
    s = p.strip()
    if len(s) >= 2 and ((s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'"))):
        quote = s[0]
        inner = s[1:-1]
        
        # Check if the closing quote was actually escaped with an odd number of backslashes (e.g. "abc\"")
        trailing_bs = 0
        for ch in reversed(inner):
            if ch == '\\':
                trailing_bs += 1
            else:
                break
        
        if trailing_bs % 2 == 0:
            # Closing quote is not escaped. Check for unescaped matching quotes in interior.
            escaped = False
            has_unescaped_matching_quote = False
            for ch in inner:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == quote:
                    has_unescaped_matching_quote = True
                    break
            if not has_unescaped_matching_quote:
                p = inner

    # 2. Fix BRE-style \| alternation -> |
    # Standalone \| (e.g. "\|" or " \| ") represents search for literal pipe, not alternation
    if r"\|" in p and p.strip() != r"\|":
        result = []
        in_bracket = False
        i = 0
        n = len(p)
        while i < n:
            if p[i] == '\\' and i + 1 < n:
                if p[i + 1] == '|' and not in_bracket:
                    bs_count = 0
                    j = i - 1
                    while j >= 0 and p[j] == '\\':
                        bs_count += 1
                        j -= 1
                    if bs_count % 2 == 0:
                        # Unescaped backslash before pipe: BRE alternation -> ERE |
                        result.append('|')
                        i += 2
                        continue
                result.append(p[i:i+2])
                i += 2
                continue
            
            # Track bracket class status respecting preceding backslashes
            bs_count = 0
            j = i - 1
            while j >= 0 and p[j] == '\\':
                bs_count += 1
                j -= 1
            is_escaped = (bs_count % 2 != 0)
            
            if p[i] == '[' and not is_escaped:
                in_bracket = True
            elif p[i] == ']' and in_bracket and not is_escaped:
                in_bracket = False
            result.append(p[i])
            i += 1
        p = "".join(result)

    return p

def _python_grep_search(directory: str = None, pattern: str = "", file_pattern: str = None, max_results: int = 30, path: str = None, query: str = None) -> str:
    """Search file contents using regex pattern (Fallback Python implementation)."""
    try:
        raw_path = directory or path or "."
        start_path = _resolve_path(raw_path)
        if not start_path.exists():
            return f"Error: Path '{raw_path}' does not exist."
        
        pattern = pattern or query or ""
        pattern = _normalize_grep_pattern(pattern)
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {e}"
        
        results = []
        skip_ext = {'.exe', '.dll', '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip', '.mp3', '.mp4', '.wav', '.asset', '.meta', '.prefab', '.unity', '.pdb', '.obj', '.bin'}
        skip_dirs = {'.git', '.argent', 'node_modules', '__pycache__', 'venv', 'env', 'Library', 'obj', 'bin', '.venv'}
        
        target_files = []
        if start_path.is_file():
            if start_path.suffix.lower() not in skip_ext:
                target_files.append(start_path)
        elif start_path.is_dir():
            glob_pattern = file_pattern or "*"
            for file_path in start_path.rglob(glob_pattern):
                if not file_path.is_file():
                    continue
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                parts_lower = [p.lower() for p in file_path.parts]
                if any(p.lower() in {d.lower() for d in skip_dirs} for p in file_path.parts):
                    continue
                if 'temp' in parts_lower and 'assets' in parts_lower:
                    continue
                if file_path.suffix.lower() in skip_ext:
                    continue
                target_files.append(file_path)
        else:
            return f"Error: '{raw_path}' is neither a file nor a directory."
        
        for file_path in target_files:
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
            return f"No matches found for pattern '{pattern}' in '{raw_path}'."
        
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

def _resolve_symbol_location(file_path: Path, symbol: str) -> tuple[int, int] | None:
    """Find line (1-indexed) and column (0-indexed) of a symbol in a file."""
    if not symbol or not str(symbol).strip():
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return None

    # Priority 1: Explicit definition syntax across major languages
    def_regex = re.compile(
        r"(?:def|class|fn|func|function|interface|struct|type|enum|trait|pub\s+fn|pub\s+struct|async\s+def)\s+("
        + re.escape(symbol.strip())
        + r")\b"
    )
    for idx, line in enumerate(lines):
        m = def_regex.search(line)
        if m:
            return (idx + 1, m.start(1))

    # Priority 2: Assignment or field declaration
    assign_regex = re.compile(r"\b(" + re.escape(symbol.strip()) + r")\s*(?::|=|\+=)")
    for idx, line in enumerate(lines):
        m = assign_regex.search(line)
        if m:
            return (idx + 1, m.start(1))

    # Priority 3: Any word-boundary token match
    word_regex = re.compile(r"\b(" + re.escape(symbol.strip()) + r")\b")
    for idx, line in enumerate(lines):
        m = word_regex.search(line)
        if m:
            return (idx + 1, m.start(1))

    return None


def _handle_code_target_search(
    start_path: Path,
    raw_path: str,
    symbol: str,
    target: str,
    direction: str = "incoming",
    max_results: int = 50,
    line: int = None,
    column: int = None,
) -> str:
    """Handle semantic symbol exploration (definition, references, implementations, call_hierarchy)."""
    from src.lsp.manager import lsp_manager

    try:
        max_results = max(1, int(max_results))
    except (ValueError, TypeError):
        max_results = 30
    if line is not None:
        try:
            line = int(line)
            if line < 1:
                line = None
        except (ValueError, TypeError):
            line = None
    if column is not None:
        try:
            column = max(0, int(column))
        except (ValueError, TypeError):
            column = 0

    # 1. Definition / Symbol search
    if target in ("definition", "symbol"):
        # If given a specific file:
        if start_path.is_file():
            target_line = line
            target_col = column
            if target_line is None or target_col is None:
                loc = _resolve_symbol_location(start_path, symbol)
                if loc:
                    target_line, target_col = loc

            if target_line is not None and lsp_manager.is_available():
                results = lsp_manager.find_definition(str(start_path), target_line, target_col or 0)
                if results:
                    out = [f"Found definition(s) for '{symbol}' via LSP:"]
                    out.append("-" * 60)
                    for d in results:
                        out.append(f"- {d['file_path']}:{d['line']}:{d['column']}")
                    return "\n".join(out)

            # Fallback for Python via Jedi if available
            try:
                from intelligence import intel
                if target_line is not None:
                    jedi_res = intel.find_definitions(str(start_path), target_line, target_col or 0)
                    if jedi_res and "error" not in jedi_res[0]:
                        out = [f"Found definition(s) for '{symbol}' (via Jedi):"]
                        out.append("-" * 60)
                        for d in jedi_res:
                            out.append(f"- {d['name']} ({d['type']}) in {d['file_path']}:{d['line']}:{d['column']}")
                        return "\n".join(out)
            except Exception:
                pass

        # If searching across workspace or file fallback:
        if lsp_manager.is_available():
            try:
                ws_results = lsp_manager.get_workspace_symbols(symbol)
                if ws_results:
                    return f"Found symbol(s) for '{symbol}' in workspace:\n" + lsp_manager.format_symbols(ws_results)
            except Exception:
                pass

        # Text/regex fallback for definition across files
        def_pattern = r"\b(?:def|class|fn|func|function|interface|struct|type|enum)\s+" + re.escape(symbol) + r"\b"
        fallback_res = grep_search(path=raw_path, pattern=def_pattern, target="all", max_results=max_results)
        if "Found" in fallback_res:
            return f"[LSP definition unresolved; showing definition match(es) for '{symbol}']:\n{fallback_res}"
        return f"No definition found for symbol '{symbol}' in '{raw_path}'."

    # 2. References search
    elif target == "references":
        target_file = None
        target_line = line
        target_col = column

        if start_path.is_file():
            target_file = start_path
            if target_line is None or target_col is None:
                loc = _resolve_symbol_location(target_file, symbol)
                if loc:
                    target_line, target_col = loc
        else:
            # Locate symbol definition first via LSP workspace symbols or fast ripgrep
            if lsp_manager.is_available():
                try:
                    ws_res = lsp_manager.get_workspace_symbols(symbol)
                    if ws_res:
                        first_sym = ws_res[0]
                        fp = first_sym.get("location", {}).get("path")
                        pos = first_sym.get("location", {}).get("range", {}).get("start", {})
                        if fp and pos:
                            target_file = Path(fp)
                            target_line = pos.get("line", 1)
                            target_col = pos.get("character", 0)
                except Exception:
                    pass

            if not target_file:
                def_pattern = r"\b(?:def|class|fn|func|function|interface|struct|type|enum)\s+" + re.escape(symbol.strip()) + r"\b"
                quick_search = grep_search(path=raw_path, pattern=def_pattern, target="all", max_results=5)
                for line_match in quick_search.splitlines():
                    m = re.match(r"^([a-zA-Z]:[\\/].+?|\.?[\w\-./\\]+):(\d+):", line_match.strip())
                    if m:
                        cand_path = Path(m.group(1).strip())
                        if cand_path.is_file():
                            loc = _resolve_symbol_location(cand_path, symbol)
                            if loc:
                                target_file = cand_path
                                target_line, target_col = loc
                                break

        if target_file and target_line is not None and lsp_manager.is_available():
            results = lsp_manager.find_references(str(target_file), target_line, target_col or 0)
            if results:
                total_count = len(results)
                capped_results = results[:max_results]
                out = [f"Found {total_count} reference(s) for '{symbol}' via LSP:"]
                out.append("-" * 60)
                for r in capped_results:
                    out.append(f"- {r['file_path']}:{r['line']}:{r['column']}")
                if total_count > max_results:
                    out.append(f"\n(Results limited to {max_results}. Found {total_count} total.)")
                return "\n".join(out)

        # Fallback to ripgrep word search
        word_pat = r"\b" + re.escape(symbol) + r"\b"
        fallback_res = grep_search(path=raw_path, pattern=word_pat, target="all", max_results=max_results)
        return f"[LSP references unavailable for this target; showing text reference(s) for '{symbol}']:\n{fallback_res}"

    # 3. Implementations search
    elif target == "implementations":
        target_file = None
        target_line = line
        target_col = column
        if start_path.is_file():
            target_file = start_path
            if target_line is None or target_col is None:
                loc = _resolve_symbol_location(target_file, symbol)
                if loc:
                    target_line, target_col = loc

        if target_file and target_line is not None and lsp_manager.is_available():
            impls = lsp_manager.find_implementations(str(target_file), target_line, target_col or 0)
            if impls:
                out = [f"Found {len(impls)} implementation(s) for '{symbol}' via LSP:"]
                out.append("-" * 60)
                for r in impls:
                    out.append(f"- {r['file_path']}:{r['line']}:{r['column']}")
                return "\n".join(out)

        return f"No concrete implementations found for '{symbol}' in '{raw_path}'."

    # 4. Call hierarchy
    elif target == "call_hierarchy":
        target_file = None
        target_line = line
        target_col = column
        if start_path.is_file():
            target_file = start_path
            if target_line is None or target_col is None:
                loc = _resolve_symbol_location(target_file, symbol)
                if loc:
                    target_line, target_col = loc

        if target_file and target_line is not None and lsp_manager.is_available():
            hierarchy = lsp_manager.get_call_hierarchy(str(target_file), target_line, target_col or 0, direction)
            if hierarchy:
                return lsp_manager.format_call_hierarchy(hierarchy, direction)

        return f"No call hierarchy found for '{symbol}' in '{raw_path}'."

    # Unknown target: fallback to all
    return grep_search(path=raw_path, query=symbol, target="all", max_results=max_results)


def grep_search(
    directory: str = None,
    pattern: str = "",
    file_pattern: str = None,
    max_results: int = 30,
    path: str = None,
    query: str = None,
    target: str = "all",
    direction: str = "incoming",
    symbol: str = None,
    line: int = None,
    column: int = None,
) -> str:
    """Universal search across project files and code structure.
    
    Modes (target):
    - 'all' (default): fast full-text/regex ripgrep search in any file (.py, .env, .json, .md, logs).
    - 'definition': find definition/declaration of a function, class, or symbol via LSP/AST.
    - 'references': find exact usages of a symbol across the project via LSP.
    - 'implementations': find concrete implementations of an interface or abstract class via LSP.
    - 'call_hierarchy': tree of callers (direction='incoming') or callees (direction='outgoing') via LSP.
    """
    import subprocess
    
    # Sanitize max_results boundary values (e.g. 0, negative, or non-int string)
    try:
        max_results = max(1, int(max_results))
    except (TypeError, ValueError):
        max_results = 30

    raw_path = directory or path or "."
    start_path = _resolve_path(raw_path)
    if not start_path.exists():
        return f"Error: Path '{raw_path}' does not exist."
    if not (start_path.is_dir() or start_path.is_file()):
        return f"Error: '{raw_path}' is neither a file nor a directory."
        
    search_term = query or pattern or symbol or ""
    normalized_target = (target or "all").strip().lower()

    # Route semantic targets to code intelligence handler
    if normalized_target in ("definition", "references", "implementations", "call_hierarchy", "symbol"):
        if not search_term or not search_term.strip():
            return "Error: A symbol name or query is required for semantic code exploration."
        return _handle_code_target_search(
            start_path=start_path,
            raw_path=raw_path,
            symbol=search_term,
            target=normalized_target,
            direction=direction or "incoming",
            max_results=max_results,
            line=line,
            column=column,
        )

    # Standard full-text / regex search (target == "all")
    clean_pattern = _normalize_grep_pattern(search_term)
        
    try:
        cmd = ["rg", "-n", "-H", "-i", "--no-heading", "-M", "200", "-m", str(max_results)]
        if start_path.is_dir():
            if file_pattern:
                cmd.extend(["-g", file_pattern])
            cmd.extend(["-g", "!*.{exe,dll,png,jpg,jpeg,gif,pdf,zip,mp3,mp4,wav,asset,meta,prefab,unity,pdb,obj,bin}"])
        cmd.append(clean_pattern)
        cmd.append(str(start_path))
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if not lines:
                return f"No matches found for pattern '{clean_pattern}' in '{raw_path}'."
                
            results = []
            for rline in lines:
                m = re.match(r'^(.+?):(\d+):(.*)$', rline)
                if m:
                    file_p = m.group(1)
                    line_num = m.group(2)
                    text = m.group(3).strip()
                    results.append({'file': file_p, 'line': line_num, 'text': text})
                elif start_path.is_file():
                    m_single = re.match(r'^(\d+):(.*)$', rline)
                    if m_single:
                        results.append({'file': str(start_path), 'line': m_single.group(1), 'text': m_single.group(2).strip()})
                if len(results) >= max_results:
                    break
            
            output = [f"Found {len(results)} match(es) for '{clean_pattern}' (via ripgrep):"]
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
            return f"No matches found for pattern '{clean_pattern}' in '{raw_path}'."
        else:
            log.warning(f"ripgrep returned non-zero code {result.returncode}, falling back to python. Error: {result.stderr}")
            pass
            
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning(f"ripgrep error: {e}, falling back to python")
        pass
        
    return _python_grep_search(directory=raw_path, pattern=clean_pattern, file_pattern=file_pattern, max_results=max_results)
