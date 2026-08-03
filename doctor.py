"""
/doctor — environment self-diagnostics.

Fast, side-effect-free checks of everything Argent depends on: provider
connectivity, model tier and which automatic adaptations are active,
optional dependency stacks, browser automation, MCP configuration.
Each check returns (status, detail) where status is "ok" / "warn" / "fail".
"""

import importlib.util
import platform
import sys
from pathlib import Path

from src.agent.shell import run_text

from rich.table import Table

from config import (
    CONFIG_FILE, get_current_model, get_provider, get_context_window,
    get_model_size_category, get_model_category_override, get_browser_mode,
    get_browser_name, get_mcp_servers, get_hooks_dir, get_skills_dir,
    get_embedding_provider, get_auto_rag,
)

OK, WARN, FAIL = "ok", "warn", "fail"

# ASCII-only markers: must render on legacy Windows consoles (cp1251) too.
_STATUS_STYLE = {
    OK: "[bold green]OK[/bold green]",
    WARN: "[bold yellow]WARN[/bold yellow]",
    FAIL: "[bold red]FAIL[/bold red]",
}


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _check_version():
    try:
        from version import __version__
        return (OK, f"Argent v{__version__}")
    except Exception:
        return (WARN, "version unknown")


def _check_python():
    v = sys.version_info
    detail = f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()}"
    return (OK if v >= (3, 10) else WARN, detail)


def _check_provider():
    provider_name = get_provider()
    try:
        from providers import create_provider
        provider = create_provider()
        error = provider.validate_config()
        if error:
            return (FAIL, f"{provider_name}: {error}")
        return (OK, f"{provider_name}: connected")
    except Exception as e:
        return (FAIL, f"{provider_name}: {e}")


def _check_model_tier():
    model = get_current_model()
    provider = get_provider()
    category = get_model_size_category(model)
    override = get_model_category_override()

    from src.agent.strategy import get_model_strategy
    strategy = get_model_strategy(model, provider)

    adaptations = []
    if strategy.wants_constrained_decoding():
        adaptations.append("constrained-decoding")
    if strategy.wants_objective_anchor():
        adaptations.append("objective-anchor")
    adaptations.append("native-tools" if strategy.supports_native_tools() else "json-steps")

    detail = (
        f"{model} -> tier '{category}'"
        + (f" (manual override)" if override else "")
        + f", {type(strategy).__name__}, ctx {get_context_window()}, "
        + ", ".join(adaptations)
    )
    return (OK, detail)


def _check_auxiliary_model():
    from config import get_auxiliary_model, get_auxiliary_provider, get_provider as _gp
    aux = get_auxiliary_model()
    if not aux:
        return (OK, "not set - service tasks use the main model")
    return (OK, f"{get_auxiliary_provider() or _gp()}:{aux} (summarization, /commit)")


def _check_browser():
    if not _module_available("playwright"):
        return (WARN, "playwright not installed — browser tools unavailable")
    try:
        from browser_detect import detect_browsers
        browsers = detect_browsers()
        mode = get_browser_mode()
        if mode == "user" and not browsers:
            return (WARN, "mode 'user' but no CDP-capable browser detected")
        names = ", ".join(b.key for b in browsers) if browsers else "none detected"
        return (OK, f"mode '{mode}' ({get_browser_name()}); detected: {names}")
    except Exception as e:
        return (WARN, f"playwright ok, browser detection failed: {e}")


def _check_rag():
    if not _module_available("chromadb"):
        return (WARN, "chromadb not installed — RAG (/enable_rag) unavailable")
    emb = get_embedding_provider()
    if emb == "sentence_transformers" and not _module_available("sentence_transformers"):
        return (WARN, "chromadb ok, but sentence-transformers missing (switch /rag_provider to ollama?)")
    from config import get_auto_retrieve, get_external_kbs
    flags = []
    if get_auto_rag():
        flags.append("auto-RAG")
    if get_auto_retrieve():
        flags.append("auto-retrieve")
    kbs = [kb.get("name", kb.get("id")) for kb in get_external_kbs() if kb.get("enabled", True)]
    detail = f"chromadb + {emb}"
    if flags:
        detail += f" ({', '.join(flags)})"
    if kbs:
        detail += f"; KBs: {', '.join(kbs)}"
    # all-MiniLM is weaker on technical docs; recommend nomic for doc-heavy KBs.
    if kbs and emb != "ollama":
        detail += " — tip: for docs, nomic-embed-text via Ollama beats MiniLM"
    return (OK, detail)


def _check_web_search():
    """Which search engines this session will actually federate."""
    from src.research.search import engine_labels
    active = engine_labels()
    detail = ", ".join(active)

    # Query language and reranker language must match, or the pipeline fights
    # itself: non-English pages get retrieved and then scored so low by an
    # English-only reranker that they never survive into the answer.
    from config import get_search_languages
    from src.research.rerank import active_model_name, _DEFAULT_MODEL
    langs = get_search_languages()
    multilingual_reranker = active_model_name() != _DEFAULT_MODEL
    detail += f"; query languages: {', '.join(langs)}"
    if langs != ["en"] and not multilingual_reranker:
        return (WARN, detail + " — but the reranker is English-only, so non-English "
                "results get buried even when relevant (set 'reranker_model')")
    if multilingual_reranker and langs == ["en"]:
        detail += " (multilingual reranker loaded but queries are English-only)"

    if "Brave" not in active:
        # Brave is a second INDEPENDENT index (DuckDuckGo's results are largely
        # Bing's), so adding it widens recall rather than reshuffling the same
        # pages — and it replaces a scraper with a contracted API.
        detail += (" — optional: add a Brave Search API key for a second "
                   "independent index (config key 'brave_api_key')")
    return (OK, detail)


def _check_optional_deps():
    deps = {
        "json5": "tolerant JSON parsing for weak models",
        "tiktoken": "precise token counting fallback",
        "jedi": "find_definition / find_references",
        "ddgs": "web search",
    }
    missing = [f"{name} ({why})" for name, why in deps.items() if not _module_available(name)]
    if missing:
        return (WARN, "missing: " + "; ".join(missing))
    return (OK, ", ".join(deps))


def _check_mcp():
    servers = get_mcp_servers()
    if not servers:
        return (OK, "no servers configured")
    names = ", ".join(f"{s['name']} ({s.get('type', 'stdio')})" for s in servers)
    return (OK, f"{len(servers)} configured: {names}")


def _check_git():
    try:
        res = run_text(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, timeout=5,
        )
        if res.returncode != 0:
            return (WARN, "not a git repository — /commit, checkpoints and rollback unavailable")
        dirty = run_text(
            ["git", "status", "--porcelain"],
            capture_output=True, timeout=5,
        ).stdout.strip()
        return (OK, "repository detected" + (f", {len(dirty.splitlines())} uncommitted change(s)" if dirty else ", clean"))
    except FileNotFoundError:
        return (FAIL, "git executable not found")
    except Exception as e:
        return (WARN, f"git check failed: {e}")


def _check_extensions():
    hooks_dir = Path(get_hooks_dir())
    skills_dir = Path(get_skills_dir())
    plugins = len(list(hooks_dir.glob("*.py"))) if hooks_dir.exists() else 0
    skills = len(list(skills_dir.glob("*.md"))) if skills_dir.exists() else 0
    return (OK, f"{plugins} plugin(s) in {hooks_dir.name}/, {skills} skill(s) in {skills_dir.name}/")


def _check_config():
    return (OK, str(CONFIG_FILE))


CHECKS = [
    ("Argent version", _check_version),
    ("Python", _check_python),
    ("Provider", _check_provider),
    ("Model & tier", _check_model_tier),
    ("Auxiliary model", _check_auxiliary_model),
    ("Browser automation", _check_browser),
    ("RAG / semantic search", _check_rag),
    ("Web search engines", _check_web_search),
    ("Optional deps", _check_optional_deps),
    ("MCP servers", _check_mcp),
    ("Git", _check_git),
    ("Plugins & skills", _check_extensions),
    ("Config file", _check_config),
]


def run_diagnostics() -> list:
    """Run all checks, return [(name, status, detail)]. Never raises."""
    results = []
    for name, check in CHECKS:
        try:
            status, detail = check()
        except Exception as e:
            status, detail = FAIL, f"check crashed: {e}"
        results.append((name, status, detail))
    return results


def print_diagnostics():
    """Render the diagnostics table to the console."""
    from ui import console
    results = run_diagnostics()

    table = Table(title="Argent Doctor", show_lines=False, expand=False)
    table.add_column("Check", style="bold cyan", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Details", overflow="fold")

    for name, status, detail in results:
        table.add_row(name, _STATUS_STYLE[status], detail)

    console.print(table)

    fails = sum(1 for _, s, _ in results if s == FAIL)
    warns = sum(1 for _, s, _ in results if s == WARN)
    if fails:
        console.print(f"[bold red]{fails} critical issue(s) found.[/bold red]")
    elif warns:
        console.print(f"[bold yellow]{warns} warning(s) — Argent works, some features are limited.[/bold yellow]")
    else:
        console.print("[bold green]All systems operational.[/bold green]")
