"""Launch Argent with the native Rust TUI (Ratatui + PyO3).

It looks like the terminal REPL because it draws the REPL's own output: blocks
are rendered by `src/bridge/render.py` (Rich + Pygments) and shown as ANSI.

Features:
- In-process, no JSON-RPC
- Typing never blocks: the input stays live during agent turns, messages queue
- Reasoning and tool cards open with a click, or Ctrl+N/P then Tab
- Mouse-wheel scrolling, drag-to-copy (Shift+drag uses the terminal's own selection)
- Redraws only when something changed: no work while idle
"""

import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENV_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
_ERROR_LOG = os.path.join(_ROOT, ".argent", "tui_error.log")
_CRASH_LOG = os.path.join(_ROOT, ".argent", "tui_crash.log")
_LAUNCH_MARK = os.path.join(_ROOT, ".argent", "launch.ok")
_crash_file = None  # kept referenced: faulthandler writes to it from a signal handler


def _announce_start() -> None:
    """First action of the process: tell Argent.bat the interpreter really started.

    The launcher waits for this file and falls back to another way of opening a
    console when it does not appear (see Argent.bat). It also arms faulthandler, so a
    native crash (the Rust extension, a DLL) leaves a traceback on disk instead of a
    window that silently closes.
    """
    global _crash_file
    try:
        os.makedirs(os.path.dirname(_LAUNCH_MARK), exist_ok=True)
        with open(_LAUNCH_MARK, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except OSError:
        pass
    try:
        import faulthandler
        _crash_file = open(_CRASH_LOG, "a", encoding="utf-8")
        faulthandler.enable(file=_crash_file, all_threads=True)
    except Exception:
        pass


_announce_start()


def _report_failure(message: str) -> None:
    """Print *message*, persist it, and hold the console open.

    The TUI is normally started from a throwaway console window (Argent.bat uses
    `start`), so simply exiting would make the message flash and vanish — which
    is exactly why a startup failure used to be undiagnosable. Keep the window
    until the user acknowledges it, and always leave a copy on disk.
    """
    print(message)

    try:
        os.makedirs(os.path.dirname(_ERROR_LOG), exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(_ERROR_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"\n=== {stamp} ===\n{os.environ.get('PYTHONEXECUTABLE', sys.executable)}\n{message}\n")
        print(f"\nЛог: {_ERROR_LOG}")
    except Exception:
        pass

    try:
        if sys.stdin.isatty():
            input("\nНажмите Enter, чтобы закрыть окно...")
    except Exception:
        pass


def _report_missing_extension(exc: BaseException) -> None:
    """Explain *why* argent_tui is missing instead of assuming it was never built.

    The usual cause is not a stale build: it is launching this file with the
    wrong interpreter. `python run_tui.py` picks up whatever `python.exe` comes
    first on PATH, which on this machine is CPython 3.14, while the extension is
    only ever installed into the project venv (CPython 3.12).
    """
    lines = [f"Ошибка импорта argent_tui: {exc}", f"Интерпретатор: {sys.executable}",
             f"Версия Python: {sys.version.split()[0]}", ""]

    if os.path.exists(_VENV_PY) and os.path.normcase(_VENV_PY) != os.path.normcase(sys.executable):
        lines += [
            "Модуль собран и установлен в venv проекта, но запущен другой Python.",
            "Запустите так:",
            f'    "{_VENV_PY}" "{os.path.join(_ROOT, "run_tui.py")}"',
            "или просто запустите Argent.bat из корня проекта.",
        ]
    else:
        lines += [
            "Соберите крейт командой:",
            f'    "{os.path.join(_ROOT, ".venv", "Scripts", "maturin.exe")}" develop -m crates/argent-tui/Cargo.toml',
        ]

    _report_failure("\n".join(lines))


try:
    import argent_tui
except ImportError as e:
    _report_missing_extension(e)
    sys.exit(1)

try:
    from src.bridge.repl_bridge import ReplBridge
except Exception:
    # Outside any other guard: without this an import error (a broken config file, a
    # locked module) printed a traceback into a window that closed at once.
    _report_failure(f"Не удалось загрузить мост с агентом:\n{traceback.format_exc()}")
    sys.exit(1)


def main():
    try:
        bridge = ReplBridge()
    except Exception:
        _report_failure(f"Не удалось создать мост с агентом:\n{traceback.format_exc()}")
        sys.exit(1)

    try:
        argent_tui.run_tui(bridge)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _report_failure(f"Ошибка во время работы Rust TUI: {e}\n\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()