"""`/tui-build` — собрать расширение TUI, поставить его и сказать, что осталось человеку.

Зачем командой, а не руками: цикл состоит из шести шагов — `maturin build --release`,
найти wheel, скопировать в `scratch/wheels`, снять остатки оборванной установки,
`pip install --force-reinstall`, перезапустить окно. Пропущенный шаг стоит дорого:
на диске остаётся старая сборка, она отвечает на «новая фича не работает», и это уже
дважды стоило полного круга отладки не того кода.

Чего команда сделать не может: перезапустить Argent. `.pyd` держит живой процесс
(в том числе тот, в котором идёт этот самый разговор), поэтому новый файл ложится на
диск рядом, а подхватывается при следующем запуске. Об этом команда говорит прямо.

Использование:
  /tui-build              собрать, установить, сказать про перезапуск
  /tui-build --build      только собрать (ничего не устанавливает)
  /tui-build --install    не собирать, поставить уже собранный wheel
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from argent.ui import console

ROOT = Path(__file__).resolve().parent.parent
CRATE = ROOT / "crates" / "argent-tui"
WHEEL_GLOB = "argent_tui-*.whl"
INSTALL_DIR = ROOT / "scratch" / "wheels"
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"


def _run(cmd, cwd, timeout=1800):
    """Run one step and return (ok, tail of output). Never raises: a failed build is an
    answer, not an exception."""
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "VIRTUAL_ENV": str(ROOT / ".venv")},
        )
    except FileNotFoundError:
        return False, f"не найдено: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, f"не уложилось в {timeout} с"
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(line for line in out.strip().splitlines()[-6:])
    took = time.time() - started
    return proc.returncode == 0, f"{tail}\n({took:.1f} с)" if tail else f"({took:.1f} с)"


def _build():
    console.print("[cyan]1/4 сборка release…[/cyan]")
    ok, tail = _run(
        [sys.executable, "-m", "maturin", "build", "--release"],
        CRATE,
    )
    if not ok:
        console.print(f"[red]Сборка не удалась:[/red]\n{tail}")
        return None
    wheels = sorted((CRATE / "target" / "wheels").glob(WHEEL_GLOB),
                    key=lambda p: p.stat().st_mtime)
    if not wheels:
        console.print("[red]Сборка прошла, но wheel не найден.[/red]")
        return None
    wheel = wheels[-1]
    console.print(f"    [green]ok[/green] · {wheel.name}")
    return wheel


def _clean_leftovers():
    """Remove what an interrupted, or over-a-running-app, install leaves behind.

    pip renames the live package to `~rgent_tui` and the loaded module to
    `*.pyd.old-<rand>` because Windows will not delete a file mapped into a running
    process. Those leftovers are re-created on EVERY install, so the sweep has to run
    after the install as well as before it.

    Returns (removed, kept). `kept` is normal, not an error: the file that could not go
    is the one this very process is running, and it disappears on the next sweep after a
    restart.
    """
    removed = 0
    kept = []
    for p in list(SITE_PACKAGES.glob("*gent_tui*")):
        if p.name in ("argent_tui", "argent_tui-0.1.0.dist-info"):
            continue
        if not (p.name.startswith("~") or "old-" in p.name):
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        if p.exists():
            kept.append(p.name)
        else:
            removed += 1
    pkg = SITE_PACKAGES / "argent_tui"
    if pkg.is_dir():
        for f in pkg.glob("*.pyd.old-*"):
            f.unlink(missing_ok=True)
            if f.exists():
                kept.append(f.name)
            else:
                removed += 1
    return removed, kept


def _loaded_pyd() -> Path:
    """The compiled module actually on disk, not the package's `__init__.py`."""
    import argent_tui

    found = Path(argent_tui.__file__)
    if found.name == "__init__.py":
        pyds = sorted(found.parent.glob("*.pyd"), key=lambda f: f.stat().st_mtime)
        if pyds:
            return pyds[-1]
    return found


def _stage(wheel: Path) -> Path:
    console.print("[cyan]2/4 подготовка…[/cyan]")
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    staged = INSTALL_DIR / wheel.name
    shutil.copy2(wheel, staged)
    removed, kept = _clean_leftovers()
    console.print(f"    [green]ok[/green] · убрано остатков: {removed}")
    if kept:
        console.print(f"    [yellow]держит запущенный процесс (уйдёт после перезапуска): {', '.join(kept)}[/yellow]")
    return staged


def _install(wheel: Path) -> bool:
    console.print("[cyan]3/4 установка…[/cyan]")
    ok, tail = _run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
         str(wheel)],
        ROOT,
        timeout=600,
    )
    if not ok:
        console.print(f"[red]pip не смог поставить:[/red]\n{tail}")
        return False
    console.print("    [green]ok[/green]")
    return True


def _report(wheel: Path):
    """Which build is on disk — and when it was built, which the file date cannot say.

    pip writes the `.pyd` at install time, so its mtime is the install, not the build.
    The wheel file's own mtime is the only timestamp that means "built at".
    """
    try:
        # The timestamp inside the wheel is fixed by maturin for reproducible builds
        # (1980-01-01), and the installed .pyd's mtime is the install, not the build.
        # The wheel file's own mtime is the only honest answer to "when was this built".
        built = time.strftime("%Y-%m-%d %H:%M", time.localtime(wheel.stat().st_mtime))
        pyd = _loaded_pyd()
        console.print(f"\n[bold]На диске:[/bold] {pyd.name}, сборка {built}")
    except Exception as exc:                                  # pragma: no cover
        console.print(f"[yellow]Проверить установленное расширение не удалось: {exc}[/yellow]")


def command_tui_build(*args):
    """Собрать и установить расширение TUI. См. докстринг модуля."""
    flags = {str(a).lower() for a in args}
    only_build = "--build" in flags
    only_install = "--install" in flags

    if not CRATE.exists():
        console.print(f"[red]Нет каталога расширения: {CRATE}[/red]")
        return

    wheel = None
    if not only_install:
        wheel = _build()
        if wheel is None:
            return
    if only_build:
        console.print("\n[yellow]Только сборка — ничего не установлено.[/yellow]")
        return

    if wheel is None:
        found = sorted((CRATE / "target" / "wheels").glob(WHEEL_GLOB),
                       key=lambda p: p.stat().st_mtime)
        if not found:
            console.print("[red]Собранного wheel нет: сначала /tui-build --build.[/red]")
            return
        wheel = found[-1]
        console.print(f"[cyan]Ставлю готовый:[/cyan] {wheel.name}")

    staged = _stage(wheel)
    if not _install(staged):
        return

    left, still = _clean_leftovers()
    if left:
        console.print(f"    [dim]после установки убрано остатков: {left}[/dim]")
    if still:
        console.print(
            "    [yellow]остатки от этой установки останутся до перезапуска: "
            f"{', '.join(still)}[/yellow]"
        )

    console.print("[cyan]4/4 проверка…[/cyan]")
    _report(wheel)

    console.print(
        "\n[bold yellow]Готово, но окно всё ещё на старом коде.[/bold yellow] "
        "`.pyd` держит запущенный процесс, поэтому новый файл подхватится при следующем "
        "запуске. Закрой все окна Argent и открой заново — после этого проверяй."
    )
