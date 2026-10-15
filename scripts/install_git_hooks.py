"""Install the repository's git hooks (hooks live outside version control): python scripts/install_git_hooks.py"""
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    git_dir = Path(subprocess.run(["git", "rev-parse", "--git-dir"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    hooks = (ROOT / git_dir if not git_dir.is_absolute() else git_dir) / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / "scripts" / "git_hooks").iterdir():
        target = hooks / src.name
        shutil.copyfile(src, target)
        target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))      # sh does not read CRLF
        target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
