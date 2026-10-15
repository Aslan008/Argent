"""Install a finished Leeth 2 run into Argent -- safely: the model it replaces is kept, and a candidate that does not load is never switched in.

    python scripts/leeth2_install.py <downloaded_run_dir> [--backup-as leeth2_v31]

``<downloaded_run_dir>`` is the folder that holds final/ (and, next to it, report.json / final_test.json / final_dev.json).

What it does:
  * copies the candidate to a staging folder beside the install location and loads it through the runtime (a smoke check on
    typed questions: finite probabilities, an answer that is one of the options); nothing is touched if that fails
  * moves the current install aside to .argent/models/<backup-as> (default leeth2_prev_<time>); a rename, so it costs no disk
  * moves the staged candidate into place and checks it once more from there; if that check fails the old install is put back
  * optionally exports the ONNX graph next to it and prints the torch-vs-ONNX parity (--onnx)

The legacy stack stays the default; enable with  /fast model leeth2  (or LEETH_BACKEND=leeth2).  To go back, rename the backup folder
back to leeth2 (the command is printed at the end).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEST = ROOT / ".argent" / "models" / "leeth2"
REQUIRED = ("model.safetensors", "leeth2_config.json", "tokenizer.json")
EXTRAS = ("report.json", "final_test.json", "final_dev.json")
TOOLS = ["read_file", "run_command", "browser_open", "NO_TOOL", "UNFAMILIAR"]


def smoke(final: Path) -> Tuple[bool, str]:
    """Load ``final`` through the runtime and ask four typed questions; (ok, what was seen)."""
    from src.leeth.primitives import ChoiceQuestion, ExtractQuestion, NoulQuestion, ScoreQuestion
    from src.leeth2.runtime import Leeth2Runtime

    try:
        rt = Leeth2Runtime.load(final)
        state = 'Incoming user request in Argent session:\n"удали папку build"\nWorking directory: C:/proj'
        questions = [
            ChoiceQuestion("tool", "Какой инструмент Argent должен быть вызван в первую очередь?", TOOLS),
            NoulQuestion("safe", "Можно ли выполнить это без подтверждения пользователя?"),
            ScoreQuestion("risk", "Оцените риск действия", levels=5),
            ExtractQuestion("what", "Что нужно удалить?"),
        ]
        rt.decide(state, questions)                               # warm-up
        b = rt.decide(state, questions)
        lines = [f"device={rt.device}  {b.elapsed_ms:.0f} ms"]
        for k, v in b.results.items():
            d = v.to_dict()
            json.dumps(d, allow_nan=False)                        # a NaN anywhere is a broken model
            lines.append(f"  {k}: {d}")
        tool = b.results["tool"].to_dict()
        if tool.get("choice") not in TOOLS:
            return False, f"the tool answer {tool.get('choice')!r} is not one of the options"
        probs = [p for p in (tool.get("probabilities") or {}).values() if isinstance(p, (int, float))]
        if probs and not math.isclose(sum(probs), 1.0, abs_tol=0.02):
            return False, f"the tool probabilities sum to {sum(probs):.3f}"
        n = sum(p.numel() for p in rt.model.parameters()) / 1e6
        lines.insert(0, f"{n:.0f}M parameters")
        return True, "\n".join(lines)
    except Exception as e:                                        # any failure to load or answer means: do not install
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--onnx", action="store_true", help="also export model.onnx and check parity with PyTorch")
    ap.add_argument("--dest", default=str(DEST), help="install location (default: .argent/models/leeth2)")
    ap.add_argument("--backup-as", default="", help="folder name (beside the install location) for the model being replaced; default leeth2_prev_<time>")
    a = ap.parse_args()

    src = Path(a.run_dir)
    final = src / "final" if (src / "final").exists() else src
    for need in REQUIRED:
        if not (final / need).exists():
            sys.exit(f"missing {need} in {final}")
    dest = Path(a.dest)
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{dest.name}_staging"
    shutil.rmtree(staging, ignore_errors=True)
    (staging).mkdir(parents=True)
    shutil.copytree(final, staging / "final")
    for extra in EXTRAS:
        if (src / extra).exists():
            shutil.copy2(src / extra, staging / extra)

    ok, seen = smoke(staging / "final")
    print(seen)
    if not ok:
        shutil.rmtree(staging, ignore_errors=True)
        sys.exit(f"the candidate does not pass the smoke check; nothing was installed ({seen})")

    backup = None
    if dest.exists():
        backup = parent / (a.backup_as or f"{dest.name}_prev_{time.strftime('%Y%m%d_%H%M%S')}")
        if backup.exists():
            shutil.rmtree(staging, ignore_errors=True)
            sys.exit(f"{backup} already exists: pick another --backup-as (the current install was left alone)")
        try:
            dest.rename(backup)
        except OSError as e:
            shutil.rmtree(staging, ignore_errors=True)
            sys.exit(f"cannot move the current install aside ({e}); close Argent (it holds the model files) and run again")
    try:
        staging.rename(dest)
    except OSError as e:
        if backup is not None:
            backup.rename(dest)
        sys.exit(f"cannot put the candidate in place ({e}); the previous install was restored")

    ok, seen = smoke(dest / "final")
    if not ok:
        shutil.rmtree(dest, ignore_errors=True)
        if backup is not None:
            backup.rename(dest)
        sys.exit(f"the installed copy failed the check ({seen}); the previous install was restored")
    print(f"installed -> {dest / 'final'}")
    if backup is not None:
        print(f"previous install kept at {backup}\nto go back:  rename {backup} to {dest.name}  (close Argent first)")

    if a.onnx:
        from src.leeth2 import export as X
        from src.leeth2.runtime import Leeth2Runtime

        rt = Leeth2Runtime.load(dest / "final")
        p = X.export(rt.model, dest / "model.onnx")
        print(f"onnx -> {p} ({p.stat().st_size / 1e6:.0f} MB); parity:", X.parity(rt.model, p))
    print("\nenable with:  /fast model leeth2   (or set LEETH_BACKEND=leeth2)")


if __name__ == "__main__":
    main()
