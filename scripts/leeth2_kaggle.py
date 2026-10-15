"""Kaggle plumbing for Leeth 2 training (code dataset, kernels, polling, output).

    python scripts/leeth2_kaggle.py code                 stage + create/version  hunzlee/leeth2-code
    python scripts/leeth2_kaggle.py argent-data          stage + create/version  hunzlee/leeth2-argent-data
    python scripts/leeth2_kaggle.py targeted-data        build + create/version  hunzlee/leeth2-targeted-v2 (fine-tune data, ~80 MB)
    python scripts/leeth2_kaggle.py targeted-v3          build (--v3) + create/version  hunzlee/leeth2-targeted-v3 (production-state fine-tune data)
    python scripts/leeth2_kaggle.py exam-src             stage + create/version  hunzlee/leeth2-exam-src (the slice of the project the exam scripts import)
    python scripts/leeth2_kaggle.py exam <name>=<kernel-slug>...   write + push the exam kernel for these models (kernel-slug = a finished kernel whose run/final is the model)
    python scripts/leeth2_kaggle.py arena <name>=<kernel-slug>...  write + push the arena kernel (CPU): these models against Laya on the same public benchmarks
    python scripts/leeth2_kaggle.py push <kernel_dir>    kaggle kernels push
    python scripts/leeth2_kaggle.py wait <owner/slug>    poll until the latest run ends, then print the log tail
    python scripts/leeth2_kaggle.py output <owner/slug> <dir> [--only glob]   download kernel output
    python scripts/leeth2_kaggle.py ls <owner/slug>                  list the files of the latest run (with sizes)
    python scripts/leeth2_kaggle.py fetch <owner/slug> <dir> <glob>  download ONLY the matching output files (e.g. 'run/final/*')

Everything runs on the user's Kaggle account via the installed ``kaggle`` CLI; nothing trains locally.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".argent" / "kaggle" / "leeth2"
OWNER = "hunzlee"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def kaggle(*args: str, cwd: Path | None = None, timeout: int = 3600) -> subprocess.CompletedProcess:
    r = subprocess.run(["kaggle", *args], cwd=cwd, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    out = "\n".join(l for l in (r.stdout + "\n" + r.stderr).splitlines() if "RequestsDependencyWarning" not in l and "warnings.warn" not in l)
    r.stdout = out
    return r


def stage_code() -> Path:
    d = WORK / "ds_code"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(ROOT / "src" / "leeth2", d / "leeth2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (d / "dataset-metadata.json").write_text('{"title": "leeth2-code", "id": "%s/leeth2-code", "licenses": [{"name": "CC0-1.0"}]}' % OWNER, encoding="utf-8")
    return d


def stage_argent() -> Path:
    d = WORK / "ds_argent"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    names = ["leeth_400m_train.jsonl", "leeth_400m_train_extra.jsonl", "leeth_400m_train_safety.jsonl", "leeth_400m_val.jsonl", "multidomain_train.jsonl", "multidomain_val.jsonl",
             "docker_train.jsonl", "docker_val.jsonl", "solver_counterexamples.jsonl", "shadow_mined.jsonl", "choice_classes.json"]
    shutil.copy2(ROOT / "scripts" / "prod_battery.json", d / "prod_battery.json")
    for sub in ("src/leeth/data_400m", "src/leeth/data"):
        for n in names:
            p = ROOT / sub / n
            if p.exists() and not (d / n).exists():
                shutil.copy2(p, d / n)
    (d / "dataset-metadata.json").write_text('{"title": "leeth2-argent-data", "id": "%s/leeth2-argent-data", "licenses": [{"name": "CC0-1.0"}]}' % OWNER, encoding="utf-8")
    return d


def stage_targeted() -> Path:
    """The v2 fine-tune data (data/finetune.py build) as a dataset folder."""
    sys.path.insert(0, str(ROOT))
    from src.leeth2.data import finetune

    d = WORK / "ds_targeted_v2"
    if d.exists():
        shutil.rmtree(d)
    finetune.build(d)
    (d / "dataset-metadata.json").write_text('{"title": "leeth2-targeted-v2", "id": "%s/leeth2-targeted-v2", "licenses": [{"name": "CC0-1.0"}]}' % OWNER, encoding="utf-8")
    return d


def stage_targeted_v3(tag: str = "v3") -> Path:
    """The v3 fine-tune data: v2 grammar share + session-context / 16-option / real-style generators.

    Kaggle auto-extracts an uploaded .zip, so the dataset is ONE zip of the jsonl files (183 MB of text -> ~15 MB to upload).
    ``tag`` names the dataset (leeth2-targeted-<tag>) so a later top-up does not overwrite the data an earlier run used."""
    sys.path.insert(0, str(ROOT))
    from src.leeth2.data import finetune

    raw, d = WORK / f"raw_targeted_{tag}", WORK / f"ds_targeted_{tag}"
    for x in (raw, d):
        if x.exists():
            shutil.rmtree(x)
    finetune.build(raw, v3=True, v4=tag.startswith("v4"))
    d.mkdir(parents=True)
    import zipfile

    with zipfile.ZipFile(d / f"targeted_{tag}.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in ("train.jsonl", "dev.jsonl", "calib.jsonl", "test.jsonl", "manifest.json"):
            z.write(raw / name, name)
    shutil.rmtree(raw)
    (d / "dataset-metadata.json").write_text('{"title": "leeth2-targeted-%s", "id": "%s/leeth2-targeted-%s", "licenses": [{"name": "CC0-1.0"}]}' % (tag, OWNER, tag), encoding="utf-8")
    return d


def stage_exam_src() -> Path:
    """The slice of the project the promotion exams import: logger, the dispatcher modules, all of leeth2, the exam scripts and their batteries."""
    d = WORK / "ds_exam"
    if d.exists():
        shutil.rmtree(d)
    (d / "proj" / "src" / "leeth").mkdir(parents=True)
    (d / "proj" / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "logger.py", d / "proj" / "logger.py")
    for name in ("__init__", "dispatcher", "model", "primitives", "runtime", "tokenizer"):
        shutil.copy2(ROOT / "src" / "leeth" / f"{name}.py", d / "proj" / "src" / "leeth" / f"{name}.py")
    shutil.copytree(ROOT / "src" / "leeth2", d / "proj" / "src" / "leeth2", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("leeth2_exam.py", "leeth2_real_exam.py", "leeth2_stress.py", "leeth21_exam.py", "leeth21_calibrate.py", "ood_exam_v3.json", "prod_battery.json", "prod_battery_v2.json", "safety_exam_v2.json"):
        shutil.copy2(ROOT / "scripts" / name, d / "proj" / "scripts" / name)
    (d / "dataset-metadata.json").write_text('{"title": "leeth2-exam-src", "id": "%s/leeth2-exam-src", "licenses": [{"name": "CC0-1.0"}]}' % OWNER, encoding="utf-8")
    return d


def stage_exam_kernel(candidates: dict) -> Path:
    """The exam kernel (WORK/kernel_exam/exam_merge.py is the template) for ``{name: kernel-slug}``: synthetic exams + the real-request exam per model."""
    import json
    import re

    src = (WORK / "kernel_exam" / "exam_merge.py").read_text(encoding="utf-8")
    cands = {n: slug.split("/")[-1] for n, slug in candidates.items()}
    src = re.sub(r"^CANDIDATES = .*$", "CANDIDATES = " + json.dumps(cands), src, count=1, flags=re.M)
    d = WORK / "kernel_exam_x"
    d.mkdir(exist_ok=True)
    (d / "exam_x.py").write_text(src, encoding="utf-8")
    meta = {"id": f"{OWNER}/leeth2-exam-x", "title": "leeth2-exam-x", "code_file": "exam_x.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{OWNER}/leeth2-exam-src", f"{OWNER}/leeth2-real-queries"], "competition_sources": [],
            "kernel_sources": [f"{OWNER}/{k}" for k in cands.values()], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def stage_arena_kernel(candidates: dict, slug: str = "leeth2-arena-final", reuse_laya: str = "", only: tuple = (), samples: int = 0) -> Path:
    """The arena kernel (WORK/kernel_arena/arena.py is the template: our models against Laya on the same public benchmarks) for ``{name: kernel-slug}``.
    CPU only -- it costs no GPU quota -- and written under its own slug, so the finished arena run stays readable."""
    import json
    import re

    src = (WORK / "kernel_arena" / "arena.py").read_text(encoding="utf-8")
    cands = {n: (k.split("/")[-1].split(":")[0], k.split(":")[1] if ":" in k else "final") for n, k in candidates.items()}   # "slug" or "slug:subfolder" (e.g. a soup that sits in soup0.5/)
    src, n = re.subn(r"^CANDIDATES = .*$", "CANDIDATES = " + json.dumps(cands), src, count=1, flags=re.M)
    assert n == 1
    if only:                                                 # a precision run on a few tasks: more samples each, everything else dropped
        src, k = re.subn(r'^ONLY_TASKS = \(\)', f'ONLY_TASKS = {tuple(only)!r}', src, count=1, flags=re.M)
        assert k == 1
        if samples:
            src, k = re.subn(r'^N_COMMON = \d+', f'N_COMMON = {samples}', src, count=1, flags=re.M)
            src, k2 = re.subn(r'^N_OURS = \d+', f'N_OURS = {samples}', src, count=1, flags=re.M)
            assert k == 1 and k2 == 1
    if reuse_laya:                                           # Laya's answers on the same samples come from that finished arena kernel
        src, n = re.subn(r'^LAYA_REUSE = ""', f'LAYA_REUSE = "{reuse_laya}"', src, count=1, flags=re.M)
        assert n == 1
    d = WORK / "kernel_arena_x"
    d.mkdir(exist_ok=True)
    (d / "arena_x.py").write_text(src, encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": "arena_x.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_internet": True, "dataset_sources": [f"{OWNER}/leeth2-code"], "competition_sources": [],
            "kernel_sources": [f"{OWNER}/{k[0]}" for k in cands.values()] + ([f"{OWNER}/{reuse_laya}"] if reuse_laya else []), "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def push_dataset(d: Path, msg: str) -> None:
    slug = f"{OWNER}/{d.name.replace('ds_', 'leeth2-').replace('leeth2-code', 'leeth2-code')}"
    r = kaggle("datasets", "create", "-p", str(d), "-r", "zip", cwd=d)
    print(r.stdout.strip())
    if "already" in r.stdout.lower() or r.returncode != 0:
        r = kaggle("datasets", "version", "-p", str(d), "-m", msg, "--dir-mode", "zip", cwd=d)
        print(r.stdout.strip())
        if r.returncode != 0:
            sys.exit(f"dataset upload failed:\n{r.stdout}")


def wait(slug: str, poll: int = 45, max_s: int = 14 * 3600) -> str:
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_s:
        r = kaggle("kernels", "status", slug)
        status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if status != last:
            print(f"[{time.strftime('%H:%M:%S')}] {status}", flush=True)
            last = status
        if any(k in status for k in ("COMPLETE", "ERROR", "CANCEL")):
            return status
        time.sleep(poll)
    return "TIMEOUT"


def _list_output(slug: str):
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = KaggleApi()
    api.authenticate()
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = slug.split("/", 1)
    with api.build_kaggle_client() as k:
        return k.kernels.kernels_api_client.list_kernel_session_output(req).files or []


def fetch(slug: str, dest: str, pattern: str) -> None:
    """Stream matching files of a kernel's output to ``dest`` without pulling the rest (state.pt alone is ~2 GB)."""
    import fnmatch

    import requests

    out = Path(dest)
    for item in _list_output(slug):
        if not fnmatch.fnmatch(item.file_name, pattern):
            continue
        target = out / item.file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with requests.get(item.url, stream=True, timeout=1800) as r:
            r.raise_for_status()
            n = 0
            with open(target, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    n += len(chunk)
        print(f"{item.file_name}: {n / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("code")
    sub.add_parser("argent-data")
    sub.add_parser("targeted-data")
    p = sub.add_parser("targeted-v3")
    p.add_argument("--tag", default="v3")
    sub.add_parser("exam-src")
    p = sub.add_parser("exam")
    p.add_argument("pairs", nargs="+", help="name=kernel-slug")
    p = sub.add_parser("arena")
    p.add_argument("pairs", nargs="+", help="name=kernel-slug")
    p.add_argument("--slug", default="leeth2-arena-final")
    p.add_argument("--only", nargs="*", default=[], help="precision run: only these tasks")
    p.add_argument("--n", type=int, default=0, help="samples per task for the precision run")
    p.add_argument("--reuse-laya", default="", help="slug of a finished arena kernel whose Laya answers are reused (same samples)")
    p = sub.add_parser("push")
    p.add_argument("kernel_dir")
    p = sub.add_parser("wait")
    p.add_argument("slug")
    p = sub.add_parser("output")
    p.add_argument("slug")
    p.add_argument("dest")
    p.add_argument("--only", default="")
    p = sub.add_parser("ls")
    p.add_argument("slug")
    p = sub.add_parser("fetch")
    p.add_argument("slug")
    p.add_argument("dest")
    p.add_argument("pattern")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if a.cmd == "code":
        push_dataset(stage_code(), "leeth2 code")
    elif a.cmd == "targeted-data":
        push_dataset(stage_targeted(), "v2 fine-tune data")
    elif a.cmd == "targeted-v3":
        push_dataset(stage_targeted_v3(a.tag), f"{a.tag} fine-tune data")
    elif a.cmd == "exam-src":
        push_dataset(stage_exam_src(), "exam sources")
    elif a.cmd == "exam":
        d = stage_exam_kernel(dict(x.split("=", 1) for x in a.pairs))
        r = kaggle("kernels", "push", "-p", str(d))
        print(r.stdout.strip())
    elif a.cmd == "arena":
        d = stage_arena_kernel(dict(x.split("=", 1) for x in a.pairs), a.slug, a.reuse_laya, tuple(a.only), a.n)
        r = kaggle("kernels", "push", "-p", str(d))
        print(r.stdout.strip())
    elif a.cmd == "argent-data":
        push_dataset(stage_argent(), "argent data")
    elif a.cmd == "push":
        r = kaggle("kernels", "push", "-p", a.kernel_dir)
        print(r.stdout.strip())
        sys.exit(r.returncode)
    elif a.cmd == "wait":
        print(wait(a.slug))
    elif a.cmd == "ls":
        for f in _list_output(a.slug):
            print(f"{getattr(f, 'size', 0) or 0:>14}  {f.file_name}")
    elif a.cmd == "fetch":
        fetch(a.slug, a.dest, a.pattern)
    elif a.cmd == "output":
        Path(a.dest).mkdir(parents=True, exist_ok=True)
        args = ["kernels", "output", a.slug, "-p", a.dest]
        r = kaggle(*args)
        print(r.stdout.strip())


if __name__ == "__main__":
    main()
