"""One-command Unity project orientation for an AI coding agent.

Usage:  python unity_context.py [project_root]

Prints the facts that CHANGE what correct code looks like: Unity version
(API set), render pipeline (shader/material property names), input handling
mode (Input vs InputSystem code), notable packages, assembly definitions,
scenes, and script conventions. Stdlib only; read-only.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Packages that change how code must be written, worth surfacing by name.
NOTABLE_PACKAGES = {
    "com.unity.render-pipelines.universal": "URP",
    "com.unity.render-pipelines.high-definition": "HDRP",
    "com.unity.inputsystem": "Input System (new)",
    "com.unity.textmeshpro": "TextMeshPro",
    "com.unity.ugui": "uGUI",
    "com.unity.ui": "UI Toolkit (runtime)",
    "com.unity.cinemachine": "Cinemachine",
    "com.unity.addressables": "Addressables",
    "com.unity.test-framework": "Unity Test Framework",
    "com.unity.netcode.gameobjects": "Netcode for GameObjects",
    "com.unity.animation.rigging": "Animation Rigging",
    "com.unity.ai.navigation": "AI Navigation (NavMesh)",
}


def detect_root(start: Path) -> Path | None:
    """Walk up from `start` to the nearest dir holding Assets/ + ProjectSettings/."""
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "Assets").is_dir() and (candidate / "ProjectSettings").is_dir():
            return candidate
    return None


def unity_version(root: Path) -> str:
    try:
        text = (root / "ProjectSettings" / "ProjectVersion.txt").read_text(encoding="utf-8")
        m = re.search(r"m_EditorVersion:\s*(\S+)", text)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def read_manifest(root: Path) -> dict:
    try:
        data = json.loads((root / "Packages" / "manifest.json").read_text(encoding="utf-8"))
        return data.get("dependencies", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def render_pipeline(root: Path, deps: dict) -> str:
    """Active pipeline: the asset assigned in GraphicsSettings decides, not the
    package list (URP can be installed yet not active)."""
    custom_assigned = None
    try:
        gfx = (root / "ProjectSettings" / "GraphicsSettings.asset").read_text(
            encoding="utf-8", errors="ignore")
        m = re.search(r"m_CustomRenderPipeline:\s*\{fileID:\s*(-?\d+)", gfx)
        if m:
            custom_assigned = m.group(1) != "0"
    except OSError:
        pass

    if custom_assigned is False:
        return "Built-in Render Pipeline"
    if custom_assigned is True:
        if "com.unity.render-pipelines.high-definition" in deps:
            return "HDRP (custom pipeline assigned)"
        if "com.unity.render-pipelines.universal" in deps:
            return "URP (custom pipeline assigned)"
        return "custom SRP assigned (package unclear)"
    # GraphicsSettings unreadable — fall back to packages as a hint.
    if "com.unity.render-pipelines.high-definition" in deps:
        return "HDRP package installed (GraphicsSettings unreadable)"
    if "com.unity.render-pipelines.universal" in deps:
        return "URP package installed (GraphicsSettings unreadable)"
    return "Built-in Render Pipeline (no SRP package)"


def input_mode(root: Path) -> str:
    """activeInputHandler: 0 = legacy Input Manager, 1 = Input System package, 2 = both."""
    try:
        settings = (root / "ProjectSettings" / "ProjectSettings.asset").read_text(
            encoding="utf-8", errors="ignore")
        m = re.search(r"activeInputHandler:\s*(\d)", settings)
        if m:
            return {
                "0": "LEGACY Input Manager only (use UnityEngine.Input)",
                "1": "NEW Input System only (use UnityEngine.InputSystem; UnityEngine.Input calls THROW)",
                "2": "BOTH enabled (either API works; match existing code)",
            }.get(m.group(1), "unknown")
    except OSError:
        pass
    return "unknown (ProjectSettings.asset unreadable — check manually)"


def list_asmdefs(root: Path, cap: int = 15) -> list:
    out = []
    for p in sorted((root / "Assets").rglob("*.asmdef")):
        try:
            name = json.loads(p.read_text(encoding="utf-8")).get("name", p.stem)
        except (OSError, json.JSONDecodeError):
            name = p.stem
        out.append(f"{name}  ({p.relative_to(root)})")
        if len(out) >= cap:
            out.append("… (more exist)")
            break
    return out


def list_scenes(root: Path, cap: int = 10) -> list:
    scenes = sorted(str(p.relative_to(root)) for p in (root / "Assets").rglob("*.unity"))
    return scenes[:cap] + (["… (more exist)"] if len(scenes) > cap else [])


def script_conventions(root: Path, sample_cap: int = 200):
    """Count scripts, find where they live, and sample namespace usage so new
    code can match the project's own conventions."""
    scripts = list((root / "Assets").rglob("*.cs"))
    by_dir = Counter(str(p.parent.relative_to(root)) for p in scripts)
    namespaces = Counter()
    for p in scripts[:sample_cap]:
        try:
            for m in re.finditer(r"^\s*namespace\s+([\w.]+)", p.read_text(encoding="utf-8", errors="ignore"), re.M):
                namespaces[m.group(1)] += 1
        except OSError:
            continue
    return len(scripts), by_dir.most_common(3), namespaces.most_common(5)


def build_report(root: Path) -> str:
    deps = read_manifest(root)
    notable = [label for pkg, label in NOTABLE_PACKAGES.items() if pkg in deps]
    n_scripts, top_dirs, top_ns = script_conventions(root)

    lines = [
        f"UNITY PROJECT: {root}",
        f"Unity version: {unity_version(root)}",
        f"Render pipeline: {render_pipeline(root, deps)}",
        f"Input handling: {input_mode(root)}",
        f"Notable packages: {', '.join(notable) if notable else '(none of the usual suspects)'}",
        "",
        f"Scripts: {n_scripts} .cs under Assets/",
    ]
    if top_dirs:
        lines.append("Most scripts in: " + "; ".join(f"{d} ({n})" for d, n in top_dirs))
    if top_ns:
        lines.append("Namespaces in use: " + ", ".join(f"{ns} ({n})" for ns, n in top_ns))
    else:
        lines.append("Namespaces in use: none found — project code is global-namespace; follow that.")

    asmdefs = list_asmdefs(root)
    lines.append("")
    lines.append("Assembly definitions (asmdef): " + (str(len(asmdefs)) if asmdefs else "none (single Assembly-CSharp)"))
    lines.extend(f"  - {a}" for a in asmdefs)

    scenes = list_scenes(root)
    lines.append("")
    lines.append(f"Scenes: {len(scenes)}")
    lines.extend(f"  - {s}" for s in scenes)
    return "\n".join(lines)


def main() -> int:
    start = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    root = detect_root(start)
    if root is None:
        print(f"Not a Unity project (no Assets/ + ProjectSettings/ found from {start.resolve()} upward).")
        return 1
    print(build_report(root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
