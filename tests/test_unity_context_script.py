"""unity_context.py (bundled with the unity-dev skill): one-command project
orientation — version, pipeline, input mode, asmdefs, conventions."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "skills" / "unity-dev" / "scripts" / "unity_context.py"


@pytest.fixture(scope="module")
def uc():
    spec = importlib.util.spec_from_file_location("unity_context", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_project(tmp_path, *, urp=True, input_handler="1"):
    root = tmp_path / "Game"
    (root / "Assets" / "Scripts" / "Player").mkdir(parents=True)
    (root / "ProjectSettings").mkdir()
    (root / "Packages").mkdir()

    (root / "ProjectSettings" / "ProjectVersion.txt").write_text(
        "m_EditorVersion: 6000.0.32f1\n", encoding="utf-8")
    deps = {"com.unity.textmeshpro": "3.2.0"}
    if urp:
        deps["com.unity.render-pipelines.universal"] = "17.0.3"
    (root / "Packages" / "manifest.json").write_text(
        json.dumps({"dependencies": deps}), encoding="utf-8")
    (root / "ProjectSettings" / "GraphicsSettings.asset").write_text(
        "  m_CustomRenderPipeline: {fileID: 11400000, guid: abc}\n" if urp
        else "  m_CustomRenderPipeline: {fileID: 0}\n", encoding="utf-8")
    (root / "ProjectSettings" / "ProjectSettings.asset").write_text(
        f"  activeInputHandler: {input_handler}\n", encoding="utf-8")

    (root / "Assets" / "Scripts" / "Player" / "PlayerMove.cs").write_text(
        "namespace Game.Core {\npublic class PlayerMove {} }\n", encoding="utf-8")
    (root / "Assets" / "Scripts" / "Game.Core.asmdef").write_text(
        json.dumps({"name": "Game.Core"}), encoding="utf-8")
    (root / "Assets" / "Main.unity").write_text("scene", encoding="utf-8")
    return root


def test_detect_root_from_nested_dir(uc, tmp_path):
    root = _make_project(tmp_path)
    nested = root / "Assets" / "Scripts" / "Player"
    assert uc.detect_root(nested) == root.resolve()


def test_detect_root_none_outside(uc, tmp_path):
    assert uc.detect_root(tmp_path) is None


def test_report_core_facts(uc, tmp_path):
    report = uc.build_report(_make_project(tmp_path))
    assert "6000.0.32f1" in report
    assert "URP" in report
    assert "NEW Input System only" in report
    assert "TextMeshPro" in report
    assert "Game.Core" in report                  # asmdef and namespace
    assert "Main.unity" in report


def test_builtin_pipeline_and_legacy_input(uc, tmp_path):
    report = uc.build_report(_make_project(tmp_path, urp=False, input_handler="0"))
    assert "Built-in Render Pipeline" in report
    assert "LEGACY Input Manager" in report


def test_no_asmdef_reports_single_assembly(uc, tmp_path):
    root = _make_project(tmp_path)
    (root / "Assets" / "Scripts" / "Game.Core.asmdef").unlink()
    report = uc.build_report(root)
    assert "single Assembly-CSharp" in report
