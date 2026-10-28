"""Съёмка экрана: регион, честный отчёт и отсутствие догадок.

Раньше инструмент молча брал `GetWindowRect` (с невидимой рамкой DWM), возвращал
`title = "HWND 984082"` для числового HWND и не говорил ни слова о том, что часть
окна за экраном. По такому отчёту нельзя понять, что попало в кадр.
"""

import pytest

from src.automation.desktop import (
    DesktopAutomation,
    _frame_is_blank,
    _offscreen_pixels,
    _parse_region,
)
from argent.tools.desktop_tools import _capture_report


# ── region ──────────────────────────────────────────────────────────────────

def test_the_interesting_strip_of_a_window_is_a_single_argument():
    # Панель у поля ввода — это низ окна, а не его верх.
    assert _parse_region("bottom:200", 1000, 400) == (0, 200, 1000, 200)
    assert _parse_region("top:120", 1000, 400) == (0, 0, 1000, 120)
    assert _parse_region("left:300", 1000, 400) == (0, 0, 300, 400)
    assert _parse_region("right:400", 1000, 400) == (600, 0, 400, 400)


def test_explicit_coordinates_are_accepted():
    assert _parse_region("10,20,300,100", 1000, 400) == (10, 20, 300, 100)
    assert _parse_region("10, 20, 300, 100", 1000, 400) == (10, 20, 300, 100)


def test_a_region_larger_than_the_frame_is_clamped_not_refused():
    x, y, w, h = _parse_region("bottom:900", 1000, 400)
    assert (y + h) == 400, "не выходим за нижнюю границу"
    assert w <= 1000 and h <= 400


def test_a_region_that_makes_no_sense_is_ignored():
    # Лучше снять всё окно, чем отдать пустоту неизвестного размера.
    assert _parse_region("nonsense", 1000, 400) is None
    assert _parse_region("", 1000, 400) is None
    assert _parse_region(None, 1000, 400) is None
    assert _parse_region("1,2,3", 1000, 400) is None


# ── the frame itself ────────────────────────────────────────────────────────

def test_a_uniform_frame_is_recognised_as_blank():
    """DirectComposition windows (Windows Terminal) can answer PrintWindow with
    an empty frame; blurting it out as truth is worse than falling back."""
    assert _frame_is_blank(b"\x00" * 40000) is True
    assert _frame_is_blank(b"") is True
    # A small live element is enough: the frame is not blank.
    varied = bytearray(b"\x00" * 40000)
    varied[20000:24000] = b"\xff" * 4000
    assert _frame_is_blank(bytes(varied)) is False


def test_pixels_outside_the_desktop_are_counted(monkeypatch):
    monkeypatch.setattr(
        "src.automation.desktop._virtual_screen",
        lambda: {"x": 0, "y": 0, "width": 1920, "height": 1080},
    )
    assert _offscreen_pixels(100, 100, 800, 600) == 0
    # Окно уходит за нижний край: 20 px по высоте, ширина 800.
    assert _offscreen_pixels(100, 1000, 800, 100) == 20 * 800


# ── the report ──────────────────────────────────────────────────────────────

BASE = {
    "success": True,
    "path": ".argent/artifacts/desktop/x.png",
    "width": 1734,
    "height": 311,
    "frame": {"width": 1734, "height": 311},
    "region": None,
    "title": r"C:\Windows\system32\cmd.exe",
    "pid": 14712,
    "rect": {"x": 829, "y": 1294, "width": 1734, "height": 311},
    "screen": {"x": 0, "y": 0, "width": 2560, "height": 1600},
    "foreground": True,
    "offscreen_px": 0,
    "method": "printwindow",
}


def test_the_report_names_the_window_and_how_the_frame_was_taken():
    text = _capture_report(dict(BASE), "window")

    assert "cmd.exe" in text
    assert "pid 14712" in text, "иначе непонятно, чьё это окно"
    assert "1734x311" in text
    assert "829,1294" in text, "позиция на экране — половина ответа на вопрос «что я вижу»"
    assert "printwindow" in text
    assert "foreground" in text
    assert "x.png" in text


def test_the_report_says_which_strip_was_cut_out():
    res = dict(BASE, width=1734, height=260, region=[0, 51, 1734, 260])
    text = _capture_report(res, "window")
    assert "region 1734x260 at 0,51" in text
    assert "1734x311" in text, "виден и целый кадр — есть с чем сравнить"


def test_the_report_warns_about_a_window_that_leaves_the_screen():
    text = _capture_report(dict(BASE, offscreen_px=9588), "window")
    assert "outside the desktop" in text
    assert "9588" in text


def test_the_report_admits_when_the_picture_came_from_the_screen():
    text = _capture_report(dict(BASE, method="bitblt-fallback"), "window")
    assert "empty frame" in text and "taken from the screen" in text


def test_an_ambiguous_title_lists_the_candidates_instead_of_picking_one(monkeypatch):
    monkeypatch.setattr(
        DesktopAutomation,
        "find_windows",
        classmethod(lambda cls, needle: [
            {"hwnd": 101, "title": "cmd.exe", "pid": 1, "x": 0, "y": 0, "width": 100, "height": 50},
            {"hwnd": 202, "title": "cmd.exe (2)", "pid": 2, "x": 0, "y": 60, "width": 100, "height": 50},
        ]),
    )
    res = DesktopAutomation.capture_window("cmd.exe")

    assert res["success"] is False
    assert "matches 2 windows" in res["error"]
    assert "[HWND: 101]" in res["error"] and "[HWND: 202]" in res["error"]
    assert len(res["candidates"]) == 2
