"""Win32 handles are pointers, not ints.

`ctypes` without `argtypes` assumes a 32-bit int for every argument and for the return
value. On a 64-bit desktop that is two bugs in one:

* passing a handle larger than 2^31 raises `OverflowError: int too long to convert`;
* receiving one truncates it silently — the worst kind, because the bogus handle fails
  later, somewhere else.

Both capture paths died on the first one: `GetWindowDC(hwnd)` for a window, and the
screen path behind it. The fix is to declare the signatures once, when the module is
imported, so no call site can forget them.
"""

import ctypes
import os
from ctypes import wintypes

import pytest

# Importing the module is what declares the signatures (side effect on purpose: every
# call site then inherits them). The tests below check the result of that import.
import src.automation.desktop  # noqa: F401

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Win32 only")


def _api():
    return ctypes.windll.user32, ctypes.windll.gdi32


class TestSignaturesAreDeclared:
    """If these declarations are lost, capture breaks only on some machines — the ones
    whose handles happen to be large. That is why they are pinned."""

    def test_a_window_dc_is_taken_and_returned_as_a_handle(self):
        user32, _ = _api()
        assert user32.GetWindowDC.argtypes == [wintypes.HWND]
        assert user32.GetWindowDC.restype is wintypes.HDC
        assert user32.GetDC.restype is wintypes.HDC

    def test_a_blit_needs_device_contexts_on_both_sides(self):
        _, gdi32 = _api()
        argtypes = gdi32.BitBlt.argtypes
        assert argtypes[0] is wintypes.HDC, "первый аргумент — целевой DC"
        assert argtypes[5] is wintypes.HDC, "шестой — исходный DC"

    def test_the_pixel_reader_declares_its_buffer(self):
        _, gdi32 = _api()
        argtypes = gdi32.GetDIBits.argtypes
        assert argtypes[0] is wintypes.HDC
        assert argtypes[1] is wintypes.HBITMAP
        assert argtypes[4] is ctypes.c_void_p

    def test_a_handle_above_two_to_the_thirty_first_fits(self):
        """The value class that used to raise 'int too long to convert'.

        A pointer keeps all 64 bits; an untyped ctypes call would reinterpret it as a
        signed 32-bit int and refuse it.
        """
        handle = wintypes.HWND(0x8000_0001)
        assert handle.value == 0x8000_0001


def _win(hwnd, title, process, pid=1000, width=800, height=600):
    return {"hwnd": hwnd, "title": title, "process": process, "pid": pid,
            "x": 0, "y": 0, "width": width, "height": height}


#: A real desktop, reduced: a browser calls its windows after the pages it shows.
DESKTOP = [
    _win(1, "Gemini — Яндекс Браузер", "browser.exe", pid=1001),
    _win(2, "Землерой — Яндекс Браузер", "browser.exe", pid=1001),
    _win(3, "Untitled - Notepad", "notepad.exe", pid=1002),
]


class TestWindowIdentity:
    """A title is not the only name a window has.

    The browser case: a person says "Яндекс Браузер", the window says the page it is
    showing, and a title-only search finds nothing — while the browser is plainly open.
    The executable name is the identity that does not change.
    """

    def setup_method(self):
        from src.automation.desktop import match_windows
        self.match = match_windows

    def test_the_executable_name_finds_a_browser(self):
        found = self.match(DESKTOP, "browser")
        assert [w["hwnd"] for w, _ in found] == [1, 2]
        assert all(how == "process" for _, how in found)

    def test_the_executable_name_with_its_extension_works_too(self):
        assert len(self.match(DESKTOP, "browser.exe")) == 2

    def test_a_single_application_is_found_by_its_name(self):
        # The title says nothing about the program — exactly the browser case.
        windows = [_win(3, "Untitled", "notepad.exe")]
        found = self.match(windows, "notepad")
        assert [w["hwnd"] for w, _ in found] == [3]
        assert found[0][1] == "process"

    def test_a_title_win_comes_before_a_process_hit(self):
        found = self.match(DESKTOP, "notepad")
        assert [w["hwnd"] for w, _ in found] == [3]
        assert found[0][1] == "title", "заголовок — более точное попадание, чем имя файла"

    def test_a_title_still_works_and_comes_first(self):
        found = self.match(DESKTOP, "Землерой")
        assert found[0][0]["hwnd"] == 2
        assert found[0][1] == "title"

    def test_a_word_from_the_page_title_still_matches(self):
        assert [w["hwnd"] for w, _ in self.match(DESKTOP, "браузер")] == [1, 2]

    def test_an_unknown_name_matches_nothing(self):
        assert self.match(DESKTOP, "совсем другое приложение") == []

    def test_loose_words_are_the_last_resort(self):
        windows = [_win(9, "Отчёт по продажам за квартал", "excel.exe")]
        found = self.match(windows, "продажам квартал")
        assert found and found[0][1] == "words"

    def test_an_exact_title_beats_a_substring(self):
        windows = [_win(4, "Notepad", "notepad.exe"), _win(5, "Untitled - Notepad", "notepad.exe")]
        assert self.match(windows, "notepad")[0][0]["hwnd"] == 4

    def test_the_suggestion_lists_executables_not_only_titles(self):
        from src.automation.desktop import _describe_windows

        text = _describe_windows(DESKTOP)
        assert "browser.exe" in text and "notepad.exe" in text
        assert "Землерой" in text


class TestTheListingCarriesTheProcess:
    """The name has to be visible, or nobody can use it."""

    @pytest.mark.skipif(os.name != "nt", reason="Win32 only")
    def test_every_listed_window_names_its_process(self):
        from src.automation.desktop import DesktopAutomation

        windows = DesktopAutomation.list_windows()
        assert windows, "на рабочем столе должно быть хотя бы одно окно"
        assert all("process" in w for w in windows)
        assert any(w["process"] for w in windows), "хотя бы у одного окна имя процесса известно"


class TestChoosingAmongMatches:
    """Several windows of one program are not an ambiguity — they are one answer."""

    @staticmethod
    def _w(hwnd, title, process, area=100):
        return {"hwnd": hwnd, "title": title, "process": process, "pid": 1,
                "x": 0, "y": 0, "width": area, "height": 1}

    def test_the_window_in_front_wins(self):
        from src.automation.desktop import _pick_among

        matches = [(self._w(1, "стр. A", "browser.exe", 10), "process"),
                   (self._w(2, "стр. B", "browser.exe", 99), "process")]
        (win, _), reason = _pick_among(matches, foreground=1)
        assert win["hwnd"] == 1 and reason == "foreground"

    def test_without_a_foreground_the_largest_window_is_taken(self):
        from src.automation.desktop import _pick_among

        matches = [(self._w(1, "малое", "browser.exe", 10), "process"),
                   (self._w(2, "большое", "browser.exe", 99), "process")]
        (win, _), reason = _pick_among(matches, foreground=0)
        assert win["hwnd"] == 2 and reason == "largest"

    def test_different_programs_still_refuse_to_guess(self):
        from src.automation.desktop import _pick_among

        matches = [(self._w(1, "a", "browser.exe"), "process"),
                   (self._w(2, "b", "notepad.exe"), "words")]
        assert _pick_among(matches, foreground=1) == (None, "")

    def test_the_report_admits_which_one_it_took(self):
        from argent.tools.desktop_tools import _capture_report

        report = _capture_report({
            "path": "x.png", "title": "страница", "width": 100, "height": 50,
            "method": "printwindow",
            "match": {"by": "process", "reason": "foreground", "others": ["другая страница"]},
        }, "window")
        assert "executable name" in report
        assert "the one in front" in report and "другая страница" in report


class TestCaptureActuallyWorks:
    """These run the real pipeline. Before the fix they ended in OverflowError, not in
    a picture — which is why they are the regression test."""

    @pytest.fixture(autouse=True)
    def on_a_desktop(self):
        if not os.environ.get("SESSIONNAME"):
            pytest.skip("no interactive desktop session")

    def test_the_screen_is_captured(self, tmp_path):
        from src.automation.desktop import DesktopAutomation

        target = tmp_path / "screen.png"
        res = DesktopAutomation.capture_screen(output_path=str(target))

        assert res.get("success"), res.get("error")
        assert target.exists() and target.stat().st_size > 1000
        assert res["width"] > 0 and res["height"] > 0

    def test_a_window_is_captured_by_handle_and_by_title(self, tmp_path):
        from src.automation.desktop import DesktopAutomation

        windows = [w for w in DesktopAutomation.list_windows() if w.get("title")]
        if not windows:
            pytest.skip("no visible window to photograph")
        w = windows[0]

        by_handle = DesktopAutomation.capture_window(str(w["hwnd"]), str(tmp_path / "a.png"))
        assert by_handle.get("success"), by_handle.get("error")

        by_title = DesktopAutomation.capture_window(w["title"][:10], str(tmp_path / "b.png"))
        assert by_title.get("success"), by_title.get("error")

    def test_the_whole_multi_monitor_desktop_is_captured(self, tmp_path):
        from src.automation.desktop import DesktopAutomation

        target = tmp_path / "all.png"
        res = DesktopAutomation.capture_screen(output_path=str(target), monitor="all")

        assert res.get("success"), res.get("error")
        assert res["width"] > 0 and res["height"] > 0
