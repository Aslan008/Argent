"""Tests for Desktop Window Automation tools in Argent."""

import pytest
from src.automation.desktop import DesktopAutomation
from tools.desktop_tools import list_desktop_windows, capture_desktop_window


class TestDesktopAutomation:
    """Verifies targeted desktop window enumeration and capture."""

    def test_desktop_automation_availability(self):
        # On Windows it should be available
        import os
        expected = os.name == "nt"
        assert DesktopAutomation.is_available() == expected

    @pytest.mark.skipif(not DesktopAutomation.is_available(), reason="Windows only")
    def test_list_windows_returns_valid_structure(self):
        windows = DesktopAutomation.list_windows()
        assert isinstance(windows, list)
        if windows:
            first = windows[0]
            for key in ("hwnd", "title", "pid", "x", "y", "width", "height"):
                assert key in first
                assert isinstance(first["hwnd"], int)
                assert isinstance(first["title"], str)

    def test_list_desktop_windows_tool_returns_string(self):
        res = list_desktop_windows()
        assert isinstance(res, str)
        if DesktopAutomation.is_available():
            assert "visible desktop window" in res or "No visible" in res
        else:
            assert "Windows" in res

    def test_capture_nonexistent_window_returns_error(self):
        res = capture_desktop_window("NonExistentWindow_XYZ_123456789")
        assert "Error" in res
        assert "not found" in res.lower() or "windows" in res.lower()
