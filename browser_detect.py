"""
browser_detect.py — Detect installed Chromium-based browsers on Windows.

Supports: Chrome, Yandex Browser, Edge, Brave.
All of these support Chrome DevTools Protocol (CDP) for remote automation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logger import get_logger

log = get_logger("browser_detect")


@dataclass
class BrowserInfo:
    """Information about a detected browser installation."""
    name: str            # Human-readable name (e.g. "Yandex Browser")
    key: str             # Short key for config (e.g. "yandex")
    exe_path: str        # Full path to browser executable
    user_data_dir: str   # Path to user profile data directory


# Known Chromium-based browser locations on Windows.
# Each entry: (key, display_name, list_of_possible_exe_paths, user_data_dir_relative_to_LOCALAPPDATA)
_BROWSER_REGISTRY = [
    (
        "chrome",
        "Google Chrome",
        [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
        ],
        os.path.join("Google", "Chrome", "User Data"),
    ),
    (
        "yandex",
        "Yandex Browser",
        [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Yandex", "YandexBrowser", "Application", "browser.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Yandex", "YandexBrowser", "Application", "browser.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Yandex", "YandexBrowser", "Application", "browser.exe"),
        ],
        os.path.join("Yandex", "YandexBrowser", "User Data"),
    ),
    (
        "edge",
        "Microsoft Edge",
        [
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        ],
        os.path.join("Microsoft", "Edge", "User Data"),
    ),
    (
        "brave",
        "Brave Browser",
        [
            os.path.join(os.environ.get("PROGRAMFILES", ""), "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
        ],
        os.path.join("BraveSoftware", "Brave-Browser", "User Data"),
    ),
]


def detect_browsers() -> list[BrowserInfo]:
    """Detect all installed Chromium-based browsers.

    Scans known installation paths and returns a list of BrowserInfo
    for each browser found. Order matches _BROWSER_REGISTRY priority.
    """
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    found: list[BrowserInfo] = []

    for key, display_name, exe_candidates, data_dir_rel in _BROWSER_REGISTRY:
        for exe_path in exe_candidates:
            if not exe_path or not os.path.isfile(exe_path):
                continue

            user_data_dir = os.path.join(local_app_data, data_dir_rel)
            info = BrowserInfo(
                name=display_name,
                key=key,
                exe_path=exe_path,
                user_data_dir=user_data_dir,
            )
            found.append(info)
            log.info("Detected %s at %s", display_name, exe_path)
            break  # Take first valid path for this browser

    return found


def find_browser(name: str) -> Optional[BrowserInfo]:
    """Find a specific browser by its key (chrome, yandex, edge, brave).

    Returns None if the browser is not installed.
    """
    name_lower = name.lower().strip()
    for browser in detect_browsers():
        if browser.key == name_lower:
            return browser
    return None


def get_default_browser() -> Optional[BrowserInfo]:
    """Get the first detected browser (priority: Chrome > Yandex > Edge > Brave).

    Returns None if no Chromium-based browser is found.
    """
    browsers = detect_browsers()
    return browsers[0] if browsers else None
