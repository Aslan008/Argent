"""The process's Windows integrity level.

Low integrity is a sandbox token: it cannot write to ordinary user files even
when their ACL grants FullControl (no-write-up). That failure looks like random
"permission denied" in the config, the project and cargo, so the UI should say
it once instead of leaving the person to guess.
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional

_LABELS = {
    0x0000: "Untrusted",
    0x1000: "Low",
    0x2000: "Medium",
    0x3000: "High",
    0x4000: "System",
}

_cache: Optional[str] = None


def label_for_rid(rid: int) -> str:
    return _LABELS.get(int(rid), f"Unknown (0x{int(rid):x})")


def _detect() -> Optional[str]:
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        TOKEN_QUERY = 0x0008
        TOKEN_INTEGRITY_LEVEL = 25

        class SID_AND_ATTRIBUTES(ctypes.Structure):
            _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

        class TOKEN_MANDATORY_LABEL(ctypes.Structure):
            _fields_ = [("Label", SID_AND_ATTRIBUTES)]

        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)
        ]
        advapi32.OpenProcessToken.restype = wintypes.BOOL

        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)
        ):
            return None
        try:
            size = wintypes.DWORD()
            advapi32.GetTokenInformation(
                token, TOKEN_INTEGRITY_LEVEL, None, 0, ctypes.byref(size)
            )
            if not size.value:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(
                token, TOKEN_INTEGRITY_LEVEL, buf, size, ctypes.byref(size)
            ):
                return None
            label = ctypes.cast(buf, ctypes.POINTER(TOKEN_MANDATORY_LABEL)).contents
            sid = label.Label.Sid
            advapi32.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
            advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
            advapi32.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
            advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
            count = advapi32.GetSidSubAuthorityCount(sid).contents.value
            rid = advapi32.GetSidSubAuthority(sid, count - 1).contents.value
            return label_for_rid(rid)
        finally:
            kernel32.CloseHandle(token)
    except Exception:
        return None


def integrity_label() -> Optional[str]:
    """'Low', 'Medium', 'High', ... on Windows; None when unknown/not Windows."""
    global _cache
    if _cache is None:
        _cache = _detect()
    return _cache


def is_low() -> bool:
    return integrity_label() == "Low"
