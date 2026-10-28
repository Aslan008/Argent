"""API keys in the OS credential store, not in a plain JSON file.

`keyring` reaches the platform's own vault: Windows Credential Manager, the
macOS Keychain, Secret Service on Linux. The config file keeps everything else
(provider, model, endpoints) and is written with the user's own permissions;
the secret itself is not in it.

This is an upgrade, not a requirement. Keyring may be absent (a minimal
install), may have no usable backend (a headless Linux, a locked keychain), or
may refuse this process (a service account, a sandbox). In every one of those
cases `set_secret` returns False and the caller keeps the key in the config
file, exactly as before — a user is never locked out of their provider because
the vault is unavailable.

Set `ARGENT_DISABLE_KEYRING=1` to force the config-file path; the test suite
does this so a run never writes into the developer's real vault.
"""

from __future__ import annotations

import os

SERVICE = "argent"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _disabled() -> bool:
    return os.environ.get("ARGENT_DISABLE_KEYRING", "").strip().lower() in _TRUTHY


def _backend():
    """The keyring module, or None when it is absent or switched off."""
    if _disabled():
        return None
    try:
        import keyring
        return keyring
    except Exception:
        return None


def _usable(keyring) -> bool:
    """A backend that is installed is not necessarily a backend that stores.

    keyring ships `fail.Keyring` and `null.Keyring` for exactly the case of "no
    usable vault"; asking it to store would raise or silently drop the secret.
    """
    try:
        backend = keyring.get_keyring()
    except Exception:
        return False
    name = f"{type(backend).__module__}.{type(backend).__name__}".lower()
    return not any(marker in name for marker in ("fail", "null", "chainer"))


def available() -> bool:
    """True when a working credential store is present."""
    keyring = _backend()
    return keyring is not None and _usable(keyring)


def get_secret(name: str) -> str | None:
    keyring = _backend()
    if keyring is None:
        return None
    try:
        return keyring.get_password(SERVICE, name) or None
    except Exception:
        return None


def set_secret(name: str, value: str) -> bool:
    """Store the secret in the vault. False means "use the config file instead"."""
    keyring = _backend()
    if keyring is None or not _usable(keyring):
        return False
    try:
        keyring.set_password(SERVICE, name, value)
        return True
    except Exception:
        return False


def delete_secret(name: str) -> None:
    keyring = _backend()
    if keyring is None:
        return
    try:
        keyring.delete_password(SERVICE, name)
    except Exception:
        # Nothing to delete, or the vault is unavailable: the caller also clears
        # the config-file copy, so a stale secret is not left behind by this.
        pass
