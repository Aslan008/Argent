import pytest

import argent.config as config


def test_a_read_only_config_does_not_crash_a_setting_change(monkeypatch):
    monkeypatch.setattr(
        "argent.atomic_io.atomic_write_text",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")),
    )
    monkeypatch.setattr(config, "_CONFIG_SAVE_WARNED", False)

    config.save_config({"model": "x"})

    assert config._CONFIG_CACHE == {"model": "x"}
    assert config._CONFIG_SAVE_WARNED is True
