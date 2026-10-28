from argent import features


def test_all_features_are_off_by_default(monkeypatch):
    for name in features.FEATURES:
        monkeypatch.delenv(f"ARGENT_FEATURE_{name.upper()}", raising=False)
    monkeypatch.setattr(features, "_from_config", lambda name: None)
    assert all(features.enabled(name) is False for name in features.FEATURES)


def test_env_switches_a_feature_on_and_off(monkeypatch):
    monkeypatch.setattr(features, "_from_config", lambda name: None)
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "1")
    assert features.enabled("replay") is True
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "off")
    assert features.enabled("replay") is False


def test_config_is_used_when_env_is_absent(monkeypatch):
    monkeypatch.delenv("ARGENT_FEATURE_BENCH", raising=False)
    monkeypatch.setattr(features, "_from_config", lambda name: name == "bench")
    assert features.enabled("bench") is True
    assert features.enabled("replay") is False


def test_a_typo_does_not_enable_a_feature(monkeypatch):
    monkeypatch.setattr(features, "_from_config", lambda name: None)
    monkeypatch.setenv("ARGENT_FEATURE_REPLAY", "tru")
    assert features.enabled("replay") is False


def test_unknown_feature_is_rejected():
    import pytest

    with pytest.raises(KeyError):
        features.enabled("does-not-exist")
