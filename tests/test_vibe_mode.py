"""/vibe: one switch that curates existing knobs — approval policy and
per-turn checkpoints. Nothing new, everything rewindable."""

import pytest

import approval
from src.agent import checkpoints


@pytest.fixture
def restore_policy():
    yield
    approval.set_policy(approval.POLICY_ASK)


def test_toggle_on_enables_auto_policy_and_checkpoints(monkeypatch, restore_policy):
    import main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)
    checkpoints.set_auto_checkpoint(False)

    assert main.toggle_vibe_mode(False) is True
    assert approval.get_policy() == approval.POLICY_AUTO
    assert checkpoints.auto_checkpoint_enabled() is True


def test_toggle_off_restores_ask_policy(monkeypatch, restore_policy):
    import main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)

    assert main.toggle_vibe_mode(True) is False
    assert approval.get_policy() == approval.POLICY_ASK


def test_destructive_actions_still_prompt_in_vibe(monkeypatch, restore_policy):
    """POLICY_AUTO never auto-approves destructive actions — the safety net
    /vibe relies on. Pin it here so a policy change can't silently break vibe."""
    import main
    monkeypatch.setattr(main, "print_system", lambda *a, **k: None)
    main.toggle_vibe_mode(False)                     # vibe ON

    asked = []
    approval.set_approval_backend(
        lambda action, destructive, grant_key: asked.append(action) or "deny")
    try:
        assert approval.request_approval("rm everything", destructive=True) is False
        assert asked == ["rm everything"]            # prompted, not auto-approved
        assert approval.request_approval("safe read", destructive=False) is True
        assert len(asked) == 1                       # safe one auto-approved
    finally:
        approval.reset_approval_backend()
