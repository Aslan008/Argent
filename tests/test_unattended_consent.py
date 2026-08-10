"""What a present human authorised does not carry over to a run at 04:00.

runner.py promises that "an unattended run REFUSES every gated action and
records what it wanted to do". It did not. request_approval answered two
shortcuts before ever consulting the installed backend:

    if not destructive:
        if grant_key in _session_grants:   -> True, gate never called
        if _policy == POLICY_AUTO:         -> True, gate never called

Both are reachable in ordinary use. /vibe and /auto set POLICY_AUTO for the
whole process, and the scheduler runs automations on a background thread of
that same process; answering "always allow git" once in the terminal grants it
for the session. Either way a scheduled job inherited the authority — and the
run log recorded nothing, because nothing had been refused.

The existing test only covered destructive=True, which routes to the backend
under every policy and so never exercised the hole.
"""

import pytest

import approval
from src.automation.runner import UnattendedGate


def _fresh_gate():
    gate = UnattendedGate()
    approval.set_approval_backend(gate)
    return gate


@pytest.fixture(autouse=True)
def clean_approval_state():
    """Policy and grants are process globals; a leaked one silently rewrites
    the next test's premise (which is the whole subject of this file)."""
    approval.set_policy(approval.POLICY_ASK)
    approval.clear_session_grants()
    approval.reset_approval_backend()
    yield
    approval.reset_approval_backend()
    approval.set_policy(approval.POLICY_ASK)
    approval.clear_session_grants()


class TestPolicyDoesNotLeakIntoAnUnattendedRun:
    def test_auto_policy_does_not_approve_for_the_gate(self):
        """The exact bypass: /vibe is on, the scheduler fires, git push runs."""
        approval.set_policy(approval.POLICY_AUTO)
        gate = _fresh_gate()
        allowed = approval.request_approval("run command: git push origin master",
                                            destructive=False, grant_key="git")
        assert allowed is False
        assert [d["action"] for d in gate.denied] == ["run command: git push origin master"]

    def test_a_session_grant_does_not_approve_for_the_gate(self):
        """"Always allow git" was answered by a human looking at their own
        terminal, about their own command."""
        approval.set_approval_backend(lambda *a: "always")
        approval.request_approval("git status", destructive=False, grant_key="git")
        assert "git" in approval.get_session_grants()

        gate = _fresh_gate()
        allowed = approval.request_approval("run command: git push --force",
                                            destructive=False, grant_key="git")
        assert allowed is False
        assert gate.denied

    def test_destructive_still_refused(self):
        approval.set_policy(approval.POLICY_AUTO)
        gate = _fresh_gate()
        assert approval.request_approval("rm -rf build", destructive=True) is False
        assert gate.denied


class TestTheInteractiveSessionIsUnaffected:
    def test_auto_policy_still_approves_for_a_human(self):
        approval.set_policy(approval.POLICY_AUTO)
        assert approval.request_approval("run command: pytest", destructive=False) is True

    def test_a_session_grant_still_works_for_a_human(self):
        approval.set_approval_backend(lambda *a: "always")
        assert approval.request_approval("git status", destructive=False,
                                         grant_key="git") is True
        # second call must not re-prompt: the backend now refuses everything,
        # and the grant is what has to carry it.
        approval.set_approval_backend(lambda *a: "deny")
        assert approval.request_approval("git log", destructive=False,
                                         grant_key="git") is True

    def test_a_plain_backend_is_not_treated_as_unattended(self):
        approval.set_approval_backend(lambda *a: "deny")
        assert approval._nobody_is_watching() is False


class TestTheGateDeclaresItself:
    def test_the_flag_is_on_the_backend(self):
        assert UnattendedGate.unattended is True

    def test_the_runner_installs_that_backend(self):
        """A gate that is never installed protects nothing."""
        import inspect
        from src.automation import runner
        source = inspect.getsource(runner.run_automation)
        assert "set_approval_backend(gate)" in source
        assert "reset_approval_backend()" in source
