"""A git global option must not hide a destructive subcommand from the classifier.

`git -C . reset --hard` used to come back "safe". In /vibe a "safe" verdict runs
with nobody asked, so a leading option was effectively an off switch for the gate.
"""

import pytest

from argent.approval import assess_command_risk, is_destructive_command

DESTRUCTIVE = [
    "git reset --hard",
    "git -C . reset --hard",
    "git -C ../repo reset --hard",
    "git --no-pager reset --hard",
    "git -c core.pager=cat reset --hard",
    "git --git-dir=/tmp/x/.git reset --hard",
    "git -C . clean -fdx",
    "git --no-pager clean -fd",
    "git -C . push --force",
    "git -C . push origin :main",
    "git -C . branch -D old",
    "git -C . stash drop",
    "git --no-pager checkout -- .",
]

ORDINARY = [
    "git status",
    "git -C . status",
    "git --no-pager log --oneline -5",
    "git -C sub diff --stat",
    "git branch",
    "git stash list",
    "git push origin main",
    "git -C . reset HEAD~1",
    "git -C . add -A",
]


@pytest.mark.parametrize("cmd", DESTRUCTIVE)
def test_a_global_option_does_not_hide_a_destructive_subcommand(cmd):
    level, _ = assess_command_risk(cmd)
    assert level != "safe", cmd
    assert is_destructive_command(cmd) is True, cmd


@pytest.mark.parametrize("cmd", ORDINARY)
def test_ordinary_git_stays_safe(cmd):
    level, _ = assess_command_risk(cmd)
    assert level == "safe", cmd
