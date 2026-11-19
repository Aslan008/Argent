"""`echo $(rm -rf dir)` runs `rm`, and the gate must see it.

The rules read the outer word (`echo`) and called the whole thing safe, so a
destructive payload inside a command substitution went through unexamined — in
/vibe, with nobody asked. The classifier now grades the substituted body too.
"""

import pytest

from argent.approval import DELETES, classify_command_risk

DESTRUCTIVE = [
    "echo $(rm -rf /tmp/x)",
    "echo `rm -rf /tmp/x`",
    "ls $(rm -rf build)",
    "git status; $(rm -rf /tmp/x)",
    'echo "$(rm -rf /tmp/x)"',
    "cat $(echo hi) && $(rm -rf /tmp/x)",
]

ORDINARY = [
    "echo hi",
    "echo $(pwd)",
    "echo $(git rev-parse HEAD)",
    'echo "today is $(date)"',
    "X=$(pwd) && echo $X",
    "echo `date`",
    "npm run $(node -p \"require('./package.json').scripts.build\")",
]


@pytest.mark.parametrize("cmd", DESTRUCTIVE)
def test_a_delete_inside_a_substitution_is_not_safe(cmd):
    risk = classify_command_risk(cmd)
    assert risk.level != "safe", cmd
    assert risk.kind == DELETES, (cmd, risk)


@pytest.mark.parametrize("cmd", ORDINARY)
def test_ordinary_substitutions_stay_safe(cmd):
    assert classify_command_risk(cmd).level == "safe", cmd


def test_nesting_is_followed():
    assert classify_command_risk("echo $(echo $(rm -rf /tmp/x))").kind == DELETES


def test_a_catastrophic_body_blocks():
    assert classify_command_risk("echo $(rm -rf /)").level == "block"


def test_unclosed_substitution_is_still_read():
    assert classify_command_risk("echo $(rm -rf /tmp/x").level != "safe"


def test_recursion_is_bounded():
    # The depth guard must stop the walk instead of looping on a self-reference.
    assert classify_command_risk("echo $(echo $(echo $(echo $(pwd))))").level == "safe"
