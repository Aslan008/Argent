"""The risk gate must not call code "safe" that it never understood.

In POLICY_AUTO (vibe mode) a command graded "safe" runs with no prompt, so the
grade is a promise. Inline python was graded by a blacklist of names — and
`subprocess.run`, `shutil.copyfile`, `os.makedirs`, `os.startfile`,
`getattr(os, "rem" + "ove")` and `Path(p).open(mode)` all passed. So did any
program handed to an interpreter through stdin (heredoc, here-string, pipe).

The grade is now an allowlist: everything in the snippet has to be recognised.
These tests pin both halves — the escapes that used to pass, and the everyday
read-only one-liners that must keep running without a question.
"""

import pytest

from approval import assess_command_risk


def _level(cmd: str) -> str:
    return assess_command_risk(cmd)[0]


class TestInlinePythonEscapesAreNoLongerSafe:
    @pytest.mark.parametrize("cmd", [
        # processes
        'python -c "import subprocess; subprocess.run([\'cmd\', \'/c\', \'del\', \'x\'])"',
        'python -c "import os; os.system(\'del x\')"',
        'python -c "import os; os.startfile(\'x.exe\')"',
        'python -c "import os; os.popen(\'del x\')"',
        # files
        'python -c "import shutil; shutil.copyfile(\'a\', \'b\')"',
        'python -c "import os; os.makedirs(\'d\')"',
        'python -c "from pathlib import Path; Path(\'x\').write_text(\'y\')"',
        'python -c "from pathlib import Path; Path(\'d\').mkdir()"',
        'python -c "from pathlib import Path; Path(\'x\').open(\'w\')"',
        'python -c "from pathlib import Path; Path(\'x\').open(mode=\'wb\')"',
        'python -c "from pathlib import Path; m = \'w\'; Path(\'x\').open(m)"',
        'python -c "print(open(\'x\', \'w\').write(\'a\'))"',
        'python -c "m = \'w\'; open(\'x\', m)"',
        'python -c "import numpy; numpy.save(\'x.npy\', [1])"',
        'python -c "import torch; torch.save({}, \'x.pt\')"',
        'python -c "import pandas; pandas.DataFrame().to_csv(\'x.csv\')"',
        # network and modules that were never allowed
        'python -c "import urllib.request as u; u.urlopen(\'http://example.com\')"',
        'python -c "import socket"',
        'python -c "import ctypes"',
        'python -c "import tempfile"',
        'python -c "import requests; requests.get(\'http://example.com\')"',
        'python -c "import importlib; importlib.import_module(\'subprocess\')"',
        'python -c "from subprocess import run"',
        'python -c "from os import *"',
        # dynamic lookup: the name is built at run time, so no list can see it
        'python -c "import os; getattr(os, \'rem\' + \'ove\')(\'x\')"',
        'python -c "import os; f = os.remove; f(\'x\')"',
        'python -c "import os as o; o.remove(\'x\')"',
        'python -c "from os import remove; remove(\'x\')"',
        'python -c "import operator, os; operator.attrgetter(\'remove\')(os)(\'x\')"',
        'python -c "import sys; sys.modules[\'subprocess\'].check_output([\'x\'])"',
        'python -c "eval(\'1+1\')"',
        'python -c "exec(\'pass\')"',
        'python -c "__import__(\'os\')"',
        'python -c "import os; vars(os)[\'remove\'](\'x\')"',
        'python -c "(lambda: 1)()"',
        # leaving the sandbox through dunders and private attributes
        'python -c "().__class__.__bases__[0].__subclasses__()"',
        'python -c "import sys; sys._getframe()"',
        'python -c "import os; os._exit(0)"',
        # unreadable
        'python -c "def ("',
    ])
    def test_it_asks(self, cmd):
        assert _level(cmd) == "warn", cmd


class TestEverydayOneLinersStaySafe:
    @pytest.mark.parametrize("cmd", [
        'python -c "print(1 + 1)"',
        'python -c "import sys; print(sys.version)"',
        'python -c "import os; print(os.getcwd())"',
        'python -c "import os; print(os.listdir(\'.\'))"',
        'python -c "import os; print(os.path.exists(\'x\'))"',
        'python -c "import json; print(json.load(open(\'package.json\'))[\'name\'])"',
        'python -c "import json; print(json.dumps({\'k\': 1}))"',
        'python -c "from pathlib import Path; print(Path(\'x\').read_text())"',
        'python -c "from pathlib import Path; print(sorted(Path(\'.\').glob(\'*.py\')))"',
        'python -c "from pathlib import Path; print(Path(\'x\').open().read())"',
        'python -c "import platform; print(platform.system())"',
        'python -c "import re; print(re.findall(r\'\\d+\', \'a1b22\'))"',
        'python -c "import datetime; print(datetime.datetime.now())"',
        'python -c "import hashlib; print(hashlib.sha256(b\'x\').hexdigest())"',
        'python -c "d = {\'a\': 1}; print(d.copy())"',
        'python -c "import importlib.metadata as m; print(m.version(\'pip\'))"',
        # "does it import?": installed third-party packages are trusted
        'python -c "import numpy; print(numpy.__version__)"',
        'python -c "import torch; print(torch.cuda.is_available())"',
        'python -c "import filelock; print(\'ok\')"',
        'python -u -c "import numpy; print(\'numpy ok\')"',
    ])
    def test_it_does_not_ask(self, cmd):
        assert _level(cmd) == "safe", cmd


class TestProgramsFedThroughStdin:
    @pytest.mark.parametrize("cmd", [
        # pipes: the program is whatever the left side prints
        'echo "import os; os.remove(\'x\')" | python',
        'echo "import os; os.remove(\'x\')" | python -',
        'cat evil.py | python',
        'Get-Content evil.py | python -',
        'echo "console.log(1)" | node',
        'echo "rm -rf x" | bash',
        'echo ok | sh',
        'python -',
        'python -u -',
        'python',
        # heredocs and here-strings whose source is not read-only
        "python - <<'EOF'\nimport os\nos.remove('x')\nEOF",
        "python <<EOF\nimport shutil\nshutil.rmtree('d')\nEOF",
        "@'\nimport os\nos.remove('x')\n'@ | python -",
        "@\"\nimport subprocess\nsubprocess.run(['x'])\n\"@ | python",
        # a vetted heredoc does not excuse the rest of the line
        "python - <<'EOF'\nprint(1)\nEOF\nrm -rf build",
    ])
    def test_it_asks(self, cmd):
        assert _level(cmd) == "warn", cmd

    @pytest.mark.parametrize("cmd", [
        # a script or module is a visible target
        "python script.py",
        "python -m pytest -q",
        "python -u script.py --flag",
        "python --version",
        "python -V",
        "echo hi | python script.py",
        "cat data.json | python -m json.tool",
        # read-only source in a heredoc / here-string is vetted like -c
        "python - <<'EOF'\nimport json, sys\nprint(sys.version)\nEOF",
        "python <<EOF\n# it's a comment with an apostrophe\nprint(1)\nEOF",
        "@'\nimport os\nprint(os.getcwd())\n'@ | python -",
    ])
    def test_it_does_not_ask(self, cmd):
        assert _level(cmd) == "safe", cmd


class TestVibeModeRefusesWhatItCannotVet:
    """What "warn" means in POLICY_AUTO: refused without a prompt, never run."""

    def test_an_unvetted_snippet_is_auto_denied(self, monkeypatch):
        import approval

        monkeypatch.setattr(approval, "_policy", approval.POLICY_AUTO)
        cmd = "python -c \"import subprocess; subprocess.run(['x'])\""
        destructive = approval.is_destructive_command(cmd)
        assert destructive
        assert approval.request_approval(f"run {cmd}", destructive=destructive) is False

    def test_a_read_only_snippet_runs_unprompted(self, monkeypatch):
        import approval

        monkeypatch.setattr(approval, "_policy", approval.POLICY_AUTO)
        cmd = 'python -c "import sys; print(sys.version)"'
        assert not approval.is_destructive_command(cmd)
        assert approval.request_approval(f"run {cmd}", destructive=False) is True
