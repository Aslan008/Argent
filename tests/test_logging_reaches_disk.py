"""A logger nobody attached a handler to is a silent failure factory.

Three times now the answer to "why did X happen?" was unavailable because the
module logged through `logging.getLogger("argent.something")` — a name with no
handler anywhere — instead of `logger.get_logger()`, which opens the file in
~/.argent/logs. The records were formatted and dropped.

    mcp_client.py   logging.getLogger("argent.mcp")            -> nothing on disk
    providers.py    logging.getLogger("argent.providers")      -> nothing on disk
    trimmer.py      logging.getLogger("argent.agent.trimmer")  -> nothing on disk

The trimmer one hurt most: it is the only record of whether a context compaction
summarised the conversation or gave up and hard-reset it, which is exactly what
you need when a model forgets the project after compaction.
"""

import ast
import logging
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SKIP = {".git", "venv_pyqt6", "node_modules", "__pycache__", ".pytest_cache",
         ".claude", "tests", "scripts", "desktop", "visuals"}


def _sources():
    for path in _ROOT.rglob("*.py"):
        if any(part in _SKIP for part in path.parts):
            continue
        yield path


class TestNoHandlerlessLoggers:
    def test_only_logger_py_calls_getlogger(self):
        """Every other module must go through get_logger(), which attaches the
        file handler. Direct getLogger() is how all three regressions happened.

        Parsed rather than grepped: the comments that document those three
        regressions quote the offending call, and a text match cannot tell a
        warning about a mistake from the mistake."""
        offenders = []
        for path in _sources():
            if path.name == "logger.py":
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "getLogger"):
                    offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno}")
        assert not offenders, (
            "эти модули берут logger в обход logger.get_logger() — у него нет "
            f"обработчика, записи никуда не попадут: {offenders}")

    def test_get_logger_actually_writes(self, tmp_path, monkeypatch):
        import logger as logger_mod

        monkeypatch.setattr(logger_mod, "LOG_DIR", tmp_path)
        name = "probe_writes"
        logging.getLogger(name).handlers.clear()
        log = logger_mod.get_logger(name)
        log.warning("hello")
        for handler in log.handlers:
            handler.flush()
        assert (tmp_path / f"{name}.log").read_text(encoding="utf-8").strip().endswith("hello")


class TestTheCompactionTrail:
    def test_the_trimmer_logs_to_disk(self):
        from src.agent import trimmer
        assert trimmer.log.handlers, "у логгера триммера снова нет обработчика"
        target = Path(trimmer.log.handlers[0].baseFilename)
        assert target.parent.name == "logs"

    @pytest.mark.parametrize("phrase", [
        "Context summarization timed out",
        "Hard context reset performed",
        "Summarization failed, falling back to hard reset",
    ])
    def test_the_decisions_worth_recording_are_recorded(self, phrase):
        """These three lines answer 'why does the model no longer know the
        project'. Losing any of them puts the question back out of reach."""
        text = (_ROOT / "src" / "agent" / "trimmer.py").read_text(encoding="utf-8")
        assert phrase in text
