"""The keyword trigger: the word `ultracode` in a message switches the mode on for this session.

Like a slash command typed in words. It only ever does what `/ultracode on` does: the mode is the session's, it
is not saved, the tool stays behind the person's own approvals. The model is told that this message named it and
decides whether the message is a job (start a workflow) or a question about the mode (answer it).
"""

from __future__ import annotations

import re
from typing import Optional

from logger import get_logger

log = get_logger("ultracode")

_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
# the word itself: not part of a path, an address, an identifier or a file name
_WORD = re.compile(r"(?<![\w/\\@.\-])ultracode(?![\w/\\@\-]|\.\w)", re.I)

NOTE = (
    "The person's message names Ultracode. If it asks for a job, do not improvise: start with "
    "ultracode(action=\"start\", goal=\"<the whole job with every requirement and the checks the message names>\"). "
    "If it only asks about the mode, answer the question and start nothing."
)


def mentions(text: str) -> bool:
    """Whether `text` names Ultracode as a word, outside code and not as a slash command."""
    if not text or text.lstrip().startswith("/"):
        return False
    return _WORD.search(_CODE.sub(" ", text)) is not None


def on_user_message(text: str) -> bool:
    """Called for every message of the person, before the turn's tools and prompt are built. True when the
    message switched the mode on (or kept it on and named it)."""
    from src.agent.workflow import runtime
    runtime.keyword_turn(False)
    try:
        import config
        if not config.is_ultracode_keyword_enabled():
            return False
        from src.agent import plan_mode
        if plan_mode.PLAN.active or not mentions(text):
            return False
    except Exception:
        log.debug("keyword trigger skipped", exc_info=True)
        return False
    was_on = runtime.enabled()
    runtime.set_enabled(True)
    runtime.keyword_turn(True)
    if not was_on:
        runtime.say("[bold green]Ultracode включён[/bold green] по слову в сообщении (только в этой сессии; "
                    "/ultracode off — выключить).")
    return True


def strip(text: str) -> Optional[str]:
    """The message without the keyword, for a goal. None when the keyword is not there."""
    if not mentions(text):
        return None
    return re.sub(r"\s{2,}", " ", _WORD.sub("", text)).strip(" :,-—")
