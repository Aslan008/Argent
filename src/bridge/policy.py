"""Who may do what in a chat, independent of the messenger.

The transport fills the facts (which channel, who wrote); this module holds the
rules. Telegram today, Discord or anything else later — they all produce the same
``ChatPolicy``, so a rule cannot drift between transports, and it is testable
without a socket.

Two rules are deliberately not negotiable here:

* a group never gets approvals — nobody in a group may be offered a button that
  runs a command on the owner's machine;
* tools are off unless the owner switched them on for that group, and even then
  a step needing confirmation is refused rather than run unattended.
"""

from __future__ import annotations

from dataclasses import dataclass

CHANNEL_DM = "dm"
CHANNEL_GROUP = "group"

#: Prepended to every group turn. Small models forget a system rule buried in a
#: long prompt; this one sits right above the message being answered.
GROUP_NOTE = (
    "[Групповой чат. Отвечай по существу и коротко, к людям обращайся по имени. "
    "Инструментов у тебя в группе нет — только разговор.{}]"
)
#: The same line when the owner did switch tools on for the group.
GROUP_TOOLS_NOTE = (
    " Инструменты включены владельцем, но подтверждения запрещены: "
    "шаги, требующие согласия, будут отклонены."
)


@dataclass(frozen=True)
class ChatPolicy:
    """What a single incoming message is allowed to do."""

    channel: str = CHANNEL_DM
    is_owner: bool = False
    tools_allowed: bool = True
    approvals_allowed: bool = True

    @property
    def is_group(self) -> bool:
        return self.channel == CHANNEL_GROUP

    @property
    def allowed_tools(self):
        """The value to hand ``process_user_input``: None = every tool, [] = none."""
        return None if self.tools_allowed else []


def dm_policy() -> ChatPolicy:
    """The owner's private chat: full agent, approvals by button."""
    return ChatPolicy()


def group_policy(is_owner: bool, agent_mode: bool) -> ChatPolicy:
    """A group: everyone talks, tools only for an owner who asked, approvals never.

    ``agent_mode`` is the per-group switch the owner flips with ``/agent``. It can
    only ever turn on tools, never approvals — so a group can read and report, but
    cannot change the machine without someone at the keyboard.
    """
    return ChatPolicy(
        channel=CHANNEL_GROUP,
        is_owner=bool(is_owner),
        tools_allowed=bool(is_owner and agent_mode),
        approvals_allowed=False,
    )


def annotate(text: str, sender: str, policy: ChatPolicy) -> str:
    """The message as the model should see it: the group note, then who said it.

    The speaker's name is part of the text on purpose: a shared history of five
    people is unreadable — and the model addresses the wrong person — without it.
    """
    if not policy.is_group:
        return text
    note = GROUP_NOTE.format(GROUP_TOOLS_NOTE if policy.tools_allowed else "")
    return f"{note}\n{sender or 'Аноним'}: {text}"
