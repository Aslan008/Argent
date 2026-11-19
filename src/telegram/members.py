"""Who is in this group, as far as Telegram lets a bot know.

There is no "list every member" call in the Bot API. A bot can see the member count, the
administrators, who joins (``new_chat_members``), and whoever writes to it. This roster is
exactly that — everybody we have actually seen — and it says so, instead of pretending to
know the full list. That honesty matters: a bot that claims to know the group and then
misses half of it is worse than one that says "я знаю тех, кто здесь писал".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: A name longer than this does not fit in a sentence the bot writes.
MAX_NAME = 32


@dataclass
class Member:
    id: str
    name: str
    username: str = ""
    is_admin: bool = False
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    messages: int = 0

    @property
    def handle(self) -> str:
        """How to address the person in text: @username when there is one."""
        return f"@{self.username}" if self.username else self.name


class Roster:
    """People seen in each chat, in memory. A restart forgets them (documented)."""

    def __init__(self) -> None:
        self._chats: Dict[str, Dict[str, Member]] = {}

    def note(self, chat_id: str | int, user: Dict, count_message: bool = True,
             at: Optional[float] = None) -> Optional[Member]:
        """Remember this person wrote here. Returns None for the bot itself and anon admins."""
        uid = str((user or {}).get("id") or "").strip()
        if not uid or (user or {}).get("is_bot"):
            return None
        now = time.time() if at is None else at
        people = self._chats.setdefault(str(chat_id), {})
        member = people.get(uid)
        if member is None:
            member = Member(id=uid, name=self._name_of(user), username=str(user.get("username") or ""),
                            first_seen=now, last_seen=now)
            people[uid] = member
        if user.get("username"):
            member.username = str(user["username"])
        # A display name can change; the newest one is the one people see.
        member.name = self._name_of(user) or member.name
        member.last_seen = now
        if count_message:
            member.messages += 1
        return member

    @staticmethod
    def _name_of(user: Dict) -> str:
        full = " ".join(p for p in (user.get("first_name"), user.get("last_name")) if p).strip()
        if full:
            return full[:MAX_NAME]
        username = str(user.get("username") or "").strip()
        return f"@{username}"[:MAX_NAME] if username else "Участник"

    def mark_admin(self, chat_id: str | int, user: Dict, is_admin: bool = True) -> None:
        member = self.note(chat_id, user, count_message=False)
        if member is not None:
            member.is_admin = is_admin

    def get(self, chat_id: str | int, user_id: str | int) -> Optional[Member]:
        return self._chats.get(str(chat_id), {}).get(str(user_id))

    def known(self, chat_id: str | int, limit: int = 30) -> List[Member]:
        """Everybody seen here, freshest first."""
        people = list(self._chats.get(str(chat_id), {}).values())
        people.sort(key=lambda m: -m.last_seen)
        return people[:limit]

    def count(self, chat_id: str | int) -> int:
        return len(self._chats.get(str(chat_id), {}))

    def forget(self, chat_id: str | int) -> None:
        self._chats.pop(str(chat_id), None)


def roster_text(members: List[Member], total: Optional[int] = None) -> str:
    """The list as a person would read it, with the honest limit spelled out."""
    from src.telegram.render import esc

    if not members:
        return ("👥 <b>Участники</b>\n\nЯ пока никого здесь не видел. "
                "Telegram не отдаёт ботам полный список участников — только тех, кто пишет, "
                "и тех, кто вступает.")
    lines = ["👥 <b>Кого я знаю в этом чате</b>", ""]
    for member in members:
        mark = " 👑" if member.is_admin else ""
        lines.append(f"• {esc(member.name)}" + (f" {esc(member.handle)}" if member.username else "")
                     + mark)
    lines.append("")
    lines.append(f"<i>Знаю {len(members)}"
                 + (f" из {total}" if isinstance(total, int) and total > 0 else "")
                 + ". Telegram не отдаёт ботам весь список — только тех, кто писал, "
                   "вступал или кого я проверял.</i>")
    return "\n".join(lines)


def member_text(member: Optional[Member]) -> str:
    from src.telegram.render import esc

    if member is None:
        return ("👤 Этого человека я пока не знаю: он здесь не писал при мне. "
                "Telegram не даёт ботам смотреть участников без запроса.")
    lines = [f"👤 <b>{esc(member.name)}</b>"]
    if member.username:
        lines.append(f"@{esc(member.username)}")
    lines.append(f"id: <code>{esc(member.id)}</code>")
    if member.is_admin:
        lines.append("👑 администратор")
    lines.append(f"сообщений при мне: {member.messages}")
    return "\n".join(lines)
