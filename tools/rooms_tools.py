"""The `rooms` tool: reusable, bounded recipes. See src/rooms/tool.py."""

from logger import get_logger

log = get_logger("tools")


def rooms(action: str = "list", room: str = "", task: str = "", params=None, ops=None, save_as: str = "") -> str:
    from src.rooms.tool import rooms_action
    return rooms_action(action, room, task, params, ops=ops, save_as=save_as)
