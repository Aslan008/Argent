from skill_manager import skill_manager


def list_skills() -> str:
    """Lists all available skills (flat markdown and SKILL.md folder skills)."""
    from src.rooms import policy as room_policy

    skills = [s for s in skill_manager.list_skills() if room_policy.skill_blocked(s["name"]) is None]
    if not skills:
        return "No skills found in the skills directory."

    output = "Available Skills:\n"
    for s in skills:
        tag = " [bundle]" if s.get("kind") == "folder" else ""
        output += f"  - {s['name']}{tag}: {s['description']}\n"
    return output


def read_skill(name: str) -> str:
    """Reads the full instructions of a specific skill."""
    from src.rooms import policy as room_policy

    blocked = room_policy.skill_blocked(name)
    if blocked:
        return f"Error: {blocked}"
    content = skill_manager.read_skill(name)
    if content:
        content = room_policy.cut_skill(name, content)
        return f"Instructions for skill '{name}':\n\n{content}"
    return f"Skill '{name}' not found."
