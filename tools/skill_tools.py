from skill_manager import skill_manager

def list_skills() -> str:
    """Lists all available markdown-based skills."""
    skills = skill_manager.list_skills()
    if not skills:
        return "No skills found. You can create one via `create_skill`."
    
    output = "Available Skills:\n"
    for s in skills:
        output += f"  - {s['name']}: {s['description']}\n"
    return output

def read_skill(name: str) -> str:
    """Reads the full instructions of a specific skill."""
    content = skill_manager.read_skill(name)
    if content:
        return f"Instructions for skill '{name}':\n\n{content}"
    return f"Skill '{name}' not found."

def create_skill(name: str, instructions: str, description: str = "") -> str:
    """Creates or updates a markdown-based skill."""
    result = skill_manager.create_skill(name, instructions, description)
    return result

def delete_skill(name: str) -> str:
    """Deletes a skill."""
    result = skill_manager.delete_skill(name)
    return result
