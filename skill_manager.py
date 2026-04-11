import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from ui import print_system, print_error
from config import get_skills_dir

class SkillManager:
    """Manages instruction-based skills for Argent."""
    
    def __init__(self):
        self.skills_dir = Path(get_skills_dir()).expanduser().resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def list_skills(self) -> List[Dict[str, str]]:
        """Lists all available skills with their descriptions."""
        skills = []
        for item in self.skills_dir.iterdir():
            if item.is_file() and item.suffix == ".md":
                skill_info = self._get_skill_metadata(item)
                skills.append(skill_info)
        return skills

    def read_skill(self, name: str) -> Optional[str]:
        """Reads the full instructions of a skill by its name (stem)."""
        if not name.endswith(".md"):
            name += ".md"
        file_path = self.skills_dir / name
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Strip frontmatter if present for the final instruction
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        return parts[2].strip()
                return content.strip()
        return None

    def create_skill(self, name: str, instructions: str, description: str = "") -> str:
        """Creates a new skill file."""
        if not name.endswith(".md"):
            name += ".md"
        file_path = self.skills_dir / name
        
        content = f"---\ndescription: \"{description}\"\n---\n\n{instructions}"
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Skill '{name}' created successfully in {self.skills_dir}."
        except Exception as e:
            return f"Error creating skill: {e}"

    def delete_skill(self, name: str) -> str:
        """Deletes a skill file."""
        if not name.endswith(".md"):
            name += ".md"
        file_path = self.skills_dir / name
        if file_path.exists():
            file_path.unlink()
            return f"Skill '{name}' deleted."
        return f"Skill '{name}' not found."

    def _get_skill_metadata(self, file_path: Path) -> Dict[str, str]:
        """Extracts metadata from a skill file (description from frontmatter)."""
        name = file_path.stem
        description = "No description provided."
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        metadata = yaml.safe_load(parts[1])
                        if isinstance(metadata, dict):
                            description = metadata.get("description", description)
        except Exception:
            pass
            
        return {"name": name, "description": description}

# Global instance
skill_manager = SkillManager()
