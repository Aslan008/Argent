import os
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ui import print_system, print_error
from config import get_skills_dir

# Agent Skills open standard (agentskills.io): a skill is a folder containing
# this file, with optional scripts/ references/ assets/ subfolders.
SKILL_FILE = "SKILL.md"
_RESOURCE_DIRS = ("scripts", "references", "assets")


class SkillManager:
    """Manages instruction-based skills for Argent.

    Supports two layouts:
    - flat:   skills/<name>.md                 (Argent's original simple skills)
    - folder: skills/<name>/SKILL.md (+ bundled scripts/references/assets)
              — the cross-platform Agent Skills (SKILL.md) standard.
    """

    def __init__(self):
        self.skills_dir = Path(get_skills_dir()).expanduser().resolve()
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    # ── frontmatter ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_frontmatter(content: str) -> Tuple[dict, str]:
        """Return (metadata, body) from markdown with --- YAML frontmatter."""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    meta = yaml.safe_load(parts[1])
                    meta = meta if isinstance(meta, dict) else {}
                except Exception:
                    meta = {}
                return meta, parts[2].strip()
        return {}, content.strip()

    # ── discovery / resolution ───────────────────────────────────────────
    def _resolve(self, name: str) -> Tuple[Optional[Path], Optional[str]]:
        """Locate a skill by name. Returns (markdown_path, kind) or (None, None)."""
        stem = name[:-3] if name.endswith(".md") else name
        folder = self.skills_dir / stem
        if folder.is_dir() and (folder / SKILL_FILE).exists():
            return folder / SKILL_FILE, "folder"
        flat = self.skills_dir / f"{stem}.md"
        if flat.exists():
            return flat, "flat"
        return None, None

    def list_skills(self) -> List[Dict[str, str]]:
        """List all skills (flat and folder-based) with their descriptions."""
        skills = []
        if not self.skills_dir.exists():
            return skills
        for item in sorted(self.skills_dir.iterdir(), key=lambda p: p.name.lower()):
            if item.is_file() and item.suffix == ".md" and item.name != SKILL_FILE:
                path, kind = item, "flat"
            elif item.is_dir() and (item / SKILL_FILE).exists():
                path, kind = item / SKILL_FILE, "folder"
            else:
                continue
            try:
                meta, _ = self._parse_frontmatter(path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            skills.append({
                "name": item.stem if kind == "flat" else item.name,
                "display_name": meta.get("name") or (item.stem if kind == "flat" else item.name),
                "description": meta.get("description", "No description provided."),
                "kind": kind,
            })
        return skills

    # ── reading ──────────────────────────────────────────────────────────
    def read_skill(self, name: str) -> Optional[str]:
        """Read a skill's instructions. For folder skills, also surface the
        allowed-tools and a manifest of bundled resources the model can use."""
        path, kind = self._resolve(name)
        if not path:
            return None
        try:
            meta, body = self._parse_frontmatter(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if kind == "flat":
            return body

        sections = [body]
        allowed = meta.get("allowed-tools") or meta.get("allowed_tools")
        if allowed:
            sections.append(f"[ALLOWED TOOLS]: {allowed}")
        when = meta.get("when_to_use") or meta.get("when-to-use")
        if when:
            sections.append(f"[WHEN TO USE]: {when}")

        folder = path.parent
        manifest = []
        for sub in _RESOURCE_DIRS:
            d = folder / sub
            if d.is_dir():
                files = [str((d / f).resolve()) for f in sorted(os.listdir(d)) if (d / f).is_file()]
                if files:
                    manifest.append(f"{sub}/:\n" + "\n".join(f"  - {p}" for p in files))
        if manifest:
            sections.append(
                f"[BUNDLED RESOURCES] Skill folder: {folder.resolve()}\n"
                "Read or run these with read_file / run_command when the instructions call for them:\n"
                + "\n".join(manifest)
            )
        return "\n\n".join(sections)

    # ── creation / deletion ──────────────────────────────────────────────
    def create_skill(self, name: str, instructions: str, description: str = "") -> str:
        """Create or update a flat markdown skill (SKILL.md-compatible body)."""
        stem = name[:-3] if name.endswith(".md") else name
        file_path = self.skills_dir / f"{stem}.md"
        frontmatter = yaml.safe_dump(
            {"name": stem, "description": description},
            allow_unicode=True, default_flow_style=False, sort_keys=False,
        ).strip()
        content = f"---\n{frontmatter}\n---\n\n{instructions}"
        try:
            file_path.write_text(content, encoding="utf-8")
            return f"Skill '{stem}.md' created successfully in {self.skills_dir}."
        except Exception as e:
            return f"Error creating skill: {e}"

    def delete_skill(self, name: str) -> str:
        """Delete a skill (flat file or folder)."""
        path, kind = self._resolve(name)
        if not path:
            return f"Skill '{name}' not found."
        try:
            if kind == "folder":
                shutil.rmtree(path.parent)
            else:
                path.unlink()
            return f"Skill '{name}' deleted."
        except Exception as e:
            return f"Error deleting skill: {e}"

    # ── import (Agent Skills standard) ───────────────────────────────────
    def import_skill(self, source: str) -> str:
        """Import a SKILL.md skill into the skills directory.

        Accepts a folder containing SKILL.md (copied with its bundled
        scripts/references/assets), a standalone SKILL.md file, or a flat .md.
        """
        src = Path(source).expanduser().resolve()
        if not src.exists():
            return f"Error: '{source}' does not exist."

        if src.is_dir():
            if not (src / SKILL_FILE).exists():
                return f"Error: '{src}' has no {SKILL_FILE}. Not an Agent Skills folder."
            dest = self.skills_dir / src.name
            if dest.exists():
                return f"Error: a skill named '{src.name}' already exists. Delete it first."
            shutil.copytree(src, dest)
            return f"Imported skill folder '{src.name}' (with bundled resources)."

        if src.name == SKILL_FILE:
            meta, _ = self._parse_frontmatter(src.read_text(encoding="utf-8"))
            fname = meta.get("name") or src.parent.name or "imported_skill"
            dest = self.skills_dir / fname
            if dest.exists():
                return f"Error: a skill named '{fname}' already exists."
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest / SKILL_FILE)
            return f"Imported skill '{fname}'."

        if src.suffix == ".md":
            dest = self.skills_dir / src.name
            if dest.exists():
                return f"Error: a skill named '{src.stem}' already exists."
            shutil.copy(src, dest)
            return f"Imported flat skill '{src.stem}'."

        return f"Error: '{source}' is not a skill (expected a folder with SKILL.md, a SKILL.md, or a .md file)."


# Global instance
skill_manager = SkillManager()
