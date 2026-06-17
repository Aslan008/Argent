import os
import re
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ui import print_system, print_error
from config import get_skills_dir

# Agent Skills open standard (agentskills.io): a skill is a folder containing
# this file, with optional scripts/ references/ assets/ subfolders.
SKILL_FILE = "SKILL.md"
_RESOURCE_DIRS = ("scripts", "references", "assets")

# Ecosystem skills are written against Claude Code / Cursor tool names. Argent's
# tools do the same jobs under different names — surface a translation so the
# model (especially a weak one) calls a tool that actually exists here.
_TOOL_NAME_MAP = {
    "WebSearch": "search_web",
    "WebFetch": "read_webpage",
    "Read": "read_file",
    "Write": "write_file",
    "Edit": "replace_in_file",
    "MultiEdit": "multi_replace_in_file",
    "Bash": "run_command",
    "Glob": "search_files",
    "Grep": "grep_search",
    "LS": "list_directory",
    "Task": "run_subagent",
    "TodoWrite": "add_work_task",
    "NotebookEdit": "replace_in_file",
}
# These names double as ordinary English words; only treat them as a tool
# reference when the text signals it (backticked, "<Name> tool", or listed on
# an allowed-tools line). The rest are distinctive enough to match anywhere.
_AMBIGUOUS_TOOL_NAMES = {"Read", "Write", "Edit", "Task", "LS", "Bash"}


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

    # ── cross-agent tool-name translation ────────────────────────────────
    @classmethod
    def _mentions_tool(cls, text: str, name: str) -> bool:
        """Whether `text` refers to a Claude Code tool named `name` (case-sensitive)."""
        if name not in _AMBIGUOUS_TOOL_NAMES:
            return re.search(rf"\b{name}\b", text) is not None
        # ambiguous (also an English word): require an explicit tool cue
        if re.search(rf"`{name}`", text) or re.search(rf"\b{name}\b\s+tool", text):
            return True
        for line in text.splitlines():
            low = line.lower()
            if "allowed" in low and "tool" in low and re.search(rf"\b{name}\b", line):
                return True
        return False

    @classmethod
    def _tool_name_hints(cls, text: str) -> str:
        """A translation note for any foreign tool names the skill references."""
        if not text:
            return ""
        found = {cc: ar for cc, ar in _TOOL_NAME_MAP.items() if cls._mentions_tool(text, cc)}
        if not found:
            return ""
        lines = "\n".join(f"  - {cc} -> {ar}" for cc, ar in found.items())
        return ("[TOOL NAMES] This skill was written for another agent. In Argent, "
                "call the equivalent tool instead:\n" + lines)

    # ── reading ──────────────────────────────────────────────────────────
    def read_skill(self, name: str) -> Optional[str]:
        """Read a skill's instructions. For folder skills, also surface the
        allowed-tools and a manifest of bundled resources the model can use.
        Foreign (Claude Code / Cursor) tool names get an Argent translation."""
        path, kind = self._resolve(name)
        if not path:
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            return None
        meta, body = self._parse_frontmatter(raw)
        hint = self._tool_name_hints(raw)
        if kind == "flat":
            return body + (f"\n\n{hint}" if hint else "")

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
        if hint:
            sections.append(hint)
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

    # ── naming / source detection ────────────────────────────────────────
    @staticmethod
    def _sanitize_name(raw: str) -> str:
        """Turn an arbitrary skill name into a safe directory name."""
        s = re.sub(r"[^\w.-]+", "-", str(raw).strip()).strip("-.")
        return s or "imported_skill"

    @staticmethod
    def _looks_like_repo(source: str) -> bool:
        """True if `source` is a git/GitHub reference rather than a local path."""
        s = source.strip()
        if s.startswith(("http://", "https://", "git@", "ssh://")) or s.endswith(".git"):
            return True
        # owner/repo shorthand — but only if it isn't an existing local path
        if re.fullmatch(r"[\w.-]+/[\w.-]+", s) and not Path(s).expanduser().exists():
            return True
        return False

    @staticmethod
    def _parse_repo_ref(ref: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """Parse a repo reference into (clone_url, branch, subpath).

        Handles GitHub web URLs (incl. /tree/<branch>/<subpath>), owner/repo
        shorthand, SSH (git@) and generic .git URLs.
        """
        ref = ref.strip().rstrip("/")
        # GitHub web URL, optionally pointing at a branch + subfolder
        m = re.match(
            r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?"
            r"(?:/tree/([^/]+)(?:/(.*))?)?$",
            ref,
        )
        if m:
            owner, repo, branch, subpath = m.groups()
            return (f"https://github.com/{owner}/{repo}.git", branch, subpath or None)
        # SSH or explicit .git URL — clone whole repo
        if ref.startswith("git@") or ref.startswith("ssh://") or ref.endswith(".git"):
            return (ref, None, None)
        # any other http(s) git host (gitlab, bitbucket, …)
        if ref.startswith(("http://", "https://")):
            return (ref, None, None)
        # owner/repo shorthand → GitHub
        if re.fullmatch(r"[\w.-]+/[\w.-]+", ref):
            return (f"https://github.com/{ref}.git", None, None)
        return None

    def _install_skills_from_tree(self, root: Path, fallback_name: str) -> List[str]:
        """Find every SKILL.md under `root` and install each as its own skill.

        Returns a list of human-readable per-skill result messages.
        """
        root = Path(root)
        skill_files = sorted(root.rglob(SKILL_FILE))
        messages: List[str] = []
        for sm in skill_files:
            folder = sm.parent
            try:
                meta, _ = self._parse_frontmatter(sm.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            is_root = folder.resolve() == root.resolve()
            raw = meta.get("name") or (folder.name if not is_root else fallback_name)
            name = self._sanitize_name(raw)
            dest = self.skills_dir / name
            if dest.exists():
                messages.append(f"skipped '{name}' (already exists)")
                continue
            try:
                if is_root:
                    # SKILL.md sits at the repo root: copy just the skill files,
                    # never the whole checkout (.git, unrelated sources, …).
                    dest.mkdir(parents=True)
                    shutil.copy(sm, dest / SKILL_FILE)
                    for sub in _RESOURCE_DIRS:
                        if (folder / sub).is_dir():
                            shutil.copytree(folder / sub, dest / sub)
                else:
                    shutil.copytree(folder, dest)
                messages.append(f"installed '{name}'")
            except Exception as e:
                messages.append(f"failed '{name}': {e}")
        return messages

    def _import_from_repo(self, ref: str) -> str:
        """Clone a git/GitHub reference and install the SKILL.md skill(s) it holds."""
        parsed = self._parse_repo_ref(ref)
        if not parsed:
            return f"Error: could not understand repository reference '{ref}'."
        clone_url, branch, subpath = parsed
        tmp = Path(tempfile.mkdtemp(prefix="argent_skill_"))
        try:
            cmd = ["git", "clone", "--depth", "1"]
            if branch:
                cmd += ["--branch", branch]
            cmd += [clone_url, str(tmp)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            except FileNotFoundError:
                return "Error: git is not installed. Install git, or import from a local path."
            except subprocess.TimeoutExpired:
                return "Error: git clone timed out (180s)."
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                hint = detail[-1] if detail else "unknown error"
                return f"Error: git clone failed: {hint[:200]}"

            search_root = tmp / subpath if subpath else tmp
            if not search_root.exists():
                return f"Error: path '{subpath}' was not found in the repository."
            repo_name = self._sanitize_name(
                clone_url.rstrip("/").split("/")[-1].removesuffix(".git")
            )
            messages = self._install_skills_from_tree(search_root, repo_name)
            if not messages:
                return f"No {SKILL_FILE} skill found in {ref}."
            installed = sum(1 for m in messages if m.startswith("installed"))
            header = (f"Installed {installed} skill(s) from {ref}:" if installed
                      else f"From {ref} (nothing new installed):")
            return header + "\n" + "\n".join(f"  - {m}" for m in messages)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── import (Agent Skills standard) ───────────────────────────────────
    def import_skill(self, source: str) -> str:
        """Import a SKILL.md skill into the skills directory.

        Accepts a GitHub/git reference (owner/repo, a full URL, or a
        /tree/<branch>/<subfolder> URL — cloned automatically), a local folder
        containing SKILL.md (copied with its bundled scripts/references/assets),
        a standalone SKILL.md file, or a flat .md.
        """
        source = source.strip()
        if self._looks_like_repo(source):
            return self._import_from_repo(source)

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
